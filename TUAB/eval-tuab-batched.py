from dotenv import load_dotenv
from pathlib import Path
import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import os
import time

import requests


### THESE CONSTS NEED TO BE UPDATED FOR EACH DATASET OR MODEL ###

DATASET = "TUAB"
MODEL = "gemini-3.5-flash"
PROMPT = "Classify this EEG image as belonging to a normal or abnormal brain and provide a text rationale."
ENV_API_KEY_NAME = "GEMINI_API_KEY"
CLASSIFICATION_POSSIBILITIES = ["normal", "abnormal"]

#################################################################

CSV_PATH = f"{MODEL}_batch_results_{DATASET}.csv"
DATA_ROOT = Path("data")
STATE_DIR = Path(".batch-tuab")
STATE_PATH = STATE_DIR / f"{MODEL}_{DATASET}_batch_state.json"

REQUEST_TIMEOUT_SECONDS = 120
NUM_RETRIES = 5
RETRY_DELAY = 1
POLL_INTERVAL_SECONDS = 30
BATCH_MAX_BODY_BYTES = 18 * 1024 * 1024  # Inline Batch API requests must stay under 20MB.
RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}
SUCCEEDED_STATES = {"JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED"}
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "BATCH_STATE_SUCCEEDED",
    "BATCH_STATE_FAILED",
    "BATCH_STATE_CANCELLED",
    "BATCH_STATE_EXPIRED",
}

BATCH_CREATE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:batchGenerateContent"
BATCH_GET_URL = "https://generativelanguage.googleapis.com/v1beta/{batch_name}"
BATCH_DOWNLOAD_URL = "https://generativelanguage.googleapis.com/download/v1beta/{file_name}:download?alt=media"

EEG_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {"type": "string", "enum": CLASSIFICATION_POSSIBILITIES},
        "rationale": {"type": "string"},
    },
    "required": ["classification", "rationale"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TUAB EEG image evals with Gemini Batch API.")
    parser.add_argument("--submit-only", action="store_true", help="Submit batch jobs, then exit without polling.")
    parser.add_argument("--poll-only", action="store_true", help="Only poll previously submitted jobs in the state file.")
    parser.add_argument("--poll-interval", type=int, default=POLL_INTERVAL_SECONDS, help="Seconds between polling attempts.")
    return parser.parse_args()


def get_api_key() -> str:
    api_key = os.getenv(ENV_API_KEY_NAME)
    if not api_key:
        raise RuntimeError(f"Set {ENV_API_KEY_NAME} in your environment or .env file before running this script.")
    return api_key


def image_to_b64(image_path: str | Path) -> dict:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/png"

    image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    return {"mimeType": mime_type, "data": image_b64}


def initialize_csv() -> None:
    csv_path = Path(CSV_PATH)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "actual", "predicted", "correct", "text_rationale", "api_json"])


def load_processed_paths() -> set[str]:
    processed_paths: set[str] = set()
    csv_path = Path(CSV_PATH)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return processed_paths

    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            path = row.get("path")
            if path:
                processed_paths.add(path)
    return processed_paths


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"jobs": [], "saved_job_names": []}

    with open(STATE_PATH, "r", encoding="utf-8") as file:
        state = json.load(file)

    state.setdefault("jobs", [])
    state.setdefault("saved_job_names", [])
    return state


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


def request_key(image_path: Path, actual: str) -> str:
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:16]
    return f"{actual}-{digest}"


def build_generate_content_request(image_path: Path) -> dict:
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": PROMPT},
                    {"inlineData": image_to_b64(image_path)},
                ],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": EEG_JSON_SCHEMA,
        },
    }


def compact_json_bytes(data: dict) -> bytes:
    return json.dumps(data, separators=(",", ":")).encode("utf-8")


def iter_batch_chunks(pending_images: list[dict]):
    current_chunk: list[dict] = []
    current_size = 0

    for image_info in pending_images:
        image_path = Path(image_info["path"])
        item = {
            "key": image_info["key"],
            "request": build_generate_content_request(image_path),
            "metadata": {"path": image_info["path"], "actual": image_info["actual"]},
        }

        item_size = len(compact_json_bytes(item)) + 2
        if item_size > BATCH_MAX_BODY_BYTES:
            raise RuntimeError(f"Single batch request is larger than {BATCH_MAX_BODY_BYTES} bytes: {image_info['path']}")

        if current_chunk and current_size + item_size > BATCH_MAX_BODY_BYTES:
            yield current_chunk
            current_chunk = []
            current_size = 0

        current_chunk.append(item)
        current_size += item_size

    if current_chunk:
        yield current_chunk


def gemini_request(method: str, url: str, api_key: str, **kwargs) -> requests.Response:
    headers = kwargs.pop("headers", {})
    headers["x-goog-api-key"] = api_key

    for attempt in range(NUM_RETRIES + 1):
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
                **kwargs,
            )
        except requests.RequestException:
            if attempt < NUM_RETRIES:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            raise

        if response.status_code in RETRYABLE_HTTP_STATUS_CODES and attempt < NUM_RETRIES:
            time.sleep(RETRY_DELAY * (attempt + 1))
            continue

        response.raise_for_status()
        return response

    raise RuntimeError("Request failed without returning a response")


def extract_batch_name(create_response: dict) -> str:
    name = create_response.get("name")
    if name:
        return name

    metadata_name = create_response.get("metadata", {}).get("name")
    if metadata_name:
        return metadata_name

    raise RuntimeError(f"Could not find batch job name in create response: {create_response}")


def submit_batch_job(batch_items: list[dict], api_key: str, job_index: int) -> dict:
    request_items = [
        {
            "request": item["request"],
            "metadata": {"key": item["key"]},
        }
        for item in batch_items
    ]
    body = {
        "batch": {
            "display_name": f"eval-{DATASET}-{MODEL}-{int(time.time())}-{job_index}",
            "input_config": {"requests": {"requests": request_items}},
        }
    }

    response = gemini_request(
        "POST",
        BATCH_CREATE_URL,
        api_key,
        data=compact_json_bytes(body),
        headers={"Content-Type": "application/json"},
    )
    response_json = response.json()
    batch_name = extract_batch_name(response_json)

    return {
        "name": batch_name,
        "created_at": time.time(),
        "requests": [
            {
                "key": item["key"],
                "path": item["metadata"]["path"],
                "actual": item["metadata"]["actual"],
            }
            for item in batch_items
        ],
    }


def collect_pending_images(processed_paths: set[str], in_flight_paths: set[str]) -> list[dict]:
    images: list[dict] = []

    for category in CLASSIFICATION_POSSIBILITIES:
        data_dir = DATA_ROOT / category
        if not data_dir.exists():
            print(f"Skipping missing data directory: {data_dir}")
            continue

        for image_path in sorted(path for path in data_dir.iterdir() if path.is_file()):
            path_string = str(image_path)
            if path_string in processed_paths or path_string in in_flight_paths:
                continue

            images.append({"key": request_key(image_path, category), "path": path_string, "actual": category})

    return images


def submit_pending_jobs(state: dict, processed_paths: set[str], api_key: str) -> None:
    saved_job_names = set(state["saved_job_names"])
    in_flight_paths = {
        request_info["path"]
        for job in state["jobs"]
        if job["name"] not in saved_job_names
        for request_info in job.get("requests", [])
    }

    pending_images = collect_pending_images(processed_paths, in_flight_paths)
    if not pending_images:
        print("No new images to submit.")
        return

    print(f"Submitting {len(pending_images)} image(s) with Batch API.")
    submitted_jobs = 0
    for index, chunk in enumerate(iter_batch_chunks(pending_images), start=1):
        job = submit_batch_job(chunk, api_key, index)
        state["jobs"].append(job)
        save_state(state)
        submitted_jobs += 1
        print(f"Submitted {job['name']} with {len(job['requests'])} image(s).")

    print(f"Submitted {submitted_jobs} batch job(s).")


def extract(api_json: dict) -> tuple[str, str]:
    parts = api_json["candidates"][0]["content"]["parts"]
    content = "".join(part.get("text", "") for part in parts)
    result = json.loads(content)
    return result["classification"], result["rationale"]


def save_prediction(image_path: str, actual: str, api_json: dict, processed_paths: set[str]) -> None:
    if image_path in processed_paths:
        return

    predicted, text_rationale = extract(api_json)
    correct = predicted == actual

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            image_path,
            actual,
            predicted,
            correct,
            text_rationale,
            json.dumps(api_json, ensure_ascii=False),
        ])

    processed_paths.add(image_path)


def save_error(image_path: str, actual: str, error: object, processed_paths: set[str]) -> None:
    if image_path in processed_paths:
        return

    error_text = error if isinstance(error, str) else json.dumps(error, ensure_ascii=False)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            image_path,
            actual,
            "ERROR",
            False,
            error_text,
            json.dumps({"error": error}, ensure_ascii=False),
        ])

    processed_paths.add(image_path)


def get_batch_status(batch_name: str, api_key: str) -> dict:
    response = gemini_request("GET", BATCH_GET_URL.format(batch_name=batch_name), api_key)
    return response.json()


def get_batch_state(status: dict) -> str | None:
    state = status.get("metadata", {}).get("state") or status.get("state")
    if state:
        return state
    if status.get("done") and status.get("response"):
        return "JOB_STATE_SUCCEEDED"
    if status.get("done") and status.get("error"):
        return "JOB_STATE_FAILED"
    return None


def download_results_file(file_name: str, api_key: str) -> list[dict]:
    response = gemini_request("GET", BATCH_DOWNLOAD_URL.format(file_name=file_name), api_key)
    lines = response.content.decode("utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def response_key(inline_response: dict, fallback_request: dict | None) -> str | None:
    return (
        inline_response.get("metadata", {}).get("key")
        or inline_response.get("key")
        or (fallback_request or {}).get("key")
    )


def save_inline_results(job: dict, inlined_responses: list[dict], processed_paths: set[str]) -> None:
    requests_by_key = {request_info["key"]: request_info for request_info in job["requests"]}
    seen_keys: set[str] = set()

    for index, inline_response in enumerate(inlined_responses):
        fallback_request = job["requests"][index] if index < len(job["requests"]) else None
        key = response_key(inline_response, fallback_request)
        request_info = requests_by_key.get(key) if key else fallback_request
        if not request_info:
            print(f"Could not match batch response to a request: {inline_response}")
            continue

        seen_keys.add(request_info["key"])
        if inline_response.get("error"):
            save_error(request_info["path"], request_info["actual"], inline_response["error"], processed_paths)
        else:
            try:
                save_prediction(request_info["path"], request_info["actual"], inline_response["response"], processed_paths)
            except Exception as error:
                save_error(request_info["path"], request_info["actual"], str(error), processed_paths)

    for request_info in job["requests"]:
        if request_info["key"] not in seen_keys:
            save_error(request_info["path"], request_info["actual"], "No response returned for request", processed_paths)


def save_file_results(job: dict, file_name: str, api_key: str, processed_paths: set[str]) -> None:
    requests_by_key = {request_info["key"]: request_info for request_info in job["requests"]}
    seen_keys: set[str] = set()

    for line in download_results_file(file_name, api_key):
        key = line.get("key") or line.get("metadata", {}).get("key")
        request_info = requests_by_key.get(key)
        if not request_info:
            print(f"Could not match result line to a request: {line}")
            continue

        seen_keys.add(key)
        if line.get("error"):
            save_error(request_info["path"], request_info["actual"], line["error"], processed_paths)
        else:
            try:
                api_json = line.get("response", line)
                save_prediction(request_info["path"], request_info["actual"], api_json, processed_paths)
            except Exception as error:
                save_error(request_info["path"], request_info["actual"], str(error), processed_paths)

    for request_info in job["requests"]:
        if request_info["key"] not in seen_keys:
            save_error(request_info["path"], request_info["actual"], "No response returned for request", processed_paths)


def normalize_inlined_responses(inlined_responses: object) -> list[dict] | None:
    if isinstance(inlined_responses, list):
        return inlined_responses
    if isinstance(inlined_responses, dict):
        nested = inlined_responses.get("inlinedResponses") or inlined_responses.get("inlined_responses")
        if isinstance(nested, list):
            return nested
    return None


def normalize_responses_file(responses_file: object) -> str | None:
    if isinstance(responses_file, str):
        return responses_file
    if isinstance(responses_file, dict):
        name = responses_file.get("name")
        if isinstance(name, str):
            return name
    return None


def save_succeeded_job_results(job: dict, status: dict, api_key: str, processed_paths: set[str]) -> None:
    response = status.get("response", {})
    inlined_responses = normalize_inlined_responses(
        response.get("inlinedResponses") or response.get("inlined_responses")
    )
    responses_file = normalize_responses_file(response.get("responsesFile") or response.get("responses_file"))

    if inlined_responses is not None:
        save_inline_results(job, inlined_responses, processed_paths)
    elif responses_file:
        save_file_results(job, responses_file, api_key, processed_paths)
    else:
        for request_info in job["requests"]:
            save_error(request_info["path"], request_info["actual"], f"No batch results found in status: {status}", processed_paths)


def save_failed_job_results(job: dict, status: dict, processed_paths: set[str]) -> None:
    state = get_batch_state(status)
    error = status.get("error") or f"Batch job ended with state {state}"
    for request_info in job["requests"]:
        save_error(request_info["path"], request_info["actual"], error, processed_paths)


def poll_jobs(state: dict, processed_paths: set[str], api_key: str, poll_interval: int) -> None:
    while True:
        saved_job_names = set(state["saved_job_names"])
        unsaved_jobs = [job for job in state["jobs"] if job["name"] not in saved_job_names]

        if not unsaved_jobs:
            print("No unsaved batch jobs left.")
            return

        made_progress = False
        for job in unsaved_jobs:
            status = get_batch_status(job["name"], api_key)
            state_name = get_batch_state(status)
            print(f"{job['name']}: {state_name or 'UNKNOWN'}")

            if state_name in SUCCEEDED_STATES:
                save_succeeded_job_results(job, status, api_key, processed_paths)
                state["saved_job_names"].append(job["name"])
                save_state(state)
                made_progress = True
                print(f"Saved results for {job['name']}.")
            elif state_name in TERMINAL_STATES:
                save_failed_job_results(job, status, processed_paths)
                state["saved_job_names"].append(job["name"])
                save_state(state)
                made_progress = True
                print(f"Saved error rows for {job['name']}.")

        if not made_progress:
            print(f"No jobs complete yet. Sleeping {poll_interval} seconds.")
            time.sleep(poll_interval)


def main() -> None:
    args = parse_args()
    load_dotenv()
    api_key = get_api_key()

    initialize_csv()
    processed_paths = load_processed_paths()
    state = load_state()

    if not args.poll_only:
        submit_pending_jobs(state, processed_paths, api_key)

    if args.submit_only:
        print(f"Submitted jobs are tracked in {STATE_PATH}")
        return

    poll_jobs(state, processed_paths, api_key, args.poll_interval)
    print("Finished job")


if __name__ == "__main__":
    main()
