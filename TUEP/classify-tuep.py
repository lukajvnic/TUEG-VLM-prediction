import argparse
import base64
import csv
import http.client
import json
import mimetypes
import os
import random
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
PREDICTION_DETAILS_CSV = "prediction_details.csv"
REQUEST_TIMEOUT_SECONDS = 120
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BASE_DELAY_SECONDS = 1.0
DEFAULT_RETRY_MAX_DELAY_SECONDS = 60.0
RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}


EEG_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {"type": "string", "enum": ["epilepsy", "no_epilepsy"]},
        "rationale": {"type": "string"},
    },
    "required": ["classification", "rationale"],
}


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_gemini_api_key() -> str:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment or in a .env file before running this script.")
    return api_key


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify TUEP EEG images with Gemini.")
    parser.add_argument(
        "--max-workers",
        type=positive_int,
        default=env_int("GEMINI_CONCURRENCY", DEFAULT_MAX_WORKERS),
        help=f"Number of concurrent Gemini requests. Default: GEMINI_CONCURRENCY or {DEFAULT_MAX_WORKERS}.",
    )
    parser.add_argument(
        "--max-retries",
        type=nonnegative_int,
        default=env_int("GEMINI_MAX_RETRIES", DEFAULT_MAX_RETRIES),
        help=f"Retries per image for 429/5xx/network failures. Default: GEMINI_MAX_RETRIES or {DEFAULT_MAX_RETRIES}.",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=positive_float,
        default=env_float("GEMINI_RETRY_BASE_DELAY", DEFAULT_RETRY_BASE_DELAY_SECONDS),
        help=f"Initial retry delay in seconds. Default: GEMINI_RETRY_BASE_DELAY or {DEFAULT_RETRY_BASE_DELAY_SECONDS}.",
    )
    parser.add_argument(
        "--retry-max-delay",
        type=positive_float,
        default=env_float("GEMINI_RETRY_MAX_DELAY", DEFAULT_RETRY_MAX_DELAY_SECONDS),
        help=f"Maximum retry delay in seconds. Default: GEMINI_RETRY_MAX_DELAY or {DEFAULT_RETRY_MAX_DELAY_SECONDS}.",
    )
    return parser.parse_args()


def image_inline_data(image_path: str) -> dict:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"

    image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    return {"mimeType": mime_type, "data": image_b64}


def retry_sleep_seconds(
    attempt: int,
    base_delay: float,
    max_delay: float,
    retry_after: str | None = None,
) -> float:
    if retry_after:
        try:
            return min(max_delay, max(0.0, float(retry_after)))
        except ValueError:
            pass

    exponential_delay = min(max_delay, base_delay * (2**attempt))
    return exponential_delay * random.uniform(0.5, 1.5)


def predict(
    model: str,
    prompt: str,
    image_path: str | Path,
    api_key: str,
    max_retries: int,
    retry_base_delay: float,
    retry_max_delay: float,
):
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inlineData": image_inline_data(str(image_path))},
                ],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": EEG_JSON_SCHEMA,
        },
    }

    api_url = f"{GEMINI_API_URL.format(model=model)}?key={api_key}"
    request_body = json.dumps(payload).encode("utf-8")

    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            api_url,
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            retryable = error.code in RETRYABLE_HTTP_STATUS_CODES
            if retryable and attempt < max_retries:
                retry_after = error.headers.get("Retry-After") if error.headers else None
                time.sleep(retry_sleep_seconds(attempt, retry_base_delay, retry_max_delay, retry_after))
                continue
            raise RuntimeError(f"Gemini request failed ({error.code}): {details}") from error
        except (
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
            http.client.HTTPException,
            ConnectionError,
        ) as error:
            if attempt < max_retries:
                time.sleep(retry_sleep_seconds(attempt, retry_base_delay, retry_max_delay))
                continue
            raise RuntimeError(f"Gemini request failed after {max_retries + 1} attempt(s): {error}") from error

    parts = data["candidates"][0]["content"]["parts"]
    content = "".join(part.get("text", "") for part in parts)

    result = json.loads(content)
    if result.get("classification") not in {"epilepsy", "no_epilepsy"}:
        raise ValueError(f"Invalid classification returned: {result}")
    return result, data


def evaluate(prediction, actual):
    return (1, 0) if prediction["classification"] == actual else (0, 1)


def initialize_details_csv(csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "filename", "actual", "prediction", "correct", "text_rationale", "api_json"])


def load_existing_details(csv_path: str | Path) -> tuple[set[str], dict[str, dict[str, int]]]:
    csv_path = Path(csv_path)
    processed_paths: set[str] = set()
    counts = {
        "epilepsy": {"correct": 0, "incorrect": 0},
        "no_epilepsy": {"correct": 0, "incorrect": 0},
    }

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return processed_paths, counts

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = row.get("path")
            actual = row.get("actual")
            if path:
                processed_paths.add(path)
            if actual in counts:
                if row.get("correct") == "True":
                    counts[actual]["correct"] += 1
                else:
                    counts[actual]["incorrect"] += 1

    return processed_paths, counts


def save_prediction_details(image_path: str | Path, actual: str, prediction: dict | None, api_json: dict) -> None:
    image_path = Path(image_path)
    predicted = prediction["classification"] if prediction else "ERROR"
    text_rationale = prediction.get("rationale", "") if prediction else ""
    correct = prediction is not None and predicted == actual

    with open(PREDICTION_DETAILS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            str(image_path),
            image_path.name,
            actual,
            predicted,
            correct,
            text_rationale,
            json.dumps(api_json, ensure_ascii=False),
        ])


def classify_image(
    image_path: Path,
    actual: str,
    model: str,
    prompt: str,
    api_key: str,
    max_retries: int,
    retry_base_delay: float,
    retry_max_delay: float,
) -> dict:
    try:
        prediction, api_json = predict(
            model,
            prompt,
            image_path,
            api_key,
            max_retries,
            retry_base_delay,
            retry_max_delay,
        )
        correct, incorrect = evaluate(prediction, actual)
    except Exception as error:
        prediction = None
        api_json = {"error": str(error)}
        correct, incorrect = 0, 1

    return {
        "image_path": image_path,
        "actual": actual,
        "prediction": prediction,
        "api_json": api_json,
        "correct": correct,
        "incorrect": incorrect,
    }


def process_dataset(
    actual: str,
    data_dir: str | Path,
    model: str,
    prompt: str,
    api_key: str,
    processed_paths: set[str],
    max_workers: int,
    max_retries: int,
    retry_base_delay: float,
    retry_max_delay: float,
) -> tuple[int, int]:
    image_paths = sorted(path for path in Path(data_dir).iterdir() if path.is_file())
    pending_paths = [path for path in image_paths if str(path) not in processed_paths]
    skipped = len(image_paths) - len(pending_paths)
    completed = skipped
    correct_total = 0
    incorrect_total = 0

    print(f"Starting {actual}: {len(pending_paths)} pending, {skipped} skipped, {len(image_paths)} total")
    if not pending_paths:
        return correct_total, incorrect_total

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                classify_image,
                image_path,
                actual,
                model,
                prompt,
                api_key,
                max_retries,
                retry_base_delay,
                retry_max_delay,
            ): image_path
            for image_path in pending_paths
        }

        for future in as_completed(futures):
            image_path = futures[future]
            result = future.result()

            # CSV writes happen only here, in the main thread.
            save_prediction_details(
                result["image_path"],
                result["actual"],
                result["prediction"],
                result["api_json"],
            )

            correct_total += result["correct"]
            incorrect_total += result["incorrect"]
            completed += 1

            if result["prediction"] is None:
                print(f"Error on {image_path}: {result['api_json'].get('error')}")
            print(f"({completed}/{len(image_paths)}) {actual}: {image_path.name}")

    return correct_total, incorrect_total


def main():
    load_dotenv()
    args = parse_args()

    model = "gemini-3.5-flash"
    prompt = "Classify this EEG image as epilepsy or no_epilepsy and provide a text rationale."
    api_key = get_gemini_api_key()

    initialize_details_csv(PREDICTION_DETAILS_CSV)
    processed_paths, existing_counts = load_existing_details(PREDICTION_DETAILS_CSV)
    epilepsy_correct = existing_counts["epilepsy"]["correct"]
    epilepsy_incorrect = existing_counts["epilepsy"]["incorrect"]
    no_epilepsy_correct = existing_counts["no_epilepsy"]["correct"]
    no_epilepsy_incorrect = existing_counts["no_epilepsy"]["incorrect"]

    print(f"Evaluating {model}")
    print(f"Concurrency: {args.max_workers}; max retries/image: {args.max_retries}")

    correct, incorrect = process_dataset(
        "epilepsy",
        "data/epilepsy",
        model,
        prompt,
        api_key,
        processed_paths,
        args.max_workers,
        args.max_retries,
        args.retry_base_delay,
        args.retry_max_delay,
    )
    epilepsy_correct += correct
    epilepsy_incorrect += incorrect

    correct, incorrect = process_dataset(
        "no_epilepsy",
        "data/no-epilepsy",
        model,
        prompt,
        api_key,
        processed_paths,
        args.max_workers,
        args.max_retries,
        args.retry_base_delay,
        args.retry_max_delay,
    )
    no_epilepsy_correct += correct
    no_epilepsy_incorrect += incorrect

    total = epilepsy_correct + epilepsy_incorrect + no_epilepsy_correct + no_epilepsy_incorrect
    print("Total", total)
    print("Epilepsy Correct:", epilepsy_correct)
    print("Epilepsy Incorrect:", epilepsy_incorrect)
    print("No Epilepsy Correct:", no_epilepsy_correct)
    print("No Epilepsy Incorrect:", no_epilepsy_incorrect)

    accuracy = (epilepsy_correct + no_epilepsy_correct) / total if total else 0
    print("Accuracy:", accuracy)

    with open("final_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "total", "epilepsy_correct", "epilepsy_incorrect", "no_epilepsy_correct", "no_epilepsy_incorrect", "accuracy"])
        writer.writerow([model, total, epilepsy_correct, epilepsy_incorrect, no_epilepsy_correct, no_epilepsy_incorrect, accuracy])


if __name__ == "__main__":
    main()
