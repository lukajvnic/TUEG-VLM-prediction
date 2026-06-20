from dotenv import load_dotenv
import os
import csv
import json
import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
import base64

### THESE CONSTS NEED TO BE UPDATED FOR EACH DATASET OR MODEL ###

DATASET = "TUAB"
MODEL = "gemini-3.5-flash"
PROMPT = "Classify this EEG image as belonging to a normal or abnormal brain and provide a text rationale."
ENV_API_KEY_NAME = "GEMINI_API_KEY"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={{key}}"
CLASSIFICATION_POSSIBILITIES = ["normal", "abnormal"]

#################################################################

CSV_PATH = f"{MODEL}_results_{DATASET}.csv"
REQUEST_TIMEOUT_SECONDS = 120
NUM_WORKERS = 4
NUM_RETRIES = 5
RETRY_DELAY = 1
RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}

EEG_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {"type": "string", "enum": CLASSIFICATION_POSSIBILITIES},
        "rationale": {"type": "string"},
    },
    "required": ["classification", "rationale"],
}


def image_to_b64(image_path: str) -> dict:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"

    image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    return {"mimeType": mime_type, "data": image_b64}


def initialize_csv() -> None:
    if Path(CSV_PATH).exists():
        return

    with open(CSV_PATH, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["path", "actual", "predicted", "correct", "text_rationale", "api_json"])


def load_processed_paths() -> set[str]:
    processed_paths: set[str] = set()
    with open(CSV_PATH, 'r') as file:
        for row in csv.DictReader(file):
            path = row.get("path")
            if path:
                processed_paths.add(path)
    return processed_paths


def extract(api_json: dict) -> tuple[str, str]:
    parts = api_json["candidates"][0]["content"]["parts"]
    content = "".join(part.get("text", "") for part in parts)
    result = json.loads(content)
    return result["classification"], result["rationale"]


def save_prediction(image_path: str, actual: str, api_json: dict) -> None:
    predicted, text_rationale = extract(api_json)
    correct = predicted == actual
    
    with open(CSV_PATH, "a") as file:
        writer = csv.writer(file)
        writer.writerow([
            image_path,
            actual,
            predicted,
            correct,
            text_rationale,
            json.dumps(api_json)
        ])


def predict(image_path: Path, actual: str) -> dict:
    # this payload might also change depending on the model
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": PROMPT},
                    {"inlineData": image_to_b64(str(image_path))},
                ],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": EEG_JSON_SCHEMA,
        },
    }

    api_url = API_URL.format(key=os.getenv(ENV_API_KEY_NAME))

    for attempt in range(NUM_RETRIES + 1):
        try:
            response = requests.post(api_url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException:
            if attempt < NUM_RETRIES:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            raise

        if response.status_code in RETRYABLE_HTTP_STATUS_CODES and attempt < NUM_RETRIES:
            time.sleep(RETRY_DELAY * (attempt + 1))
            continue

        response.raise_for_status()
        return response.json()

    raise RuntimeError("Request failed without returning a response")


def process_dataset(category: str, processed_paths: set[str]) -> None:
    data_dir = Path("data") / category
    image_paths = sorted(path for path in data_dir.iterdir() if path.is_file())
    pending_paths = [path for path in image_paths if str(path) not in processed_paths]
    completed = len(image_paths) - len(pending_paths)

    if not pending_paths:
        return

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(predict, image_path, category): image_path for image_path in pending_paths}

        for future in as_completed(futures):
            image_path = futures[future]

            try:
                api_json = future.result()
                save_prediction(str(image_path), category, api_json)
            except Exception as error:
                with open(CSV_PATH, "a") as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        str(image_path),
                        category,
                        "ERROR",
                        False,
                        str(error),
                        json.dumps({"error": str(error)}),
                    ])
                print(f"Error on {image_path}: {error}")

            completed += 1
            print(f"({completed}/{len(image_paths)}) {category}: {image_path.name}")


def main():
    load_dotenv()
    initialize_csv()

    processed_paths = load_processed_paths()
    for category in CLASSIFICATION_POSSIBILITIES:
        process_dataset(category, processed_paths)

    print("Finished job")


if __name__ == "__main__":
    main()

