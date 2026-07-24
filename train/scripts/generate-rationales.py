'''
This program takes one LLM model, provides the image and a prompt containing the ground truth answer, and tells it to create a text rationale.
It will save it to rationales.csv (?) under each of the datasets (e.g. datasets/TUEP/rationales.csv)
this will just be a copy of labels.csv with an additional column "rationale" that is generated based on the prior information.
'''

import base64
import csv
import mimetypes
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


MODEL = "qwen2.5vl:3b"
# The 100–120 word target usually fits in this, with room for labels.
MAX_OUTPUT_TOKENS = 256

PROMPTS = {
    "TUAB": "The EEG is labeled {labels}. Write a rationale for this classification using visible EEG features. State the label explicitly.",
    "TUAR": "The EEG contains these artifacts: {labels}. Write a rationale explaining why each listed artifact is visible. State every label explicitly.",
    "TUEP": "The EEG is from a person labeled {labels}. Write a rationale for this classification using visible EEG features. State the label explicitly.",
    "TUEV": "The EEG contains these events: {labels}. Write a rationale explaining why each listed event is visible. State every label explicitly.",
    "TUSL": "The EEG contains these events: {labels}. Write a rationale explaining why each listed event is visible. State every label explicitly.",
    "TUSZ": "The EEG contains these seizure labels: {labels}. Write a rationale explaining why each listed seizure type is visible. State every label explicitly.",
}

REPORT_FORMAT = (
    "Write a focused 3–4 sentence EEG mini-report (100–120 words). State every label "
    "explicitly. Describe only visible background, morphology, distribution, timing, or "
    "artifacts. No heading, repetition, or unsupported findings."
)

def get_datasets():
    datasets = Path(__file__).parents[2] / "datasets"
    return sorted(path for path in datasets.iterdir() if (path / "labels.csv").is_file())

def create_prompt(dataset, row):
    labels = ", ".join(label for label, value in row.items() if value.strip().lower() == "true") or "none"
    return f"{PROMPTS[Path(dataset).name].format(labels=labels)} {REPORT_FORMAT}"

def get_completed(dataset):
    with (dataset / "rationales.csv").open(newline="", encoding="utf-8") as file:
        return {
            row["image_path"]
            for row in csv.DictReader(file)
            if row["ground_truth_rationales"].strip()
        }


def get_pending_count():
    total = 0
    for dataset in get_datasets():
        completed = get_completed(dataset)
        with (dataset / "labels.csv").open(newline="", encoding="utf-8") as file:
            total += sum(
                row["image_path"] not in completed and (dataset / row["image_path"]).is_file()
                for row in csv.DictReader(file)
            )
    return total


def build_batch():
    for dataset in get_datasets():
        completed = get_completed(dataset)
        with (dataset / "labels.csv").open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["image_path"] in completed:
                    continue
                path = dataset / row["image_path"]
                if not path.is_file():
                    continue
                try:
                    mime_type = mimetypes.guess_type(path)[0] or "image/png"
                    image = base64.b64encode(path.read_bytes()).decode()
                except Exception as error:
                    print(f"Skipping unreadable image {path}: {error}", file=sys.stderr, flush=True)
                    continue
                yield path, [HumanMessage(content=[
                    {"type": "text", "text": create_prompt(dataset, row)},
                    {"type": "image_url", "image_url": f"data:{mime_type};base64,{image}"},
                ])]

def init_model():
    return ChatOllama(model=MODEL, num_predict=MAX_OUTPUT_TOKENS)


def get_dataset(path):
    return next(parent for parent in path.parents if (parent / "labels.csv").is_file())


def load_rationales(dataset):
    with (dataset / "rationales.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return reader.fieldnames, list(reader)


def save_rationales(dataset, fields, rows):
    path = dataset / "rationales.csv"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def send(batch, total):
    model = init_model()
    rationales = {}

    for index, (path, messages) in enumerate(batch, start=1):
        dataset = get_dataset(path)
        print(f"[{index}/{total}] {dataset.name}: {path.relative_to(dataset)}", flush=True)
        if dataset not in rationales:
            fields, rows = load_rationales(dataset)
            rationales[dataset] = fields, rows, {row["image_path"]: row for row in rows}

        fields, rows, by_path = rationales[dataset]
        try:
            response = model.invoke(messages)
        except Exception as error:
            # Leave this row blank so a later run retries it; do not abort the job.
            print(f"Skipping {path}: {error}", file=sys.stderr, flush=True)
            continue
        by_path[path.relative_to(dataset).as_posix()]["ground_truth_rationales"] = response.content
        save_rationales(dataset, fields, rows)

def init_csv():
    for dataset in get_datasets():
        path = dataset / "rationales.csv"
        if path.exists():
            continue
        with (dataset / "labels.csv").open(newline="", encoding="utf-8") as source, \
             path.open("w", newline="", encoding="utf-8") as target:
            reader = csv.reader(source)
            writer = csv.writer(target)
            writer.writerow(next(reader) + ["ground_truth_rationales"])
            writer.writerows(row + [""] for row in reader)


def main():
    init_csv()
    total = get_pending_count()
    print(f"Generating {total} rationales", flush=True)
    send(build_batch(), total)


if __name__ == "__main__":
    main()
