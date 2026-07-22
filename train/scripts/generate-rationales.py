'''
This program takes one LLM model, provides the image and a prompt containing the ground truth answer, and tells it to create a text rationale.
It will save it to rationales.csv (?) under each of the datasets (e.g. datasets/TUEP/rationales.csv)
this will just be a copy of labels.csv with an additional column "rationale" that is generated based on the prior information.
'''

import base64
import csv
import mimetypes
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


MODEL = "qwen2.5vl:7b"

PROMPTS = {
    "TUAB": "The EEG is labeled {labels}. Write a concise rationale for this classification using visible EEG features. State the label explicitly.",
    "TUAR": "The EEG contains these artifacts: {labels}. Write a concise rationale explaining why each listed artifact is visible. State every label explicitly.",
    "TUEP": "The EEG is from a person labeled {labels}. Write a concise rationale for this classification using visible EEG features. State the label explicitly.",
    "TUEV": "The EEG contains these events: {labels}. Write a concise rationale explaining why each listed event is visible. State every label explicitly.",
    "TUSL": "The EEG contains these events: {labels}. Write a concise rationale explaining why each listed event is visible. State every label explicitly.",
    "TUSZ": "The EEG contains these seizure labels: {labels}. Write a concise rationale explaining why each listed seizure type is visible. State every label explicitly.",
}

def get_datasets():
    datasets = Path(__file__).parents[2] / "datasets"
    return sorted(path for path in datasets.iterdir() if (path / "labels.csv").is_file())

def create_prompt(dataset, row):
    labels = ", ".join(label for label, value in row.items() if value.strip().lower() == "true") or "none"
    return PROMPTS[Path(dataset).name].format(labels=labels)

def build_batch():
    for dataset in get_datasets():
        with (dataset / "rationales.csv").open(newline="", encoding="utf-8") as file:
            completed = {
                row["image_path"]
                for row in csv.DictReader(file)
                if row["ground_truth_rationales"].strip()
            }

        with (dataset / "labels.csv").open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["image_path"] in completed:
                    continue
                path = dataset / row["image_path"]
                if not path.is_file():
                    continue
                mime_type = mimetypes.guess_type(path)[0] or "image/png"
                image = base64.b64encode(path.read_bytes()).decode()
                yield path, [HumanMessage(content=[
                    {"type": "text", "text": create_prompt(dataset, row)},
                    {"type": "image_url", "image_url": f"data:{mime_type};base64,{image}"},
                ])]

def init_model():
    return ChatOllama(model=MODEL)


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


def send(batch):
    model = init_model()
    rationales = {}

    for path, messages in batch:
        dataset = get_dataset(path)
        if dataset not in rationales:
            fields, rows = load_rationales(dataset)
            rationales[dataset] = fields, rows, {row["image_path"]: row for row in rows}

        fields, rows, by_path = rationales[dataset]
        response = model.invoke(messages)
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
    send(build_batch())


if __name__ == "__main__":
    main()
