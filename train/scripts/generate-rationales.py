'''
This program takes one LLM model, provides the image and a prompt containing the ground truth answer, and tells it to create a text rationale.
It will save it to rationales.csv under each of the datasets (e.g. datasets/TUEP/rationales.csv):
a copy of labels.csv with an additional column "ground_truth_rationales" generated from the labels and the image.
'''

import base64
import csv
import mimetypes
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


MODEL = "qwen2.5vl:3b"
# Grounded, per-label rationales run longer than a plain summary; give them room.
MAX_OUTPUT_TOKENS = 512
# Concurrent in-flight requests. Match the server's OLLAMA_NUM_PARALLEL so they batch.
PARALLEL_REQUESTS = int(os.environ.get("OLLAMA_NUM_PARALLEL", "4"))
# Persist progress every this many completed rationales (bounds O(N^2) CSV rewrites).
FLUSH_EVERY = 25

IMAGE_INTRO = (
    "This image is a multi-channel EEG waveform plot: time runs along the horizontal "
    "axis and each row is the signal from one electrode channel. "
)

LABEL_CLAUSES = {
    "TUAB": "This recording is labeled {labels}.",
    "TUEP": "This recording is from a person labeled {labels}.",
    "TUAR": "This recording contains these artifacts: {labels}.",
    "TUEV": "This recording contains these events: {labels}.",
    "TUSL": "This recording contains these events: {labels}.",
    "TUSZ": "This recording contains these seizure labels: {labels}.",
}

GROUNDED = (
    "Justify this labeling using only visible EEG features. For each finding, give the "
    "specific evidence: which channel(s) it appears on, where along the time axis, and "
    "what the curve looks like there (shape, frequency, amplitude). State every label "
    "explicitly. Write a focused 3-5 sentence mini-report; do not describe the image "
    "format, define terminology, repeat yourself, or add unsupported findings."
)


def get_datasets():
    datasets = Path(__file__).parents[2] / "datasets"
    return sorted(path for path in datasets.iterdir() if (path / "labels.csv").is_file())


def is_train(row):
    # Fine-tune on train patients only; split absent = old unsplit dataset (use all).
    return row.get("split") in (None, "train")


def create_prompt(dataset, row):
    labels = ", ".join(label for label, value in row.items() if value.strip().lower() == "true") or "none"
    return f"{IMAGE_INTRO}{LABEL_CLAUSES[dataset.name].format(labels=labels)} {GROUNDED}"


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
                is_train(row) and row["image_path"] not in completed and (dataset / row["image_path"]).is_file()
                for row in csv.DictReader(file)
            )
    return total


def build_batch():
    for dataset in get_datasets():
        completed = get_completed(dataset)
        with (dataset / "labels.csv").open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if not is_train(row) or row["image_path"] in completed:
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
    # Pin num_predict AND num_ctx on the request so the report isn't cut short by a
    # server-side default (rationales were truncating mid-sentence at ~128 tokens).
    # num_ctx covers the ~3050-token image + prompt + report; temperature=0 keeps
    # the reports focused and reproducible.
    return ChatOllama(
        model=MODEL,
        num_predict=MAX_OUTPUT_TOKENS,
        num_ctx=8192,
        temperature=0,
    )


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


def init_csv():
    # Rebuild each rationales.csv from the current labels.csv, carrying over any
    # rationales already written. This keeps the two files in sync, so a labels.csv
    # that gained or lost rows never desyncs the rationale lookup during a run.
    for dataset in get_datasets():
        with (dataset / "labels.csv").open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fields = [*reader.fieldnames, "ground_truth_rationales"]
            label_rows = [row for row in reader if is_train(row)]

        existing = {}
        path = dataset / "rationales.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as file:
                existing = {
                    row["image_path"]: row.get("ground_truth_rationales", "")
                    for row in csv.DictReader(file)
                }

        rows = [
            {**row, "ground_truth_rationales": existing.get(row["image_path"], "")}
            for row in label_rows
        ]
        save_rationales(dataset, fields, rows)


def generate(model, path, messages):
    try:
        response = model.invoke(messages)
        return path, str(response.content), None
    except Exception as error:
        return path, None, error


def send(batch, total):
    model = init_model()
    datasets = {}
    dirty = set()
    processed = 0

    def flush():
        for dataset in dirty:
            fields, rows, _ = datasets[dataset]
            save_rationales(dataset, fields, rows)
        dirty.clear()

    def handle(result):
        nonlocal processed
        path, content, error = result
        dataset = get_dataset(path)
        processed += 1
        print(f"[{processed}/{total}] {dataset.name}: {path.relative_to(dataset)}", flush=True)
        if error is not None:
            # Leave this row blank so a later run retries it; do not abort the job.
            print(f"Skipping {path}: {error}", file=sys.stderr, flush=True)
            return
        if dataset not in datasets:
            fields, rows = load_rationales(dataset)
            datasets[dataset] = (fields, rows, {row["image_path"]: row for row in rows})
        _, _, by_path = datasets[dataset]
        by_path[path.relative_to(dataset).as_posix()]["ground_truth_rationales"] = content
        dirty.add(dataset)
        if processed % FLUSH_EVERY == 0:
            flush()

    items = iter(batch)
    with ThreadPoolExecutor(max_workers=PARALLEL_REQUESTS) as executor:
        # Keep the pool fed without materialising every image at once: only pull
        # (and base64-encode) the next item when an in-flight request completes.
        inflight = set()
        for _ in range(PARALLEL_REQUESTS * 2):
            item = next(items, None)
            if item is None:
                break
            inflight.add(executor.submit(generate, model, *item))

        while inflight:
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                handle(future.result())
                item = next(items, None)
                if item is not None:
                    inflight.add(executor.submit(generate, model, *item))

    flush()


def main():
    init_csv()
    total = get_pending_count()
    print(f"Generating {total} rationales", flush=True)
    send(build_batch(), total)


if __name__ == "__main__":
    main()
