'''
This program takes one LLM model, provides the image and a prompt containing the ground truth answer, and tells it to create a text rationale.

Two splits, two output files:

  --split train (default) -> datasets/<DS>/rationales.csv
      A copy of the train rows of labels.csv plus a "ground_truth_rationales"
      column. This is the fine-tuning target set.

  --split test            -> datasets/<DS>/rationales-test.csv
      The same thing for the test windows the benchmark actually scores (the
      stratified sample from eval/scripts/sample.py). These are the *reference*
      rationales that eval/agreement.py compares each model's zero-shot
      rationale against.

The two files are deliberately separate. rationales-test.csv is written from
held-out **test** patients, so feeding it to the fine-tune would void the
patient-level split -- never glob rationales*.csv into training data. Keeping
them apart also protects the train file: init_csv() rebuilds its target file
from labels.csv on every run, so a shared filename would blank one split's work
each time the other ran.
'''

import argparse
import base64
import csv
import importlib.util
import mimetypes
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


# Not qwen2.5vl: it locks onto a single token and repeats it for the whole
# num_predict budget on these plots (ollama/ollama#10767 -- 7b/32b/72b all
# affected, so a bigger Qwen is not the fix). Ollama has no repetition-loop
# detection (#16179) and exposes neither DRY nor XTC (#7504), so no sampler
# setting prevents this; a different model family is the only lever.
MODEL = "gemma3:12b"
# Grounded, per-label rationales run longer than a plain summary; give them room.
MAX_OUTPUT_TOKENS = 512
# Concurrent in-flight requests. Match the server's OLLAMA_NUM_PARALLEL so they batch.
PARALLEL_REQUESTS = int(os.environ.get("OLLAMA_NUM_PARALLEL", "4"))
# Persist progress every this many completed rationales (bounds O(N^2) CSV rewrites).
FLUSH_EVERY = 25
# Once the Ollama runner goes bad it stays bad, so stop rather than spend the rest
# of the allocation writing garbage. See is_degenerate.
MAX_CONSECUTIVE_DEGENERATE = 20

RATIONALE_FILES = {"train": "rationales.csv", "test": "rationales-test.csv"}
RATIONALE_COLUMN = "ground_truth_rationales"

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


def rationales_path(dataset, split="train"):
    return dataset / RATIONALE_FILES[split]


NON_LABEL_COLUMNS = ("image_path", "split", "assessed")


def create_prompt(dataset, row):
    # `assessed` is a bookkeeping flag whose value is literally "true"/"false", so
    # it has to be excluded by name or it lands in the prompt as a class label.
    labels = ", ".join(
        label
        for label, value in row.items()
        if label not in NON_LABEL_COLUMNS and value.strip().lower() == "true"
    ) or "none"
    return f"{IMAGE_INTRO}{LABEL_CLAUSES[dataset.name].format(labels=labels)} {GROUNDED}"


# ---------------------------------------------------------------- target rows


def load_sample_module():
    # eval/scripts/sample.py defines the benchmark's test subsample. Loaded by path
    # (rather than duplicated) so the reference rationales cover exactly the windows
    # eval.py scores -- the same reason scrub-degenerate.py loads this module.
    path = Path(__file__).parents[2] / "eval" / "scripts" / "sample.py"
    spec = importlib.util.spec_from_file_location("sample", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_test_policy():
    # The sampling caps live in eval/config.yml so the reference set and the
    # evaluated set are driven by one number, not two copies of it.
    path = Path(__file__).parents[2] / "eval" / "config.yml"
    with path.open() as file:
        settings = yaml.safe_load(file).get("settings") or {}
    return settings.get("test-sample") or {}


def load_labels(dataset):
    with (dataset / "labels.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames), list(reader)


def evaluated_names(dataset, run):
    """Image basenames that appear in <run>'s results CSVs for this dataset.

    Driving the reference set off a finished run guarantees every scored window
    has a reference, even if the sampling policy changed after that run was
    submitted. Costs a dependency on the run having produced results already.
    """
    results = Path(__file__).parents[2] / "eval" / "runs" / run / "results"
    names = set()
    for path in sorted(results.glob(f"*-{dataset.name}.csv")):
        with path.open(newline="", encoding="utf-8") as file:
            names.update(Path(row["path"]).name for row in csv.DictReader(file))
    return names


def target_rows(dataset, split, policy, run):
    """The labels.csv rows this split should hold a rationale for."""
    _, rows = load_labels(dataset)

    if split == "train":
        # Split absent = old unsplit dataset, so fall through and use everything.
        return [row for row in rows if row.get("split") in (None, "train")]

    test_rows = [row for row in rows if row.get("split") == "test"]
    if not test_rows:
        return []
    if run:
        wanted = evaluated_names(dataset, run)
    else:
        wanted = {
            Path(name).name
            for name in load_sample_module().select(test_rows, dataset.name, policy)
        }
    return [row for row in test_rows if Path(row["image_path"]).name in wanted]


def build_plan(split, policy=None, run=None):
    """[(dataset, fields, rows)] -- what each rationales file should contain.

    Computed once and shared by init_csv / pending_count / build_batch so the
    three can never disagree about which rows are in scope.
    """
    plan = []
    for dataset in get_datasets():
        fields, _ = load_labels(dataset)
        rows = target_rows(dataset, split, policy, run)
        if rows:
            plan.append((dataset, [*fields, RATIONALE_COLUMN], rows))
    return plan


# ------------------------------------------------------------------- csv i/o


def load_rationales(dataset, split="train"):
    with rationales_path(dataset, split).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return reader.fieldnames, list(reader)


def save_rationales(dataset, fields, rows, split="train"):
    path = rationales_path(dataset, split)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def existing_rationales(dataset, split):
    path = rationales_path(dataset, split)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        return {
            row["image_path"]: row.get(RATIONALE_COLUMN, "")
            for row in csv.DictReader(file)
        }


def init_csv(plan, split):
    # Rebuild each rationales file from the current labels.csv, carrying over any
    # rationales already written. This keeps the two files in sync, so a labels.csv
    # that gained or lost rows never desyncs the rationale lookup during a run.
    for dataset, fields, rows in plan:
        existing = existing_rationales(dataset, split)
        save_rationales(
            dataset,
            fields,
            [{**row, RATIONALE_COLUMN: existing.get(row["image_path"], "")} for row in rows],
            split,
        )


def pending_rows(dataset, rows, split):
    written = existing_rationales(dataset, split)
    return [
        row for row in rows
        if not written.get(row["image_path"], "").strip()
        and (dataset / row["image_path"]).is_file()
    ]


def pending_count(plan, split):
    return sum(len(pending_rows(dataset, rows, split)) for dataset, _, rows in plan)


# ------------------------------------------------------------------ requests


def build_batch(plan, split):
    for dataset, _, rows in plan:
        for row in pending_rows(dataset, rows, split):
            path = dataset / row["image_path"]
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


class Degenerate(Exception):
    pass


def is_degenerate(text):
    # A wedged runner returns the same token repeated for the whole num_predict
    # budget (a 512-char run of "?"). Greedy decoding makes it identical every
    # time, so it looks like a real answer to every check except this one. Treat
    # it as a failure: the row stays blank and a later run retries it, instead of
    # baking garbage into the fine-tuning set.
    words = text.split()
    if len(text) < 40 or len(words) < 8:
        return True
    return len(set(text)) < 12 or len(set(words)) / len(words) < 0.2


def generate(model, path, messages):
    try:
        response = model.invoke(messages)
        content = str(response.content).strip()
        if is_degenerate(content):
            return path, None, Degenerate(f"{len(content)} chars: {content[:60]!r}")
        return path, content, None
    except Exception as error:
        return path, None, error


def send(batch, total, split):
    model = init_model()
    datasets = {}
    dirty = set()
    processed = 0
    streak = 0

    def flush():
        for dataset in dirty:
            fields, rows, _ = datasets[dataset]
            save_rationales(dataset, fields, rows, split)
        dirty.clear()

    def handle(result):
        nonlocal processed, streak
        path, content, error = result
        dataset = get_dataset(path)
        processed += 1
        print(f"[{processed}/{total}] {dataset.name}: {path.relative_to(dataset)}", flush=True)
        if error is not None:
            # Leave this row blank so a later run retries it; do not abort the job.
            print(f"Skipping {path}: {error}", file=sys.stderr, flush=True)
            streak = streak + 1 if isinstance(error, Degenerate) else 0
            if streak >= MAX_CONSECUTIVE_DEGENERATE:
                flush()
                raise SystemExit(
                    f"{streak} degenerate responses in a row -- the Ollama runner is wedged. "
                    "Restart it (and check the GPU) before resuming; completed rows are saved."
                )
            return
        streak = 0
        if dataset not in datasets:
            fields, rows = load_rationales(dataset, split)
            datasets[dataset] = (fields, rows, {row["image_path"]: row for row in rows})
        _, _, by_path = datasets[dataset]
        by_path[path.relative_to(dataset).as_posix()][RATIONALE_COLUMN] = content
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


# ----------------------------------------------------------------------- cli


def load_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=sorted(RATIONALE_FILES),
        default="train",
        help="train (default) writes the fine-tuning targets to rationales.csv; "
             "test writes reference rationales for the benchmark's sampled test "
             "windows to rationales-test.csv. Never train on the test file.",
    )
    parser.add_argument(
        "--run",
        metavar="RUN",
        help="--split test only: take the target windows from eval/runs/RUN's "
             "results instead of re-deriving them from the sampling policy. Use "
             "this to guarantee every scored window has a reference.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report target/written/pending counts per dataset and write nothing.",
    )
    return parser.parse_args()


def report(plan, split):
    print(f"split={split} -> {RATIONALE_FILES[split]}")
    total = pending = 0
    for dataset, _, rows in plan:
        missing = pending_rows(dataset, rows, split)
        written = len(rows) - len(missing)
        absent = sum(not (dataset / row["image_path"]).is_file() for row in rows)
        note = f", {absent} images missing on disk" if absent else ""
        print(f"  {dataset.name}: target {len(rows)}, written {written}, pending {len(missing)}{note}")
        total += len(rows)
        pending += len(missing)
    print(f"  TOTAL target {total}, pending {pending}")


def main():
    args = load_args()
    if args.run and args.split != "test":
        raise SystemExit("--run applies to --split test only")

    policy = load_test_policy() if args.split == "test" else None
    plan = build_plan(args.split, policy, args.run)

    if args.dry_run:
        report(plan, args.split)
        return

    init_csv(plan, args.split)
    total = pending_count(plan, args.split)
    print(f"Generating {total} {args.split} rationales with {MODEL}", flush=True)
    send(build_batch(plan, args.split), total, args.split)


if __name__ == "__main__":
    main()
