import csv
import json
import os
import sys
from base64 import b64decode, b64encode
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from helpers.pipeline import DATASETS, RATIONALE, ROOT, db, image_message, log_failure, read_csv, submit_array

MODEL = "gemma3:12b"  # not qwen2.5vl: repetition loop on these plots, ollama#10767
NUM_PREDICT = 512
NUM_CTX = 8192
PARALLEL = 4
RETRY_TEMPERATURE = 0.3
TIME, RAM, GPUS = "12:00:00", "16G", "a100_3g.20gb:1"
FLUSH_EVERY = 25
MAX_CONSECUTIVE_DEGENERATE = 20

IMAGE_INTRO = ("This image is a multi-channel EEG waveform plot: time runs along the horizontal "
               "axis and each row is the signal from one electrode channel. ")

LABEL_CLAUSES = {
    "TUAB": "This recording is labeled {labels}.",
    "TUEP": "This recording is from a person labeled {labels}.",
    "TUAR": "This recording contains these artifacts: {labels}.",
    "TUEV": "This recording contains these events: {labels}.",
    "TUSL": "This recording contains these events: {labels}.",
    "TUSZ": "This recording contains these seizure labels: {labels}.",
}

GROUNDED = ("Justify this labeling using only visible EEG features. For each finding, give the "
            "specific evidence: which channel(s) it appears on, where along the time axis, and "
            "what the curve looks like there (shape, frequency, amplitude). State every label "
            "explicitly. Write a focused 3-5 sentence mini-report; do not describe the image "
            "format, define terminology, repeat yourself, or add unsupported findings.")


def true_labels(row):
    return [c for c, v in row.items() if c not in ("path", RATIONALE) and v.strip().lower() == "true"]


def sampled_paths(dataset):
    rows = db().execute("SELECT DISTINCT path FROM pipeline WHERE dataset = ? AND sampled = 1",
                        (dataset,))
    return {p.split("/", 1)[1] for (p,) in rows}


def pending(dataset, rows):
    sampled = sampled_paths(dataset)
    return [r for r in rows
            if not r[RATIONALE].strip() and true_labels(r)
            and (r["path"].startswith("train/") or r["path"] in sampled)]


def create_prompt(dataset, row):
    labels = ", ".join(true_labels(row))
    return f"{IMAGE_INTRO}{LABEL_CLAUSES[dataset].format(labels=labels)} {GROUNDED}"


def is_degenerate(text):
    words = text.split()
    if len(text) < 40 or len(words) < 8:
        return True
    return len(set(text)) < 12 or len(set(words)) / len(words) < 0.2


def save(dataset, fields, rows):
    path = ROOT / "datasets" / dataset / "labels.csv"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def generate_all(dataset):
    from langchain_ollama import ChatOllama
    folder = ROOT / "datasets" / dataset
    rows = read_csv(folder / "labels.csv")
    todo = sorted(pending(dataset, rows), key=lambda r: r["path"])  # test refs first
    fields = list(rows[0].keys())
    by_path = {r["path"]: r for r in rows}
    kwargs = dict(model=MODEL, base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                  num_predict=NUM_PREDICT, num_ctx=NUM_CTX)
    llm = ChatOllama(temperature=0, **kwargs)
    retry = ChatOllama(temperature=RETRY_TEMPERATURE, **kwargs)

    def generate(row):
        message = image_message(folder / row["path"], create_prompt(dataset, row))
        text = str(llm.invoke([message]).content).strip()
        if is_degenerate(text):
            text = str(retry.invoke([message]).content).strip()  # temp 0 makes degenerate loops deterministic
        return text

    done = streak = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futures = {pool.submit(generate, row): row["path"] for row in todo}
        for future in as_completed(futures):
            path = futures[future]
            done += 1
            try:
                text = future.result()
            except Exception as e:
                print(f"failed {path}: {e}", file=sys.stderr, flush=True)
                log_failure("rationale", dataset, path, MODEL, e)
                streak = 0
                continue
            if is_degenerate(text):
                streak += 1
                print(f"degenerate {path}: {text[:60]!r}", file=sys.stderr, flush=True)
                log_failure("rationale", dataset, path, MODEL, f"degenerate: {text[:60]!r}")
                if streak >= MAX_CONSECUTIVE_DEGENERATE:
                    save(dataset, fields, rows)
                    pool.shutdown(wait=False, cancel_futures=True)
                    sys.exit(f"{streak} degenerate responses in a row - runner is wedged")
                continue
            streak = 0
            by_path[path][RATIONALE] = text
            if done % FLUSH_EVERY == 0:
                save(dataset, fields, rows)
                print(f"{done}/{len(todo)}", flush=True)
    save(dataset, fields, rows)
    print(f"{dataset}: {len(todo)} attempted")


def main():
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        targets = json.loads(b64decode(os.environ["RATIONALE_DATASETS"]))
        generate_all(targets[int(os.environ["SLURM_ARRAY_TASK_ID"])])
        return
    targets = [ds for ds in DATASETS if pending(ds, read_csv(ROOT / "datasets" / ds / "labels.csv"))]
    if not targets:
        print("nothing to generate")
        return
    payload = b64encode(json.dumps(targets).encode()).decode()
    out = submit_array("eeg-vlm-rationales", TIME, RAM, GPUS, len(targets) - 1, len(targets),
                       PARALLEL, f"python {ROOT}/eval/ground-truth/generate-rationales.py",
                       {"RATIONALE_DATASETS": payload})
    print(f"{out} - datasets: {', '.join(targets)}")


if __name__ == "__main__":
    main()
