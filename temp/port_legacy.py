"""Import a legacy evaluation run into the current eval/runs layout.

Examples:
    python temp/port_legacy.py  # imports the archived run configured below
    python temp/port_legacy.py /path/to/archive
    python temp/port_legacy.py /path/to/results/run-... /path/to/logs/run-...

A legacy result is successful only when it contains exactly as many rows as the
current on-disk dataset.  A partial result, or a task identified in a Slurm
output log with no result CSV, is imported as a failure.
"""

import argparse
import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path


DATASETS = ("TUEP", "TUAB", "TUEV", "TUAR", "TUSZ", "TUSL")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "eval"
LEGACY_ARCHIVE = Path("/home/luka/scratch/tueg-vlm-legacy/20260722-162258")
LEGACY_RUN = "run-20260712-143321-451122"
DEFAULT_RESULTS = LEGACY_ARCHIVE / "results" / LEGACY_RUN
DEFAULT_LOGS = LEGACY_ARCHIVE / "logs" / LEGACY_RUN


def safe_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        type=Path,
        nargs="?",
        help="Legacy results directory, or an archive containing results/ and logs/.",
    )
    parser.add_argument(
        "logs",
        type=Path,
        nargs="?",
        help="Legacy logs directory (defaults to the archived legacy run when no arguments are given).",
    )
    parser.add_argument("--name", help="New run directory name (default: run-<timestamp>).")
    return parser.parse_args()


def resolve_inputs(args):
    if args.results is None:
        results_dir, logs_dir = DEFAULT_RESULTS, DEFAULT_LOGS
    else:
        results_dir = args.results.resolve()
        logs_dir = args.logs.resolve() if args.logs else None
    if logs_dir is None:
        logs_dir = results_dir / "logs"
        results_dir = results_dir / "results"

    if not results_dir.is_dir():
        raise SystemExit(f"ERROR: results directory does not exist: {results_dir}")
    if not logs_dir.is_dir():
        raise SystemExit(f"ERROR: logs directory does not exist: {logs_dir}")
    return results_dir, logs_dir


def expected_image_counts():
    counts = {}
    for dataset in DATASETS:
        dataset_dir = PROJECT_ROOT / "datasets" / dataset
        labels = dataset_dir / "labels.csv"
        if not labels.is_file():
            raise SystemExit(f"ERROR: missing labels file: {labels}")
        with labels.open(newline="", encoding="utf-8") as file:
            counts[dataset] = sum(
                (dataset_dir / row["image_path"]).is_file()
                for row in csv.DictReader(file)
            )
    return counts


def model_from_result(path):
    """Read the Ollama model name from a legacy result CSV."""
    try:
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                try:
                    payload = json.loads(row.get("api_json", ""))
                except json.JSONDecodeError:
                    continue
                metadata = payload.get("response_metadata", {})
                model = metadata.get("model") or metadata.get("model_name")
                if model:
                    return model
    except (csv.Error, OSError, UnicodeDecodeError) as error:
        print(f"WARNING: cannot read {path}: {error}")
    return None


def dataset_from_filename(path):
    match = re.search(r"-(%s)$" % "|".join(DATASETS), path.stem)
    return match.group(1) if match else None


def count_result_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return sum(1 for _ in csv.DictReader(file))


def result_index(results_dir, tasks):
    results = {}
    for path in sorted(results_dir.rglob("*.csv")):
        if path.name in {"summary.csv", "rank.csv"}:
            continue
        dataset = dataset_from_filename(path)
        model = model_from_result(path)
        if not model and dataset:
            matches = [
                task_model for task_model, task_dataset in tasks
                if task_dataset == dataset
                and path.name == f"{safe_filename(task_model)}-{dataset}.csv"
            ]
            model = matches[0] if len(matches) == 1 else None
        if not dataset or not model:
            print(f"WARNING: cannot identify model/dataset for result: {path}")
            continue
        task = model, dataset
        if task in results:
            print(f"WARNING: duplicate result for {model} / {dataset}; keeping newest")
            if path.stat().st_mtime <= results[task].stat().st_mtime:
                continue
        results[task] = path
    return results


def logged_tasks(logs_dir):
    """Extract model/dataset pairs printed by the legacy array script."""
    pattern = re.compile(r"\bmodel=(\S+)\s+dataset=(%s)\b" % "|".join(DATASETS))
    tasks = set()
    for path in logs_dir.rglob("*.out"):
        try:
            tasks.update(pattern.findall(path.read_text(errors="replace")))
        except OSError as error:
            print(f"WARNING: cannot read {path}: {error}")
    return tasks


def build_rows(results, tasks, expected_counts, destination):
    rows = []
    for model, dataset in sorted(set(results) | tasks):
        result = results.get((model, dataset))
        expected = expected_counts[dataset]
        if result is None:
            rows.append({
                "model": model, "dataset": dataset, "status": "fail",
                "reason": "legacy task produced no result CSV",
                "total_images": expected, "completed_images": 0,
            })
            continue

        completed = count_result_rows(result)
        target = destination / "results" / f"{safe_filename(model)}-{dataset}.csv"
        shutil.copy2(result, target)
        if completed == expected:
            status, reason = "success", ""
        else:
            status = "fail"
            reason = f"incomplete legacy result ({completed}/{expected} images evaluated)"
        rows.append({
            "model": model, "dataset": dataset, "status": status, "reason": reason,
            "total_images": expected, "completed_images": completed,
        })
    return rows


def create_run(name):
    name = name or "run-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = EVAL_DIR / "runs" / name
    if run_dir.exists():
        raise SystemExit(f"ERROR: destination already exists: {run_dir}")
    (run_dir / "results").mkdir(parents=True)
    (run_dir / "logs").mkdir()
    shutil.copy2(EVAL_DIR / "config.yml", run_dir / "config.yml")
    return run_dir


def write_metadata(run_dir, rows):
    fields = ("model", "dataset", "status", "reason", "total_images", "completed_images")
    with (run_dir / "status.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "tasks.json").write_text(
        json.dumps([{"model": row["model"], "dataset": row["dataset"]} for row in rows], indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    results_dir, logs_dir = resolve_inputs(args)
    tasks = logged_tasks(logs_dir)
    results = result_index(results_dir, tasks)
    if not results and not tasks:
        raise SystemExit("ERROR: found no identifiable legacy results or array tasks")

    run_dir = create_run(args.name)
    rows = build_rows(results, tasks, expected_image_counts(), run_dir)
    write_metadata(run_dir, rows)
    successes = sum(row["status"] == "success" for row in rows)
    print(f"Imported {len(rows)} task(s): {successes} success, {len(rows) - successes} fail")
    print(run_dir)


if __name__ == "__main__":
    main()
