"""Import completed and failed legacy tasks into a new-format run directory.

Usage:
    python temp/port_legacy.py /path/to/legacy-archive
"""

import argparse
import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path


DATASETS = {"TUEP", "TUAB", "TUEV", "TUAR", "TUSZ", "TUSL"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "eval"
BASE_CONFIG = EVAL_DIR / "scripts" / "config-base.yml"


def safe_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_run", type=Path, help="Archived legacy run containing results/ and logs/.")
    parser.add_argument("--name", help="Name for the imported run directory.")
    return parser.parse_args()


def load_job_ids(legacy_run):
    job_ids = set()
    for name in ("squeue-before.txt", "sacct.txt", "sacct-before.txt", "sacct-after.txt"):
        path = legacy_run / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            match = re.match(r"\s*(\d+)", line)
            if match:
                job_ids.add(match.group(1))
    if not job_ids:
        raise SystemExit("ERROR: no job IDs found in the legacy archive")
    return job_ids


def load_task_datasets(logs):
    tasks = {}
    pattern = re.compile(r"\bjob=(\d+)\s+task=(\d+)\s+model=(\S+)\s+dataset=(\S+)")
    for path in logs.rglob("*.out"):
        for match in pattern.finditer(path.read_text(errors="replace")):
            job_id, task_id, model, dataset = match.groups()
            if dataset in DATASETS:
                tasks[job_id, task_id] = model, dataset
    return tasks


def dataset_from_row(row, task_datasets):
    dataset = next((value for value in row[4:] if value in DATASETS), None)
    if dataset:
        return dataset
    task = task_datasets.get((row[1], row[2]))
    return task[1] if task else None


def load_csv_outcomes(logs, filename, status, task_datasets, job_ids):
    outcomes = {}
    for path in logs.rglob(filename):
        with path.open(newline="") as file:
            for row in csv.reader(file):
                if len(row) < 4 or row[1] not in job_ids:
                    continue
                dataset = dataset_from_row(row, task_datasets)
                if not dataset:
                    print(f"WARNING: no dataset found for legacy task {row[1]}_{row[2]}")
                    continue
                outcomes[row[3], dataset] = "" if status == "success" else ", ".join(row[4:])
    return outcomes


def load_json_outcomes(logs, job_ids):
    successes = {}
    failures = {}
    for path in logs.rglob("task_status/*.json"):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if str(record.get("job_id")) not in job_ids:
            continue
        task = record.get("model"), record.get("dataset")
        if not all(task) or task[1] not in DATASETS:
            continue
        if record.get("outcome") == "completed":
            successes[task] = ""
        elif record.get("outcome") == "failed":
            failures[task] = record.get("error", "legacy task failed")
    return successes, failures


def api_model(path):
    with path.open(newline="") as file:
        row = next(csv.DictReader(file), None)
    if not row:
        return None
    try:
        payload = json.loads(row["api_json"])
    except (KeyError, json.JSONDecodeError):
        return None
    metadata = payload.get("response_metadata", {})
    return metadata.get("model") or metadata.get("model_name")


def dataset_from_filename(path):
    match = re.search(r"(TUEP|TUAB|TUEV|TUAR|TUSZ|TUSL)", path.stem)
    return match.group(1) if match else None


def index_results(results_dir):
    index = {}
    for path in results_dir.rglob("*.csv"):
        if path.name in {"summary.csv", "rank.csv"}:
            continue
        model = api_model(path)
        dataset = dataset_from_filename(path)
        if model and dataset:
            index[model, dataset] = path
    return index


def find_result(results_dir, results, task):
    model, dataset = task
    matches = list(results_dir.rglob(f"{safe_filename(model)}-{dataset}.csv"))
    if matches:
        return max(matches, key=lambda path: path.stat().st_mtime)
    return results.get(task)


def count_rows(path):
    with path.open(newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def create_run(name):
    name = name or "run-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = EVAL_DIR / "runs" / name
    (run_dir / "results").mkdir(parents=True)
    (run_dir / "logs").mkdir()
    return run_dir


def write_status(run_dir, rows):
    fields = ("model", "dataset", "status", "reason", "total_images", "completed_images")
    with (run_dir / "status.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def copy_config(legacy_run, run_dir):
    shutil.copy2(BASE_CONFIG, run_dir / "config.yml")
    legacy_config = legacy_run / "config.yml"
    if legacy_config.exists():
        shutil.copy2(legacy_config, run_dir / "legacy-config.yml")


def import_outcomes(legacy_run, run_dir, job_ids):
    logs = legacy_run / "logs"
    task_datasets = load_task_datasets(logs)
    successes = load_csv_outcomes(logs, "completed_models.csv", "success", task_datasets, job_ids)
    failures = load_csv_outcomes(logs, "failed_models.csv", "fail", task_datasets, job_ids)
    json_successes, json_failures = load_json_outcomes(logs, job_ids)
    successes.update(json_successes)
    failures.update(json_failures)
    results_dir = legacy_run / "results"
    results = index_results(results_dir)
    rows = []

    for task in successes:
        result = find_result(results_dir, results, task)
        if not result:
            print(f"WARNING: no result CSV for completed task {task[0]} / {task[1]}")
            continue
        destination = run_dir / "results" / f"{safe_filename(task[0])}-{task[1]}.csv"
        shutil.copy2(result, destination)
        completed = count_rows(destination)
        rows.append({
            "model": task[0], "dataset": task[1], "status": "success", "reason": "",
            "total_images": completed, "completed_images": completed,
        })

    for (model, task_dataset), reason in failures.items():
        if (model, task_dataset) not in successes:
            rows.append({
                "model": model, "dataset": task_dataset, "status": "fail", "reason": reason,
                "total_images": 0, "completed_images": 0,
            })
    return rows


def write_tasks(run_dir, rows):
    tasks = [{"model": row["model"], "dataset": row["dataset"]} for row in rows]
    (run_dir / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")


def main():
    args = parse_args()
    legacy_run = args.legacy_run.resolve()
    run_dir = create_run(args.name)
    copy_config(legacy_run, run_dir)
    rows = import_outcomes(legacy_run, run_dir, load_job_ids(legacy_run))
    write_status(run_dir, rows)
    write_tasks(run_dir, rows)
    print(f"Imported {len(rows)} task(s) into {run_dir}")


if __name__ == "__main__":
    main()
