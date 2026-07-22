"""Import completed and failed legacy tasks into a new-format run directory.

Usage:
    python temp/port_legacy.py /path/to/legacy-archive --dataset TUEP
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
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--job-id", required=True, help="Legacy array job ID to import.")
    parser.add_argument("--name", help="Name for the imported run directory.")
    return parser.parse_args()


def dataset_from_row(row, default):
    return next((value for value in row[4:] if value in DATASETS), default)


def load_outcomes(path, status, default_dataset, job_id):
    if not path.exists():
        return {}

    outcomes = {}
    with path.open(newline="") as file:
        for row in csv.reader(file):
            if len(row) < 4 or row[1] != job_id:
                continue
            dataset = dataset_from_row(row, default_dataset)
            if dataset:
                outcomes[row[3], dataset] = "" if status == "success" else ", ".join(row[4:])
    return outcomes


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
    for path in results_dir.glob("*.csv"):
        if path.name in {"summary.csv", "rank.csv"}:
            continue
        model = api_model(path)
        dataset = dataset_from_filename(path)
        if model and dataset:
            index[model, dataset] = path
    return index


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


def import_outcomes(legacy_run, run_dir, dataset, job_id):
    logs = legacy_run / "logs"
    successes = load_outcomes(logs / "completed_models.csv", "success", dataset, job_id)
    failures = load_outcomes(logs / "failed_models.csv", "fail", dataset, job_id)
    results = index_results(legacy_run / "results")
    rows = []

    for task in successes:
        result = results.get(task)
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
    rows = import_outcomes(legacy_run, run_dir, args.dataset, args.job_id)
    write_status(run_dir, rows)
    write_tasks(run_dir, rows)
    print(f"Imported {len(rows)} task(s) into {run_dir}")


if __name__ == "__main__":
    main()
