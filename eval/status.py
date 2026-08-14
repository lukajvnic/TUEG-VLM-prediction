import argparse
import csv
import fcntl
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml


def load_config(run_dir):
    with (run_dir / "config.yml").open() as file:
        return yaml.safe_load(file)


def read_statuses(file):
    return {
        (row["model"], row["dataset"]): row
        for row in csv.DictReader(file)
    }


def load_statuses(run_dir):
    with (run_dir / "status.csv").open(newline="") as file:
        return read_statuses(file)


def load_tasks(run_dir, config):
    path = run_dir / "tasks.json"
    if path.exists():
        return json.loads(path.read_text())

    return [
        {"model": model, "dataset": dataset}
        for model, model_config in config["models"].items()
        for dataset in model_config["datasets"]
    ]


def run_slurm(command):
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True).stdout.splitlines()
    except FileNotFoundError:
        return []


def get_squeue_states(job_ids):
    states = {}
    command = ["squeue", "-h", "-r", "-j", ",".join(job_ids), "-o", "%A|%a|%T|%R"]
    for line in run_slurm(command):
        job_id, task_id, state, reason = line.split("|", 3)
        if task_id.isdigit():
            states[(job_id, int(task_id))] = (state, reason)
    return states


def get_sacct_states(job_ids):
    # JobID, not JobIDRaw: raw ids are per-element and carry no `<array>_<task>`
    # suffix, so every finished array element fails the parse below.
    states = {}
    command = ["sacct", "-n", "-P", "-X", "-j", ",".join(job_ids), "--format=JobID,State,Reason"]
    for line in run_slurm(command):
        job_task, state, reason = line.split("|", 2)
        if "_" not in job_task:
            continue
        job_id, task_id = job_task.rsplit("_", 1)
        if task_id.isdigit():
            states[(job_id, int(task_id))] = (state, reason)
    return states


def get_slurm_states(tasks):
    job_ids = sorted({task["job_id"] for task in tasks if "job_id" in task})
    if not job_ids:
        return {}

    states = get_sacct_states(job_ids)
    states.update(get_squeue_states(job_ids))
    return states


def get_slurm_update(task, statuses, slurm_states):
    key = task["model"], task["dataset"]
    if statuses.get(key, {}).get("status") in {"success", "fail"}:
        return None

    slurm_state = slurm_states.get((task.get("job_id"), task.get("task_id")))
    if not slurm_state:
        return None

    state, reason = slurm_state
    if state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
        return None
    if state == "COMPLETED":
        return "success", ""
    return "fail", reason if reason and reason != "None" else state


def save_slurm_updates(run_dir, tasks, statuses, slurm_states):
    updates = {
        (task["model"], task["dataset"]): update
        for task in tasks
        if (update := get_slurm_update(task, statuses, slurm_states))
    }
    if not updates:
        return

    path = run_dir / "status.csv"
    with path.open("a+", newline="", encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.seek(0)
        rows = read_statuses(file)
        for (model, dataset), (status, reason) in updates.items():
            existing = rows.get((model, dataset), {})
            if existing.get("status") in {"success", "fail"}:
                continue
            rows[model, dataset] = {
                **existing,
                "model": model,
                "dataset": dataset,
                "status": status,
                "reason": reason,
                "total_images": existing.get("total_images", 0),
                "completed_images": existing.get("completed_images", 0),
            }
        file.seek(0)
        file.truncate()
        writer = csv.DictWriter(file, fieldnames=("model", "dataset", "status", "reason", "total_images", "completed_images"))
        writer.writeheader()
        writer.writerows(rows.values())


def get_task_status(task, statuses, slurm_states):
    status = statuses.get((task["model"], task["dataset"]), {}).get("status")
    if status in {"success", "fail"}:
        return status

    slurm_state = slurm_states.get((task.get("job_id"), task.get("task_id")))
    if slurm_state:
        state, _ = slurm_state
        if state == "PENDING":
            return "waiting"
        if state in {"RUNNING", "CONFIGURING", "COMPLETING"}:
            return "running"
        return "success" if state == "COMPLETED" else "fail"

    return status or "waiting"


def get_elapsed(run_name):
    started = datetime.strptime(run_name.removeprefix("run-"), "%Y%m%d-%H%M%S-%f")
    return str(datetime.now() - started).split(".")[0]


def print_report(run_name, tasks, statuses, slurm_states):
    counts = Counter(get_task_status(task, statuses, slurm_states) for task in tasks)
    finished = counts["success"] + counts["fail"]

    print(f"Run: {run_name}")
    print(f"Elapsed: {get_elapsed(run_name)}")
    print(f"Total jobs: {len(tasks)}")
    print(f"Finished: {finished} (success: {counts['success']}, failed: {counts['fail']})")
    print(f"Running: {counts['running']}")
    print(f"Waiting for allocation: {counts['waiting']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()

    run_dir = Path(__file__).parent / "runs" / args.run
    config = load_config(run_dir)
    tasks = load_tasks(run_dir, config)
    statuses = load_statuses(run_dir)
    slurm_states = get_slurm_states(tasks)
    save_slurm_updates(run_dir, tasks, statuses, slurm_states)
    print_report(args.run, tasks, load_statuses(run_dir), slurm_states)


if __name__ == "__main__":
    main()
