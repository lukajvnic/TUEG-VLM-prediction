import csv
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yml"
LOG_DIR = PROJECT_ROOT / "logs"


DATASETS = ("TUEP", "TUAB", "TUEV", "TUAR", "TUSZ", "TUSL")


def load_tasks(run_dir: Path | None) -> set[tuple[str, str]]:
    """Load the exact task set submitted for this run, not the mutable live config."""
    if run_dir is not None:
        manifest = run_dir / "tasks.json"
        if manifest.exists():
            return {tuple(task) for task in json.loads(manifest.read_text(encoding="utf-8"))}

    config_path = run_dir / "config.yml" if run_dir is not None and (run_dir / "config.yml").exists() else CONFIG_PATH
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    models = list(config.get("models", {})) if config.get("model") == "all" else [config.get("model")]
    datasets = DATASETS if config.get("dataset") == "all" else (config.get("dataset"),)
    return {(model, dataset) for model in models for dataset in datasets if model and dataset}


def last_job_ids() -> list[str]:
    path = LOG_DIR / "last_array_job.txt"
    if not path.exists():
        return []
    return [job_id for job_id in path.read_text(encoding="utf-8").strip().split(",") if job_id]


def logged_tasks(log_dir: Path, job_ids: list[str], outcome: str, default_dataset: str | None = None) -> set[tuple[str, str]]:
    """Read atomic per-task status files, with CSV support for legacy runs."""
    status_dir = log_dir / "task_status"
    if status_dir.exists():
        tasks = set()
        for path in status_dir.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if record.get("outcome") != outcome or (job_ids and record.get("job_id") not in job_ids):
                continue
            model, dataset = record.get("model"), record.get("dataset")
            if model and dataset:
                tasks.add((model, dataset))
        return tasks

    # Legacy runs appended rows to shared CSV files.
    path = log_dir / ("completed_models.csv" if outcome == "completed" else "failed_models.csv")
    if not path.exists():
        return set()
    tasks = set()
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if len(row) < 4 or (job_ids and row[1] not in job_ids):
                continue
            dataset = row[4] if len(row) >= 5 and row[4] in DATASETS else default_dataset
            if dataset:
                tasks.add((row[3], dataset))
    return tasks


def last_summary_job_id() -> str | None:
    path = LOG_DIR / "last_summary_job.txt"
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def last_run_dir() -> Path | None:
    path = LOG_DIR / "last_run_dir.txt"
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return Path(value) if value else None


def last_log_dir() -> Path:
    path = LOG_DIR / "last_log_dir.txt"
    if not path.exists():
        return LOG_DIR
    value = path.read_text(encoding="utf-8").strip()
    return Path(value) if value else LOG_DIR


def format_duration(seconds: int) -> str:
    days, remainder = divmod(max(0, seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days}d {hours}h {minutes}m {seconds}s"


def elapsed_since_run_start(run_dir: Path | None) -> str | None:
    if run_dir is None:
        return None

    name = run_dir.name
    if name.startswith("run-"):
        timestamp = name.removeprefix("run-")
    elif name.startswith("smoketest-"):
        timestamp = name.removeprefix("smoketest-")
    else:
        return None

    try:
        started = datetime.strptime(timestamp, "%Y%m%d-%H%M%S-%f")
    except ValueError:
        try:
            started = datetime.strptime(timestamp, "%Y%m%d-%H%M%S")
        except ValueError:
            return None

    return format_duration(int((datetime.now() - started).total_seconds()))


def slurm_rows(job_ids: list[str]) -> list[dict[str, str]]:
    if not job_ids:
        return []
    try:
        result = subprocess.run(
            ["squeue", "-h", "-r", "-j", ",".join(job_ids), "-o", "%i|%T|%M|%R"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []

    rows = []
    for line in result.stdout.splitlines():
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        rows.append({"id": parts[0], "state": parts[1], "elapsed": parts[2], "reason": parts[3]})
    return rows


def main() -> None:
    job_ids = last_job_ids()
    summary_job_id = last_summary_job_id()
    run_dir = last_run_dir()
    log_dir = last_log_dir()
    tasks = load_tasks(run_dir)
    config_path = run_dir / "config.yml" if run_dir is not None and (run_dir / "config.yml").exists() else CONFIG_PATH
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    default_dataset = config.get("dataset") if config.get("dataset") != "all" else None

    completed = logged_tasks(log_dir, job_ids, "completed", default_dataset) & tasks
    failed = logged_tasks(log_dir, job_ids, "failed", default_dataset) & tasks
    done = completed | failed
    total = len(tasks)
    done_percent = 100 * len(done) / total if total else 0

    rows = slurm_rows(job_ids)
    states = Counter(row["state"] for row in rows)
    reasons = Counter(row["reason"] for row in rows if row["state"] == "PENDING")
    waiting = states.get("PENDING", 0)
    running = states.get("RUNNING", 0)
    queued_or_running = waiting + running
    remaining = max(0, total - len(done))

    print(f"Run dir: {run_dir if run_dir else 'unknown'}")
    print(f"Log dir: {log_dir}")
    elapsed = elapsed_since_run_start(run_dir)
    if elapsed:
        print(f"Elapsed: {elapsed}")
    print(f"Array job(s): {', '.join(job_ids) if job_ids else 'unknown'}")
    if summary_job_id:
        summary_rows = slurm_rows([summary_job_id])
        summary_state = summary_rows[0]["state"] if summary_rows else "not in queue / finished"
        print(f"Summary job: {summary_job_id} ({summary_state})")

    print(f"Done: {len(done)}/{total} ({done_percent:.1f}%)")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")
    print(f"Remaining: {remaining}")
    print(f"Waiting for allocation: {waiting}")
    print(f"Running: {running}")
    print(f"Queued/running model tasks: {queued_or_running}")

    if states:
        print("Slurm states:", ", ".join(f"{state}={count}" for state, count in sorted(states.items())))
    if reasons:
        print("Pending reasons:", ", ".join(f"{reason}={count}" for reason, count in reasons.most_common()))


if __name__ == "__main__":
    main()
