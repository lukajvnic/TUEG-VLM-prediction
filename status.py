import csv
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yml"
LOG_DIR = PROJECT_ROOT / "logs"


def load_models() -> list[str]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    models = list(config.get("models", {}))
    return models if config.get("model") == "all" else [config.get("model")]


def last_job_ids() -> list[str]:
    path = LOG_DIR / "last_array_job.txt"
    if not path.exists():
        return []
    return [job_id for job_id in path.read_text(encoding="utf-8").strip().split(",") if job_id]


def logged_models(path: Path, job_ids: list[str]) -> set[str]:
    if not path.exists():
        return set()

    models = set()
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if len(row) >= 4 and (not job_ids or row[1] in job_ids):
                models.add(row[3])
    return models


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
    models = set(load_models())
    job_ids = last_job_ids()
    summary_job_id = last_summary_job_id()
    run_dir = last_run_dir()
    log_dir = last_log_dir()

    completed = logged_models(log_dir / "completed_models.csv", job_ids) & models
    failed = logged_models(log_dir / "failed_models.csv", job_ids) & models
    done = completed | failed
    total = len(models)
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
