import csv
import subprocess
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yml"
LOG_DIR = PROJECT_ROOT / "logs"


def load_models() -> list[str]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    models = list(config.get("models", {}))
    return models if config.get("model") == "all" else [config.get("model")]


def last_job_id() -> str | None:
    path = LOG_DIR / "last_array_job.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def logged_models(path: Path, job_id: str | None) -> set[str]:
    if not path.exists():
        return set()

    models = set()
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if len(row) >= 4 and (job_id is None or row[1] == job_id):
                models.add(row[3])
    return models


def slurm_states(job_id: str | None) -> Counter:
    if not job_id:
        return Counter()
    try:
        result = subprocess.run(
            ["squeue", "-h", "-r", "-j", job_id, "-o", "%T"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return Counter()
    return Counter(line.strip() for line in result.stdout.splitlines() if line.strip())


def main() -> None:
    models = set(load_models())
    job_id = last_job_id()
    completed = logged_models(LOG_DIR / "completed_models.csv", job_id) & models
    failed = logged_models(LOG_DIR / "failed_models.csv", job_id) & models
    done = completed | failed
    total = len(models)
    percent = 100 * len(done) / total if total else 0

    print(f"Job: {job_id or 'unknown'}")
    print(f"Done: {len(done)}/{total} ({percent:.1f}%)")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")
    print(f"Remaining: {total - len(done)}")

    states = slurm_states(job_id)
    if states:
        print("Slurm:", ", ".join(f"{state}={count}" for state, count in sorted(states.items())))


if __name__ == "__main__":
    main()
