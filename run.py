import subprocess
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yml"
MODELS_FILE = PROJECT_ROOT / "models.txt"
SLURM_SCRIPT = PROJECT_ROOT / "scripts" / "run_array.sbatch"
LOG_DIR = PROJECT_ROOT / "logs"


def line_count(path: Path) -> int:
    if not path.exists():
        raise SystemExit(f"ERROR: missing {path}")
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config()
    concurrency = int(config.get("array-concurrency", 4))

    if concurrency < 1:
        raise SystemExit("ERROR: array-concurrency must be at least 1")
    if not SLURM_SCRIPT.exists():
        raise SystemExit(f"ERROR: missing {SLURM_SCRIPT}")

    n_models = line_count(MODELS_FILE)
    if n_models < 1:
        raise SystemExit(f"ERROR: {MODELS_FILE} is empty")

    LOG_DIR.mkdir(exist_ok=True)
    array = f"0-{n_models - 1}%{concurrency}"
    cmd = ["sbatch", f"--chdir={PROJECT_ROOT}", f"--array={array}", str(SLURM_SCRIPT)]

    print(f"Submitting {n_models} model(s): {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
