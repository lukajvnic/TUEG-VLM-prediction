import subprocess
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yml"
SLURM_SCRIPT = PROJECT_ROOT / "scripts" / "run_array.sbatch"
LOG_DIR = PROJECT_ROOT / "logs"


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

    models = config.get("models") or {}
    selected_model = config.get("model")
    models_to_run = list(models) if selected_model == "all" else [selected_model]

    if not models_to_run or not models_to_run[0]:
        raise SystemExit("ERROR: config.yml must set model")
    for model in models_to_run:
        if model not in models:
            raise SystemExit(f"ERROR: model {model!r} is not listed under models in config.yml")

    n_models = len(models_to_run)

    LOG_DIR.mkdir(exist_ok=True)
    array = f"0-{n_models - 1}%{concurrency}"
    cmd = ["sbatch", f"--chdir={PROJECT_ROOT}", f"--array={array}", str(SLURM_SCRIPT)]

    print(f"Submitting {n_models} model(s): {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
