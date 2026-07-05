import subprocess
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yml"
SLURM_SCRIPT = PROJECT_ROOT / "scripts" / "run_with_gpu.sh"
LOG_DIR = PROJECT_ROOT / "logs"

JOB_NAME = "ollama-vlm"
ACCOUNT = "def-milad777"

STDOUT_PATTERN = str(LOG_DIR / "%x-%j.out")
STDERR_PATTERN = str(LOG_DIR / "%x-%j.err")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config()
    models = config["models"]
    selected_model = config["model"]

    if selected_model == "all":
        models_to_run = models.items()
    else:
        models_to_run = [(selected_model, models[selected_model])]

    LOG_DIR.mkdir(exist_ok=True)

    previous_job_id = None
    for model, settings in models_to_run:
        cmd = [
            "sbatch",
            f"--job-name={JOB_NAME}",
            f"--account={ACCOUNT}",
            f"--gres=gpu:{settings['gpus']}",
            f"--mem={settings['ram']}",
            f"--time={settings['time']}",
            f"--output={STDOUT_PATTERN}",
            f"--error={STDERR_PATTERN}",
            f"--export=ALL,EVAL_MODEL={model}",
        ]

        if previous_job_id:
            cmd.append(f"--dependency=afterok:{previous_job_id}")

        cmd.append(str(SLURM_SCRIPT))

        print("Submitting:")
        print(" ".join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout.strip())

        previous_job_id = result.stdout.strip().split()[-1]


if __name__ == "__main__":
    main()
