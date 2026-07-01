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
    slurm = config.get("slurm", {})

    LOG_DIR.mkdir(exist_ok=True)

    cmd = [
        "sbatch",
        f"--job-name={JOB_NAME}",
        f"--account={ACCOUNT}",
        f"--gres={slurm['gres']}",
        f"--mem={slurm['mem']}",
        f"--time={slurm['time']}",
        f"--output={STDOUT_PATTERN}",
        f"--error={STDERR_PATTERN}",
        str(SLURM_SCRIPT),
    ]

    print("Submitting:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
