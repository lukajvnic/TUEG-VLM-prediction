"""sbatch plumbing shared by train/train.py, train/run-eval.py and train/run-custom.py."""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pipeline import ACCOUNT, ROOT  # noqa: E402

SBATCH = """#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --account={account}
#SBATCH --time={time}
#SBATCH --mem={ram}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpus}
{array}#SBATCH --output={logs}/{job}-{jobid}.out

set -euo pipefail
log_task() {{ ( flock -x 9; echo "$(date -Iseconds),{job_env},{task_env},{job},${{1:-}},${{2:-}}" >&9 ) 9>>{logs}/tasks.csv; }}
log_task start
trap 'log_task end $?' EXIT
module load StdEnv/2023 python/3.11 cuda arrow  # arrow: LLaMA-Factory (datasets) needs the Alliance pyarrow, no wheel on PyPI works
source {root}/{venv}/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$SCRATCH/hf-cache
export TOKENIZERS_PARALLELISM=false
cd {root}
{command}
"""


def script(job, time, ram, cpus, gpus, command, array_last=None, venv=".venv"):
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    single = array_last is None
    return SBATCH.format(
        job=job, account=ACCOUNT, time=time, ram=ram, cpus=cpus, gpus=gpus, logs=logs, root=ROOT,
        command=command, venv=venv,
        array="" if single else f"#SBATCH --array=0-{array_last}\n",
        jobid="%j" if single else "%A_%a",
        job_env="$SLURM_JOB_ID" if single else "$SLURM_ARRAY_JOB_ID",
        task_env="0" if single else "$SLURM_ARRAY_TASK_ID")


def job_name(model, dataset):
    # both keys in the name, so the queue is readable and a resubmission can skip the pairs already pending
    return f"eeg-vlm-train-{model.replace(':', '-')}-{dataset}"


def queued():
    # names of this user's pending/running jobs; empty when squeue is unavailable (local dry runs)
    try:
        result = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
                                text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return set()
    return set(result.stdout.split()) if result.returncode == 0 else set()


def submit(text, env=None):
    exports = "".join(f",{k}={v}" for k, v in (env or {}).items())
    result = subprocess.run(["sbatch", f"--export=ALL{exports}"], input=text, text=True, capture_output=True)
    if result.returncode:
        sys.exit(f"sbatch failed: {result.stderr.strip()}")
    return result.stdout.strip()
