import base64
import json
import subprocess
from collections import defaultdict
from datetime import datetime
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

    groups = defaultdict(list)
    for model in models_to_run:
        spec = models[model] or {}
        key = (spec.get("time", "12:00:00"), spec.get("ram", "32G"), int(spec.get("gpus", 1)))
        groups[key].append(model)

    LOG_DIR.mkdir(exist_ok=True)
    run_id = "run-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = PROJECT_ROOT / "results" / run_id
    run_log_dir = LOG_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir.mkdir(parents=True, exist_ok=True)
    job_ids = []
    for (time_limit, ram, gpus), group_models in groups.items():
        n_models = len(group_models)
        array = f"0-{n_models - 1}%{min(concurrency, n_models)}"
        encoded_models = base64.b64encode(json.dumps(group_models).encode("utf-8")).decode("ascii")
        export = f"ALL,PI_RUN_ID={run_id},PI_LOG_DIR={run_log_dir},PI_MODEL_LIST_B64={encoded_models}"
        cmd = [
            "sbatch",
            f"--chdir={PROJECT_ROOT}",
            f"--array={array}",
            f"--time={time_limit}",
            f"--mem={ram}",
            f"--gres=gpu:{gpus}",
            f"--export={export}",
            f"--output={run_log_dir}/%x-%A_%a.out",
            f"--error={run_log_dir}/%x-%A_%a.err",
            str(SLURM_SCRIPT),
        ]

        print(f"Submitting {n_models} model(s) with gpu:{gpus} mem:{ram} time:{time_limit}: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output = result.stdout.strip()
        print(output)
        job_ids.append(output.split()[-1])

    summary_job_id = ""
    if job_ids:
        dependency = "afterany:" + ":".join(job_ids)
        cmd = [
            "sbatch",
            f"--chdir={PROJECT_ROOT}",
            f"--dependency={dependency}",
            "--job-name=tueg-vlm-summary",
            "--account=def-milad777",
            "--time=00:30:00",
            "--mem=4G",
            "--cpus-per-task=1",
            f"--output={run_log_dir}/%x-%j.out",
            f"--error={run_log_dir}/%x-%j.err",
            "--wrap",
            f"source .venv/bin/activate && python scripts/summarize.py {run_dir}",
        ]
        print(f"Submitting summary job: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output = result.stdout.strip()
        print(output)
        summary_job_id = output.split()[-1]

    (LOG_DIR / "last_array_job.txt").write_text(",".join(job_ids) + "\n", encoding="utf-8")
    (LOG_DIR / "last_run_dir.txt").write_text(f"{run_dir}\n", encoding="utf-8")
    (LOG_DIR / "last_log_dir.txt").write_text(f"{run_log_dir}\n", encoding="utf-8")
    if summary_job_id:
        (LOG_DIR / "last_summary_job.txt").write_text(f"{summary_job_id}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
