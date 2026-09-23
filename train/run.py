import argparse
import json
import os
import subprocess
import sys
from base64 import b64encode
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pipeline import ACCOUNT, ROOT, base_spec, checkpoint_dir, config, sync

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

snapshot_dir = import_module("hf-install").snapshot_dir

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


def job_name(model):
    # one job per base and the key in the name, so the queue is readable and a resubmission can skip pending ones
    return "eeg-vlm-train-" + model.replace(":", "-")


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


def train(args):
    cfg = config()
    spec = cfg["train"]
    dataset = args.dataset or spec["dataset"]
    if args.model:
        models = [args.model]
    else:  # every base without a `status` (the custom-code ones train outside this script)
        models = [k for k, v in cfg.get("bases", {}).items() if not v.get("status")]
    if not args.dry_run:
        for name in ("sft_train.jsonl", "sft_val.jsonl"):
            if not (ROOT / "datasets" / dataset / name).exists():
                sys.exit(f"missing datasets/{dataset}/{name} - run helpers/build-sft-jsonl.py {dataset} first")
    pending = queued()
    if "eeg-vlm-train" in pending and not args.model and not args.dry_run:
        sys.exit("jobs named plain `eeg-vlm-train` (submitted before per-base names, 2026-09-23) are queued or "
                 "running and cannot be matched to a base: wait for them, or submit with --model KEY for each base "
                 "that is not among them")
    for model in models:
        base = base_spec(cfg, model)
        if base.get("status"):
            sys.exit(f"{model}: {base['status']} - see config.yml bases:")
        out = checkpoint_dir(model, dataset)
        if (out / "manifest.json").exists():
            print(f"{model} on {dataset}: {out}/manifest.json exists (finished), skipping")
            continue
        if job_name(model) in pending:
            print(f"{model}: {job_name(model)} is already queued or running, skipping")
            continue
        if not args.dry_run and not (snapshot_dir(base["repo"]) / "config.json").exists():
            print(f"{model}: {snapshot_dir(base['repo'])} not staged (compute nodes are offline) - "
                  f"run train/hf-install.py {model} first, skipping")
            continue
        text = script(job_name(model), base["time"], base["ram"], spec["cpus"], base["gpus"],
                      f"python {ROOT}/train/scripts/finetune_sample.py")
        env = {"TRAIN_MODEL": model, "TRAIN_DATASET": dataset}
        if args.dry_run:
            if model == models[0]:
                print(text)
            print(f"# {model} on {dataset} -> {out}  base: {base['repo']}  {base['time']}/{base['ram']}/gpu:{base['gpus']}")
            continue
        print(f"{submit(text, env)} - {model} on {dataset} -> {out}, {base['time']}/{base['ram']}/gpu:{base['gpus']}")


def predict(args):
    specs = config()["models"]
    conn = sync()
    groups = defaultdict(list)
    for name, spec in specs.items():
        if spec.get("backend", "ollama") != "hf":
            continue
        if "checkpoint" in spec and not (ROOT / spec["checkpoint"] / "manifest.json").exists():
            print(f"{name}: {spec['checkpoint']} has no manifest.json (training not finished), skipping")
            continue
        for dataset, count in conn.execute(
                "SELECT dataset, COUNT(*) FROM pipeline WHERE model = ? AND scope = 'full' AND evaled = 0 "
                "GROUP BY dataset ORDER BY dataset", (name,)):
            groups[(spec["time"], spec["ram"], spec["cpus"], spec["gpus"])].append(
                {"model": name, "dataset": dataset, "pending": count})
    if not groups:
        print("nothing to predict")
        return
    for (time, ram, cpus, gpus), tasks in sorted(groups.items()):
        text = script("eeg-vlm-predict", time, ram, cpus, gpus, f"python {ROOT}/train/predict.py", len(tasks) - 1)
        listed = ", ".join(f"{t['model']} {t['dataset']} ({t['pending']})" for t in tasks)
        if args.dry_run:
            print(text)
            print(f"# tasks: {listed}")
            continue
        payload = b64encode(json.dumps(tasks).encode()).decode()
        print(f"{submit(text, {'PREDICT_TASKS': payload})} - {listed}, {time}/{ram}/gpu:{gpus}")


def main():
    parser = argparse.ArgumentParser(description="fine-tune track jobs: train (one job per `bases:` entry) or predict (backend: hf models)")
    parser.add_argument("command", nargs="?", choices=["train", "predict"], default="train")
    parser.add_argument("--dry-run", action="store_true", help="print the sbatch script(s), submit nothing")
    parser.add_argument("--model", help="train: only this `bases:` key (Ollama tag); default is every base without a `status`")
    parser.add_argument("--dataset", help="train: dataset instead of config.yml train.dataset")
    args = parser.parse_args()
    (train if args.command == "train" else predict)(args)


if __name__ == "__main__":
    main()
