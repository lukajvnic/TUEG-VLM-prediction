import argparse
import json
import subprocess
import sys
from base64 import b64encode
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pipeline import ACCOUNT, ROOT, base_spec, checkpoint_dir, config, sync

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
module load StdEnv/2023 python/3.11 cuda
source {root}/.venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$SCRATCH/hf-cache
export TOKENIZERS_PARALLELISM=false
cd {root}
{command}
"""


def script(job, time, ram, cpus, gpus, command, array_last=None):
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    single = array_last is None
    return SBATCH.format(
        job=job, account=ACCOUNT, time=time, ram=ram, cpus=cpus, gpus=gpus, logs=logs, root=ROOT,
        command=command,
        array="" if single else f"#SBATCH --array=0-{array_last}\n",
        jobid="%j" if single else "%A_%a",
        job_env="$SLURM_JOB_ID" if single else "$SLURM_ARRAY_JOB_ID",
        task_env="0" if single else "$SLURM_ARRAY_TASK_ID")


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
    for model in models:
        base = base_spec(cfg, model)
        if base.get("status"):
            sys.exit(f"{model}: {base['status']} - see config.yml bases:")
        out = checkpoint_dir(model, dataset)
        if (out / "manifest.json").exists():
            print(f"{model} on {dataset}: {out}/manifest.json exists (finished), skipping")
            continue
        text = script("eeg-vlm-train", base["time"], base["ram"], spec["cpus"], base["gpus"],
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
