import csv
import json
import os
import subprocess
import sys
from base64 import b64encode
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pipeline import ROOT, base_spec, config, sync  # noqa: E402
from helpers.slurm import script, submit  # noqa: E402

# custom-code families are scored in the venv that trained them: their remote code targets transformers 4.x
# (DeepSeek-OCR imports LlamaFlashAttention2, MiniCPM uses 4.x mask utils) and does not import under the main
# venv's 5.8. Keys are `bases:` family names; anything else scores in .venv
EVAL_VENVS = {"minicpm": ".venv-llamafactory", "deepseek-ocr": ".venv-unsloth"}
# a (model, dataset) with more pending windows than this is split over several array tasks, each taking an
# interleaved slice (train/scripts/eval.py); measured 22-40 s/image on the 24B-class bases (2026-09-26), so TUSZ's 8,460
# windows would need ~68 h in one task against a 48 h walltime. 1,500 windows is ~17 h at the slowest rate
SHARD_WINDOWS = 1500
EVAL_RECORD = ROOT / "logs" / "eval-sft-tasks.csv"  # array job, task index, model, dataset per submitted task


def active_tasks():
    # (model, dataset) pairs whose array task is still queued or running: their rows are not in any eval file
    # yet, so a resubmission would run them twice and append duplicate rows
    if not EVAL_RECORD.exists():
        return set()
    try:
        result = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-o", "%F %K"],
                                text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return set()
    live = {tuple(line.split()) for line in result.stdout.splitlines() if line.strip()} if result.returncode == 0 else set()
    with EVAL_RECORD.open() as f:
        return {(job, task, model, dataset)[2:] for job, task, model, dataset in csv.reader(f) if (job, task) in live}


def eval_venv(base):
    return EVAL_VENVS.get(base.get("family"), ".venv")


def record_tasks(job, tasks):
    with EVAL_RECORD.open("a") as f:
        csv.writer(f).writerows([job, str(i), t["model"], t["dataset"]] for i, t in enumerate(tasks))


def main():
    cfg = config()
    specs = cfg["models"]
    conn = sync()
    groups = defaultdict(list)
    active = active_tasks()

    for name, spec in specs.items():
        if spec.get("backend", "ollama") != "hf":
            continue
        if "checkpoint" in spec and not (ROOT / spec["checkpoint"] / "manifest.json").exists():
            print(f"{name}: {spec['checkpoint']} has no manifest.json (training not finished), skipping")
            continue

        for dataset, count in conn.execute(
                "SELECT dataset, COUNT(*) FROM pipeline WHERE model = ? AND scope = 'full' AND evaled = 0 "
                "GROUP BY dataset ORDER BY dataset", (name,)):
            if (name, dataset) in active:
                print(f"{name} {dataset}: an eval task is still queued or running, skipping")
                continue
            shards = -(-count // SHARD_WINDOWS)  # ceil
            for shard in range(shards):
                groups[(spec["time"], spec["ram"], spec["cpus"], spec["gpus"], eval_venv(base_spec(cfg, spec["base"])))].append(
                    {"model": name, "dataset": dataset, "pending": count, "shard": shard, "shards": shards})
    if not groups:
        print("nothing to evaluate")
        return

    for (time, ram, cpus, gpus, venv), tasks in sorted(groups.items()):
        text = script("eeg-vlm-eval-sft", time, ram, cpus, gpus, f"python {ROOT}/train/scripts/eval.py", len(tasks) - 1, venv=venv)
        listed = ", ".join(f"{t['model']} {t['dataset']} ({t['pending']}" + (f" {t['shard'] + 1}/{t['shards']})" if t["shards"] > 1 else ")")
                           for t in tasks)
        payload = b64encode(json.dumps(tasks).encode()).decode()
        submitted = submit(text, {"SFT_EVAL_TASKS": payload})  # "Submitted batch job N"
        record_tasks(submitted.split()[-1], tasks)
        print(f"{submitted} - {listed}, {time}/{ram}/gpu:{gpus} ({venv})")


if __name__ == "__main__":
    main()
