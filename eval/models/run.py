import json
import math
import os
import sys
from base64 import b64encode
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from helpers.pipeline import ROOT, config, submit_array, sync

def task_expr(key):
    return ("$(python -c 'import base64, json, os; "
            "print(json.loads(base64.b64decode(os.environ[\"EVAL_TASKS\"]))"
            f"[int(os.environ[\"SLURM_ARRAY_TASK_ID\"])][\"{key}\"])')")


def staged(model):
    name, _, tag = model.partition(":")
    manifests = Path(os.environ.get("SCRATCH", "")) / "ollama/models/manifests/registry.ollama.ai/library"
    return (manifests / name / (tag or "latest")).exists()


def pending_tasks(conn):
    return conn.execute(
        "SELECT model, dataset, COUNT(*) FROM pipeline WHERE scope = 'full' AND evaled = 0 "
        "GROUP BY model, dataset ORDER BY model, dataset").fetchall()


def full_sizes(conn):
    return dict(conn.execute("SELECT dataset, COUNT(DISTINCT path) FROM pipeline WHERE scope = 'full' GROUP BY dataset"))


MIN_MINUTES, LOAD_MINUTES, HEADROOM = 20, 10, 2.0


def scaled_time(configured, pending, full):
    # a model's configured walltime covers a full pass over its largest dataset; a retry of a few windows
    # scaled the same way (2x headroom + model load) queues in minutes instead of behind a 12 h request
    h, m, _ = (int(x) for x in configured.split(":"))
    minutes = h * 60 + m
    scaled = math.ceil(minutes * pending / max(full, 1) * HEADROOM) + LOAD_MINUTES
    minutes = max(MIN_MINUTES, min(minutes, scaled))
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def submit(group, tasks, settings):
    time, ram, gpus = group
    payload = b64encode(json.dumps(tasks).encode()).decode()
    out = submit_array("eeg-vlm-eval", time, ram, gpus, len(tasks) - 1,
                       settings["array-concurrency"], task_expr("parallel"),
                       f"python {ROOT}/eval/models/eval.py", {"EVAL_TASKS": payload},
                       context=task_expr("context"))
    pending = sum(t["pending"] for t in tasks)
    print(f"{out} - {len(tasks)} tasks, {pending} images, {time}/{ram}/gpu:{gpus}")


def main():
    cfg = config()
    conn = sync()
    groups = defaultdict(list)
    rows = [r for r in pending_tasks(conn) if cfg["models"][r[0]].get("backend", "ollama") == "ollama"]
    if os.environ.get("SCRATCH"):
        missing = sorted({model for model, _, _ in rows if not staged(model)})
        if missing:
            print(f"not staged in $SCRATCH/ollama/models, skipping: {', '.join(missing)}")
            rows = [r for r in rows if r[0] not in missing]
    full = full_sizes(conn)
    for model, dataset, count in rows:
        spec = cfg["models"][model]
        parallel = spec.get("parallel-requests", cfg["settings"]["parallel-requests"])
        groups[(scaled_time(spec["time"], count, full.get(dataset, count)), spec["ram"], str(spec["gpus"]))].append(
            {"model": model, "dataset": dataset, "parallel": parallel,
             "context": spec.get("context", 8192), "pending": count})
    if not groups:
        print("nothing to run")
        return
    for group in sorted(groups):
        submit(group, groups[group], cfg["settings"])


if __name__ == "__main__":
    main()
