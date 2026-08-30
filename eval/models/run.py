import json
import sys
from base64 import b64encode
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from helpers.pipeline import ROOT, config, submit_array, sync

PARALLEL_EXPR = ("$(python -c 'import base64, json, os; "
                 "print(json.loads(base64.b64decode(os.environ[\"EVAL_TASKS\"]))"
                 "[int(os.environ[\"SLURM_ARRAY_TASK_ID\"])][\"parallel\"])')")


def pending_tasks(conn):
    return conn.execute(
        "SELECT model, dataset, COUNT(*) FROM pipeline WHERE sampled = 1 AND evaled = 0 "
        "GROUP BY model, dataset ORDER BY model, dataset").fetchall()


def submit(group, tasks, settings):
    time, ram, gpus = group
    payload = b64encode(json.dumps(tasks).encode()).decode()
    out = submit_array("eeg-vlm-eval", time, ram, gpus, len(tasks) - 1,
                       settings["array-concurrency"], PARALLEL_EXPR,
                       f"python {ROOT}/eval/models/eval.py", {"EVAL_TASKS": payload})
    pending = sum(t["pending"] for t in tasks)
    print(f"{out} - {len(tasks)} tasks, {pending} images, {time}/{ram}/gpu:{gpus}")


def main():
    cfg = config()
    conn = sync()
    groups = defaultdict(list)
    for model, dataset, count in pending_tasks(conn):
        spec = cfg["models"][model]
        parallel = spec.get("parallel-requests", cfg["settings"]["parallel-requests"])
        groups[(spec["time"], spec["ram"], str(spec["gpus"]))].append(
            {"model": model, "dataset": dataset, "parallel": parallel, "pending": count})
    if not groups:
        print("nothing to run")
        return
    for group in sorted(groups):
        submit(group, groups[group], cfg["settings"])


if __name__ == "__main__":
    main()
