import csv
import sqlite3
import sys
from pathlib import Path

limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(limit)
        break
    except OverflowError:
        limit //= 10

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["TUAB", "TUAR", "TUEP", "TUEV", "TUSL", "TUSZ"]
RATIONALE = "ground_truth_rationale"

SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline (
    path TEXT,
    model TEXT,
    dataset TEXT,
    split TEXT,
    sampled INTEGER DEFAULT 0,
    preprocessed INTEGER DEFAULT 0,
    rationale INTEGER DEFAULT 0,
    evaled INTEGER DEFAULT 0,
    judged INTEGER DEFAULT 0,
    PRIMARY KEY (path, model)
)"""

UPSERT = """
INSERT INTO pipeline (path, model, dataset, split, preprocessed, rationale, evaled, judged)
VALUES (?, ?, ?, ?, 1, ?, ?, ?)
ON CONFLICT(path, model) DO UPDATE SET
    preprocessed = excluded.preprocessed,
    rationale = excluded.rationale,
    evaled = excluded.evaled,
    judged = excluded.judged
"""

SUMMARY = """
SELECT dataset,
       COUNT(DISTINCT path),
       COUNT(DISTINCT CASE WHEN sampled THEN path END),
       COUNT(DISTINCT CASE WHEN rationale THEN path END),
       SUM(evaled),
       SUM(judged)
FROM pipeline GROUP BY dataset ORDER BY dataset
"""


def config():
    import yaml
    return yaml.safe_load((ROOT / "config.yml").read_text())


def db():
    conn = sqlite3.connect(ROOT / "pipeline.db")
    conn.execute(SCHEMA)
    return conn


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def pairs_in(path):
    return {(r["path"], r["model"]) for r in read_csv(path)}


def sync():
    models = list(config()["models"])
    conn = db()
    conn.execute("CREATE TEMP TABLE valid (path TEXT PRIMARY KEY)")
    for ds in DATASETS:
        folder = ROOT / "datasets" / ds
        images = read_csv(folder / "labels.csv")
        evaled = pairs_in(folder / "eval-baseline.csv")
        judged = pairs_in(folder / "judge-baseline.csv")
        conn.executemany(UPSERT, [
            (f"{ds}/{img['path']}", model, ds, img["path"].split("/")[0],
             int(bool((img[RATIONALE] or "").strip())),
             int((img["path"], model) in evaled),
             int((img["path"], model) in judged))
            for img in images for model in models])
        conn.executemany("INSERT OR IGNORE INTO valid VALUES (?)",
                         [(f"{ds}/{img['path']}",) for img in images])
    conn.execute("DELETE FROM pipeline WHERE path NOT IN (SELECT path FROM valid)")
    conn.execute(f"DELETE FROM pipeline WHERE model NOT IN ({','.join('?' * len(models))})", models)
    conn.commit()
    return conn


if __name__ == "__main__":
    for ds, images, sampled, rationales, evaled, judged in sync().execute(SUMMARY):
        print(f"{ds}: {images} images, {sampled} sampled, {rationales} rationales, {evaled} evaled, {judged} judged")


def append_row(path, header, row):
    import fcntl
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(header)
        writer.writerow(row)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def log_failure(stage, dataset, path, model, error):
    import datetime
    import os
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    error = " ".join(str(error).split())[:500]
    append_row(logs / "failures.csv",
               ["time", "stage", "dataset", "path", "model", "job", "task", "error"],
               [datetime.datetime.now().isoformat(timespec="seconds"), stage, dataset, path, model,
                os.environ.get("SLURM_ARRAY_JOB_ID", ""), os.environ.get("SLURM_ARRAY_TASK_ID", ""),
                error])


def image_message(path, text):
    import base64
    import mimetypes
    from langchain_core.messages import HumanMessage
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return HumanMessage(content=[
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": f"data:{mime};base64,{data}"},
    ])


ACCOUNT = "def-milad777"
CONTEXT_LENGTH = 8192

SBATCH = """#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --account={account}
#SBATCH --time={time}
#SBATCH --mem={ram}
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:{gpus}
#SBATCH --array=0-{last}%{concurrency}
#SBATCH --output={logs}/{job}-%A_%a.out

port=$((20000 + (SLURM_ARRAY_JOB_ID + SLURM_ARRAY_TASK_ID) % 40000))
export OLLAMA_HOST=127.0.0.1:$port
export OLLAMA_BASE_URL=http://127.0.0.1:$port
export OLLAMA_MODELS=$SCRATCH/ollama/models
export OLLAMA_CONTEXT_LENGTH={context}
export OLLAMA_NUM_PARALLEL={parallel}
log_task() {{ ( flock -x 9; echo "$(date -Iseconds),$SLURM_ARRAY_JOB_ID,$SLURM_ARRAY_TASK_ID,{job},$1,$2" >&9 ) 9>>{logs}/tasks.csv; }}
log_task start
apptainer exec --nv $SCRATCH/ollama/ollama.sif ollama serve > {logs}/ollama-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.log 2>&1 &
trap 'code=$?; kill %1 2>/dev/null; log_task end $code' EXIT
for i in $(seq 120); do curl -s $OLLAMA_BASE_URL > /dev/null && break; sleep 2; done
{command}
"""


def submit_array(job, time, ram, gpus, last, concurrency, parallel, command, env):
    import subprocess
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    script = SBATCH.format(job=job, account=ACCOUNT, time=time, ram=ram, gpus=gpus, last=last,
                           concurrency=concurrency, context=CONTEXT_LENGTH, parallel=parallel,
                           logs=logs, command=command)
    exports = "".join(f",{k}={v}" for k, v in env.items())
    result = subprocess.run(["sbatch", f"--export=ALL{exports}"], input=script,
                            text=True, capture_output=True, check=True)
    return result.stdout.strip()
