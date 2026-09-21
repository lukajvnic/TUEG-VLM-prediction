import csv
import re
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
HASHES = "hashes.csv"  # path,md5 per rendered PNG, written by helpers/hash-images.py

SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline (
    path TEXT,
    model TEXT,
    dataset TEXT,
    split TEXT,
    labeled INTEGER DEFAULT 0,
    scope TEXT DEFAULT 'none',
    sampled INTEGER DEFAULT 0,
    preprocessed INTEGER DEFAULT 0,
    rationale INTEGER DEFAULT 0,
    evaled INTEGER DEFAULT 0,
    judged INTEGER DEFAULT 0,
    judged_gpt INTEGER DEFAULT 0,
    done INTEGER DEFAULT 0,
    zero_shot INTEGER DEFAULT 1,
    PRIMARY KEY (path, model)
)"""

UPSERT = """
INSERT INTO pipeline (path, model, dataset, split, labeled, preprocessed, rationale, evaled, judged, judged_gpt)
VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
ON CONFLICT(path, model) DO UPDATE SET
    labeled = excluded.labeled,
    preprocessed = excluded.preprocessed,
    rationale = excluded.rationale,
    evaled = excluded.evaled,
    judged = excluded.judged,
    judged_gpt = excluded.judged_gpt
"""

SCOPE_UPDATE = """
UPDATE pipeline SET scope = CASE
    WHEN split = 'test' AND sampled = 1 THEN 'full'
    WHEN split = 'train' AND sampled = 1 AND labeled = 1 THEN 'rationale'
    ELSE 'none' END
"""

DONE_UPDATE = """
UPDATE pipeline SET done = CASE scope
    WHEN 'full' THEN rationale AND evaled AND judged AND judged_gpt
    WHEN 'rationale' THEN rationale
    ELSE 1 END
"""

SUMMARY = """
SELECT dataset,
       COUNT(DISTINCT path),
       COUNT(DISTINCT CASE WHEN sampled AND split = 'test' THEN path END),
       COUNT(DISTINCT CASE WHEN sampled AND split = 'train' THEN path END),
       COUNT(DISTINCT CASE WHEN rationale THEN path END),
       SUM(evaled),
       SUM(judged),
       SUM(judged_gpt),
       SUM(done),
       COUNT(*)
FROM pipeline GROUP BY dataset ORDER BY dataset
"""


def config():
    import yaml
    return yaml.safe_load((ROOT / "config.yml").read_text())


def db():
    conn = sqlite3.connect(ROOT / "pipeline.db")
    conn.execute("PRAGMA journal_mode=MEMORY")  # derived db: rebuildable, skip lustre fsync cost
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(pipeline)")}
    for col, default in (("judged_gpt", 0), ("done", 0), ("zero_shot", 1)):
        if col not in cols:
            conn.execute(f"ALTER TABLE pipeline ADD COLUMN {col} INTEGER DEFAULT {default}")
    return conn


# phrases that say the writer could not see what it was asked to justify; checked only on rows whose
# label is a positive finding, since "no evidence of spikes" is the right thing to say for normal/bckg
HEDGE = re.compile(r"\b(cannot (be )?(determine|confirm|identify|verify|assess)|can't (determine|confirm|identify)"
                   r"|unable to|not possible to|difficult to (identify|determine|discern|confirm)"
                   r"|as an ai|i (cannot|can't|am unable)|without (more|additional|further) (information|context|data))\b",
                   re.I)
NEGATIVE_LABELS = {"normal", "no_epilepsy", "bckg"}


def is_degenerate(text):
    words = text.split()
    return len(text) < 40 or len(words) < 8 or len(set(text)) < 12 or len(set(words)) / len(words) < 0.2


def rationale_problems(text, positives):
    text = text.strip()
    if not text:
        return []
    problems = []
    if is_degenerate(text):
        problems.append("degenerate")
    if text[-1] not in ".!?\")":
        problems.append("truncated")  # ran into the num_predict cap mid-sentence
    if positives - NEGATIVE_LABELS and HEDGE.search(text):
        problems.append("hedge")
    return problems


def has_labels(row):
    return any(v.strip().lower() == "true" for c, v in row.items() if c not in ("path", RATIONALE))


def positives(row):
    return frozenset(c for c, v in row.items() if c not in ("path", RATIONALE) and v.strip().lower() == "true")


def parse_name(path):
    patient, scan, window = Path(path).stem.rsplit("_", 2)
    return patient, f"{patient}_{scan}", int(window)


def spread(items, cap):
    if cap is None or cap <= 0 or cap >= len(items):
        return list(items)
    if cap == 1:
        return [items[len(items) // 2]]
    picked = {round(i * (len(items) - 1) / (cap - 1)) for i in range(cap)}
    return [items[i] for i in sorted(picked)]


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_hashes(dataset):
    return {r["path"]: r["md5"] for r in read_csv(ROOT / "datasets" / dataset / HASHES)}


def duplicate_images():
    """Byte-identical renders across all six corpora (measured 2026-09-21: 2,352 groups, 85 spanning
    different patient tokens). Returns (hashes per dataset, md5 -> [(dataset, path)] for repeated images,
    patient -> canonical patient), where patients sharing an image are merged into one."""
    hashes = {ds: read_hashes(ds) for ds in DATASETS}
    for ds in DATASETS:
        if not hashes[ds]:
            sys.exit(f"datasets/{ds}/{HASHES} missing - run helpers/hash-images.py {ds}")
    by_hash = {}
    for ds in DATASETS:
        for path, md5 in hashes[ds].items():
            by_hash.setdefault(md5, []).append((ds, path))
    groups = {md5: items for md5, items in by_hash.items() if len(items) > 1}
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for items in groups.values():
        patients = sorted({parse_name(p)[0] for _, p in items})
        for other in patients[1:]:
            a, b = find(patients[0]), find(other)
            if a != b:
                parent[max(a, b)] = min(a, b)
    canon = {p: find(p) for p in parent}
    return hashes, groups, canon


def pairs_in(path):
    return {(r["path"], r["model"]) for r in read_csv(path)}


def base_spec(cfg, key):
    # `bases:` maps an Ollama tag to its HF weights (+ resources, trust-remote-code); a bare HF repo id also works
    bases = cfg.get("bases", {})
    spec = dict(bases[key]) if key in bases else {"repo": key}
    spec.setdefault("gpus", cfg["train"]["gpus"])
    spec.setdefault("ram", cfg["train"]["ram"])
    return spec


def checkpoint_dir(model_key, dataset):
    return ROOT / "checkpoints" / model_key.replace(":", "-") / dataset


def model_datasets(spec):
    # a model entry may name the datasets it is evaluated on (fine-tunes); Ollama models run on all six
    return [ds for ds in DATASETS if ds in spec.get("datasets", DATASETS)]


def sync():
    specs = config()["models"]
    models = list(specs)
    conn = db()
    conn.execute("CREATE TEMP TABLE valid (path TEXT PRIMARY KEY)")
    for ds in DATASETS:
        folder = ROOT / "datasets" / ds
        images = read_csv(folder / "labels.csv")
        evaled = pairs_in(folder / "eval-baseline.csv")
        judged = pairs_in(folder / "judge-baseline.csv")
        judged_gpt = pairs_in(folder / "judge-gpt.csv")
        ds_models = [m for m in models if ds in model_datasets(specs[m])]
        conn.executemany(UPSERT, [
            (f"{ds}/{img['path']}", model, ds, img["path"].split("/")[0], int(has_labels(img)),
             int(bool((img[RATIONALE] or "").strip())),
             int((img["path"], model) in evaled),
             int((img["path"], model) in judged),
             int((img["path"], model) in judged_gpt))
            for img in images for model in ds_models])
        conn.executemany("DELETE FROM pipeline WHERE dataset = ? AND model = ?",
                         [(ds, m) for m in models if m not in ds_models])
        conn.executemany("INSERT OR IGNORE INTO valid VALUES (?)",
                         [(f"{ds}/{img['path']}",) for img in images])
        print(f"sync {ds}: {len(images)} images, {len(evaled)} evals, "
              f"{len(judged)} judgements, {len(judged_gpt)} gpt judgements", flush=True)
    conn.execute("DELETE FROM pipeline WHERE path NOT IN (SELECT path FROM valid)")
    conn.execute(f"DELETE FROM pipeline WHERE model NOT IN ({','.join('?' * len(models))})", models)
    # `sampled` is a property of the path, set by the samplers on the rows that existed at the time;
    # rows for a model added to config.yml later inherit it here instead of needing the samplers re-run
    conn.execute("UPDATE pipeline SET sampled = 1 WHERE sampled = 0 "
                 "AND path IN (SELECT path FROM pipeline WHERE sampled = 1)")
    # zero_shot = 1 for the Ollama roster, 0 for backend: hf entries (the base-as-is and the fine-tunes), so
    # "what is left for the zero-shot benchmark" is WHERE zero_shot AND NOT done
    hf = [m for m in models if specs[m].get("backend", "ollama") == "hf"]
    placeholders = ",".join("?" * len(hf)) or "''"
    conn.execute(f"UPDATE pipeline SET zero_shot = CASE WHEN model IN ({placeholders}) THEN 0 ELSE 1 END", hf)
    conn.execute(SCOPE_UPDATE)
    conn.execute(DONE_UPDATE)
    conn.commit()
    return conn


if __name__ == "__main__":
    for ds, images, sampled, train_sampled, rationales, evaled, judged, judged_gpt, done, total in sync().execute(SUMMARY):
        print(f"{ds}: {images} images, {sampled} sampled, {train_sampled} train sampled, {rationales} rationales, "
              f"{evaled} evaled, {judged} judged, {judged_gpt} gpt judged, {done}/{total} done")


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

set -euo pipefail
log_task() {{ ( flock -x 9; echo "$(date -Iseconds),$SLURM_ARRAY_JOB_ID,$SLURM_ARRAY_TASK_ID,{job},${{1:-}},${{2:-}}" >&9 ) 9>>{logs}/tasks.csv; }}
log_task start
trap 'code=$?; kill %1 2>/dev/null || true; log_task end $code' EXIT
module load StdEnv/2023 apptainer/1.4.5
source {root}/.venv/bin/activate
port=$((20000 + (SLURM_ARRAY_JOB_ID + SLURM_ARRAY_TASK_ID) % 40000))
export OLLAMA_HOST=127.0.0.1:$port
export OLLAMA_BASE_URL=http://127.0.0.1:$port
export OLLAMA_MODELS=$SCRATCH/ollama/models
export OLLAMA_CONTEXT_LENGTH={context}
export OLLAMA_NUM_PARALLEL={parallel}
export APPTAINERENV_OLLAMA_MODELS=$OLLAMA_MODELS
export APPTAINERENV_OLLAMA_HOST=$OLLAMA_HOST
export APPTAINERENV_OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL
export APPTAINERENV_OLLAMA_CONTEXT_LENGTH=$OLLAMA_CONTEXT_LENGTH
apptainer exec --nv $SCRATCH/ollama/ollama.sif ollama serve > {logs}/ollama-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.log 2>&1 &
for i in $(seq 120); do curl -s $OLLAMA_BASE_URL > /dev/null && break; sleep 2; done
curl -s $OLLAMA_BASE_URL > /dev/null || {{ echo "ollama never became ready" >&2; exit 1; }}
{command}
"""


def submit_array(job, time, ram, gpus, last, concurrency, parallel, command, env, context=CONTEXT_LENGTH):
    import subprocess
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    script = SBATCH.format(job=job, account=ACCOUNT, time=time, ram=ram, gpus=gpus, last=last,
                           concurrency=concurrency, context=context, parallel=parallel,
                           logs=logs, command=command, root=ROOT)
    exports = "".join(f",{k}={v}" for k, v in env.items())
    result = subprocess.run(["sbatch", f"--export=ALL{exports}"], input=script,
                            text=True, capture_output=True)
    if result.returncode:
        raise SystemExit(f"sbatch failed: {result.stderr.strip()}")
    return result.stdout.strip()
