# Operations — end-to-end workflow

How to go from raw EDFs to a scored benchmark, and the cluster specifics.

## Cluster

- **Narval** (`narval.alliancecan.ca`), Digital Research Alliance of Canada,
  account `def-milad777`. Also Rorqual/Nibi exist.
- **MIG** GPU slices available (`a100_3g.20gb` etc.). Monitor usage in the Metrix
  portal (`portail.narval.calculquebec.ca`).
- Motivation for many efficiency choices: a **resource-waste warning** about
  under-utilized full-GPU jobs.

## Full pipeline

### 1. Generate images (locally or on cluster; CPU-only, no GPU)
```bash
./generate-all.sh                 # all six in parallel, --overwrite
./generate-all.sh --train-windows 3 --test-windows 6   # lighter
```
Each `generate.py` is single-threaded; `generate-all.sh` runs the six in parallel.
Source EDFs must be present under `datasets/<DS>/v*.*.*/`.

### 1b. Fix labels (only after a regeneration; not needed otherwise)

> **Both scripts are currently deleted from the working tree** (still in HEAD;
> `git checkout -- datasets/relabel.py datasets/recover_tuar_background.py` to get
> them back). Their output is already applied to all six `labels.csv`, so this
> step is only needed after a regeneration. See `known-issues.md`.

```bash
python datasets/recover_tuar_background.py   # TUAR only; needs source EDFs+annotations
python datasets/relabel.py                   # all six; adds the `assessed` column
```
Both rewrite `labels.csv` **only** — no image is re-rendered, so existing PNGs
(including the copies already on the cluster) stay valid. Both are idempotent and
both support `--dry-run`. `relabel.py` needs nothing but `labels.csv`;
`recover_tuar_background.py` needs `datasets/TUAR/v3.0.1/` and refuses to write
unless every existing label reproduces exactly from source.

Since `labels.csv` travels via git, the cluster picks these up with `git pull` —
no re-transfer of images.

### 2. QC gate (always run before committing to a big run)

> **`qc.py` is not in the working tree and is gitignored, so it is not
> recoverable from git.** It has to be rewritten before this step can run.

```bash
python qc.py     # exit 0 = clean; hard failures = missing/blank images, size drift, patient leakage
```
Under-supported rare classes show as `~~` warnings (expected, non-blocking).

### 3. Bundle + transfer
```bash
./zip-datasets.sh                 # -> zips/TUAB.zip … (each has train/ + test/)
zip -0 -r zips.zip zips           # optional: one archive of the zips
rsync -avP zips.zip <user>@narval.alliancecan.ca:~/   # resumable (scp can't resume 40 GB)
```
Zips contain **only `train/` and `test/`** — `labels.csv` and code travel via git.
Clear stale image dirs before a fresh generation:
`rm -rf datasets/*/train datasets/*/test`.

### 4. On the cluster: put data in place
- `git pull` (code + the current `labels.csv` — must match the transferred images).
- Extract images: `unzip TUAB.zip -d datasets/TUAB` (train/ test/ land in place).
- Verify `labels.csv` matches images:
```bash
for d in datasets/*/; do ds=$(basename "$d")
  miss=$(comm -23 <(awk -F, 'NR>1{print $1}' "$d/labels.csv"|sort) <(cd "$d"&&find train test -name '*.png'|sort)|grep -c .)
  echo "$ds referenced-but-missing=$miss"; done
```
(both this and `python qc.py` catch a stale `labels.csv`.)

### 4b. Stage Ollama models from a login node

**Compute nodes have no outbound internet.** `registry.ollama.ai` and
`ollama.com` time out there, so a model must be in `$OLLAMA_MODELS`
(`$SCRATCH/ollama/models`) *before* submission — the roster's 35, the
`gemma3:12b` teacher, and `config.yml`'s `judge.model`.

This is not a limitation to work around, it is just a staging step. Runtime
pulling was never part of the eval sweep: `run_array.sbatch` has no pull and
never did. The judge and rationale sbatches had picked up an
`ollama show || ollama pull` fallback, which on a compute node can only hang for
~30 s and then fail — reporting a network timeout rather than "model not
staged". That cost a whole judge array on 2026-08-16. Both now fail fast and
name the model, and `run-judge.py check` / `submit` verify the judge manifest
before queueing anything.

List what is staged, from a login node:
```bash
export OLLAMA_MODELS=$SCRATCH/ollama/models
apptainer exec $SCRATCH/ollama/ollama.sif ollama list
```

Stage a missing one (login node has internet; no GPU needed):
```bash
export OLLAMA_MODELS=$SCRATCH/ollama/models APPTAINERENV_OLLAMA_MODELS=$SCRATCH/ollama/models
apptainer exec $SCRATCH/ollama/ollama.sif ollama serve &
apptainer exec $SCRATCH/ollama/ollama.sif ollama pull qwen2.5:14b-instruct
```
A wrong tag fails here immediately, which is the cheap place to find out.

### 4c. Stage Hugging Face checkpoints (fine-tune track only)

`hf-install.py` downloads full HF snapshots (weights + processor + chat template)
because Ollama's GGUF weights cannot feed a PyTorch LoRA fine-tune. Same network
constraint as 4b: **run it from a login node**, since compute nodes cannot reach
`huggingface.co`. The committed `hf-install.sbatch` therefore cannot work as
written — a compute-node job has no route out.

**Do not try to pull all ~40 repositories in one pass.** That run is heavy enough
(parallel connections, decompression and hashing across repos that are tens of GB
each) that it exceeds what a login node allows one user, and it dies partway
through with an error. Run it repeatedly over a handful of repos at a time
instead: `is_complete()` re-checks each repo against the current remote revision
and skips anything already fully on disk, so a re-run costs only the remainder.
The original download was spread across roughly a week that way.

Gated repos (Llama, Gemma) need their licence accepted on Hugging Face first and
fail loudly until it is. Both `hf-install.py` and `hf-install.sbatch` are
currently deleted from the working tree — `git checkout -- hf-install.py
hf-install.sbatch`.

### 5a. Generate rationales (train split, fine-tune data)
```bash
python eval/scripts/generate-rationales.py --split train --dry-run   # free preflight
sbatch eval/scripts/generate-rationales.sbatch     # from the project dir
```
- 1 GPU, 32G, `--time=2-00:00:00`, Ollama **via Apptainer**, teacher
  `gemma3:12b`. Resumable — resubmit if it times out.
- Monitor: `squeue -u $USER`, `tail -f generate-rationales-<jobid>.out`,
  `wc -l datasets/*/rationales.csv`.
- Degenerate output (a wedged runner repeating one token) is caught at write
  time: the row is left blank for a later run to retry, and 20 in a row aborts
  the job. No post-hoc cleanup step is needed.

### 5b. Run the eval benchmark (test split, 35 VLMs)

Check what will actually be evaluated first — this is free and catches a stale
`labels.csv` immediately:
```bash
python eval/scripts/sample.py         # per-dataset sampled counts + class support
```

**Calibrating throughput.** Nothing about throughput is measured; walltimes carry
2.5x headroom instead. A `limit: 50` pilot would measure it directly, but on a
busy cluster the queue wait for the pilot can exceed what it saves. The cheaper
route is to submit, then read `seff <jobid>` off the *first tasks that finish* and
`scontrol update` anything still pending — see "Adjusting jobs that are already
queued" below.

**Stage 1 — screen (optional).** Set `probe-images: 350`, run all 35 models, score, and keep
the models whose recording-level macro-F1 CI clears the constant-predictor
baseline that `summarize.py` prints.

**Stage 2 — full run.** Blank `probe-images`, comment out the models that did not
clear the floor, and run.

```bash
python eval/run-eval.py               # submits one array job per resource group
```

**If you have higher-priority work still pending**, do not submit the sweep
naked — 210 tasks will compete with it:
```bash
squeue -u $USER                                   # get the pending job's ID
python eval/run-eval.py --after <jobid> --nice 10000
```
`--after` (not `afterok`) releases the arrays once that job is *running*. At that
point it holds its GPU, Alliance partitions do not preempt, and nothing the sweep
does can take it away. `--nice` keeps the sweep below your other work for the
whole run rather than only at submission.

Useful while waiting: `squeue -u $USER --start` (estimated start),
`sprio -j <jobid>` (priority breakdown), `sshare -A def-milad777_gpu` (fair-share).
- Monitor: `cat eval/runs/<run>/status.csv`, `tail -f eval/runs/<run>/logs/*.out`,
  `wc -l eval/runs/<run>/results/*.csv`.
- Prereqs on cluster: `$SCRATCH/ollama/ollama.sif`, `$SCRATCH/ollama/models`, venv.

### 6. Retry failures, merge, then score
```bash
python eval/create-retry-config.py <run>   # disables succeeded tasks + sets resume-from
python eval/run-eval.py                    # runs only the unfinished work -> NEW run dir
python eval/scripts/merge-runs.py <run> <retry-run> --dry-run
python eval/scripts/merge-runs.py <run> <retry-run>   # -> <retry-run>-merged
python eval/summarize.py <retry-run>-merged # per-class + recording-level metrics
# charts plot recording balanced accuracy; --metric macro-f1 for the reportable one
```
The merge is **not optional**: the retry writes to a new run dir and carries only
the images it redid, so scoring either dir alone under-reports. See
[eval-pipeline.md](eval-pipeline.md).

### 7. Rationale agreement (optional, after scoring)

Grades *why* each model answered, by comparing its zero-shot rationale to a
reference rationale for the same window. See the "Rationale agreement" section of
[eval-pipeline.md](eval-pipeline.md).

**Step 0 is not optional.** Reference rationales cover the test split, which the
train-only rationale pass never touched — without it every join is empty.

```bash
# 0. reference rationales for the 14,850 sampled test windows (teacher: gemma3:12b)
python eval/scripts/generate-rationales.py --split test --dry-run   # free preflight
sbatch --export=ALL,SPLIT=test eval/scripts/generate-rationales.sbatch

# 1. confirm the join before spending any GPU time
python eval/run-judge.py check <run>

# 2. judge: one array task per model (judge model from config.yml's `judge:` key)
python eval/run-judge.py submit <run> [--nice 10000]

# 3. aggregate
python eval/run-judge.py report <run>
```

`check` also verifies the judge model is staged (step 4b) and exits non-zero if
not, and `submit` refuses outright — 35 tasks that each die on a registry timeout
is a whole allocation spent on nothing.

Step 0 can run **concurrently with the eval sweep** — it derives its target
windows from `eval/config.yml`'s sampling policy, not from a finished run. If the
policy changed after the sweep was submitted, use
`--split test --run <run>` instead to take the windows straight from that run's
results and guarantee full coverage.

Outputs land in the run dir: `agreement/<model>-<dataset>.csv` per pair,
`rationale-agreement.csv` per model×dataset, `rationale-agreement-by-model.csv`
as the leaderboard. Re-run `python eval/summarize.py <run>` afterwards for the
`agreement-<DATASET>.png` charts; it reads `agreement/` directly. Read `clears_control` before the rank — a model that does not
beat its own shuffled-reference floor is writing EEG boilerplate.

Both steps are resumable (per row, per pair), so requeue by resubmitting.

## Monitoring cheat-sheet

| What | Command |
|---|---|
| Queue | `squeue -u $USER` |
| Estimated start / priority | `squeue -u $USER --start` / `sprio -j <jobid>` |
| Rationale progress | `tail -f generate-rationales-*.out` / `wc -l datasets/*/rationales*.csv` |
| Agreement join coverage | `python eval/run-judge.py check <run>` |
| Agreement progress | `wc -l eval/runs/<run>/agreement/*.csv` |
| **Eval run rollup** | **`python eval/status.py <run>`** |
| Eval progress | `tail -f eval/runs/<run>/logs/*.out` / `cat eval/runs/<run>/status.csv` |
| Eval exact counts | `wc -l eval/runs/<run>/results/*.csv` |
| Resource efficiency | `seff <jobid>` / `sacct -j <jobid> --format=JobID,ReqMem,MaxRSS,Elapsed,Timelimit,State` |

`eval/status.py <run>` is the one to reach for: it reconciles `status.csv` against
live `squeue`/`sacct` state, **writes back** terminal outcomes for tasks that died
without updating their own row (a task killed by the walltime never gets to mark
itself failed), and prints a rollup of finished / running / waiting-for-allocation.

## Adjusting jobs that are already queued

Editing `eval/scripts/eval.py` or `labels.csv` affects any task that has **not yet
started** — the sbatch runs the script from disk at task start, so a `git pull` is
enough. What is frozen at submission is `config.yml` (copied into
`eval/runs/<run>/config.yml`), the baked-in task list, and the Slurm resource
request. Those can still be changed on *pending* jobs without resubmitting and
losing accrued priority:

```bash
scontrol update JobId=<arrayjobid> Nice=10000
scontrol update JobId=<arrayjobid> Dependency=after:<other-jobid>
scontrol update JobId=<arrayjobid> MinMemoryNode=24G
```

## Resume semantics (important)

- **Rationales:** just resubmit — `init_csv` reconciles, done rows skipped.
- **Eval:** `create-retry-config.py <run>` (task-level, auto-sets resume-from) +
  resubmit (image-level skip), then `scripts/merge-runs.py` to union the two run dirs
  before scoring. A same-dir Slurm requeue also auto-resumes.
- A partially-failed eval task exits 0 and is marked `fail` in `status.csv` (not
  in Slurm's accounting) — judge success by `status.csv`, not `squeue`.

## Walltime philosophy

Over-requesting walltime does **not** waste allocation (Slurm bills actual
runtime) but **slows scheduling** (harder to backfill). Because every long job is
resumable, request tight limits for faster starts -- but weigh that against a
requeue costing days on a busy cluster (eval walltimes carry 2.5x headroom for
exactly this reason).
