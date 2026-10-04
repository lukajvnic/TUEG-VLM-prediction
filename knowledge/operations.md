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

The old QC gate script no longer exists. Until a replacement is written, check
manually before a big run: the labels↔images set comparison in step 4, no
blank/truncated PNGs, and patient-level split integrity if generation changed.

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
(this catches a stale `labels.csv`.)

### 4b. Stage Ollama models from a login node

**Compute nodes have no outbound internet.** `registry.ollama.ai` and
`ollama.com` time out there, so a model must be in `$OLLAMA_MODELS`
(`$SCRATCH/ollama/models`) *before* submission — the roster's 34, the
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

Ollama's GGUF weights cannot feed a PyTorch LoRA fine-tune, so the base model
comes from Hugging Face. Compute nodes cannot reach `huggingface.co`, so
everything here runs on a **login node**:

```bash
module load StdEnv/2023 python/3.11 cuda arrow
source .venv/bin/activate                          # the one venv; requirements.txt carries the torch stack too
pip install --no-index -r requirements.txt         # Alliance wheelhouse; drop --no-index if a wheel is missing
python train/scripts/hf-install.py --dry-run               # shows $SCRATCH/hf-checkpoints/Qwen--Qwen2.5-VL-7B-Instruct
python train/scripts/hf-install.py                         # ~16 GB, resumable
python train/scripts/hf-install.py --all --dry-run         # every base in config.yml bases: without a `status` (31 repos, custom-code ones included)
python train/scripts/hf-install.py gemma3:12b llava:7b     # keys or repo ids
```

Gated repos (Llama, Gemma) need their licence accepted on Hugging Face first
and fail loudly until it is. The module names above are what the sbatch in
`helpers/slurm.py` loads and have not been verified on Narval yet (2026-09-16);
fix both places together if they differ.

### 4d. Run the fine-tune

One fine-tune per (base, dataset) since 2026-09-30; pooling is gone
(methodology-decisions.md). `config.yml` `train.experiment` names the round
(currently `labels`): checkpoints `checkpoints/<key>/<DS>-<experiment>`, job
`eeg-vlm-train-<key>-<DS>-<experiment>`, scored as `<key>-sft-<DS>-<experiment>`,
results via `python helpers/experiment-results.py` (experiments.md).
`train.target` picks the answer: `labels` (booleans only, `sft_labels_*.jsonl`)
or `rationale` (rationale then booleans, `sft_*.jsonl`). The builder always
writes both from one split. A job reads both keys when it starts, so don't change
them while jobs are queued; start a new round under a new experiment name.

```bash
python helpers/build-sft-jsonl.py TUAB          # both targets; needs rationales, so cluster-side (5a); no args = all six. Writes nothing on any train/val/test overlap
python helpers/check-splits.py                  # re-check train.target's jsonl on disk: per-corpus overlap table, exit 1 on a leak
python train/train.py qwen2.5vl:7b TUAB --dry-run
python train/train.py qwen2.5vl:7b TUAB         # one sbatch job
python train/train.py all TUAB                  # one job per base; `qwen2.5vl:7b all` = one per dataset; `all all`
python train/train.py qwen2.5vl:7b TUAB --here  # train in this process: what the job runs; on an salloc GPU node for debugging
```
**One job trains, then scores (2026-10-03).** The job's command is `train.py <key> <DS> --here &&
train/scripts/eval.py` for that pair (`job_command()`). The test-set predictions come from the same allocation,
so there is no second ~2-day queue wait. They land in `datasets/<DS>/eval-<key>-sft-<DS>-<experiment>.csv` as if
`run-eval.py` had run them. The scoring step needs the model's `pipeline.db` rows: run `python helpers/pipeline.py`
(sync) once the checkpoint dir exists, which happens on submission or on the first run. Back up `pipeline.db` first,
since a sync killed mid-commit can corrupt it. If scoring stops early, `run-eval.py` resumes the pending windows.

**Rehearse before submitting (2026-10-03).** Run `python rehearsal/rehearse.py cpu <key> <DS> [--side]` on a login
node (~1 h; run it detached, see tooling.md). For the real model on a GPU, also run `python rehearsal/rehearse.py
gpu <key> <DS> [--side]` (a <=2 h job). Submit the real job only when the cpu report says `RESULT: PASS`.

A pair is skipped, with the reason printed, when the base has a `status` or a
`family` (custom stack, below), `checkpoints/<key>/<DS>/manifest.json`
exists, its job `eeg-vlm-train-<key>-<DS>` is queued or running, the snapshot
is not staged, `train.target`'s jsonl is missing, or it fails `check-splits.py`'s
overlap check (2026-09-30: the cluster's 2026-09-16 per-corpus jsonl leak
test patients and must be rebuilt first, see known-issues). A dir with
`checkpoint-*` and no manifest resumes from its latest checkpoint on
resubmission; delete the dir to restart a pair from scratch.

Walltime is `train.time` (24 h) unless the base overrides it: 48 h for the
multi-GPU bases. Those were sized for the pooled runs (6,563 train rows); a
per-dataset run is at most TUSZ's ~2,600, so they are generous (an estimate
by row count, not measured). Measured on the first full batch (pooled, 2
epochs, sacct elapsed, 2026-09-24):

| base | GPUs | elapsed |
|---|---|---|
| qwen3-vl:2b, medgemma:4b, medgemma1.5:4b, glm-ocr, llava:13b, llava-phi3, gemma4:e2b | 1 | 6.9 to 7.5 h |
| gemma4:e4b, qwen3-vl:4b, gemma3:12b | 1 | 9 to 9.4 h |
| llava-llama3:8b, gemma4:26b, qwen3-vl:8b, minicpm-v4.6 | 1-2 | 10.5 to 12 h |
| granite3.2-vision (7.4k image tokens), qwen3-vl:30b-a3b, gemma3:27b, gemma4:31b | 1-3 | 13 to 15 h |
| qwen2.5vl:7b | 1 | 16.9 h |
| mistral-small3.1 / 3.2:24b | 2 | 26.2 / 26.6 h |
| qwen2.5vl:32b, qwen3-vl:32b | 3 | still running at 31 h / 20 h |

So 24 h covers every 1-GPU base with margin; the 24B Mistrals need the 48 h;
the 32B Qwens are the ones to watch against 48 h.
Resubmitting after a partial batch only queues the unfinished pairs.
Rollout (project lead, 2026-09-30): qwen2.5vl:7b on TUAB alone, through
scoring, before any batch. Submitted 2026-09-30 as job 4342993
(label-only target, 12 h walltime: `qwen2.5vl:7b` `time: '12:00:00'` in
`bases:` puts it in the ≤12 h `gpubase_bygpu_b2` tier; ~5.5 h expected,
estimated from the first round's s/step, not measured).

Deploy of 2026-09-30 (`0796505`): commit + push locally, `git pull --ff-only`
on Narval. The cluster had untracked copies of files the commit added
(`datasets/*/hashes.csv`, same rows in another order; an older
`helpers/inspect-sft.py`), which would abort the pull; they were moved to
`$SCRATCH/deploy-backup-20260930/`. The cluster's `labels.csv` (rationales,
not in git) are never part of a commit. Long login-node checks
(`preflight.py`) run detached (`setsid nohup ... &`, log in `logs/`), because
the SSH master dropped once mid-run and took the process with it.
Custom-code bases (`family:` in `bases:`) have their own submitter,
`train/run-custom.py`. Their venvs, and the main one, are built and checked
by one script on a login node (internet):
```bash
bash helpers/cluster-setup.sh              # all three venvs; or: main | llamafactory | unsloth
python train/scripts/hf-install.py minicpm-v:8b minicpm-v4.5:8b deepseek-ocr:3b   # MiniCPM-V-2_6 is gated: accept + `hf auth login`
python helpers/export-sft.py TUAB --format llamafactory
python train/run-custom.py --dataset TUAB --dry-run
python train/run-custom.py --dataset TUAB
```
`cluster-setup.sh` creates each venv with Alliance's `virtualenv --no-download`
after `module load arrow` (the only way pyarrow is visible), applies the pins,
and runs `helpers/preflight.py <venv>`, which must pass: it imports every
package the code path needs, loads every base's config, tokenizer and
processor from the staged snapshot (no weights), renders and tokenises the
training template the way the collator does, builds one schema enforcer, and
imports the custom models' remote modeling code. Every startup failure of the
first three batches (2026-09-23 to 26) would have been caught by it. Pins and
why: `.venv-llamafactory` transformers 4.56.2 and datasets 3.6.0 (LLaMA-Factory
bounds `<=5.8.0` / `<=4.0.0`, and the wheelhouse's `+computecanada` tags sort
above those), `.venv-unsloth` unsloth with `--no-deps` (every release caps
torch below the wheelhouse 2.14) plus the deps by hand, including the ones
DeepSeek-OCR's modeling file imports (matplotlib, einops, addict, easydict).
The custom families are also *scored* in those venvs (`EVAL_VENVS` in
`train/run-eval.py`): their remote code does not import under transformers 5.

The LLaMA-Factory yaml is generated from `config.yml` `train:` (same rank,
alpha, lr, epochs, eval/save steps; `lora_target: all`, vision tower unfrozen
when `lora.vision` is true, `image_max_pixels` at the plots' native 1536²),
checkpoint chosen on `eval_loss` since that stack has no recording metric.
`train/custom/finish.py` writes the `manifest.json` after `llamafactory-cli`
exits 0; the Unsloth script writes its own. `run-eval.py` picks them up
once the manifest exists (their `models:` entries are derived like every
other fine-tune's). moondream stays deferred (FINETUNE-TODO item 21).
`run-custom.py` needs `--dataset` since `train.dataset` left `config.yml`
(2026-09-30). Bases over 40 GB get 2 GPUs
from their `bases:` entry (and a longer `time`). Checkpoints land in
`checkpoints/<key>/<DS>/` (gitignored). Logs in
`logs/eeg-vlm-train-<key>-<DS>-<jobid>.out`; the job also writes `logs/tasks.csv` like
the array jobs, so `status.py` shows it. A finished run leaves
`checkpoints/<key>/<DS>/manifest.json` (with its `wandb_id`); scoring derives
the `<key>-sft-<DS>` model entry from the dir and refuses to start without it.

**Loss curves: wandb (2026-09-30).** The generic trainer logs to entity
`l3jovano-krembil-research-institute`, project `eeg-vlm-finetune`
(`config.yml` `train.wandb`), run `<key>-<DS>`, group `<DS>`. Compute nodes
have no internet, so runs are offline in `logs/wandb/offline-run-*`; upload
from a login node (`~/.netrc` there holds the key):
```bash
wandb sync logs/wandb/offline-run-*               # after a run; syncing a live run is untried
wandb sync --append logs/wandb/offline-run-<...>  # a resumed job's later segment (same run id, new dir)
```
**The wandb key on Narval is stale** (2026-09-30: `wandb.Api()` there says
"relogin required"). Offline logging is unaffected, but before the first sync
run `wandb login --relogin` on a login node (needs the API key), or copy
`logs/wandb/offline-run-*` to a machine with a working key and sync there.
A requeued job reuses the run id stored in `checkpoints/<key>/<DS>/wandb-id`,
but offline wandb (0.27) writes it to a new dir, so the later segments need
`--append`. That is from the wandb source, untried. The custom stacks still
log nothing. On disk, the history is also `log_history` in
`checkpoints/<key>/<DS>/checkpoint-<last>/trainer_state.json`, which
`python helpers/inspect-sft.py <key>` prints per eval step with the best
checkpoint marked. Pull just those files with
`rsync -am --include='*/' --include=trainer_state.json --include=manifest.json --exclude='*' narval.alliancecan.ca:/scratch/luka/TUEG-VLM-prediction/checkpoints/ checkpoints/`.
`report/figures/loss-curves.png` (2026-09-30) is the 28 first-round pooled
runs plotted from those files (now in `archive/round1-pooled/`).

### 4e. Score the fine-tune (and later the DPO model)

```bash
python train/run-eval.py             # one array task per (model, dataset) with a manifest; resources from its models: entry
python eval/score.py TUAB               # table + datasets/TUAB/summary.csv + chart-recording_balanced_accuracy.png (zero-shot vs fine-tuned per base)
python judge/judge.py                   # optional: judges pick the new model up like any other
```
`run-eval.py` also runs `qwen2.5vl:7b-hf`, the base model with no adapter, on TUAB
and TUEP: that row is the "before" to compare the fine-tune against (the Ollama
`qwen2.5vl:7b` row differs in quantisation and image resize). It needs no
checkpoint, so it can run as soon as the torch stack is installed. A fine-tune's entry
(`qwen2.5vl:7b-sft-TUAB`, `backend: hf`) lists `datasets: [TUAB]`, so only TUAB
rows exist for it. To score transfer, add an explicit `models:` entry with
more datasets (explicit entries win). Measured 2026-09-26:
22 to 40 s/image for the 24B-class bases on 2 GPUs (schema-constrained
decoding, batch 1), so a pair with more than 1,500 pending windows is split
over several array tasks (`SHARD_WINDOWS` in `train/run-eval.py`), each taking an
interleaved slice and appending to the same file under a lock. Every base
without a `status` gets a `<key>-sft-<DS>` entry for each
`checkpoints/<key>/<DS>/` that exists, derived in `config()`
(`finetune_models()` in `helpers/pipeline.py`, per dataset since 2026-09-30:
backend hf, the base key, that dir, `datasets: [DS]`, resources from
`train.eval`, its `large` block for bases that train on more than one GPU; an
explicit entry in `config.yml` overrides; `pooled` dirs are never derived),
so one `run-eval.py` submits a task per finished pair; predictions land in
`datasets/<DS>/eval-<model>.csv`, one file per hf model. Failures go to
`logs/failures.csv` under stage `predict`; re-running `run-eval.py` retries only
the pending rows, and a timed-out task resumes where its file stops.

### 5a. Generate rationales (train split, fine-tune data)
```bash
python helpers/hash-images.py --check          # datasets/<DS>/hashes.csv came with the pull and covers labels.csv
python train/scripts/sample-train-split.py --dry-run   # which train windows are in scope (free)
python train/scripts/sample-train-split.py             # flag them in pipeline.db (idempotent)
python eval/scripts/generate-rationales.py --split train --dry-run   # free preflight
sbatch eval/scripts/generate-rationales.sbatch     # from the project dir
```
The train sampler must have run once on the cluster's `pipeline.db` since
2026-09-16, or no train row is in scope and the teacher has nothing to do
(`status.py` prints "N train sampled" per dataset; 0 means run it). It is
cheap (about 2 s) and safe to re-run any time; it only touches `split='train'`.

Once rationales exist the sampler also screens them (truncated / hedge /
degenerate / duplicate) and drops flagged windows from scope, picking
replacements. So the loop is: generate -> `sample-train-split.py --dry-run
--show hedge` (read the examples, adjust `HEDGE` in `helpers/pipeline.py` if
it is catching good text) -> `sample-train-split.py` -> generate again for
the replacements -> sampler once more -> `build-sft-jsonl.py`.
- 1 GPU, 32G, `--time=2-00:00:00`, Ollama **via Apptainer**, teacher
  `gemma3:12b`. Resumable — resubmit if it times out.
- Monitor: `squeue -u $USER`, `tail -f generate-rationales-<jobid>.out`,
  `wc -l datasets/*/rationales.csv`.
- Degenerate output (a wedged runner repeating one token) is caught at write
  time: the row is left blank for a later run to retry, and 20 in a row aborts
  the job. No post-hoc cleanup step is needed.

### 5b. Run the eval benchmark (test split, 34 VLMs)

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

**Stage 1 — screen (optional).** Set `probe-images: 350`, run all 34 models, score, and keep
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
not, and `submit` refuses outright — 34 tasks that each die on a registry timeout
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

Walltime per array task is the model's `time` scaled to the pending count
(2026-09-21): `time x pending / full-dataset x 2 + 10 min`, floor 20 min, cap
the configured value. A retry of 19 windows asks for 20 minutes, not 12 hours,
so it stops queueing behind whole sweeps. Tasks with different scaled times
land in different array jobs, as with ram/gpu differences.

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

**Moving a pending fine-tune to a MIG slice** (the GPU type can't be changed with `scontrol`), as done for the
label-only batch on 2026-10-04:

```bash
python rehearsal/probe-memory.py submit KEY ...                 # <=2 h on an a100_3g.20gb slice: peak memory per base
python helpers/plan-resources.py --probe <merged results.jsonl>  # bases under 17 GB -> a100_3g.20gb:1, 40G
scancel <the base's PENDING eeg-vlm-train-<key>-<DS>-labels jobs>  # never a RUNNING one
python train/train.py KEY all                                    # resubmits from train/resources.csv
```

`train.py` skips a pair whose job is still queued, so cancel first. The resubmission loses the queue age the
cancelled job had, which costs little: MIG slices had ~8 jobs pending against ~600 for full A100s. The script
used is `$SCRATCH/rehearsal/to-mig.sh`, with its log in `to-mig-20261004.txt`. `plan-resources.py` rewrites the
tracked `train/resources.csv`; commit the new one, and `git checkout -- train/resources.csv` on the cluster
before the next pull.

## Resume semantics (important)

- **Rationales:** just resubmit — `init_csv` reconciles, done rows skipped.
- **Eval:** resubmit (image-level skip), then `scripts/merge-runs.py` to union
  the two run dirs before scoring. A same-dir Slurm requeue also auto-resumes.
- A partially-failed eval task exits 0 and is marked `fail` in `status.csv` (not
  in Slurm's accounting) — judge success by `status.csv`, not `squeue`.

## Queue waits after a big batch (measured 2026-10-02)
Round 1 (about 30 GPU jobs of 7 to 30 h, Sep 23 to 28) left `def-milad777_gpu` at 1.6x its fair share
(`sshare`: EffectvUsage 0.000173 vs NormShares 0.000106, FairShare 0.32). A 1-GPU 12 h job submitted on Sep 30
was still pending on Oct 2, with Slurm estimating a start ~48 h after submission. A round-1 job submitted on
Sep 23 had waited 1.4 h. The pending reason `ReqNodeNotAvail, UnavailableNodes:ng[...]` only lists the 5
drained GPU nodes; the real cause is priority. Check with `sprio -j <id>` (the fairshare term dominates) and
`scontrol show job <id>` (StartTime, SchedNodeList). Usage decays over time, so the next big batch should be
planned around this.

## Walltime philosophy

Over-requesting walltime does **not** waste allocation (Slurm bills actual
runtime) but **slows scheduling** (harder to backfill). Because every long job is
resumable, request tight limits for faster starts -- but weigh that against a
requeue costing days on a busy cluster (eval walltimes carry 2.5x headroom for
exactly this reason).
