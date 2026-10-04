# Tooling inventory

Every script in the repo, what it is for, and whether it is live. Written because
several of these had no documentation at all and it was not obvious from the
filename which were load-bearing and which were leftovers.

## Live — data pipeline

| Script | Purpose |
|---|---|
| `datasets/<DS>/generate.py` | Render EDFs → windowed PNGs + `labels.csv`. One per dataset. |
| `datasets/render.py` | Shared windowing/rendering/filtering used by all six. Rewritten and tracked on 17 August 2026 after the gitignored `_render.py` was lost; see [data-generation.md](data-generation.md). |
| `datasets/verify_render.py` | Checks `render.py` against the shipped `labels.csv` and PNGs (window selection, canvas, layout box, ink). Run before regenerating anything. |
| `generate-all.sh` | Run all six `generate.py` in parallel (`--overwrite` by default). |
| `datasets/relabel.py` | Add the `assessed` column; relabel TUSZ's unannotated windows `bckg`. Pure `labels.csv` transform, idempotent, `--dry-run`. *Deleted from the working tree; still in HEAD. Its output is already applied.* |
| `datasets/recover_tuar_background.py` | Restore TUAR's discarded `bckg` labels from source. Self-verifying, `--dry-run`. *Deleted from the working tree; still in HEAD. Its output is already applied.* |

Neither relabel script re-renders anything, so both are safe to run against
images already shipped to the cluster (`labels.csv` travels via git).

## Live — evaluation

**Scoring as of 2026-09-16 is `eval/score.py [DATASET ...] [--models ...]
[--bootstrap N]`.** The `summarize.py` described below was deleted in the
`bb5b4be` redesign (its metric core is what `score.py` ports; the chart
renderer came back as `eval/chart.py` on 2026-09-29). `score.py` reads every `datasets/<DS>/eval-*.csv` (`eval-baseline.csv`
for the Ollama roster, `eval-<model>.csv` per `backend: hf` model, via
`eval_rows()`) plus `labels.csv`, keeps only rows whose path is `scope='full'` in `pipeline.db`
(older, larger samples and duplicate rows are dropped), and writes
`datasets/<DS>/summary.csv` (one row per model: coverage, recording-level
balanced accuracy and macro-F1 with a 1000-resample cluster-bootstrap CI, the
constant-predictor floor, `clears_baseline`, degeneracy, `few_patients`) and
`summary-classes.csv` (per-class precision/recall/F1/support at window and
recording level). The printed table sorts by coverage then recording balanced
accuracy. Same formulas as the old script: majority vote per recording for
binary datasets, any-window-positive for multi-label, `MIN_SUPPORT=20`
recordings per scored class, balanced accuracy over the same class set as
macro-F1. Verified on synthetic data: a perfect predictor scores 1.0/1.0, a
constant one exactly 0.5 balanced accuracy and is flagged degenerate.

The paragraph below predates the redesign and describes scripts that no longer
exist; kept for the history of the metrics.

See [eval-pipeline.md](eval-pipeline.md) for detail. `run-eval.py` →
`run_array.sbatch` → `eval.py` (+ `sample.py`, `structure.py`), then `status.py`
to monitor, `scripts/merge-runs.py` to union a retry back into the base run, then `summarize.py` to score and report
(`summary.csv`, `classes.csv`, `rank.csv`, one `rank-<DATASET>.png` per
dataset, plotting recording-level balanced accuracy; `--metric macro-f1` charts
that instead). It also draws `agreement-<DATASET>.png` from the judge stage's
`agreement/*.csv` when that directory exists, so one command produces every chart
in a run. `summarize.py` also
**contains** the metric-agnostic bar-chart renderer that draws those charts
(formerly `eval/chart.py`), which makes it the only eval script needing
matplotlib.

Layout convention: the things **you** invoke live in `eval/`
(`run-eval.py`, `run-judge.py`, `status.py`, `summarize.py`); the things a **Slurm task or submitter** invokes live
in `eval/scripts/`.

`eval/scripts/merge-runs.py <base> <retry> [--into NAME] [--dry-run]` exists because a
retry lands in a **new** run dir and `resume-from` only suppresses re-work — it
never copies base rows forward, so neither dir is scoreable alone. Union keyed on
`path` (the two are disjoint by construction), retry status wins per task, and it
refuses to overwrite an existing destination.

## Live — rationale agreement

Grades *why* a model answered, not just *what*. See the "Rationale agreement"
section of [eval-pipeline.md](eval-pipeline.md).

| Script | Purpose |
|---|---|
| `eval/run-judge.py` | Driver: `check` (join coverage, no GPU), `submit` (array job), `report` (aggregate CSVs). |
| `judge/dedup.py` | Drops duplicate (path, model) rows from judge CSVs, keeps first; `--dry-run`, `--file`. |
| `eval/scripts/judge.py` | Worker — one array task per model; joins results ↔ references, judges each pair, resumable per pair. |
| `eval/scripts/judge_array.sbatch` | Per-task Ollama-in-Apptainer wrapper for `judge.py`. |
| `eval/config.yml` → `judge:` key | Judge model + resources. Read live, never from a run's frozen copy. |

**Prerequisite:** `sbatch --export=ALL,SPLIT=test eval/scripts/generate-rationales.sbatch`.
Without it there are no reference rationales and every join is empty.
`run-judge.py check` is the free local guard against discovering that after
queueing 34 tasks.

## Live — rationales / fine-tune

See [rationale-generation.md](rationale-generation.md).
`eval/scripts/generate-rationales.sbatch` → `eval/scripts/generate-rationales.py`
(`--split train|test`, `--dry-run`). Degenerate output is caught at write time,
so the old `scrub-degenerate.py` cleanup script was removed (2026-08-16).

It sits under `eval/` despite feeding the fine-tune because it is the teacher
pass the agreement stage depends on, and because it shares `sample.py` and
`config.yml`'s `test-sample` policy with the eval track — as a sibling it can now
`from sample import select` instead of loading that module by path.
`train/train.py MODEL|all DATASET|all [--here | --dry-run]` (2026-09-30;
replaced `train/run.py` + `train/scripts/finetune_sample.py`) is the one
fine-tune entry point, one run per (base, dataset) for `config.yml`
`train.target` (`labels` or `rationale`), named by `train.experiment`.
`helpers/pipeline.py` `run_name(DS, experiment)` (`TUAB-labels`) names the
checkpoint dir, job, wandb run and scored model; `trained_runs(key)` lists a
base's `<DS>-<experiment>` dirs; `sft_file(DS, split, target)` names the jsonl.
Every script goes through these, so they all agree. The manifest records
`experiment` and `target`.

**`helpers/experiment-results.py [EXPERIMENT]` (2026-09-30).** One round's
results: every base's zero-shot row beside its `<key>-sft-<DS>-<experiment>`
row from each `summary.csv` (recording balanced accuracy, delta, coverage, top
answer), printed and written to `experiments/<experiment>.csv` (tracked).
Defaults to `train.experiment`. Run `eval/score.py` first. Without `--here` it
submits one Slurm job per pair, skipping with a printed reason (`skip_reason()`)
a base with a `status` or `family` (custom-code bases are `run-custom.py`'s),
a finished pair (`manifest.json`), a queued pair (job `eeg-vlm-train-<key>-<DS>`),
an unstaged snapshot, a missing jsonl, and a jsonl that fails
`helpers/check-splits.py`'s overlap check (loaded once, only for pairs that get
that far). `--dry-run` prints the same lines and submits nothing. Resources
come from `config.yml` `train:` (`time`, `cpus`) and the base (`gpus`,
`ram`). The job activates `.venv` (one `requirements.txt` for both tracks
since 2026-09-21; install the torch stack into it between sweeps, not while
an eval array is running from it), sets `HF_HUB_OFFLINE=1`, and runs
`train/train.py <key> <DS> --here`; torch, transformers, peft and wandb are
imported only on that path, so submitting works in a light venv.
`--here` loads the base model from the snapshot `train/scripts/hf-install.py`
staged under `$SCRATCH/hf-checkpoints/` (falls back to the hub id when no
snapshot exists, which only works with internet), reads the SFT JSONL with a
plain reader (the `datasets` library needed pyarrow/`module load arrow` and
rejected `Path` objects), and refuses to start on an empty split. As of
2026-09-16: bf16 (the 4-bit path was removed on 2026-09-29 with llama4 off);
LoRA targets are a regex over language layers plus, with `lora.vision: true`,
the vision tower and merger at `vision-lr-scale` x the learning rate (AdamW
param groups handed to the Trainer, which builds the warmup + cosine schedule
on top); the collator masks the prompt by locating the last
`<|im_start|>assistant\n` marker in the token ids (one processor call per
example, padding-side independent); `compute_metrics` logs
`eval_label_balanced_accuracy`, teacher-forced accuracy at the target's
`true`/`false` token positions over the whole val set (2026-09-29, replacing
the generation-based val metric, see methodology-decisions.md);
`load_best_model_at_end` selects on it; the final save is that
best adapter plus `manifest.json` (dataset, rows, config, `rationale_first`,
commit, `wandb_id`, timestamp). A dir with `checkpoint-*` and no manifest
resumes from the latest (`resume_point()`). It logs to wandb offline
(`start_wandb()`: `WANDB_MODE` defaults to offline, run `<key>-<DS>`, group
`<DS>`, dir `logs/`, config = the train settings, base spec and row counts;
the HF `WandbCallback` logs into that run; the run id is kept in
`<out>/wandb-id` so a resumed job reuses it). Neither the teacher-forced
metric nor `train.py --here` has run on a GPU yet (2026-09-30).

**Multi-base trainer (2026-09-21).** The trainer (`train/train.py --here`) is model-agnostic:
`AutoModelForImageTextToText` / `AutoProcessor` with `trust-remote-code` from
the base entry, answer masking by prompt-only template length (two processor
calls per batch, the second only for its attention mask), LoRA targets
enumerated from the loaded modules (`lora_targets()` / `is_vision()` /
`VISION_HINTS` in `train/scripts/eval.py`, shared with the scorer), output dir
`checkpoints/<base key with : as ->/<DS>`, model and dataset from the
command line. `config.yml` `bases:` maps each Ollama tag to `{repo, gpus?,
ram?, trust-remote-code?, status?}`; `helpers/pipeline.py` `base_spec()` and
`checkpoint_dir()` resolve them.
`status` in `bases:` now means "off" (llama4, moondream); the custom-code bases
are marked by `family` alone. `train/scripts/hf-install.py --all` stages every
base without a `status`, custom ones included; positional
arguments accept keys or repo ids. `train/scripts/eval.py` `base:` accepts a key or a repo
id. `eval/score.py` `RECORDING_EXCLUDE` (TUSZ bckg) and `recording_classes()`
define the recording-level class set for both the scoreboard and the trainer.

**Layout (2026-09-29, trainer merged 2026-09-30).** `train/` holds what you
invoke: `train.py` (submitter and trainer), `run-eval.py` (scoring submitter,
was `run.py predict`), `run-custom.py` (custom-stack submitter, needs
`--dataset`). `train/scripts/` holds what a Slurm task or a one-off invokes:
`eval.py` (scorer, was `train/predict.py`; also the processor / template /
LoRA-target helpers the trainer imports), `hf-install.py`,
`sample-train-split.py`. The sbatch
plumbing the submitters share (`script`, `submit`, `job_name`, `queued`) is
`helpers/slurm.py`; `helpers/pipeline.py` `script_module(name)` imports a
`train/scripts/*.py` by path, since the names carry hyphens and `eval.py`
would collide with `eval/models/eval.py` on the trainer's sys.path. Scoring
jobs are named `eeg-vlm-eval-sft`; the stage stays `predict` in
`logs/failures.csv` and `status.py`.

**`helpers/inspect-sft.py [KEY ...]` (2026-09-29, per dataset since
2026-09-30).** Per (base, dataset) with a `checkpoints/<key>/<DS>/` dir: the
val curve from the latest `checkpoint-*/trainer_state.json` (step, epoch,
`eval_loss`, `eval_label_balanced_accuracy`, best checkpoint marked) and the
zero-shot vs `<key>-sft-<DS>` `summary.csv` rows with coverage and top
answer. Says so when a run has no manifest (still training, or died) or no
checkpoint yet. The first thing to run when a fine-tune scores like its
zero-shot row.

**Charts (2026-09-29).** `eval/chart.py` draws one PNG per dataset,
`datasets/<DS>/chart-<metric>.png`: every `bases:` key as a pair of bars,
zero-shot (its Ollama row) beside the fine-tune on that dataset (`<key>-sft-<DS>`, since 2026-09-30), sorted
by the fine-tuned score, a dashed reference at chance (0.5) or, for
`--metric recording_macro_f1`, at the constant-predictor floor, one direct
label per series at its peak, a dagger on degenerate models. A base with only
one row keeps its single bar. A row scored on under 99% of the test windows
(`coverage` in `summary.csv`, i.e. a predict run still in progress) is drawn
hatched and lighter and sorted after the complete fine-tunes, so a
half-finished model is never mistaken for a result. `score.py` calls it after writing `summary.csv`;
standalone: `python eval/chart.py [DS ...] [--metric ...]`. The renderer is the
pre-redesign `summarize.py` one (rounded data-ends, recessive grid) with a
second series; the two hues were checked with the data-viz palette validator.

**Pooled run (2026-09-21 to 09-30), removed.** One run per base over all six
corpora from `datasets/pooled/`; the builder's `pooled` mode and every pooled
special case are gone (per-dataset training, methodology-decisions.md). The
first-round files are archived on Narval in `archive/round1-pooled/` (2026-09-30;
README, results table and manifest of moved paths tracked in git, the bulk
ignored); nothing derives or reads them. See experiments.md.

**`helpers/check-splits.py [DS ...]` (2026-09-30).** The train/val/test
leakage gate. `Leakage()` loads every test path of all six corpora, the
merged patients (`duplicate_images()`) and the md5s from `hashes.csv` once;
`overlaps(ds, train, val)` returns, per check, the keys shared between train
and val (merged patient, raw token, recording, image md5) and between
train+val and test (merged patient, across all corpora when
`exclude-cross-dataset-test-patients` is on; image md5, all corpora).
`excerpt_warnings()` lists TUEV numeric excerpt ids (warning only). The CLI
reads `datasets/<DS>/sft_{train,val}.jsonl`, prints a per-corpus table and
`LEAK` lines, exits 1 on any overlap. `build-sft-jsonl.py` and `train.py`
load it by path.

**Custom-code bases (2026-09-21).** `bases:` entries with `family:`
(`minicpm`, `moondream`, `deepseek-ocr`) are not trained by the generic
trainer (`train.py` skips them). `helpers/export-sft.py <DS>
--format llamafactory|minicpm|unsloth` re-exports the same `sft_*.jsonl` for
their stacks (sharegpt JSON + `dataset_info.json`; OpenBMB finetune JSON;
Unsloth messages JSONL), absolute image paths, under
`datasets/<DS>/export-<format>/`. `train/scripts/eval.py` `CustomFamily` loads and
prompts them through their own APIs (`model.chat`, `model.query`,
`model.infer`), with or without a PEFT adapter, unconstrained decoding.

`bases:` fields the trainer and `train/scripts/eval.py` both honour (2026-09-23), all
routed through `load_processor` / `render` / `encode_kwargs` in
`train/scripts/eval.py`: `chat-template` (a path under `train/`, e.g.
`templates/llava-1.5.jinja`, for repos without a usable one; the file's exact
bytes become the processor's template, since whitespace is part of what the
model sees; four files serve five bases as of 2026-09-29),
`processor-kwargs` (AutoProcessor overrides), `template-kwargs`
(`apply_chat_template` kwargs such as `enable_thinking`), `answer-suffix` (a
stop token appended to the training target), `processor-files-from`
(`hf-install.py` copies a sibling's processor files). Jobs are named
`eeg-vlm-train-<key>-<DS>` (2026-09-30); both submitters skip a pair whose job is in `squeue`.

`helpers/cluster-setup.sh` + `helpers/preflight.py` (2026-09-26): build or
repair the three cluster venvs and check each without a GPU (imports,
version bounds, every base's processor and template from the snapshot, the
enforcer, the custom models' remote code). Run after any pull that touches
`requirements.txt`, `bases:` or a venv. `run-eval.py` groups tasks by venv too.

`train/run-custom.py` (2026-09-23) submits them: MiniCPM-V 2.6 / 4.5 via
`llamafactory-cli train` on a yaml it writes into the checkpoint dir from
`config.yml` `train:` (venv `.venv-llamafactory`), DeepSeek-OCR via
`train/custom/unsloth_deepseek_ocr.py` (venv `.venv-unsloth`; Unsloth's
notebook collator, reads `sft_*.jsonl` directly, eval loss on val, writes the
manifest). `train/custom/finish.py` writes `manifest.json` for stacks that do
not, and refuses without adapter weights. `helpers/slurm.py`'s `script()` takes a
`venv` argument for this. Same manifest-exists skip rule as `train.py`;
`--dataset` is required since 2026-09-30. Still `report_to="none"` (no wandb).
Unverified on a GPU. Plan and status per base: FINETUNE-TODO item 21.

**`rehearsal/rehearse.py cpu|gpu MODEL DS [--side]` (2026-10-03): dress rehearsal before a submission.**
- Loads `train/train.py` (and, with `--side`, `side/rationale-w10/train.py`) by path and runs their real
  `train()` on a sample of the real jsonl and images.
- Run 1 is crashed on purpose after the first checkpoint. The crash hook goes on the Trainer the script actually
  trains with (`init_trainer`, or the side's `build_trainer`). Run 2 must resume from it and finish: eval with
  label positions, best-checkpoint reload, adapter saved, manifest complete.
- Then it scores a few test windows through `train/scripts/eval.py`'s own `load` / `enforcer` / `predict`, and
  `eval/score.py`'s `evaluate`. Replies that don't parse are counted (a few-step model rambles past the cap);
  any other exception fails.
- Swapped, and nothing else:
  - output goes to `$SCRATCH/rehearsal/<stamp>-...` (or `~/rehearsal/` on a Mac);
  - the run is shortened: 6 or 8 optimizer steps, 24 or 96 train rows;
  - **cpu tier only**: a 2-layer random model built from the base's own config and processor, no bf16
    autocast, 448 px plots in the collator, fork workers, no MPS, and `bitsandbytes` hidden.
- **cpu tier on your Mac (the practical way):**
  - Once: `bash rehearsal/setup-local.sh qwen2.5vl:7b TUAB`. This builds `.venv-rehearsal` with Narval's exact
    versions (`rehearsal/requirements-local.txt`) and copies the base's config/tokenizer/processor files (no
    weights) into `hf-checkpoints-local/` and the sft jsonl from Narval.
  - Then run
    `HF_CHECKPOINTS=$PWD/hf-checkpoints-local .venv-rehearsal/bin/python rehearsal/rehearse.py cpu qwen2.5vl:7b TUAB --side`.
  - Measured 2026-10-03: ~70 min for main + side, `RESULT: PASS`.
  - On a login node the same tier works, but it is slow and heavy there.
- **gpu tier:** submits a <=2 h job with the real model. It reports s/step, peak GPU memory, and whether the
  estimated run fits the walltime.
  - Measured 2026-10-03: 30.9 s/step, 25.5 GB peak, ~5.1 h for TUAB labels.
  - The real job then ran at 26.7 s/step.
- Exit 1 and `RESULT: FAIL` on any failed check.
- History of 2026-10-03: the tool's own bugs (tiny-model `layer_types`, login-node SIGILL, a local class that
  wouldn't pickle, the crash hook missing the side's Trainer) were all fixed that day.

**`side/rationale-w10/` (2026-10-01): a side experiment, not the pipeline.** `train.py MODEL DS
[--here|--dry-run]` loads `train/train.py` by path and reuses its config, data, model, `init_trainer`, leak check
and manifest unchanged. It swaps in a collator that adds `label_weights` (10 on the JSON boolean values) and a
`Trainer.compute_loss` that weights the CE. `score.py MODEL DS` scores the adapter with `train/scripts/eval.py`'s
`load` / `enforcer` / `predict` and `eval/score.py`'s `evaluate`. Checkpoints, Slurm logs, wandb runs and results
all stay under `side/rationale-w10/`, so nothing in the main tree derives or reads them. See
`side/rationale-w10/README.md` and experiments.md.

**Label-only fine-tunes in scoring and judging (2026-09-30).**
`train/scripts/eval.py` reads the manifest's `target`. For `labels` it
constrains decoding to the booleans-only schema (`get_structure(DS, ...,
rationale=False)`), prompts with `prompt(DS, rationale=False)`, skips the
degenerate-rationale check and writes an empty `rationale` column.
`finetune_models()` tags derived entries with `target`; `judge/judge.py` and
`judge-openai.py` skip `target: labels` models, since there is no rationale to
judge. `eval/chart.py` pairs each base with its `train.target` fine-tune.

`helpers/eval-interactive.sh MODEL DATASET` (2026-09-21) runs `eval.py` for
one (model, dataset) on a GPU node, requesting a 30-minute allocation itself
via `srun --pty` when run from a login node: starts Ollama via
Apptainer on a random port, fakes `EVAL_TASKS` / `SLURM_ARRAY_TASK_ID`, and
processes only the rows pending in `pipeline.db`. For the last handful of
windows when an array job's queue wait is the cost. Login nodes have no GPU.

`train/scripts/sample-train-split.py [--dry-run]` (2026-09-16) decides which train
windows the fine-tune (and therefore the teacher pass) uses, and flags them
`sampled=1` in `pipeline.db`, the train-side twin of `eval/sample-test-split.py`.
Policy is `config.yml` `settings.train-sample`: `recordings-per-patient` (8),
`background-per-recording` (1), `majority-ratio` (1.5),
`reject-bad-rationales` (true), `exclude-cross-dataset-test-patients` (true).
Rows are screened first: unassessed, any patient present in any corpus's test
split, any train image **byte-identical to a test image** in any corpus, any
image that appears twice in its corpus with different labels, and (once rationale
text exists) rationales flagged by
`rationale_problems()` (truncated / hedge / degenerate) or exact duplicates,
with counts per reason printed and `--show REASON` printing examples. Then
binary datasets: cap recordings per patient, then cap the majority class at
ratio x minority, thinning round-robin across patients. Multi-label: every
event window is kept; pure-bckg windows are capped per recording, then per
patient, then at ratio x event count. Deterministic, no RNG, idempotent.
Patient identity comes from `duplicate_images()` in `helpers/pipeline.py`:
tokens that share an image anywhere are merged into one patient before the
caps and the test screen (TUEV's numeric eval dirs, `aaaaaduk`/`aaaaadul`, ...).
With `exclude-cross-dataset-test-patients: false` only the corpus's own test
split counts, for both the patient and the image screen. The image hashes are
`datasets/<DS>/hashes.csv` (path, md5), written by
`helpers/hash-images.py [DS ...] [--check]` (about 5 min for all 76k PNGs;
`--check` confirms every `labels.csv` row has a hash) and tracked in git so the
cluster never re-hashes; the sampler refuses to run on a row without a hash.
Result on the current `labels.csv` with everything on (2026-09-21): 6,904
windows (TUAB 2,424, TUAR 403, TUEP 922, TUEV 385, TUSL 49, TUSZ 2,721) and
zero fine-tune images identical to a sampled test image; 2026-09-16, before the
image screen and the patient merge, it was 7,351. The shared pieces (`parse_name`, `spread`,
`positives`, `is_degenerate`, `rationale_problems`) live in
`helpers/pipeline.py`.

`train/run-eval.py [--dry-run]` + `train/scripts/eval.py` (2026-09-16) score a
fine-tuned checkpoint on the sampled test split **HF-side** (never through
Ollama; see methodology-decisions.md). A fine-tune is registered under
`config.yml` `models:` with `backend: hf`, `base` (HF repo id, resolved to the
`$SCRATCH/hf-checkpoints` snapshot), `checkpoint` (adapter dir), optional
`datasets: [..]` (which corpora it is scored on; default all six), and
`time/ram/cpus/gpus`. An entry **without `checkpoint`** (`qwen2.5vl:7b-hf`,
2026-09-21) runs the base model as-is through the same path: the like-for-like
"before" for the fine-tunes, in the zero-shot field order unless it sets
`rationale-first: true`. `sync()` creates its `pipeline.db` rows only for those
datasets and deletes any others; `eval/models/run.py` skips `backend: hf`
models. `run-eval.py` submits one array task per (model, dataset) with
pending `scope='full'` rows, and refuses a checkpoint without `manifest.json`
(written by the trainer when a run finishes: dataset, rows, config, commit).
`train/scripts/eval.py` loads base + adapter (merged), uses the exact eval prompt via
`structure.prompt()`, greedy decoding constrained by `lm-format-enforcer` (the
HF equivalent of Ollama's json mode) to the schema in the order the checkpoint
was trained on (`rationale_first` from `manifest.json`), 512-token cap,
degenerate-rationale check, one sampled retry at 0.3, and appends rows to
`datasets/<DS>/eval-<model>.csv` (`:` written `-`, e.g.
`eval-qwen2.5vl-7b-sft-TUAB.csv`; `eval_file()` in `helpers/pipeline.py`,
2026-09-24) under the config name. `sync`, `score.py` and both judges read
every `eval-*.csv` in the folder (`eval_rows()`), so from there it is treated
like any other model. Batch size 1
(~5 s/image estimated, unmeasured). Same runner scores the DPO model later;
the trainer imports its processor, template and LoRA-target helpers.

`helpers/build-sft-jsonl.py [DATASET ...] [--dry-run]` (2026-09-13; no args =
all six; per dataset only since 2026-09-30) builds
`datasets/<DS>/sft_train.jsonl` + `sft_val.jsonl` for `train/train.py`, and
since 2026-09-30 `sft_labels_train.jsonl` + `sft_labels_val.jsonl` from the same
windows and split: booleans-only answer, prompt from `structure.prompt(DS,
rationale=False)` (no evidence sentence, no no-meta line; the zero-shot prompt
is `prompt(DS)` and unchanged byte for byte, because `config.yml` now keeps the
evidence sentence in its own `binary-rationale` / `multilabel-rationale` key).
For the rationale pair:
instruction = the exact eval prompt via `structure.prompt()`, output = the
eval-schema JSON with `text_rationale` **first** when `train.rationale-first`
is on (2026-09-16), pydantic-validated per line before writing. Eligibility
comes from `pipeline.db` (`scope='rationale'`, set by the train sampler above)
plus a non-empty `ground_truth_rationale`; it refuses to run if a dataset has
no train rows in scope, and asserts no `test/` path ever qualifies. Val is
**patient-level and class-stratified** (`train.val.fraction` 0.05,
`min-patients` 10 since 2026-09-21 (3 gave TUEP 17 windows), capped at half
the pool; patients merged through shared images as in the sampler): every class with >= 2 patients
gets one val patient first, rarest class first, then md5-hash order fills the
rest; the builder prints val class counts; deterministic rebuilds. Before
writing anything it runs `check-splits.py`'s overlap check on every requested
dataset's in-memory split and writes nothing if any dataset overlaps or has an
empty split (2026-09-30; it replaced a train/val assert that could not fail).
`--dry-run` runs the check too. **Must run where rationales exist**: the
local `labels.csv` copies have empty rationale columns (all ~32k generated
rationales are cluster-side only as of 2026-09-13), so a local run stops on
empty splits and writes nothing.

### One master `config.yml` (2026-09-13)
`train/config.yml` was merged into the root `config.yml` under a `train:` key,
and all prompt **wording** moved there under `prompts:` (`intro`, `no-meta`,
`eval.binary`/`eval.multilabel` templates + per-dataset `tasks`/`kinds`,
`rationale.label-clauses`/`rationale.grounded`). Assembly stays in code:
`eval/models/structure.py::prompt()` and
`eval/ground-truth/generate-rationales.py::create_prompt()` read the templates
via `helpers.pipeline.config()` and `.format()` them. Verified byte-identical
to the old hardcoded strings for all 6 eval prompts and 6 rationale prompts
before the swap. Purpose: the SFT JSONL builder can pull the exact eval prompt
from one place. Caveats: the YAML strings carry load-bearing leading/trailing
spaces (quoted, one line each — do not reflow).

## Live — data acquisition

| Script | Purpose |
|---|---|
| `eval/scripts/download.sh` | rsync a TUEG subcorpus from `isip.piconepress.com`. Usage: `download.sh <remote_path> [destination]`. |
| `zip-datasets.sh` | Bundle `train/` + `test/` per dataset into `zips/`. |
| `train/scripts/hf-install.py [REPO ...] [--dry-run]` | Download HF snapshots for the fine-tune into `$SCRATCH/hf-checkpoints/<repo-with-dashes>/` (no default: name keys or `--all`). Login node only. Resumable (skips complete snapshots). Rewritten 2026-09-16 from the old root-level `hf-install.py` (in history at `bb5b4be^`), which pulled all ~40 roster repos; the Ollama-tag-to-HF map lives there if ever needed. |

## Report

`report/project-report.tex`, built with `tectonic report/project-report.tex`
(10 pages as of 17 August 2026). **Audience is the PI**, and the brief is in
`Planning.md` at the repo root: a professional technical and architectural
write-up of what has been done. It follows that outline — Ollama setup, Hugging
Face setup, dataset generation, rationale generation, eval — rather than an
arbitrary structure, so rewrites should start from `Planning.md`.

**Typography (17 August 2026).** Tectonic runs XeTeX, so the preamble uses
`libertinus` (serif + sans + mono + math from one family), `hyphenat[htt]` so
file paths never hyphenate mid-token, `tcolorbox[most]` for the summary panel and
callouts, and `longtable` for the glossary. Headings are sans in the `accent`
navy with a hairline rule; the §6 status table uses coloured chips. All packages
resolve from Tectonic's bundle, but a **cold cache needs network** on first build.

**The Google Doc and the LaTeX have diverged (17 August 2026).** There is a
review copy at `docs.google.com/document/d/16Z7oKlputqVM8FH5UksI-pi4eG-8H5g4jJEtjVMJUxM`,
styled to match the LaTeX: Times New Roman, black only, justified body, centred
title block, booktabs-style tables (horizontal rules, no fills, no verticals).
No colour and no sans anywhere — that was tried and rejected.
Luka reviewed it in Suggesting mode and those suggestions were accepted, which cut
material the LaTeX still carries: the whole "What the project is testing" section,
the `def-milad777` account name, the `rsync -auvxL` clause, the images-vs-labels
transfer paragraph, and `<DS>` became `<TU??>`. The Doc is therefore the *shorter*
version and sections renumber 1--6. **Reconcile before either is treated as
canonical** — do not regenerate one from the other assuming they match.

**Register (17 August 2026).** Plain words and short sentences, but written for a PI,
not for Slack. An earlier pass leaned too far into `write-like-luka` and read as
casual ("repos", "forty-odd", "killed the process", "eats most of the compute"). Keep
the simplicity, keep the formal register. Also: no italics anywhere, and no bold
labels leading bullet items.

**Two editorial rules the report now follows**, both from PI review comments:
1. **Gloss the jargon on first use** (MIG, Apptainer, GGUF, LoRA, flash
   attention, KV cache, MoE, macro-F1, structured output, vision tokens). There
   is also an appendix glossary, `\appendix` §A.
2. **Name the incident, not just the design.** Where a choice traces to something
   that broke — the login-node HF download, the compute-node `ollama pull` that
   cost a judge array, the `JobIDRaw` bug, the two-script scoring trap, the
   `_render.py` scaling attempts — the report says so. That is the detail the PI
   asked for and it is what makes the design defensible.

The report summarises this knowledge base for an outside reader, so
**`knowledge/` stays the source of truth** and the report is regenerated from it,
never the other way round.

There is exactly one report file, and there should stay exactly one. A 22-page
version existed until 17 August 2026 and was deleted after checking every fact in
it already appeared here (the 2,488 GB to 1,008 GB RAM retune, the `JobIDRaw`
bug, the Ollama env vars, `p99 * 1.15` scaling and the crest-factor rationale all
did). `report/` is untracked, so nothing in it is recoverable from git.

Note the report names two things this directory did not previously record: there
is **no `ollama-install.py`** despite `Planning.md` asking for one (setup is the
manual `$SCRATCH/ollama/ollama.sif` Apptainer build), and several documented
scripts are missing from the working tree (see `known-issues.md`).

## Situational

| Script | Purpose |
|---|---|
| `datasets/<DS>/generate-annotated.py` | Render windows with annotation overlays drawn on. A **debugging/inspection** aid for checking that labels line up with the waveforms — not part of the benchmark pipeline, and its output must never be fed to a model (the overlay leaks the label). |
| `temp/port_legacy.py` | One-off import of an archived pre-restructure eval run into the current `eval/runs/` layout. *Deleted from the working tree; still in HEAD.* |

## Cruft — safe to delete

| Script | Why |
|---|---|
| `datasets/TUEP/scan.py` | Ad-hoc row counter that reads `row[1]` as a label. `row[1]` is the `split` column, so it has counted 0 epileptic rows since the split column was added — it predates the current schema and was never updated. |
| `README.md` (repo root) | Empty. The real documentation is this directory. |

## Note on what is gitignored

`.gitignore` excludes `knowledge/`. So **this knowledge base does not travel to
the cluster** with `git pull`. That is deliberate (it is a local working aid),
but it means anything the cluster needs must live in a tracked file, not here.
