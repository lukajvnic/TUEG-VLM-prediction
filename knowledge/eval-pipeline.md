# Evaluation pipeline

Zero-shot benchmark of 35 VLMs on a stratified sample of the **test** split, run
as Slurm array jobs on Narval with Ollama serving the models inside Apptainer.

## Files

Top-level entrypoints live in `eval/`; everything a Slurm task or a submitter
calls lives in `eval/scripts/`.

- `eval/config.yml` — judge, models, datasets/prompts, settings. **The one config.**
- `eval/run-eval.py` — reads config, groups runs by resource, submits array jobs.
- `eval/run-judge.py` — rationale-agreement driver (`check` / `submit` / `report`).
- `eval/create-retry-config.py` — regenerate config to retry only failures.
- `eval/status.py` — live run rollup; reconciles `status.csv` with Slurm.
- `eval/summarize.py` — scoring and reporting: per-class + recording-level
  metrics with bootstrap CIs, written to `summary.csv`, `classes.csv`,
  `rank.csv`, `rank-<DATASET>.png`. **Includes the bar-chart renderer**
  (`write_bar_chart`, `zoomed_range`) — see "`rank-<DATASET>.png`" below.
- `eval/scripts/run_array.sbatch` — per-array-task: start Ollama, run `eval.py`.
- `eval/scripts/eval.py` — the actual inference + logging.
- `eval/scripts/judge_array.sbatch` — per-array-task: start Ollama, run `judge.py`.
- `eval/scripts/judge.py` — the per-model judge worker.
- `eval/scripts/generate-rationales.py` + `.sbatch` — the teacher pass that writes
  reference rationales (see [rationale-generation.md](rationale-generation.md)).
- `eval/scripts/sample.py` — deterministic stratified test-set selection (also a
  standalone report: `python eval/scripts/sample.py`).
- `eval/scripts/structure.py` — pydantic output schemas per dataset.
- `eval/scripts/merge-runs.py` — union a base run with its retry run.
- `eval/scripts/config-base.yml` — full model template (used by create-retry).

## `eval/config.yml`

- **`judge`**: the rationale-agreement stage's model and resources. Read **live**
  from this file, never from a run's frozen copy — the stage runs after and
  independently of the sweep. See "Rationale agreement" below.
- **`settings`**: `array-concurrency` (max simultaneous array tasks per group),
  `eval-retries: 1`, `limit: -1` (all images), `parallel-requests: 4`,
  `resume-from:` (blank = fresh run), `model-kwargs` (temperature 0),
  `structured-output: {method: json_schema}`, `probe-images:` (blank = full run;
  set it for a stage-1 screening run), `test-sample:` (the per-recording caps —
  see [methodology-decisions.md](methodology-decisions.md)).
- **`parallel-requests` is per-model-overridable.** Models on a full A100 or 4
  GPUs carry `parallel-requests: 8`; models on a 20 GB MIG slice inherit the
  global 4. `run_array.sbatch` reads the same value out of the same config so
  `OLLAMA_NUM_PARALLEL` and the client thread pool cannot drift apart.
  **This value is not measured** — confirm it on one model before a full sweep.
- **`datasets`**: each has a `prompt`. Prompts describe a *multi-channel EEG
  waveform plot* and demand **grounded per-item evidence** ("which channel, where
  in time, what the curve looks like"), and put all reasoning in `text_rationale`.
- **`models`**: 35 active entries (10 commented out with their reason — 3 suspected
  Ollama tag aliases, 2 that do not fit Narval's hardware, and 5 of the 6
  `-thinking` variants, since the reasoning ablation is run once at 8b against
  `qwen3-vl:8b-instruct`). Each has `time`, `ram`, `gpus`, `parallel-requests`,
  `datasets`. GPU allocation:
  - **21 models with ≤7 GB of weights** → `gpus: "a100_3g.20gb:1"` (a **20 GB MIG
    slice**), 4 requests in flight
  - **13 mid/large** → `gpus: 1` (full A100), 8 in flight
  - **1 MoE** (`llama4:16x17b`, 67 GB) → `gpus: 4`
  - `qwen3-vl:8b-thinking` keeps a full A100 despite its size: it emits several
    times the tokens of any other model and is the run's long pole.
  - The same assignments are mirrored into `config-base.yml` so retries don't
    revert them.

### Sizing rules (all three gate whether a job runs at all)

- **VRAM** — Narval is A100-**40 GB** (MIG `3g.20gb` = 20 GB, 4 GPUs = 160 GB).
  A model needs `weights x 1.15 + KV x parallel-requests` to fit, or Ollama
  silently offloads layers to CPU and runs orders of magnitude slower.
- **RAM** — `1.6 x weights + 8 GB`, and **never above ~498 GB for a 4-GPU job**,
  which is a whole Narval GPU node. A request above the node total can never be
  satisfied. Roster total went 2,488 GB → 1,008 GB, which is the single biggest
  lever on queue wait.
- **Walltime** — 2.5x the estimated TUSZ task (the largest dataset, so it sets the
  bound for all six), rounded to a clean bucket. Over-requesting does not waste
  allocation (Slurm bills actual runtime), it only slows backfill — and since
  every task is resumable, the cost of *under*-requesting is a requeue, which on
  a busy cluster is far more expensive than a loose bound.

## Dispatch: `eval/run-eval.py`

- `group_runs` buckets models by `(time, ram, gpus)`.
- `dispatch_job` submits one `sbatch` per group with
  `--array=0-(N-1)%<concurrency>`, `--gres=gpu:<gpus>` (so a string like
  `a100_3g.20gb:1` becomes `--gres=gpu:a100_3g.20gb:1`), `--time`, `--mem`, and an
  `--export` carrying `RUN_DIR` and the base64 task list.
- Each run gets a fresh `eval/runs/run-<timestamp>/` with `logs/`, `results/`,
  `status.csv`, and a copy of `config.yml`.
- **`array-concurrency` is per resource group.** If it's ≥ the largest group's
  task count, it does nothing (e.g. 78 == largest group == same as 200).

## Per-task: `run_array.sbatch`

- Runs Ollama in **Apptainer** (`ollama.sif`). The rationale job does too now —
  native `ollama` is not on the compute node's PATH.
- Derives `OLLAMA_NUM_PARALLEL` from the model's own `parallel-requests` in
  `config.yml`, so server and client always agree.
- Sets `OLLAMA_CONTEXT_LENGTH=8192` (measured: a
  full-res image + prompt ≈ 4033 tokens; the 4096 default truncated the JSON
  output). Passed into the container via `APPTAINERENV_*`.
- Decodes its `(model, dataset)` from the array index and runs
  `python scripts/eval.py --model … --dataset …`.

## Inference: `eval/scripts/eval.py`

- **Test-only:** `load_dataset` skips rows whose `split` isn't `test` (and treats
  `split`/`assessed` as non-label). Single pass over `labels.csv` builds paths +
  ground truth, then `sample.select` (or `sample.probe`) drops unassessed windows
  and applies the per-recording caps. Deterministic — every model gets the same
  set, and there is no sample file that can go stale.
- **Concurrency:** `ThreadPoolExecutor(max_workers=parallel-requests)` — continuous
  batching against Ollama; lazy per-image base64 encoding in the worker.
- **No fail-fast:** a per-image error is logged and the run continues; the task is
  marked `fail` with an `"N images failed"` reason at the end (exit 0). Only
  infra errors (config load, model init) abort.
- **Structured output:** `ChatOllama(...).with_structured_output(get_structure(ds),
  method="json_schema", include_raw=True)`. `get_prediction` maps the parsed model
  to an uppercase label set.
- **Retry backoff is conditional.** The 10 s sleep runs only for transport-level
  errors. A schema-validation or JSON-parse failure is deterministic, and pausing
  before retrying it just burns walltime — at 35 models × thousands of images
  that was potentially hours of GPU time on weak-JSON models.
- **wandb logs aggregates**, not one event per image (that was ~15k events per
  task and duplicated what the results CSV already holds).
- **Resume:** `load_completed_paths` reads completed image paths from the
  `resume-from` run's results CSV **and** the current run's own — a same-dir Slurm
  requeue auto-resumes; cross-run resume needs `resume-from`.
- **Live monitoring:** prints `[done/total] dataset ok/FAIL: file` per image and
  refreshes `status.csv` every 50 images.
- Outputs `results/<model>-<dataset>.csv` with columns
  `path, prediction, actual, correct, text_rationale, api_json` (prediction/actual
  are stringified Python sets).

## Retry loop: `create-retry-config.py`

`python eval/create-retry-config.py <run>`:
- reads the run's `status.csv`, finds succeeded `(model, dataset)` pairs,
- rebuilds `config.yml` from `config-base.yml` (models) + the run's
  `judge`/`settings`/`datasets`, commenting out the succeeded pairs,
- **auto-sets `resume-from: <run>`** so surviving tasks skip completed images.
Then `python eval/run-eval.py` runs only the unfinished work.

**Every top-level key must be both loaded *and* rendered.** `render_config` emits
`judge`, `settings`, `datasets`, `models`; a key that is loaded but not rendered
is silently dropped on the first retry. This is why the judge config used to live
in its own `judge.yml` — it is now in `config.yml` and explicitly rendered, and a
round-trip test is `python eval/create-retry-config.py <run>` followed by checking
`judge` is still there. Note the rewrite goes through `yaml.safe_dump`, so **the
explanatory comments in `settings`/`datasets`/`judge` are stripped** on retry;
that was already true for the other two and this doc is the durable copy.

### The retry lands in a new run dir — merge before scoring

`run-eval.py` always mints a fresh `run-<timestamp>` dir, and `resume-from` only tells
`eval.py` which images to *skip* (`load_completed_paths`) — it never copies the
base run's rows forward. So after a retry the results are split, and **neither dir
is scoreable alone**: the base is missing the retried images, the retry holds only
those, and fully-succeeded tasks appear in the base only. `summarize.py` globs a
single run's `results/`, so pointing it at either one silently under-reports.

```bash
python eval/scripts/merge-runs.py <base-run> <retry-run> --dry-run   # counts per file
python eval/scripts/merge-runs.py <base-run> <retry-run>             # -> <retry-run>-merged
python eval/summarize.py <retry-run>-merged
```

The union is keyed on `path` and the two sets are disjoint by construction (that
is exactly what `resume-from` guarantees), so any duplicate it reports means an
overlap that should not exist — it drops them and prints the count.

## Scoring and reporting: `eval/summarize.py`

`python eval/summarize.py <run> [--bootstrap N]`:
- Parses each `results/*.csv`; derives recording from filename (`<pat>_<scan>`).
- **Window-level** per-class precision/recall/F1.
- **Recording-level** (the meaningful unit): aggregate a recording's windows into
  one prediction — **majority vote** for binary (TUAB/TUEP), **any-positive
  detection** for multi-label — then per-class P/R/F1.
- **`MIN_SUPPORT = 20`:** macro-F1 is computed only over classes with ≥20 test
  positives; under-supported classes are listed with their counts (transparent,
  not silently dropped). This is how rare classes are handled.
- **95% bootstrap CI** on both macro-F1 figures, resampling whole *recordings*
  (cluster bootstrap — windows within a recording are correlated). 1000 resamples
  by default; `--bootstrap 0` disables.
- **Constant-predictor baseline**: the best macro-F1 an image-blind constant
  answer could score on the same data, restricted to answers the schema can emit.
  This is the stage-1 promotion rule.
- **Degeneracy check**: flags a model answering the same thing on ≥95% of windows
  (`known-issues.md` listed this as an open decision; it is now implemented).
- **Patient-count caveat**: datasets under 20 independent patients (TUSL has 8)
  are marked descriptive-only.

Recording-level scoring is what fixes the TUAB/TUEP "grade a normal-looking window
against a recording-level abnormal label" problem.

### Output files

- **`summary.csv`** — one row per model×dataset. Carries the headline figures
  (window and recording macro-F1, each with CI bounds, plus
  `recording_balanced_accuracy`) *and*, on the same row,
  everything that decides whether they mean anything: `baseline_macro_f1`,
  `clears_baseline`, `degenerate`, `few_patients`, `scored_classes`,
  `low_support_classes`, and the task's `status`.
- **`classes.csv`** — the per-class P/R/F1/support table at both levels, with
  `in_macro` marking which classes cleared `MIN_SUPPORT`. This is the stdout
  breakdown in a form you can filter.
- **`rank.csv`** — models ordered by mean recording-level macro-F1.
- **`agreement-<DATASET>.png`** — one chart per dataset the judge has scored,
  drawn from `agreement/*.csv` if that directory exists. Absent (with a printed
  note) before the judge stage has run.
- **`rank-<DATASET>.png`** — one chart per dataset present in the run (six on a
  full sweep), each ranking the models on *that* dataset by recording-level
  **balanced accuracy**. `--metric macro-f1` writes
  `rank-<DATASET>-macro-f1.png` alongside rather than over them. Datasets whose every
  task failed get no chart; the closing "wrote ..." line names what was written,
  so an absence is visible.

### Why this used to be two scripts

Until 2026-08-16 there was a separate `score.py` (rigorous metrics, stdout only)
and `summarize.py` (window-level accuracy, wrote the CSVs and the chart). The
split was a trap: the *misleading* view was the one that persisted to disk and
travelled into slides, while the reportable one evaporated when the terminal
closed, and `rank.csv` ranked a model that only ever says "background" near the
top. They are now one script — the artifacts carry the reportable metrics, and
every row carries its own caveats.

Window-level `accuracy` and `balanced_accuracy` are still in `summary.csv` for
continuity with earlier runs. **They are not the headline.** Accuracy on a
background-dominated test set is exactly what the recording-level macro-F1 exists
to avoid; quote `recording_macro_f1` with its CI and its baseline, or quote
nothing.

### `rank-<DATASET>.png`

One chart per dataset, one bar per model, tallest first, plotting that dataset's
**recording-level balanced accuracy** — mean per-class (recall + specificity)/2,
over the classes with support >= `MIN_SUPPORT`, after each recording's windows
have been aggregated into one prediction.

**The charted metric and the reportable metric are deliberately different.** The
charts plot balanced accuracy because a flat 0.50 floor reads without a
paragraph of explanation; `summary.csv` and `rank.csv` still lead with
recording-level macro-F1, which is what a number in the writeup should be.
`--metric macro-f1` charts that instead. See "Choosing the charted metric".

Until 2026-08-18 this was a single `rank.png` of each model's *mean* across
datasets, which hid the thing worth seeing: a model carried by one easy dataset
drew the same bar as one that was even across all six, and the six rankings do
genuinely differ. The mean ranking still lives in `rank.csv` — the split is a
charting change only, and no metric moved. Same completeness filter as
`rank.csv`: only `success`/`unknown` tasks are charted.

The y-axis used to be fixed to 0–1, but with 33 models all landing within ~0.04
of chance, that made every bar look identical — the chart was useless as a
leaderboard. It now **zooms** to `[min − 0.05, max + 0.05]` (clamped to
`[0, 1]`), computed by `zoomed_range()`, so the ranking is actually
legible. This reintroduces the truncated-axis problem the fixed range was chosen
to avoid — gaps that are mostly noise now look bigger than they are.

The dashed reference line is the **mean constant-predictor baseline**, not the
0.50 chance level. Chance is not a meaningful landmark for macro-F1 on a
class-imbalanced multi-label set; the floor an image-blind model reaches is, and
a bar below that line is a model that failed the stage-1 promotion rule. The
baseline is included in `zoomed_range` so the line is always on-canvas. Only the
top bar is direct-labelled; `rank.csv` carries the rest.

There used to be a subtitle repeating a reportability caveat ("quick
leaderboard, not the reportable metric") on the image itself, from when the
chart plotted window-level balanced accuracy. The chart now plots the
reportable metric, so the caveat no longer applies; `write_bar_chart`'s
`subtitle` parameter still exists, just unused.

### `agreement-<DATASET>.png`

One chart per dataset, one bar per model: the **share of that model's judged
pairs the judge called `agree`**, on a full 0–100% axis rather than the zoomed
one the ranking charts use, since a rate has a real zero. Written by
`summarize.py` (it reads `agreement/*.csv` directly) even though the judge stage
produces the data, so one command draws every picture for a run.

Control pairs are excluded from the rate. They pair a rationale against a
*different* recording's reference as a boilerplate floor, so counting them would
inflate it. Verified 2026-08-18 on a synthetic 48-task run with controls mixed
in: every chart value matched `run-judge.py report`'s `agreement_rate` to within
4.4e-05, which is that CSV's 4-decimal rounding.

**The denominator is not on the chart, and it is not constant.** Only windows
carrying a reference rationale are judged, so a model whose teacher pass finished
short is graded on fewer pairs, and 80% of 40 draws the same bar as 80% of 480.
Check `pairs` in `rationale-agreement.csv` before comparing two bars. An earlier
version drew the denominator as a gray track behind each bar and was replaced
(2026-08-18) with this simpler form at the maintainer's request.

Also not on the chart: `control_agreement_rate`, the per-model false-agreement
floor. `agreement_rate` alone is not evidence of anything until it clears that
floor, so it is `clears_control` in the CSV that decides whether a tall bar
means the model read the plot or writes EEG boilerplate.

### Choosing the charted metric

`--metric {balanced-accuracy,macro-f1}` (default `balanced-accuracy`). Both
numbers are in `summary.csv` either way, and `rank.csv` always ranks on
recording macro-F1; the flag only decides what the bars are. Both are
**recording-level** and both are computed over the same `MIN_SUPPORT`-cleared
classes, so switching cannot flatter a model by grading it on more classes.

| | `balanced-accuracy` (charted default) | `macro-f1` (reportable) |
|---|---|---|
| column | `recording_balanced_accuracy` | `recording_macro_f1` |
| formula | mean per-class (recall + specificity)/2 | mean per-class F1 |
| uses precision | **no** | yes |
| constant-predictor floor | exactly 0.50, always | class-balance dependent, computed per dataset |
| reference line | flat 0.50 | `baseline_macro_f1` |

The tradeoff, which is live and worth re-reading before quoting a chart.
Balanced accuracy is easier to explain and its floor is a flat 0.50 that needs
no defending, which is why it is what the charts show. It also **ignores
precision**: a model that fires a rare class on nearly every window keeps recall
1 and loses only specificity, so it can clear 0.50 while its predictions are
mostly false positives — the exact failure mode a background-dominated,
window-tiled set invites. So a bar above 0.50 is a claim about recall and
specificity only. Before that bar becomes a sentence in the writeup, check
`recording_macro_f1` and `degenerate` on the same row.

Verified 2026-08-18 on a synthetic run carrying a deliberately image-blind model
(`always-majority:0b`, same answer on every window): its
`recording_balanced_accuracy` is exactly 0.500 on TUAB and TUEP, and its
`recording_macro_f1` equals `baseline_macro_f1` to three decimals (0.947 and
0.933) — i.e. the baseline column really is "what a constant answer scores".
On TUSL and TUSZ the same model scored macro-F1 0.000 against a baseline of
0.727 / 0.689, because the baseline is the *best* constant answer and this model
happened to pick a rare class.

Caveat shared by both: when no class clears `MIN_SUPPORT` the metric is written
as `0.0`, which for balanced accuracy reads as far below its 0.50 floor. Check
`scored_classes` before believing a zero.

What the charts still cannot show is the CI on each bar or the degeneracy flag,
so a model that only ever answered "background" draws an honest-looking bar.
`summary.csv` carries `clears_baseline` and `degenerate` per model×dataset (and
`rank.csv` the counts); read one of them before quoting the picture.

Rendering is the `# --- chart` section at the top of `summarize.py` (until
2026-08-16 a separate `eval/chart.py`; it had exactly one caller, and the import
only resolved because `eval/` happened to be on `sys.path`). It knows nothing
about ranks — `write_bar_chart` takes labels, values on a 0-based scale, an
optional `y_range`, and an optional `subtitle`; the scoring section supplies the
dataset-specific title and the zoomed range via `zoomed_range(values)`. Reuse it
for any other per-model bar chart rather than re-deriving the styling. It pulls
in **matplotlib**, which the other eval scripts do not (`summarize.py` also needs
numpy; neither is in `requirements.txt`) — so `summarize.py` is now the only eval
script that cannot run in a bare environment. Pass `y_range=None` (the default)
to get the old fixed-0–1 behaviour for a chart where the data spans a wide enough
range that truncation isn't needed.

## Rationale agreement: `eval/run-judge.py`

The benchmark scores *what* a model answered. This stage scores *why*: it
compares each model's zero-shot `text_rationale` against a **reference
rationale** written for the same window by the `gemma3:12b` teacher, which was
told the true label.

```
eval run (test split)  ──▶ runs/<run>/results/<model>-<ds>.csv   [text_rationale]
teacher --split test   ──▶ datasets/<ds>/rationales-test.csv     [ground_truth_rationales]
                                     └── join on image basename ──▶ judge ──▶ agreement/
```

### The prerequisite that bites

Reference rationales do not exist by default. `generate-rationales.py` was
train-only and `eval.py` is test-only, so the two sets were **disjoint by
construction** — a naive join yields zero pairs. `--split test` (see
[rationale-generation.md](rationale-generation.md)) is what creates the overlap,
and it must be run before anything is queued here:

```bash
sbatch --export=ALL,SPLIT=test eval/scripts/generate-rationales.sbatch
```

### Commands

```bash
python eval/run-judge.py check  <run>   # join coverage, no GPU -- always run first
python eval/run-judge.py submit <run>   # one array task per model
python eval/run-judge.py report <run>   # aggregate to CSV
```

`check` prints, per model × dataset, how many evaluated rationales have a
reference. It is free and local, and it is the guard against queueing 35 judge
tasks against a missing or half-finished teacher pass.

### The judge

Configured under the **`judge`** key of `eval/config.yml` (until 2026-08-16 a
separate `eval/judge.yml`, because `create-retry-config.py` used to drop any
top-level key it did not render — it now renders `judge` explicitly). Both
`run-judge.py` and `scripts/judge.py` read it **live** from `eval/config.yml`,
not from the run's frozen copy, since the stage runs after and independently of
the sweep. `judge_array.sbatch` reads `model`, `parallel-requests` and `num-ctx`
out of the same key so server and client cannot drift apart.

- **`mistral-small3.2:24b`**, used text-only (it reads two rationales, never the
  image). Picked from the 35 already-staged models: not the `gemma3:12b` teacher,
  no gemma lineage, smallest capable family. It **is** one of the graded models,
  so its own row grades its own prose — see
  [known-issues.md](known-issues.md) before quoting it, and
  [methodology-decisions.md](methodology-decisions.md) for the full constraint
  list.
- **Whatever judge you set must already be staged in `$OLLAMA_MODELS`.** Compute
  nodes cannot reach the Ollama registry; `check` and `submit` both verify the
  manifest exists and refuse to queue otherwise, and the sbatch fails fast rather
  than attempting a pull that can only time out.
- Verdict is **two booleans**, not one: `same_conclusion` (same finding(s)
  reported) and `same_evidence` (same channel, time region, morphology). Both →
  `agree`, one → `partial`, neither → `disagree`. More robust across models than
  a three-way enum, and it grades for free.
- The prompt explicitly says a confident tone is not correctness and that a
  generic description fitting any EEG is not matching evidence.
- `num-predict: 96` and a 15-word cap on `reason`: at ~520k pairs the reason
  field is most of the decode cost.
- **One array task per model**, iterating its six datasets, so the judge's
  weights load once per ~14.8k pairs instead of once per dataset.
- Resources `gpus: 1` / `32G` / 8 in flight / `num-ctx 4096`. A 24B Q4 is ~14 GB
  × 1.15 ≈ 16 GB before KV, so it needs a full A100 rather than the 20 GB MIG
  slice. **Walltime `12:00:00` is an estimate**, like every other walltime here;
  tasks resume per pair, so an under-request costs a requeue, not work.

### Outputs

`runs/<run>/agreement/<model>-<dataset>.csv` — one row per judged pair.
`runs/<run>/rationale-agreement.csv` — one row per model × dataset.
`runs/<run>/rationale-agreement-by-model.csv` — one row per model, **macro-averaged
over datasets** so TUSZ (8,460 of the 14,850 windows) does not dominate.
`runs/<run>/agreement-<DATASET>.png` — written by **`summarize.py`**, not by
`run-judge.py`; see "`agreement-<DATASET>.png`" below.

| column | meaning |
|---|---|
| `agreement_rate` | fraction judged `agree` — the headline |
| `agreement_rate_correct` | agreement **restricted to correctly-labelled windows** |
| `lenient_rate` | `agree` + `partial` |
| `control_agreement_rate` | the false-agreement floor |
| `clears_control` | whether `agreement_rate > control_agreement_rate` |

**`agreement_rate_correct` is the one to read.** Overall agreement conflates "the
model predicted the wrong label, so of course its rationale differs" with "the
model predicted the right label for the wrong reasons". Restricting to the
windows a model labelled correctly separates right-for-the-right-reason from
right-by-luck — which is the question a capability probe exists to answer.

**`control_agreement_rate` is the floor**, and it plays the same role the
constant-predictor baseline plays in `summarize.py`. A deterministic 5% of pairs are
re-judged with a *different recording's* reference; a model whose real agreement
rate does not clear its own control rate is writing EEG boilerplate that would
match any plot. Read `clears_control` before reading the rank.
