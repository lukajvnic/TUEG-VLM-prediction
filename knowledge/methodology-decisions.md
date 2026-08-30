# Methodology decisions (and why)

Non-obvious choices, recorded so they aren't re-litigated or accidentally undone.

## Images are waveform plots, not spectrograms
The rendered images are stacked multi-channel **waveforms** (time on x, µV on y).
Prompts once said "spectrogram" — factually wrong, and models wasted rationale
tokens noting the mismatch. Corrected everywhere.

## No label leakage in the image
The EDF filename encodes the class for some datasets (`pled_…`, `normal/…`).
Rendering the filename as a plot title let models OCR the answer and cheat. Fixes:
(1) never draw the filename on the image; (2) PNG filenames are
`<patient>_<scan>_<window>` — anonymized, label-free, dataset-free.

## Patient-level train/test split
No patient's scans appear in both splits (avoids memorizing an individual's brain).
70/30 by scan count, whole patients assigned together, deterministic by seed.

## Windowing: train = sample, test = dense
Whole recordings (4–50 min) can't be shown in one image at readable temporal
resolution (a VLM's token budget caps effective resolution; ~1 px/s destroys
morphology). So recordings are cut into 20 s windows:
- **Train:** a small mixed sample (~4 windows: events + background) so the model
  learns both the events and "nothing happening", cheaply.
- **Test:** densely tiled (evenly-spaced background **∪ every event window**) so it
  reflects a full recording *and* rare events are actually present.
This resolves the "first-20 s only" bug (which missed events entirely) without a
400k-image explosion, and mirrors deployment: train on a smart sample, test on the
real thing.

## Per-channel amplitude scaling (not global)
Each channel is scaled to its own peaks (`p99*1.15`). This keeps all 30–41
channels readable, but **loses cross-channel amplitude** (asymmetry/attenuation
cues). Global scaling was prototyped and rejected: with a 25× amplitude spread,
loud channels clip to unreadable solid blocks. Decision (made on side-by-side
renders): readability > amplitude comparability, and a VLM can't reliably measure
amplitude ratios off a plot anyway. A scale bar is therefore impossible/omitted.

## Resolution: 1536×1536
Font size, not pixel count, drove label legibility — 1536 vs 2048 gave identical
model read-back of channel labels. `fontsize 5 → 8 bold` fixed it. 1536 ≈ 3052
vision tokens (vs 1920 ≈ 4123). Bigger canvas would cost tokens for no gain.

## Signal preprocessing
Bandpass 0.5–70 Hz + 60 Hz notch (US mains), applied once per recording, on real
signal channels only (derived IBI/BURSTS/SUPPR/PHOTIC skipped — filtering them
rings). Makes recordings comparable regardless of machine and removes drift/hum.

## Context length
- **Eval:** 8192 (image+prompt ≈ 4033 tokens; 4096 default truncated the JSON).
- **Rationales:** 8192 (image+prompt ≈ 3050 tokens, leaving room for the
  512-token report; `init_model` also pins `num_ctx`/`num_predict` per request).

## MIG for small models (resource efficiency)
Prompted by a cluster resource-waste warning. Models ≤~8B run on a 20 GB MIG
slice (`a100_3g.20gb:1`) instead of a full A100; larger models keep full/quad GPUs.
Also: request batching (`OLLAMA_NUM_PARALLEL`), right-sized context, right-sized
walltime. Over-requesting walltime doesn't waste allocation (billed on actual
runtime) but slows scheduling — hence tighter, resumable jobs.

## "No annotation" is not the same as "nothing happening"
`labels_for()` returns the empty set both when the annotations say a window is
quiet *and* when they never covered that window. Generation wrote both as an
all-false row, so `summarize.py` graded unassessed windows as "every class absent" —
they could only ever produce false positives and could never contribute to
recall. 70% of TUSZ, 79% of TUEV and 75% of TUSL test windows were in this state.

Which meaning applies is a property of each corpus, established by measurement
(see [datasets.md](datasets.md), "Annotation coverage"), and the two cases get
opposite treatment:
- **TUSZ** — absence *is* background, so unannotated windows are relabelled
  `bckg` and kept as real negatives.
- **TUEV, TUSL, TUAR** — absence means "outside the annotated excerpt", so those
  windows are marked `assessed=false` and excluded from evaluation.

Carried by an `assessed` column in `labels.csv`, written by
`datasets/relabel.py`. **No images are re-rendered** — this is a pure labels.csv
transform, so the PNGs on disk and on the cluster stay valid.

## TUAR's background labels were recoverable
TUAR's `generate.py` keeps only `ARTIFACTS` and `SEIZURES`, discarding the source
annotations' 3122 `bckg` rows. So `bckg` was never in TUAR ground truth while
`structure.py` and the prompt both offered BCKG — an option that could never be
right, and a test set with no true negatives at all.
`datasets/recover_tuar_background.py` restores them, again without re-rendering:
a window's time span comes from its filename index (`index * 20 s`) and its
recording from generate.py's own deterministic `scan_names` map. The script
refuses to write unless all 4778 existing labels reproduce exactly from source
first — that gate is what makes the recovered labels trustworthy.

## Test-set sampling is capped per recording, not per window
Scoring happens at the recording level (and for TUEP the truth is per *patient*),
so precision is set by the number of recordings/patients while cost scales with
windows. `eval/scripts/sample.py` keeps every window carrying an under-supported
class, caps abundant signatures and background per recording, and caps recordings
per patient for TUEP. 42,843 → 14,850 test images with **scoring-unit class
support identical to the full assessed set** in every dataset (verify any time
with `python eval/scripts/sample.py`).

Selection is by even spacing over the window index with no RNG, so every model
evaluates the identical set and the sample is reproducible from `labels.csv`
alone. TUAR is barely reduced on purpose: with only 94 test recordings, every one
of its classes is under-supported and therefore protected.

## Rare classes: handle by support, not engineering
Classes like `mysz`(2), `spsz`(4), `elpp`(4), `tnsz`(10) have too few source
recordings to measure. No sampling/splitting trick fixes that. So: keep them in
the prompt (don't change the task), but **exclude them from headline metrics by a
support threshold** (`MIN_SUPPORT=20` in `summarize.py`), reported transparently with
their counts. Exclude by *support*, never by *performance*. Alternative if the
distinctions matter: collapse the taxonomy into coarser buckets.

## Scoring at the recording level
Binary datasets are labelled per recording, so grading individual windows against
a recording-level label is unfair. `summarize.py` aggregates a recording's windows
(majority vote / any-positive) and scores the recording. Also reports per-class
recall (not raw accuracy) so a background-dominated test set can't flatter a model
that only ever says "background".

## Agreement is charted as a rate, with the denominator left to the CSV
`summarize.py` draws `agreement-<DATASET>.png` per dataset: one bar per model,
the share of its judged pairs the judge called `agree`, on a full 0–100% axis.
Control pairs are excluded, matching `run-judge.py report` — they are the
boilerplate floor, so counting them would inflate the rate.

This started (2026-08-18) as a two-series chart: a gray track for the pairs
judged with the agreed subset drawn in front, so the denominator was visible.
The maintainer asked for the simpler single-bar percentage instead, and that is
what ships.

**The known cost, since the reason for the track was real.** Judge coverage is
not uniform — only windows with a reference rationale are judged, and the teacher
pass finishing unevenly across models is the expected case — so two equal bars
can rest on very different numbers of pairs. The chart cannot show that; `pairs`
in `rationale-agreement.csv` can. Same shape of caveat as `clears_control`, which
the chart also cannot show: read the CSV before a bar becomes a sentence.

## The charts plot balanced accuracy; macro-F1 stays the reportable metric
The per-dataset charts plot recording-level balanced accuracy by default
(2026-08-18, maintainer's call after the tradeoff below was put to them);
`--metric macro-f1` charts the other, and both columns are in `summary.csv`
regardless. `rank.csv` still ranks on recording macro-F1.

Why balanced accuracy on the picture: macro-F1 needs a paragraph of explanation
and a computed, class-balance-dependent floor before a bar means anything (on
TUAB that floor is ~0.95, so an unexplained chart looks like every model failed).
Balanced accuracy has a flat 0.50 floor that any audience already reads
correctly, because a constant image-blind answer scores recall 1 / specificity 0
on the class it names and recall 0 / specificity 1 on every other, giving exactly
0.5 whatever the class balance.

Why macro-F1 remains the number in the writeup: balanced accuracy **ignores
precision**. A model that fires a rare class on nearly every window keeps recall
1 and loses only specificity, so it can sit above 0.50 while nearly all its
positives are false — precisely the behaviour a background-dominated,
window-tiled set invites. Macro-F1 charges it for the false positives.

**The known cost of this split**: the durable artifact that travels into slides
is now the precision-blind one, which is a weaker form of the 2026-08-16 trap
recorded below. It is weaker because the chart is recording-level, is computed
over the same `MIN_SUPPORT`-cleared classes as macro-F1, and sits beside a
`rank.csv` that ranks on macro-F1 — but a bar above 0.50 still is not evidence
the model is useful. Check `recording_macro_f1` and `degenerate` on the same row
before writing a sentence about a bar.

## One ranking chart per dataset, not one averaged chart
`summarize.py` writes `rank-<DATASET>.png` for every dataset in the run (six on a
full sweep) instead of a single `rank.png` of each model's mean (2026-08-18). The
mean was the wrong unit for a *capability probe*: it answers "which model is best
overall" when the question is "does any model transfer to this modality, and on
which findings" — and a model carried by one easy dataset drew the same bar as one
that was even across all six. The six rankings genuinely differ, and the averaged
chart could not show that. Each chart carries its own dataset's
constant-predictor floor, which is the honest comparison anyway: averaging floors
across datasets with different class schemas produced a line that was not the
threshold for any of them.

Charting change only — no metric moved, and the cross-dataset mean ranking still
lives in `rank.csv`. Costs: six images to place instead of one, and the
per-dataset bars are noisier than the mean (fewer recordings behind each).

## One scoring script, and the artifacts carry the reportable metric
Scoring used to be split: `score.py` computed the defensible metrics but only
printed them, while `summarize.py` wrote `summary.csv` / `rank.csv` / `rank.png`
from window-level accuracy. That put the caveats in the ephemeral output and the
misleading numbers in the durable one — a PNG travels into slides detached from
any doc, and `rank.csv` ranked an always-says-"background" model near the top.

Merged into `summarize.py` (2026-08-16). Verified equivalent before deleting
`score.py`: on a synthetic 3-task run every printed metric was byte-identical,
and `accuracy` / `balanced_accuracy` / pooled `tp,fp,tn,fn` matched the old
`summary.csv` exactly. The ranking now sorts on mean **recording-level macro-F1**
instead of balanced accuracy, and each row carries `baseline_macro_f1`,
`clears_baseline`, `degenerate`, `few_patients` and `status` beside its score, so
a number cannot be read without its caveat. On the synthetic run this flipped the
order: the deliberately degenerate model ranked 1st by balanced accuracy (0.500,
saying "normal" every time) and last by recording macro-F1.

Two behaviours changed on purpose. Tasks are discovered by globbing `results/`
rather than by walking `status.csv` for `success`, so a task that produced rows
but never reported success is scored *and labelled* instead of silently dropped;
the ranking still excludes non-success tasks, since a half-finished task would be
compared against complete ones on a fraction of the test set. And `status.csv` is
now the source of model names — results filenames are `safe_filename(model)`,
which flattens the `:` in an Ollama tag, so the old `score.py` was reporting
`gemma3-12b` for `gemma3:12b`.

## Two-stage model screening
Stage 1 runs every model over a stratified probe (`probe-images` in
`config.yml`); stage 2 runs the full sampled test set only for models that clear
a **pre-registered** floor. The floor is computable, not a judgement call:
`summarize.py` prints the best macro-F1 a *constant* predictor could reach on the
same data, and a model is promoted when its recording-level macro-F1 95% CI lower
bound exceeds it. The probe stratifies by label signature (not `limit`, which
slices the front of a sorted list) so rare classes are present in the screen —
verified: all classes survive the probe in every dataset.

The candidate set for that baseline is restricted to answers the schema can
actually emit. TUAB/TUEP resolve to exactly one class, so "predict everything"
is not available to them and must not be used to set their bar (it would put
TUAB's floor at 0.666 instead of 0.346 and screen out every model).

## Confidence intervals resample recordings, not windows
Windows from one recording share a patient, montage and session, so resampling
individual windows would understate the interval. `summarize.py` uses a cluster
bootstrap over whole recordings for both the window-level and recording-level
macro-F1. Reported as a 95% percentile CI; 1000 resamples costs well under a
second per result file.

## Concurrency + resume everywhere
Both eval and rationale generation use bounded thread-pool concurrency (with
`OLLAMA_NUM_PARALLEL`), tolerate single-request failures, and resume from partial
progress. This is both a speed and a resource-efficiency measure.

## Never auto-commit
Working style: changes are made and left for review; commits are done manually by
the maintainer. (See the agent memory / project convention.)

## Rationale agreement needs a reference on the *test* split
The obvious way to grade rationale quality — compare each model's zero-shot
rationale to the "ground truth rationale" — does not work off the shelf here:
`generate-rationales.py` was train-only and `eval.py` is test-only, and the split
is patient-level, so the two sets are **disjoint by construction**. The join
yields zero rows.

Fixed by giving the generator a `--split test` mode that writes a *separate*
`rationales-test.csv` over the sampled test windows. Alternatives rejected:
- **Compare against train rationales of the same label signature.** Cheap, needs
  no new generation, but it is a style match, not a per-image comparison — it
  cannot distinguish a model that read *this* plot from one that wrote a
  plausible paragraph about that class.
- **Skip the reference; ask the judge whether the rationale supports the true
  label.** Also cheap, but it grades plausibility rather than agreement, and it
  gives the judge no anchor for what the plot actually shows.

## Full coverage, not a subsample
Every one of the 14,850 sampled test windows gets a reference rationale, and
every model's rationale for it is judged (~520k pairs). A ~250/dataset subsample
would have cost ~10× less with an agreement-rate SE under 3%, which is ample for
a table. Chose full coverage anyway because the reference pass is a one-time cost
that makes every later judging question re-runnable without new teacher GPU time,
and because per-pair resume makes the judge job cheap to interrupt. The subsample
path still exists as `judge.py --limit`, and it spreads rather than head-slices.

## The judge is picked from the already-staged roster
`mistral-small3.2:24b`. The constraints, in order:
1. **Not `gemma3:12b`, and not the gemma family at all** (8 gemma + 2 medgemma).
   `gemma3:12b` writes every reference rationale, so a gemma judge would be
   scoring similarity to prose from its own lineage.
2. **Not `qwen2.5vl`** (3 models). Rejected for the single-token repetition loop
   (ollama/ollama#10767) — the same defect that disqualified it as teacher.
3. **Not a `-thinking` model.** Token cost per pair must be predictable across
   ~520k pairs; a reasoning model makes walltime a function of how much it
   decides to think, which is exactly what makes `qwen3-vl:8b-thinking` the eval
   sweep's long pole.
4. **Big enough to follow the schema.** The verdict is two booleans plus a
   15-word reason; 8B-class models emit that unreliably, 24B does not.
5. **Smallest capable family**, to minimise how many rows share the judge's
   lineage. Mistral has 2 entries; llava has 5, qwen 6, gemma 8.

**Why from the roster at all:** the judge must already be in `$OLLAMA_MODELS`,
because compute nodes have no route to the Ollama registry. The eval sweep ran
against all 35 models in `config.yml`, so all 35 are known-staged — that list is
the available pool, and it is large enough that constraints 1–5 still bite.

**The cost, stated plainly:** every model in that pool is one of the 35 being
graded, so the judge grades its own prose on its own row. That is a real hole and
it is recorded in [known-issues.md](known-issues.md) — treat the
`mistral-small3.2:24b` row as non-comparable. An outside judge such as
`qwen2.5:14b-instruct` would close it, at the cost of staging one model from a
login node first.

Complete family independence was never achievable anyway: the roster spans llava,
gemma, qwen, llama, mistral, minicpm, granite, moondream and the OCR models, so
any open judge shares a family with *something* being graded. The control floor
below is what carries that weight instead of a family-purity claim.

## Verdict is two booleans, not a three-way label
`same_conclusion` (same finding(s) reported) and `same_evidence` (same channel,
time region, morphology), combined into agree / partial / disagree. Two booleans
are more robustly emitted across models than an enum, and they decompose the
actual question: a model can name the right label off the wrong feature, and that
is a different failure from naming the wrong label.

## The control floor, not a human-labelled validation set
A deterministic 5% of pairs are judged a second time against a **different
recording's** reference. Their agreement rate is the judge's false-agreement
floor, and `clears_control` gates the ranking. This is the same move `summarize.py`
makes with the constant-predictor baseline: rather than assert the metric is
valid, publish the score an image-blind answer would get on the same data.

It specifically catches the failure mode that makes LLM judges useless here — a
generic "rhythmic activity is visible across several channels" matches every
reference. If a model's agreement rate does not clear its own control rate, its
rationales carry no window-specific information, whatever the headline says.
(The judge prompt also states outright that a confident tone is not correctness
and that a description fitting any EEG is not matching evidence.)

## One config file, and `create-retry-config.py` must render every key
The judge stage used to be configured in a separate `eval/judge.yml`, because
`create-retry-config.py` rebuilds `config.yml` from `config-base.yml` and
rendered only `settings`, `datasets` and `models` — any other top-level key was
**silently dropped on the first retry**.

Folded into `config.yml` under a `judge:` key (2026-08-16), with
`render_config` extended to emit it. The real invariant is not "keep judge in its
own file" but **every key `load_retry_config` loads, `render_config` must
render** — the separate file was a workaround for a renderer that silently lost
data, and the workaround left two config files to keep in sync instead of fixing
the loss. Verified by round-tripping a synthetic run: `judge` comes back
byte-identical under `yaml.safe_load`, `resume-from` is set, and the succeeded
pair is commented out.

Two consequences to keep in mind:
- **Comments do not survive the retry rewrite.** `render_config` goes through
  `yaml.safe_dump`, so the rationale comments in `judge`/`settings`/`datasets`
  are stripped the first time a retry config is generated. This was already true
  of the other two keys; `knowledge/` is the durable copy.
- **`config-base.yml` deliberately has no `judge` key.** It supplies only
  `models`; a second copy of the judge settings there would be a silent
  drift hazard, and `config.yml` is the single source of truth.

The judge config is still read **live** from `eval/config.yml` by `run-judge.py`,
`scripts/judge.py` and `judge_array.sbatch` — never from a run's frozen copy,
since the stage runs after and independently of the sweep.

## `rationales-test.csv` must never reach the fine-tune
It is generated from held-out test patients. Training on it voids the
patient-level split that the whole benchmark rests on. Hence a distinct filename
rather than a `split` column inside `rationales.csv` — and never glob
`rationales*.csv` into training data. The separation is also mechanical:
`init_csv()` rebuilds its target file from `labels.csv` every run, so a shared
filename would blank one split's work each time the other ran.
