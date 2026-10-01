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

## Train-set sampling: cap patients, keep events, bound the majority
The fine-tune's training set is a subset of the train windows, chosen by
`train/scripts/sample-train-split.py` (2026-09-16) and recorded as `sampled=1` in
`pipeline.db`, the same mechanism the test side uses. Reasons, in order:

- **Patient memorisation.** Unsampled, one TUEP patient held 1,233 of 6,868
  windows (18%) and one TUSZ patient 1,274. A model sees that person's brain
  hundreds of times and learns them, not the class. Capping at 8 recordings
  per patient (4 windows each) makes the worst case 32.
- **Class prior.** TUEP train was 5,720 epilepsy vs 1,148 no_epilepsy windows
  from 45 vs 42 patients: the skew was entirely those few heavy patients. After
  the cap it is 878/432; the majority is then thinned to 1.5x the minority,
  round-robin across patients so the cut is spread, giving 642/428.
- **Background flood.** TUSZ train was 92% pure-bckg windows. Every event
  window is kept (they are the scarce thing), pure-bckg is capped at one per
  recording, 8 recordings per patient, and 1.5x the event count.

The policy lives in `config.yml` `settings.train-sample` and mirrors the test
policy's vocabulary. It is deterministic (even spacing and sorted round-robin,
no RNG) so the training set is reproducible from `labels.csv` alone.

Rationale generation follows the same flag: `scope='rationale'` now means
"sampled train window with a label", so the teacher pass covers 8,559 windows
instead of 32,410. Rationales already written for now-unsampled rows stay in
`labels.csv` and are simply unused. Rejected: doing this inside
`build-sft-jsonl.py` or the trainer, because then the teacher would keep
spending GPU days on windows the fine-tune never sees, and two places would
have to agree on what "the training set" is.

What this does not fix: TUEP's label is per patient, so a sampled window from
an epilepsy patient can still look normal. That is inherent to the corpus and
is why TUAB (recording-level, visible in the trace) is the better first
target; see `FINETUNE-TODO.md`.

**Two more screens in the same sampler (2026-09-16).** (1) A rationale that
is truncated, degenerate, an exact duplicate, or hedges against a positive
label (`rationale_problems()` in `helpers/pipeline.py`) drops its window from
scope and the budget picks another. Regenerating instead was rejected: the
teacher decodes at temperature 0, so a blanked row comes back identical. The
hedge patterns were written without seeing the text and must be checked on
the cluster with `--show hedge` before trusting the counts. (2)
`exclude-cross-dataset-test-patients`: 117 of 1,947 patients are train in one
corpus and test in another (measured 2026-09-16), so with the flag on, no
patient in any test split is trained on anywhere. Cost: TUSL 141 -> 49
windows, TUEP 87 -> 76 patients, TUSZ 3,556 -> 3,058 windows. Default on;
flipping it is a maintainer decision recorded in `FINETUNE-TODO.md` item 9.

**Image-level screen (2026-09-21).** Hashing every PNG showed the corpora
overlap at the recording level (2,352 identical-image groups) and that 85 of
those groups span different patient tokens, so a patient-token screen alone
still let 10 fine-tune windows into the sampled test set (151 with the
cross-dataset flag off). The sampler now also drops any train image identical
to a test image, merges patient tokens that share an image before applying the
caps, and drops images that appear twice in a corpus with different labels.
Hashes ship in git (`datasets/<DS>/hashes.csv`) because the cluster copy of the
images is the same bytes and re-hashing 76k files there is wasted lustre time.
Rejected: perceptual/near-duplicate detection. Exact bytes already catch the
same-EDF case, which is the leak; near-duplicates would be a judgment call with
no ground truth. Result: 6,904 fine-tune windows, zero identical to a sampled
test window.

## The fine-tune target is rationale first, then the label
The SFT target JSON puts `text_rationale` before the booleans
(`train.rationale-first: true`, 2026-09-16). With labels first the model
commits to the answer and then writes a paragraph that cannot change it, so
training on the paragraph teaches prose, not deciding. With the description
first, the label is predicted with the description in context, which is the
only way the rationale can carry the decision. It does not change the loss
share (the booleans are still a few tokens). Label-token upweighting was the
planned fallback and is rejected (see "No token weighting" below). Zero-shot
models keep the original order so their results stand; the fine-tune's
order is recorded in `checkpoints/<key>/<DS>/manifest.json` and `train/scripts/eval.py`
enforces and parses whatever the checkpoint was trained on. Same rationale
text; nothing regenerated.

## The checkpoint is picked by teacher-forced label accuracy, not by eval loss
Eval loss on this target is dominated by rationale tokens, so it tracks how
gemma-like the prose is; the first batch (2026-09-29) showed it falling from
2.0 to 0.25 while classification stayed at chance. From 2026-09-21 to 09-29
the trainer instead *generated* answers on a val slice and scored them like
the scoreboard; at `generate: 64` windows over six corpora that gave
per-corpus balanced accuracies of exactly 0, 0.5 or 1, so the selection was
noise, and making the slice meaningful would have cost a fifth of each run.

Replaced (2026-09-29) by a metric that costs nothing: at every `true` /
`false` token position of the target, does the model's top predicted token
match? Computed from the same forward pass as val loss, over the whole val
set, balanced over true and false positions (`eval_label_balanced_accuracy`,
`compute_metrics` in `train/train.py`; the boolean token ids
are found by decoding the vocabulary once). It is teacher-forced, so
optimistic (the model sees gemma's rationale before the label, not its own),
and `load_best_model_at_end` selects on it. **It may barely discriminate
(measured 2026-09-30):** the teacher was given the label ("This recording is
labeled {labels}", "State every label explicitly"), and of the 2,800 TUAB
train rationales on the cluster 1,951 name only the true class and 849 name
both. So the label is mostly readable from the gold rationale in context,
whatever the checkpoint learned about the image. Unverified until a run
logs it: expect values near 1 from early steps. A
tokenizer that splits `true`/`false` into pieces yields no positions and the
metric logs 0; none of the 31 bases does, to be checked in the first run. Val
is patient-level and class-stratified (each class with >= 2 patients gets a
val patient before hash order fills the rest), at least 10 patients
(`train.val.min-patients`), capped at half the patients, carved per corpus.

Val is held to the same standard as test (project lead, 2026-09-30). No
merged patient, raw token, recording or image hash is shared between train
and val. No train or val patient is a test patient of any corpus, and no
train or val image is byte-identical to a test image.
`helpers/check-splits.py` checks this and `build-sft-jsonl.py` refuses to write
on any hit. The one gap is TUEV's numeric excerpt ids, which are not subjects
(known-issues).

## Vision tower adapted on purpose, in bf16
LoRA targets are a regex over the language layers and, by default
(`train.lora.vision: true`), the vision tower's attention (`qkv`, `proj`),
MLP and patch merger, at half the language learning rate via optimizer
param groups. Before this the bare names `gate_proj/up_proj/down_proj`
matched the vision MLP by accident and nothing else in the tower. The image
type is far outside the encoder's training distribution, so leaving it
frozen throws away the part of the model that has to change most; the
off setting exists for the ablation. There is no 4-bit path any more (removed
2026-09-29; it only existed for llama4, which it could not fit): NF4 would
have quantised the vision tower too and blurred
the fine trace detail the task depends on, and a 7B bf16 LoRA at batch 1
with gradient checkpointing fits the A100-40GB.

## Fine-tuned models are scored HF-side, never through Ollama
`train/scripts/eval.py` runs a checkpoint with transformers + peft on the test
windows and writes the same `eval-baseline.csv` rows the Ollama runner writes
(2026-09-16). Converting to GGUF and serving through Ollama was considered
because it would reuse `eval.py` untouched, and rejected: it quantises the
weights that were just trained, Ollama's own image resize for qwen2.5vl may
not match the resolution the model trained at (unverified; HF keeps 1536×1536
native, ~3025 image tokens), it is the same Ollama qwen2.5vl runner that loops
on these plots, and the conversion would be redone for every DPO checkpoint.
The HF runner reuses the prompt, schema, label parsing and CSV writer, and
`lm-format-enforcer` gives the same schema-constrained decoding Ollama's json
mode gave the zero-shot models, so the comparison stays like-for-like apart
from the serving stack. Cost: a second inference path to keep in step with
`eval.py` (prompt, header, retry policy).

**The "before" is the HF base model, not the Ollama row (2026-09-21).** The
zero-shot `qwen2.5vl:7b` result came through Ollama: GGUF quantisation and
Ollama's own image resize (still unverified). The fine-tune is scored in bf16
at native 1536x1536. A gap between those two would mix serving stack and
training, so `config.yml` registers `qwen2.5vl:7b-hf` (base weights, no
adapter) and `train/scripts/eval.py` runs it through the identical path; that is
the number a fine-tune has to beat. The Ollama row stays in the benchmark as
what it is: the zero-shot entry for that model.

## The fine-tune is SFT then expert DPO; the teacher does not need to be better
Plan as of 2026-09-16: LoRA SFT on gemma3:12b's label-conditioned rationales,
benchmark it on the sampled test split, then DPO on expert-analyzed
preferences over the SFT model's own rationales, and benchmark again with the
same scoreboard.
So SFT's job is a well-formed, specific starting policy that experts can rank,
and DPO is where correctness comes from. Two consequences: the teacher's
ceiling (known-issues.md) stops being the binding limit, and mode collapse
becomes the thing to avoid. Templated, near-identical rationales give DPO
nothing to prefer between, so dedup/boilerplate filtering (FINETUNE-TODO.md
item 7), fewer epochs, and temperature > 0 when sampling DPO candidates all
matter more than they would for SFT alone.

**Rejected: a frontier-model teacher (gpt-5.6-luna) for the train rationales.**
Considered because the sampled train scope (8,559 windows) made it cheap.
Rejected by the maintainer: the previous paper found GPT-class models at about
chance on these labels, so their rationales are confabulated to the label as
well, just more fluently; DPO is the planned correction instead. It would also
have needed the test references regenerated and the 34-model judge pass
re-run, and raised a data-use question about sending TUEG images to an outside
API (so far only rationale text has left the cluster).

**Rejected: a label-only SFT control run.** Rationale vs no rationale was
settled in the previous paper; the rationale stays in every training target.

## TUSZ bckg co-labels stay; bckg leaves the recording-level mean (2026-09-21)
372 sampled TUSZ train windows carry `bckg` together with a seizure class.
Measured against the annotation csvs they are onset/offset windows and
channel-wise partial seizures with a median 92% seizure coverage (45 under
25%), so both labels are true of the picture under the prompt's "select every
one present" semantics, and the target keeps them. Rejected: flipping bckg to
false when a seizure is present, which would contradict the background
stretch the teacher rationale describes and change the semantics of one class
only. At recording level bckg is true in 2,418 of 2,443 sampled test
recordings, so its balanced accuracy is ~0.5 for every model; `eval/score.py`
`RECORDING_EXCLUDE` drops it from the TUSZ recording-level mean (macro-F1,
balanced accuracy, CI, floor) while the per-class table and the window-level
metrics keep it. Every model's TUSZ headline moves up by the same mechanism;
the trainer's val metric applies the same exclusion so checkpoint selection
and the scoreboard agree.

## Every benchmark model is fine-tuned; one generic trainer (2026-09-21)
Maintainer decision: every model in the roster gets the SFT track, not only
Qwen2.5-VL-7B. The trainer therefore loads any base through
`AutoModelForImageTextToText` + `AutoProcessor`, measures the answer span by
running the prompt-only chat template through the processor (image-token
expansion included; no per-model marker string), and adapts every `Linear`
except head and embeddings, split language/vision by module path so the
vision learning-rate scale work for any tower.
`config.yml` `bases:` is the Ollama-tag -> HF-repo map with gpus/ram for the
bases over 40 GB in bf16 (2 GPUs via `device_map="auto"`, naive model
parallel). Seven bases carry a `status` (custom-code families, no chat
template, or Llama 4's 218 GB) and are skipped by `train/train.py` until decided.
Rejected: one script per family. 29 of 33 go through the same path and the
per-family differences are all in the processor and chat template, which
transformers already abstracts. The four with their own model code
(MiniCPM-V 2.6 / 4.5, moondream2, DeepSeek-OCR) train through the stacks
that already support them (OpenBMB's finetune, LLaMA-Factory, Unsloth) on a
re-export of the same JSONL, and are scored by `train/scripts/eval.py` through their own
APIs; writing a fourth training loop per family blind was rejected as the
least likely thing to work. moondream2 has no published local LoRA path.

**One run per (base, dataset); pooling is dropped (project lead,
2026-09-30).** Each base is fine-tuned separately on each corpus, so every
fine-tune is compared against the same base's zero-shot result on the same
dataset and its checkpoint is selected on that dataset's val set alone. The
lead's reasoning: more specialised results per dataset. It replaces pooled
training (2026-09-21 to 09-30: one run per base over all six corpora, 33
runs instead of 198, chosen because TUAR/TUEV/TUSL only have enough windows
that way). That cost is now accepted. Per-corpus train/val windows from the
current sample, simulated with `build-sft-jsonl.py`'s split, not measured on
the cluster: TUAB 2,284/140, TUSZ 2,604/117, TUEP 838/84, TUAR 372/31, TUEV
368/17, TUSL 26/23. TUSL is about 3 optimizer steps per epoch at an
effective batch of 8. The first-round pooled runs (`checkpoints/<key>/pooled/`,
`eval-<key>-sft-pooled.csv`) stay on disk as history; no code derives them.
Rollout is one pair first (qwen2.5vl:7b on TUAB), end to end through scoring,
before any batch.

## The first fine-tune answers labels only (maintainer, 2026-09-30)
`config.yml` `train.target: labels`: the target is the booleans-only JSON
(`{"is_abnormal": false}`), the prompt drops the evidence sentence and the
no-meta line, and the rationale comes back (`train.target: rationale`) only
if this learns. Reasoning:
- **Loss share.** The rationale is most of the target. Median answer length
  over the six train sets is 772 characters with it and 112 without
  (measured on the cluster's rationales, 2026-09-30), so the loss was mostly
  prose. With labels only, every gradient step is about the label, and the
  loss can only drop below the class prior by reading the plot.
- **An honest checkpoint metric.** Teacher-forced label accuracy no longer
  sees a gold rationale that names the answer, so it is the model's actual
  window-level accuracy (for multi-label, each boolean after the gold earlier
  ones).

This does not reopen the settled question of whether rationales help (a
previous paper answered it). It is a staged test of whether any label signal
can be learned from these plots at all before rationales are layered on.
`build-sft-jsonl.py` writes both targets from the same windows and the same
split (`sft_labels_{train,val}.jsonl` beside `sft_{train,val}.jsonl`; checked
in a mirror of the cluster data: same images, same order, same booleans, and
the rationale files byte-identical to before). Runs go to
`checkpoints/<key>/<DS>-labels` and score as `<key>-sft-<DS>-labels`, so the
two targets never collide. The zero-shot prompt is unchanged byte for byte.

## No token weighting (project lead, 2026-09-30)
The loss is the model's default mean cross-entropy over the whole target,
rationale and booleans alike. The plan after the first round collapsed
(plan.md change 1: mark the `true`/`false` positions in the collator and
multiply their loss by `train.training.label-weight`, start at 10) was never
implemented and is rejected as overengineering the problem. The failure it
targeted (loss ~95% rationale prose, answers collapse to the class prior)
is re-tested under per-dataset training first. Don't reintroduce it without
the lead.

## Training logs to wandb, offline (2026-09-30)
The lead wants runs trackable without pulling files. Before this, training
ran with `report_to="none"` and the only record was `log_history` in
`trainer_state.json` plus the job log, which meant an rsync and a plot per
look. Now `train/train.py` logs to entity `l3jovano-krembil-research-institute`,
project `eeg-vlm-finetune` (`config.yml` `train.wandb`), one run per
(base, dataset) named `<key>-<DS>`, grouped by dataset. Narval compute
nodes have no internet, so runs are written offline and uploaded with
`wandb sync` from a login node. A requeued job resumes the same run id.
`trainer_state.json` stays the source of truth on disk. Only the generic
trainer logs; the custom stacks (`train/run-custom.py`) still run with
`report_to="none"`.

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
   decides to think, which is exactly what made `qwen3-vl:8b-thinking` the eval
   sweep's long pole before it was dropped (2026-09-17, see known-issues.md).
4. **Big enough to follow the schema.** The verdict is two booleans plus a
   15-word reason; 8B-class models emit that unreliably, 24B does not.
5. **Smallest capable family**, to minimise how many rows share the judge's
   lineage. Mistral has 2 entries; llava has 5, qwen 6, gemma 8.

**Why from the roster at all:** the judge must already be in `$OLLAMA_MODELS`,
because compute nodes have no route to the Ollama registry. The eval sweep ran
against all 34 models in `config.yml`, so all 34 are known-staged — that list is
the available pool, and it is large enough that constraints 1–5 still bite.

**The cost, stated plainly:** every model in that pool is one of the 34 being
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

## One config file
The judge stage used to be configured in a separate `eval/judge.yml` — a
workaround for a config-rewriting retry step (since retired) that silently
dropped any top-level key it did not render on the first retry. Folded into
`config.yml` under a `judge:` key (2026-08-16). The decision that stands:
**`config.yml` is the single source of truth** — no second config file, no
per-stage copies to drift. `prompts` and `train` were folded in on 2026-09-13
on the same principle (see tooling.md).

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
