# Known issues & open decisions

Limitations that are understood and (mostly) accepted, plus decisions still open.
None are silent bugs — they're flagged here on purpose.

## Accepted limitations

### Rare classes are unmeasurable at the source
`mysz`(2), `spsz`(4), `elpp`(4), `tnsz`(10) appear in too few recordings to
evaluate. Handled by a support threshold in `summarize.py` (excluded from headline
macro-F1, reported with counts) and by `qc.py` treating under-support as a warning
rather than a failure. No engineering fixes this — it's a data limit. Optional
mitigation: collapse the taxonomy into coarser buckets that have support.

### Per-channel scaling hides cross-channel amplitude
Asymmetry/attenuation cues (relevant for TUAB) aren't conveyed, because each
channel is autoscaled. Rejected global scaling because it makes loud channels
unreadable. Accepted; documented in [methodology-decisions.md](methodology-decisions.md).

### Binary datasets are weakly supervised in training
For TUAB/TUEP, every train window inherits the recording-level label, so some
"abnormal" windows look normal. This is standard weak/MIL supervision. At *scoring*
time it's handled by recording-level aggregation; in *training* it's accepted.

### VLM-on-images ≠ EEG classification
Downsampling a 40-channel plot loses the fine morphology a 1D signal model reads
directly. This benchmarks VLM *transfer* to EEG images, not state-of-the-art EEG
classification. Keep the claim framed as a capability probe, or a reviewer will
ask "why not use the signal?".

### Fine-tune inherits its teacher's ceiling
Rationales come from `gemma3:12b`; the fine-tuned student can't exceed that
teacher's reasoning quality. Consider targeting ground-truth labels with rationales
as auxiliary, if rationale quality proves weak. (The teacher was `qwen2.5vl:3b`
until Qwen2.5-VL's repetition-loop bug forced a family change — see
[rationale-generation.md](rationale-generation.md).)

### TUEV/TUSL are small once unassessed windows are removed
Excluding windows the annotators never covered leaves TUEV with 354 test windows
(151 recordings) and TUSL with 99 (34 recordings, **8 patients**). That is the
honest size of those corpora for this task — the earlier larger numbers were made
up of rows with no ground truth. TUSL cannot support an inferential claim and
`summarize.py` says so.

### TUAR still has few recordings
94 test recordings, and every class is under `MIN_SUPPORT` at the recording level
except `musc`/`elec`/`eyem`. The sampler therefore protects nearly every TUAR
window from capping, so TUAR is ~25% of the whole sampled benchmark despite being
one of the smaller corpora. Correct, but worth knowing.

### The judge is one of the 35 graded models (`mistral-small3.2:24b`)
The judge must already be staged in `$OLLAMA_MODELS` (compute nodes cannot reach
the Ollama registry), and the known-staged pool is exactly the 35 models the eval
sweep ran. So the judge is necessarily one of the models it grades.

`mistral-small3.2:24b` minimises the damage — it is not the `gemma3:12b` teacher,
shares no lineage with it, and comes from the smallest capable family (2 entries)
— but two caveats travel with every agreement number:

- **Its own row grades its own prose.** Treat `mistral-small3.2:24b`'s
  `agreement_rate` as non-comparable, or drop it from the table.
- **`mistral-small3.1:24b` shares its family**, so that row may carry a
  stylistic-match advantage the control floor does not absorb. Flag it if the two
  Mistrals top the table by a small margin.

To close this properly, stage an outside judge from a login node
(`ollama pull qwen2.5:14b-instruct`) and re-run — it is one array job and needs no
new teacher GPU time. See [operations.md](operations.md).

### The rationale judge is unvalidated against human labels
`eval/run-judge.py` asks a single LLM whether two rationales say the
same thing. Nobody has checked its verdicts against a human reading of the same
pairs, so the *absolute* agreement rate should not be quoted as "N% of rationales
were correct" — it is "N% as judged by one model". What is defensible is the
*relative* ordering of models, and the gap between a model's agreement rate and
its own `control_agreement_rate` (see
[methodology-decisions.md](methodology-decisions.md)).

Two cheap strengthenings, neither done: hand-label ~100 pairs and report the
judge's accuracy against them, and re-judge a subsample with a second judge from
a different family and report Cohen's κ. The second is a one-line config change
plus a re-run into a different output dir.

### No open judge is fully independent of the roster
The 35 benchmarked models span llava, gemma, qwen, llama, mistral, minicpm,
granite, moondream and the OCR families, so any available judge shares a family
with something it grades — this would be true even of an outside judge. The
control floor absorbs systematic leniency; it does not absorb a per-family bias.
With `mistral-small3.2:24b` judging, the exposed rows are itself and
`mistral-small3.1:24b`; see the self-grading entry above.

### Reference rationales inherit the teacher's ceiling
Same caveat as the fine-tune: the reference is `gemma3:12b`'s reading of the
plot, not a neurologist's. A model that disagrees with the reference may be
right. This measures *agreement with the teacher*, which is a proxy for rationale
quality, not rationale quality itself.

### Judge throughput and walltime are unmeasured
`12:00:00` per model task and `parallel-requests: 8` are estimates from weights
size, on the same footing as every walltime in `config.yml`. ~14,850 pairs per
model at a guessed 1–2 pairs/s is ~2–4 h. Judging resumes per pair, so an
under-request costs a requeue, not work.

The judge runs on a **full A100** (`gpus: 1`, `ram: 32G`), not a MIG slice:
`mistral-small3.2:24b` is ~14 GB of Q4 weights × 1.15 ≈ 16 GB before any KV
cache, against the 19.6 GiB a `3g.20gb` slice reports. Those values mirror the
model's own roster entry, which is known to schedule. Walltime is kept at 12 h
rather than the roster's 24 h because text-only pairs cost far less than a ~4033
token image. If Ollama logs CPU offload, drop `parallel-requests` to 4.

### Several documented scripts are missing from the working tree

Checked 17 August 2026 with `git cat-file -e HEAD:<path>` against `ls`. The docs
in this directory describe all of these as if they were runnable. They are not,
right now.

**Gone entirely** (gitignored, so not in HEAD *and* not on disk, and **not
recoverable from git**):

| File | What it blocks |
|---|---|
| ~~`datasets/_render.py`~~ | **Resolved 17 August 2026.** Rewritten as `datasets/render.py`, now tracked, with the six generators repointed at it and `datasets/verify_render.py` checking it against the corpus. See [data-generation.md](data-generation.md). |
| `qc.py` | The QC gate in `operations.md` step 2. |

**`generate-all.sh` was rewritten at `datasets/generate-all.sh`** (checked
19 August 2026) but is still caught by `.gitignore`'s `generate-all.sh` pattern,
so it does not reach the cluster and would be lost like the original. `qc.py`
remains missing and ignored. Un-ignoring both is on `HARDENING.md` at the repo
root, which tracks the August 2026 bulletproofing pass.

**Deleted from the working tree but still in HEAD** (recover with
`git checkout -- <path>`):
`datasets/relabel.py`, `datasets/recover_tuar_background.py`, `hf-install.py`,
`hf-install.sbatch`, `temp/port_legacy.py`.

Two things stop this being urgent. The label transforms have **already been
applied** — `assessed` is present in all six `labels.csv`, with `assessed=false`
counts of TUAR 538, TUEV 2378, TUSL 472 and zero for TUAB/TUEP/TUSZ (train+test
combined, counted directly from the manifests). And the standing rule is never to
regenerate images.

The rendering *method* is now archived in `datasets/render.py`, with its window
selection verified exact against the corpus. The exact *bytes* of the 76,334
images are still only reproducible on the original library versions, so treat
the shipped PNGs as the artifact of record. `hf-install.py` also has to come
back before the fine-tune
track can proceed.

## Open decisions (not yet done)

### Class-stratified splits
Rare classes concentrated in few patients land entirely in one split. A
class-stratified patient split would help, but is genuinely hard for multi-label
data and can't rescue a 2-recording class. Currently: not done; handled by support
threshold instead.

### Statistical rigor — partly done
Bootstrap CIs on macro-F1 and a constant-predictor baseline **are** implemented in
`summarize.py` (cluster bootstrap over recordings). Still outstanding: a split-seed
sensitivity check on a couple of models, and multiple-comparison control across
35 models × 6 datasets × many classes. Everything still rides on one 70/30 seed.

### Throughput numbers are unmeasured
`parallel-requests` and every walltime in `config.yml` are derived from a
weights-size/active-parameter model, not measured. Walltimes carry 2.5× headroom
over that estimate for exactly this reason, and every task is resumable, so the
failure mode is a requeue rather than lost work. The largest uncertainty by far is
`qwen3-vl:8b-thinking`: its cost is set by how many reasoning tokens it emits
before the JSON, which could plausibly be half or double the 3.5× assumed here.

### qwen3-vl:235b-a22b does not fit Narval
Its Q4 weights are ~142 GB against 160 GB of VRAM on 4× A100-40GB — too little
headroom for activations, the vision tower and KV cache, so Ollama would offload
layers to CPU and crawl. Its old `ram: 512G` also exceeded the ~498 GB total on a
Narval GPU node, meaning that job could never have been scheduled at all.
Excluded, not deleted; it needs 80 GB-class cards (an H100 partition) to run.

### Image resolution below 1536 is untested
1536 vs 2048 was tested and made no difference to channel-label read-back; 1024
was never tried. Since ~3052 of the ~4033 prompt tokens are vision tokens,
prefill dominates and 1024 could nearly halve inference cost. Not changed here —
it would alter the benchmark and needs the legibility check run first.

### Cross-dataset patient leakage
The six corpora share the same `aaaaXXXX` anonymization, so the same subject may
appear across datasets. Splits are **per dataset**. If datasets are ever pooled or
used for transfer, a **global** patient split is required, or train/test leak
across datasets.

## Things that ARE handled (don't re-fix)

- Unassessed windows graded as all-negative — fixed (`assessed` column; TUSZ
  unannotated windows relabelled `bckg`, TUEV/TUSL/TUAR excluded).
- TUAR offering BCKG while its ground truth had none — fixed (background labels
  recovered from source by `datasets/recover_tuar_background.py`).
- Filename label leakage — fixed (no title on image; anonymized filenames).
- Spectrogram/waveform prompt mismatch — fixed.
- First-20 s-only sampling — fixed (windowing + event coverage).
- Flat/clipping renders — fixed (`p99*1.15` per-channel scaling).
- Unreadable labels — fixed (`fontsize 8 bold`).
- Line-noise/drift — fixed (bandpass + notch).
- Rationale-gen crash on changed `labels.csv` — fixed (reconciling `init_csv`).
- Even-spaced test sampling missing events — fixed (`pick_test` ∪ event windows).
- Serial/underutilized inference — fixed (concurrency + MIG + right-sized context).
- QC only findable by eyeballing — fixed (`qc.py` gate).
- Rationale references and eval rationales covering disjoint splits — fixed
  (`generate-rationales.py --split test` → `rationales-test.csv`; the train file
  is untouched and `run-judge.py check` verifies the join before any GPU time).
- Judge settings vanishing on the first eval retry — fixed. They now live under
  `config.yml`'s `judge:` key and `create-retry-config.py` **renders** it; the
  earlier fix (a separate `judge.yml` the rewriter never touched) worked but left
  two config files to keep in sync. Do not add a top-level key to `config.yml`
  without adding it to `render_config` — that is the actual trap.
- Whole judge array failing on `ollama pull` (2026-08-16) — fixed. Compute nodes
  have no route to `registry.ollama.ai`, and `judge_array.sbatch` had a
  `ollama show || ollama pull` fallback that could only hang for 30 s and then
  fail, reporting a network timeout instead of "model not staged".
  `run_array.sbatch` never had such a fallback — the eval sweep has always
  assumed pre-staged models, which is correct. Both the judge and rationale
  sbatches now fail fast naming the model, and `run-judge.py check` / `submit`
  verify the judge manifest is in `$OLLAMA_MODELS` before anything is queued.
  **The fix was never to change the judge model** — doing that would trade a
  one-off pull for a permanent methodology hole.
- The durable artifacts (`summary.csv`, `rank.csv`, the ranking chart) carrying
  window-level accuracy while the reportable metrics were print-only — fixed
  (`score.py` merged into `summarize.py`; ranking is recording-level macro-F1 and
  every row carries its baseline, CI and degeneracy flag).
- Model names mangled in scoring output (`gemma3-12b` for `gemma3:12b`) — fixed
  (real names read from `status.csv`, not from the sanitised results filename).
- **`qwen2.5vl:3b` cannot reach full coverage** (open; decide before reporting).
  It fails with langchain's `No data received from Ollama stream` — the stream
  yields zero chunks and the logged `raw` is empty (502/502 failed rows on
  run-20260809's TUAB). Not the flash-attention repetition bug: that produces a
  flood of chunks, not none. The failure is largely image-deterministic — on
  run-20260814's retry, previously-failed images failed again at 72-82%
  (TUAB 393/502, TUEP 355/452, TUAR 1229/1504, TUEV 76/105, TUSL 22/28) versus
  42% on first exposure, so each pass recovers only ~20% of the remainder and
  never converges. Either exclude the model or report it with reduced coverage
  and say so; do not keep spending cluster days on retries.
- `status.py` reporting tasks as `running` forever after their job left the queue
  — fixed. `get_sacct_states` asked sacct for `JobIDRaw`, which numbers array
  *elements* individually (`556372`) instead of `<array>_<task>` (`555281_0`), so
  the `"_" not in job_task` guard skipped every finished element and the
  write-back never fired. Confirmed on run-20260809-233354-623815, where 11 tasks
  sat at `running` with an empty `squeue`. Now queries `JobID`; an unexpanded
  pending array (`555281_[6-20]`) still parses as non-digit and is skipped, which
  is correct because `squeue` covers pending tasks.

## Reproducibility notes

Pin what affects outputs: Ollama version + model **digests** (quantization
changes results), matplotlib/font versions (fonts differ across machines → render
differences), and the generation `--seed`. Render on one environment where possible.
