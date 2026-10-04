# Known issues & open decisions

Limitations that are understood and (mostly) accepted, plus decisions still open.
None are silent bugs — they're flagged here on purpose.

## Accepted limitations

### Rare classes are unmeasurable at the source
`mysz`(2), `spsz`(4), `elpp`(4), `tnsz`(10) appear in too few recordings to
evaluate. Handled by a support threshold in `summarize.py` (excluded from headline
macro-F1, reported with counts). No engineering fixes this — it's a data limit. Optional
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

### The judge is one of the 34 graded models (`mistral-small3.2:24b`)
The judge must already be staged in `$OLLAMA_MODELS` (compute nodes cannot reach
the Ollama registry), and the known-staged pool is exactly the 34 models the eval
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
The 34 benchmarked models span llava, gemma, qwen, llama, mistral, minicpm,
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

**`generate-all.sh` was rewritten at `datasets/generate-all.sh`** (checked
19 August 2026) but is still caught by `.gitignore`'s `generate-all.sh` pattern,
so it does not reach the cluster and would be lost like the original.
Un-ignoring it is on `HARDENING.md` at the repo root, which tracks the
August 2026 bulletproofing pass.

**Deleted from the working tree but still in HEAD** (recover with
`git checkout -- <path>`):
`datasets/relabel.py`, `datasets/recover_tuar_background.py`,
`temp/port_legacy.py`. (`hf-install.py` / `hf-install.sbatch` were later
removed from HEAD too, in `bb5b4be`; a trimmed `train/scripts/hf-install.py` replaced
them on 2026-09-16.)

Two things stop this being urgent. The label transforms have **already been
applied** — `assessed` is present in all six `labels.csv`, with `assessed=false`
counts of TUAR 538, TUEV 2378, TUSL 472 and zero for TUAB/TUEP/TUSZ (train+test
combined, counted directly from the manifests). And the standing rule is never to
regenerate images.

The rendering *method* is now archived in `datasets/render.py`, with its window
selection verified exact against the corpus. The exact *bytes* of the 76,334
images are still only reproducible on the original library versions, so treat
the shipped PNGs as the artifact of record.

### Scratch purge on 2026-10-15: moving everything we keep to /project (found 2026-10-03)
`/scratch/to_delete/luka` (root-owned, written 2026-10-02) lists 38,559 files, 2.1 TB. The deletion date is
2026-10-15 (Alliance email to the maintainer).
- **Model weights.** All the `hf-checkpoints` snapshots: the 27 bases in use (822 GB) and 14 unused ones
  (1,309 GB, mostly Llama-4 at 1,045 GB).
- **The job venv.** 12,263 files of `.venv`. Imports read `__pycache__/*.pyc` and only `stat()` the `.py`
  files, so their atimes stay at June/July. Once a `.py` is gone, its package imports as an empty namespace.
- **Old files.** 744 `.git/objects`, plus the July `temp/` logs and results, old `eval/runs`, and 613 July
  `wandb/offline-run-*` dirs.
- **Not listed** (all accessed recently): the images, the rationale-filled `labels.csv`, results CSVs,
  `pipeline.db`, checkpoints, `archive/`.

Plan (maintainer, 2026-10-03). Nothing we created is lost; scratch only holds temporary job output.
1. Copy to `/project/def-milad777/luka/` now. `migration-logs/migrate.sh` makes a snapshot copy and is
   re-runnable as a final sync:
   - the repo minus `.venv` (~130 GB, 140k files) to `TUEG-VLM-prediction/`;
   - the weights in use, minus the LLaVA-1.5 family and llava-34b, to `hf-checkpoints/` (~830 GB);
   - the group quota is 1,000 GB / 500k files, shared with the other group members.
2. Quota: the maintainer is asking Alliance support for 2 TB. Once granted, copy the LLaVA-1.5 and llava-34b
   weights too.
3. After the queued jobs finish (they were submitted with `/scratch` paths) and before Oct 15:
   - run a final `rsync` of the repo;
   - rebuild the venv in the project checkout (`helpers/cluster-setup.sh`);
   - set `HF_CHECKPOINTS=/project/def-milad777/luka/hf-checkpoints`;
   - run `helpers/preflight.py main` and `rehearsal/rehearse.py gpu`;
   - submit from `/project` from then on.
4. The 14 unused snapshots go with the purge (maintainer's call). They are public and can be re-downloaded
   with `train/scripts/hf-install.py`: Llama-4 Maverick and Scout, Qwen3-VL 2B/4B/8B/30B-A3B/32B Thinking,
   Qwen3-VL-235B-A22B, llava-v1.6-mistral-7b, Phi-3-vision, Qwen2.5-VL-3B, DeepSeek-OCR (the deepseek-ai copy;
   unsloth's is in use), moondream2, MiniCPM-V-4_6.

Touching files to dodge the purge is against Alliance rules, so everything here is a copy to `/project`.

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
34 models × 6 datasets × many classes. Everything still rides on one 70/30 seed.

### First fine-tune batch: startup failures (2026-09-23)

All 29 generic jobs died at startup on the first submission because the cluster
`.venv` had never had `requirements.txt`'s torch stack installed. After that,
19 of 29 failed in the first minutes; the 9 that loaded cleanly (gemma3 x3,
medgemma x2, llava 7b/13b, bakllava, granite) kept running. Causes and fixes:

- Bare `ImportError:` from transformers on every Qwen, Mistral and GLM-OCR
  load, and `Could not import module 'Gemma4Processor'`: `torchvision` was
  not installed; transformers 5's image processors need it. Added to
  `requirements.txt`.
- `couldn't connect to huggingface.co` for llava:34b, llava-phi3:3.8b,
  minicpm-v4.6:1b: snapshots not staged. `run.py` now skips a base whose
  `$SCRATCH/hf-checkpoints/<repo>` has no `config.json` and names the
  `hf-install.py` command, instead of letting the trainer fall back to the
  hub id offline.
- `this processor does not have a chat template` for llava-llama3:8b: the
  xtuner repo ships none. `bases:` takes a `chat-template` (Jinja) that the
  trainer and `train/scripts/eval.py` set on the processor; llava-llama3's is the Llama 3
  format with `<image>`. Prefix masking checked locally by rendering both turns.
- llama4:16x17b is off (`status` in `bases:`): transformers 5.8 keeps the 16
  experts per layer as one 3-D parameter, bitsandbytes only converts
  `nn.Linear`, so NF4 leaves ~190 GB of experts in bf16 and LoRA never reaches
  them. Needs an MoE-aware quantiser or a pre-quantised checkpoint.
- (4-bit path removed 2026-09-29.) 4-bit loading dropped transformers' default `lm_head` skip because a
  `llm_int8_skip_modules` list was passed; `lm_head` is named now. That list
  matches module-name prefixes/suffixes, not substrings.
- DeepSeek-OCR: the `deepseek-ai` repo's modeling code imports
  `LlamaFlashAttention2`, gone since transformers 4.48, so it only loads on
  4.46. `bases:` now points at `unsloth/DeepSeek-OCR` (same config and
  weights, patched imports), which is what Unsloth's notebook trains from.
  `train/scripts/eval.py` passes `eval_mode=True` and a real `output_path` to `infer()`,
  which otherwise returns None and crashes on `makedirs(None)`.
- Static review of every base against transformers 5.8.0 (2026-09-23, three
  passes over library source and model repos; nothing GPU-verified):
  - gemma4:12b is `gemma4_unified`, saved by transformers 5.10; not in 5.8.
    `status` set, off the batch until a newer-transformers venv exists.
  - gemma4:26b/31b generation prompt ends with an empty thinking channel the
    assistant turn never renders; the collator splices it in
    (`EMPTY_THOUGHT` in `train/scripts/eval.py`), so the target starts exactly
    where generation starts.
  - mistral-small3.2 ships weights and `tekken.json` only; `hf-install.py`
    copies 3.1's processor/tokenizer/template files (`processor-files-from`).
    Both Mistral repos also carry a 48 GB `consolidated.safetensors` that is
    now ignored on download.
  - llava-phi3 and llava-llama3 have no `processor_config.json` (patch_size
    None crashes LlavaProcessor): `processor-kwargs` supplies llava-1.5's
    values. llava-phi3 also gets a Phi-3 `chat-template`.
  - llava:7b/13b render no stop token after the assistant turn and bakllava
    renders the literal typo `<\s>`: `chat-template` overrides end the turn
    with `</s>`. GLM-OCR renders none either; `answer-suffix: <|user|>` (one
    of its eos ids) plus `enable_thinking: false` so the empty
    `<think></think>` sits in the prompt, not the target. MiniCPM-V-4.6 gets
    the same thinking switch.
  - Templates that emit `{{ bos_token }}` (gemma, medgemma, mistral) were
    tokenised with `add_special_tokens` on, giving a double BOS;
    `encode_kwargs()` turns it off when the rendered text already starts
    with BOS, in the trainer and in `train/scripts/eval.py`.
  - 32B-class dense bases (llava:34b, qwen2.5vl:32b, qwen3-vl:32b) are
    66-70 GB in bf16, 5-10% headroom on 2 x 40 GB: now 3 GPUs / 192G, as are
    gemma4:31b and qwen3-vl:30b-a3b (62 GB).
  - MoE bases (gemma4:26b, qwen3-vl:30b-a3b) keep experts as fused
    parameters; LoRA reaches attention and vision only. Accepted.
  - gemma4's audio tower is skipped by `lora_targets` (`SKIP_HINTS`).
  - granite3.2-vision tiles 1536² into ~7.4k image tokens: fits, slowest
    1-GPU run.
  - GLM-OCR's config was saved by transformers 5.17; 5.8 keeps unknown keys,
    weight names unverified until the first load.
- 2026-09-24, second batch: 24 of 27 generic bases finished in 7 to 27 h
  (see the elapsed table in `knowledge/operations.md`); llava:34b died
  loading its tokenizer (`tiktoken` missing, added to `requirements.txt`).
- 2026-09-25, first predict batch: every task died at the first generate
  with lm-format-enforcer's "transformers is not installed". The enforcer
  imports `transformers.tokenization_utils`, which transformers 5 removed
  (slow tokenizers), and swallows the ImportError. `enforcer()` in
  `shim_tokenization_utils()` in `train/scripts/eval.py` puts the class where the
  enforcer looks: 5.8 had no module at all (alias the base module), 5.14
  aliases the name to `tokenization_utils_sentencepiece` without the class
  (set the attribute). The first shim only handled the 5.8 case. The 2026-09-23 review had called this import 5.x-safe;
  it was not. Same path in the trainer: val generation during the first
  training batch ran unconstrained (its ImportError fallback), so the
  checkpoint choice there used a parsed-JSON rate of whatever the models
  produced unaided. Retrain is not warranted; note it when reading
  `eval_json_ok` in those logs.
- 2026-09-26, third batch, all 162 failed at startup again: the enforcer
  shim had not been pulled before submitting (same error as the day before);
  llava:34b's repo has only `tokenizer.model`, and without `sentencepiece`
  transformers 5 tries the tiktoken parser on it; `.venv-unsloth` had been
  rebuilt from the first install line, without matplotlib; LLaMA-Factory
  also bounds `datasets <= 4.0.0`, which the wheelhouse `4.0.0+computecanada`
  fails. Root cause across all three batches: every install was hand-typed
  from chat and never verified before submitting. `helpers/cluster-setup.sh`
  and `helpers/preflight.py` replace that. Also found: `train/scripts/eval.py` was
  scoring the custom families in the main venv, whose transformers 5.8
  cannot import their remote code; they now score in their training venv.
- The wheelhouse `bitsandbytes` (0.50.1+computecanada) CPU library faults
  with SIGILL on Narval's EPYC 7532 / Milan CPUs (no AVX-512), login and
  compute nodes alike. On a GPU node its CUDA library loads instead, so
  training and predict are unaffected; only a GPU-less import of
  bitsandbytes, or of peft >= 0.20 / trl / unsloth (they import it eagerly)
  crashes. `helpers/preflight.py` skips those imports without a GPU
  (`NEEDS_GPU`) and `cluster-setup.sh PREFLIGHT_GPU=1` runs the full check
  through `srun` on a GPU node. Found 2026-09-26.
- 2026-09-27, fourth batch: predict runs (23 tasks done overnight, rows
  flowing), but four train jobs died. gemma4:12b: host-RAM OOM at 64G during
  load, now 128G. MiniCPM (both): LLaMA-Factory also bounds `trl <= 0.24.0`
  and the wheelhouse has `0.24.0+computecanada`; the preflight had only
  checked two of its five bounds, now all five, and the venv pins trl,
  accelerate and peft below the bounds. DeepSeek-OCR: the Unsloth script
  decoded every training image into RAM up front (~48 GB) and PIL failed
  part-way; images are paths now, opened per batch. llava:34b on 3 GPUs was
  still running at 37 h.
- 2026-09-28: 52 of the fourth batch's predict tasks were OOM-killed (host
  RAM, 64G / 128G) after 1 to 24 h of correct output. lm-format-enforcer's
  TokenEnforcer keeps an allowed-token tensor for every token prefix it has
  seen and never evicts; one prefix function was reused across every image
  of a task, so ~200 entries per image accumulated. `Enforcer.fresh()` in
  `train/scripts/eval.py` now builds a new one per generation (the vocabulary
  table is still cached once). Progress lines print peak RSS so the next run
  shows it flat. Rows written before the kill were kept and the resubmit
  resumes from them.
- 2026-09-29, first look at the fine-tunes (`helpers/inspect-sft.py` on
  glm-ocr, bakllava, gemma3:27b, qwen2.5vl:7b, gemma3:4b; test coverage
  partial): training loss fell from ~2.0 to ~0.25 in every run and the val
  JSON-valid rate was 1.0, but recording balanced accuracy stayed at chance
  on val throughout and on the test set the fine-tunes mostly answer one
  class (bakllava TUAB "normal" 100%, TUEP "no_epilepsy" 98%; gemma3:27b
  TUEP "epilepsy" 98%, TUEV "pled" 99%, TUSZ "bckg" 100%; glm-ocr TUAB
  "abnormal" 100%). Reading: the loss is ~95% rationale prose, so the models
  learned to write gemma-style paragraphs and emit the class prior, not to
  read the plot (FINETUNE-TODO item 4's second bullet, label-token
  upweighting, was never done). Exceptions worth noting: gemma3:27b TUAB
  0.635 (67% abnormal, 2 classes), qwen2.5vl:7b TUEP 0.564 and TUSL 0.569.
  Second problem: the trainer's val slice (`train.val.generate: 64` windows
  over six corpora) gives per-corpus balanced accuracies of exactly 0, 0.5
  or 1, so checkpoint selection on it is noise.
- 2026-09-30, loss curves of the 27 finished pooled runs (`trainer_state.json`
  `log_history`, medians): train loss falls below 0.6 by step 50, val loss
  goes 0.475 (step 100) to 0.287 (step 1642). At the epoch-2 boundary (step
  820) train loss steps down 0.330 to 0.283 while val barely moves (0.328 to
  0.325), so the end gap (train 0.247, val 0.287) is second-epoch
  memorisation; val never rises. Final val spans 0.225 to 0.347, so loss does
  not separate the bases. Consequence of the noisy selection above: 7 of 27
  kept checkpoints are step 100 (bakllava, gemma4:e2b, llava:7b,
  llava-llama3, minicpm-v4.6, qwen3-vl:2b, qwen3-vl:4b), so those "fine-tunes"
  are 100 steps in.
- 2026-09-30, minicpm-v4.6:1b's logged *train* loss is ~8x its val loss at
  every eval step (ratios 7.55 to 8.03 from its `trainer_state.json`;
  every other pooled run sits at ~1.0). 8 is `gradient-accumulation`, so the
  logged number is a sum over micro-batches, not a mean. Likely cause, not
  verified: its forward takes `**kwargs`, so the Trainer skips the divide by
  accumulation steps and expects the model to normalise by
  `num_items_in_batch`, which MiniCPM's code ignores. If so the gradient was
  8x too; AdamW mostly absorbs that, but clipping acts on the inflated norm.
  Val loss (0.32 at the end) is unaffected. `report/figures/loss-curves.png`
  plots its train loss divided by 8.
- 2026-09-30, per-dataset split sizes. These come from the cluster's
  rationales and pipeline.db scope, run through the new builder in a local
  mirror, not yet on the cluster. Train/val windows (patients): TUAB 2283 (443) / 140 (24),
  TUAR 371 (110) / 31 (10), TUEP 834 (65) / 83 (10), TUEV 366 (155) / 17
  (10), TUSL 25 (7) / 23 (7), TUSZ 2603 (392) / 117 (21). **TUSL is open:**
  `min-patients: 10` is capped at half the patients, so val takes 23 of 48
  windows and training gets ~3 optimizer steps per epoch. Needs a lead
  decision (smaller val for TUSL, more sampled windows, or leave TUSL out
  of the fine-tune track).
- 2026-09-30, TUEV numeric excerpt ids (knowledge/datasets.md) in the
  per-dataset split: 43 ids in train (125 windows), 3 in val (6 windows).
  They are excerpt ids, not subjects, so train/val subject-disjointness is
  unprovable for them; only the image-hash merge links excerpts of one
  subject. `helpers/check-splits.py` prints them as a warning, not a failure.
- 2026-09-30, label-only fine-tunes (`target: labels` in their manifest) are never
  judged, so their `pipeline.db` rows never reach `done` (which needs
  `judged` and `judged_gpt`) and `helpers/status.py` counts them as
  unfinished. Cosmetic: scoring doesn't read `done`.
- 2026-10-04, **scoring cost of the multi-label datasets.** Label-only answers are ~10 tokens on TUAB and TUEP,
  but ~58 to 74 on TUAR and TUSZ (8 per boolean field). Constrained decoding is sequential, so TUSZ's 8,460 test
  windows are estimated at ~12 h for qwen2.5vl:7b and 20 to 60 h for the Gemma and 24-32B bases (estimates in
  `helpers/plan-resources.py`, not measured).
  - A faster scorer would teacher-force the fixed JSON keys and read only the true/false decision at each value
    position: about K short forward passes in place of ~74 single-token decodes, an estimated 5-10x. It must
    match constrained greedy decoding first.
- 2026-10-04, **llava:34b's tokenizer vs lm-format-enforcer.** On TUAR, TUEV and TUSL (`rehearsal/check-pairs.py`),
  the enforcer rejects the exact tokens llava:34b trains on. It still produces valid JSON by another tokenization,
  so the risk is a small accuracy loss for that base on those datasets, not a failure.
- Custom stacks: DeepSeek-OCR's modeling file imports `matplotlib`
  (`.venv-unsloth`); LLaMA-Factory rejects the wheelhouse
  `transformers==5.8.0+computecanada` against its `<=5.8.0` bound (local tags
  sort above), so `.venv-llamafactory` pins 4.56.2. Both venvs must be built
  with Alliance's `virtualenv --no-download` after `module load arrow`, or
  pyarrow is invisible.

### Throughput numbers are unmeasured
`parallel-requests` and every walltime in `config.yml` are derived from a
weights-size/active-parameter model, not measured. Walltimes carry 2.5× headroom
over that estimate for exactly this reason, and every task is resumable, so the
failure mode is a requeue rather than lost work. The largest uncertainty was
`qwen3-vl:8b-thinking`, whose cost was set by how many reasoning tokens it emitted
before the JSON; it was dropped 2026-09-17 (see below).

### judge-gpt.csv had duplicate rows from concurrent runs (2026-09-09)
Two instances of judge/judge-openai.py overlapped (an earlier one never died),
so both judged the same pending pairs: TUAB ended up with 76,248 rows against
~74k eval pairs. Duplicates never corrupted pipeline.db (sync uses sets) but
waste API spend and would double-count in analysis. `judge/dedup.py` drops
duplicate (path, model) rows keeping the first; it supports `--dry-run` and
`--file` (works on judge-baseline.csv too). Run it only while no judge is
appending, and check `pgrep -af judge-openai` before starting a new run.

### qwen3-vl:8b-thinking truncated JSON at 16k context (fixed 2026-09-08)
~992 eval units failed with "Invalid json output" or partial schema objects
(e.g. a TUSZOutput with 2 of ~10 fields), concentrated on the long-schema
datasets (TUAR 530, TUSZ 444) and near-absent on binary ones (TUAB 11, TUEP 4).
Cause: reasoning tokens exhausted the 16384 context (~4k of it prompt) before
the constrained JSON completed; `temperature: 0` made every retry fail
identically, so requeues burned attempts (963 on TUAR alone) without progress.
Fix: `context: 32768`, `parallel-requests: 4` (KV cache doubles; 8 in flight
would not fit the A100's 40 GB). `run.py` reads `context` at submit time, so
the fix applies on resubmission without touching running jobs.

### qwen3-vl:8b-thinking dropped from the benchmark (2026-09-17)
Cut from the roster: commented out in `config.yml` with the other five
`-thinking` variants, and its 76,334 `pipeline.db` rows zeroed (`sampled`,
`evaled`, `judged`, `judged_gpt`, `done` = 0, `scope` = none) so it counts as
neither pending nor done. The next `helpers/pipeline.py` sync deletes the rows
outright, since sync drops any model absent from `config.yml`; that is expected.
Its `eval-baseline.csv` / `judge-*.csv` rows, if any exist on the cluster, are
untouched and are simply never re-imported. Consequence: the benchmark has no
reasoning-vs-instruct ablation; the two entries below are kept as the record of
why (unbounded thinking with no Ollama budget made its cost unpredictable and its
failures unfixable by retry). The `-thinking` judge constraint in
methodology-decisions.md is now moot but still correct.

### qwen3-vl:8b-thinking thinking loops — planned workaround (researched 2026-09-13, superseded by the drop above)
Beyond the 16k truncation above, the model can fall into thinking loops that
eat the full 32k context: `</think>` is an ordinary sampled token and repetition
loops suppress it, so no budget means no guaranteed answer. Ollama has **no
thinking budget** (open proposal [ollama#17561](https://github.com/ollama/ollama/issues/17561),
unimplemented as of 2026-09; vLLM and llama.cpp enforce one during decoding by
force-injecting `</think>`). Plan when this is picked up again:

- **Cap `num_predict` ~6k** for this model so a runaway think can't consume 32k
  (answers are short structured JSON; 4–6k is generous). Per #17561's
  benchmarks, budgeting costs a few accuracy points but removes all
  context-exhaustion failures.
- **Streaming loop abort in `eval.py`**: sliding-window n-gram repetition check
  on streamed tokens (or crudely, same ~50-char chunk 3+ times), abort the
  request early. `eval.py` is read at task start, so this deploys without
  resubmitting.
- **Retry the failed tail with `think: false`** via the existing blank-row retry
  infra. Caveat: qwen3-vl's Ollama template silently ignored `think: false`
  ([ollama#14798](https://github.com/ollama/ollama/issues/14798)) — verify on
  the cluster's Ollama version that nothink actually suppresses thinking before
  relying on it.
- **Do not** resize `num_ctx` dynamically per request — it forces a model
  reload, which serializes everything at `parallel-requests: 4`. The "dynamic"
  part lives in the output budget + retry ladder, not the context.

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

### Cross-dataset patient leakage, and duplicate images the patient split cannot see
The six corpora share the same `aaaaXXXX` anonymization, so the same subject may
appear across datasets (117 patients are train in one corpus and test in another,
measured 2026-09-16). `train/scripts/sample-train-split.py` screens those out when
`settings.train-sample.exclude-cross-dataset-test-patients` is on.

**Measured 2026-09-21 by md5-hashing all 76,334 rendered PNGs** (same window of
the same EDF renders byte-identically, so a hash match is the same EEG):

- 2,352 groups of identical images, 4,797 images. Almost all are the same
  recording present in two corpora: TUEP/TUSZ 1,266 groups, TUAR/TUSZ 333,
  TUAB/TUEP 207, TUSL/TUSZ 197, TUAR/TUSL 127, TUAB/TUSZ 116; within one corpus
  TUEV 74, TUSZ 50, TUEP 2.
- **85 groups span different patient ids.** Two kinds: TUEV's numeric eval dirs
  (`001`, `004`, `015`, `091` hold the same EEG; the number is an excerpt id, not
  a subject, so the "patient" split over those 80 dirs is not a patient split),
  and near-identical tokens across corpora (`aaaaaduk` train vs `aaaaadul` test in
  TUEV, `aaaaatjz` in TUAR vs `aaaaalqk` in TUEP). The patient-id screen cannot
  see either.
- 870 train windows corpus-wide are identical to some test window (TUSZ 445,
  TUEP 149, TUSL 120, TUAR 66, TUAB 62, TUEV 28).
- Leak into the **fine-tune sample vs the sampled test set**: with the exclusion
  flag on, 10 windows (TUEV 9, TUAR 1). With it off, 151 (TUAB 40, TUSL 41,
  TUSZ 35, TUAR 22, TUEV 9, TUEP 4). This is the strongest argument for keeping
  the flag on (FINETUNE-TODO item 9), and for adding a hash screen so the flag's
  blind spots close too.
- 19 same-corpus duplicate groups carry conflicting labels (identical image, two
  label sets): label noise, small.

**Fixed 2026-09-21** in `train/scripts/sample-train-split.py`: `helpers/hash-images.py`
writes `datasets/<DS>/hashes.csv` (tracked in git), the sampler drops any train
image identical to a test image, merges patient tokens that share an image
before the caps, and drops same-corpus images carrying two different label
sets. Fine-tune sample: 6,904 windows, zero identical to a sampled test window.
The 45 same-patient same-corpus groups (e.g. two windows of one TUEP recording
rendering identically) are left alone. Caveat: most multi-patient window-0
groups are calibration signal, not shared EEG (see below), so the patient merge
over-links a few unrelated tokens (largest merged cluster 12 tokens); the cost
is a few more train windows held out, not a leak.

### Fine-tune track review (2026-09-16)
Review of `helpers/build-sft-jsonl.py`, `train/scripts/finetune_sample.py` and
`config.yml`'s `train:` key, with train-split counts measured from the local
`labels.csv` (rationales themselves are cluster-only, so rationale text was not
inspected). Nothing below is fixed yet; each is a decision. The working
checklist, in plain language and in the order to tackle them, is
`FINETUNE-TODO.md` at the repo root; tick items off there and update here.

1. **No evaluation path for a fine-tuned checkpoint.** `eval/models/eval.py`
   only speaks Ollama tags from `config.yml`, and no scoring script exists in the
   tree (`summarize.py` is gone). Needs an HF-side inference script that writes
   `eval-baseline.csv` rows under a model name like `qwen2.5vl:7b-ft-<DS>` so the
   existing judge/scoring path applies. Until then checkpoint choice is blind.
2. **`dataset: TUEP` is the worst starting corpus.** 87 train patients, 5720
   epilepsy vs 1148 no_epilepsy windows, one patient holds 1233 windows, labels
   are patient-level so most windows carry no visible signal. TUAB is the sane
   first target: 511 patients, 1436/1364, max 36 windows/patient.
   *Sampling side fixed 2026-09-16* by `train/scripts/sample-train-split.py`: TUEP is
   now 642/428 from 87 patients, max 32/patient. The patient-level-label
   problem remains and cannot be sampled away.
3. **Loss is dominated by teacher prose; labels come first in the JSON.** A
   handful of boolean tokens vs ~150-250 rationale tokens, and the model commits
   to the label before the rationale, so the rationale cannot act as reasoning.
   Options: rationale-first field order for the FT schema (grammar-constrained
   decoding enforces property order) and per-token label upweighting. No
   label-only ablation: rationale vs no rationale was settled in the previous
   paper and is not this project's question. *Rationale-first done
   2026-09-16* (methodology-decisions.md); upweighting waits for a number.
4. **No rebalancing.** TUSZ train is 19,581 pure-bckg of 21,207; absz 22,
   mysz 8, spsz 8; TUSL bckg 7; TUAR shiv 4. Uniform sampling means rare
   classes get almost no gradient. Cap bckg per recording / windows per patient
   (TUSZ max 1274, TUEP 1233), or oversample event windows.
   *Done 2026-09-16* via `pipeline.db` scope, see methodology-decisions.md
   "Train-set sampling". TUSZ is now 3,556 windows with every event window
   kept. The 8-window classes stay unlearnable; that is a data limit.
5. **Val split ignores size and class.** 5% of patients by hash; on TUEP that is
   5 patients and could be one 1233-window patient. Val loss is token-level
   prose loss and does not track classification. Stratify, cap, and compute a
   parsed-JSON macro-F1 at eval steps (`load_best_model_at_end` on it).
   *Done 2026-09-16*, unverified on a GPU.
6. **Teacher rationales are unfiltered.** No filter for hedging against the
   label, 512-token truncation (mid-sentence endings), meta commentary, or
   exact duplicates from temperature-0 decoding. Needs a cluster-side pass.
   *Filter in the train sampler 2026-09-16*; hedge patterns written blind,
   check with `--show hedge` on the cluster.
7. **bckg + seizure co-labels in TUSZ train** (bckg+fnsz 201, bckg+cpsz 87,
   bckg+gnsz 66). The teacher is told to justify both. *Decided 2026-09-21:
   kept* (median 92% seizure coverage in those windows, both labels true of
   the picture); bckg is excluded from the TUSZ recording-level mean instead.
   See methodology-decisions.md.
8. **LoRA `target_modules` by bare name also hit the vision tower's MLP**
   (`Qwen2_5_VLMLP` has `gate_proj/up_proj/down_proj`, confirmed in
   transformers source) but not vision attention (`qkv`, `proj`) or the patch
   merger. Vision is adapted halfway by accident. Make it deliberate either way.
   *Done 2026-09-16*: regex targets, vision on by default at half LR.
9. **NF4 quantizes the vision tower too.** bitsandbytes quantizes every Linear
   not skipped. Prefer bf16 LoRA (fits an A100-40GB at batch 1, ~3.3k tokens,
   grad checkpointing) or pass `llm_int8_skip_modules=["visual"]`.
   *Done 2026-09-16*: bf16 default.
10. Minor: no warmup/cosine, 3 epochs over highly redundant windows, collator
    runs the image processor twice per example, prefix masking assumes right
    padding (true today: Qwen tokenizer config sets no `padding_side`), no
    Slurm wrapper, `hf-install.py` deleted, torch/transformers/peft absent from
    `requirements.txt`. Small multi-label sets (TUAR 574, TUEV 438, TUSL 141
    labeled train windows) argue for pooling all six with a global patient split.
    *All done 2026-09-16 except pooling itself; the cross-dataset overlap is
    measured (117 patients) and screened by the sampler.*
11. Unverified: whether Ollama keeps a 1536x1536 image native for qwen2.5vl. HF
    does (`max_pixels` 12,845,056, ~3025 visual tokens). If it downscales, an
    exported GGUF would see a different resolution than training; another
    reason to evaluate the checkpoint HF-side. Consequence (2026-09-21): the
    zero-shot `qwen2.5vl:7b` row (Ollama, GGUF quantisation, its own resize) is
    not a like-for-like "before" for the HF bf16 fine-tune. Run the HF base
    model with no adapter through `train/scripts/eval.py` as the baseline before
    claiming an improvement.

### Fine-tune data audit (2026-09-21)
Measured locally from `labels.csv`, the samplers' dry runs and the PNGs. Numbers
in FINETUNE-TODO items 14-18. Headlines:

- Duplicate images across corpora and splits (section above).
- **Every model is to be fine-tuned, pooled** (maintainer, 2026-09-21; pooling superseded by per-dataset runs 2026-09-30, methodology-decisions.md).
  Trainer made generic (29 of 33 bases); llama4 runs NF4 on a 4-GPU node;
  MiniCPM-V 2.6 / 4.5 and DeepSeek-OCR go through their own stacks from
  `helpers/export-sft.py`; moondream2 has no local LoRA path (its repo pulls
  fine-tuned "variants" from Moondream's cloud). FINETUNE-TODO items 18, 21.
- **The recording-level metric has structural ceilings.** Sampled test, truth
  per recording: TUSZ `bckg` is true in 2,418 of 2,443 recordings, so its
  balanced accuracy is ~0.5 whatever the model does and the 4-class TUSZ mean
  tops out near 0.875. TUAR `elec`/`musc`/`eyem` are true in 87/91/83 of 94
  recordings, so their specificity rests on 3 to 11 recordings. Multi-label
  recording truth and prediction are unions over windows, so one false-positive
  window flags a recording; a fine-tune trained on 44% event windows must be
  conservative per window to gain recording-level BA.
- **Val sets were too small to pick checkpoints on** for four corpora: TUEP 17
  windows / 4 patients, TUSL 10, TUEV 21, TUAR 25 (TUAB 140, TUSZ 124 are fine).
  Fixed 2026-09-21: `train.val.min-patients` 3 -> 10 (TUEP 84 windows / 10
  patients, TUAR 31, TUEV 17 (numeric-dir patients are tiny), TUSL 23 of 49),
  and the val metric is now the scoreboard's recording-level balanced accuracy.
- **Too few steps** at batch 1 x accumulation 8 x 2 epochs: TUSL 9, TUAR 95,
  TUEV 103 optimizer steps. Those three only make sense pooled (item 9).
- Images are 1536x1536 RGBA with alpha all 255, and every `labels.csv` row has
  its PNG on disk (0 missing, 0 extra, all six corpora). HF's default
  `max_pixels` keeps them native (~3,025 visual tokens).
- **Calibration windows**: window 0 of some recordings is a calibration pulse
  in every channel. A periodicity check over all in-scope window 0/1 images
  found 27 (TUSZ 23, TUEP 3, TUAR 1), all test-side, all negative-labelled,
  none in the fine-tune sample; `datasets/calibration-windows.csv`. Open
  decision in FINETUNE-TODO item 19.
- Rationale text is cluster-only and was not inspected; hedge rate, truncation
  rate, duplicate rate and fabricated evidence on TUEP/TUAB negative windows
  are still unmeasured.


## Things that ARE handled (don't re-fix)

- **The repo's `datasets/` folder shadowed the Hugging Face `datasets` library**. Found 2026-10-03 and fixed
  the same day in `3d1a441`.
  - Jobs 4342993 and 4400514 waited ~2 days in the queue, then died 12 min in with `AttributeError: module
    'datasets' has no attribute 'Dataset'` in `Trainer._get_dataloader`.
  - Cause: train.py puts the repo root first on sys.path. transformers 5.14's `is_datasets_available()` is a
    bare `find_spec`, so it saw the folder as an installed package (HF `datasets` is not installed). Round 1
    never hit this: its trainer imported transformers before adding the root to sys.path.
  - Fix: `sys.modules.setdefault("datasets", None)` right after the sys.path insert.
  - Scoring (`train/scripts/eval.py`) never reaches the Trainer paths, so it needs no guard.
  - The preflight only builds processors, so it could not catch this. That is why `rehearsal/rehearse.py`
    exists.
- **`bitsandbytes` is still installed in `.venv`** (left from the removed 4-bit path). PEFT imports it while
  adding LoRA layers.
  - On a GPU node it loads its CUDA library and works.
  - On the login nodes (AMD EPYC 7532) its CPU library dies with SIGILL (exit 132). Only the rehearsal's cpu
    tier runs there, so the rehearsal hides the module. Uninstalling it would also work.

- **The per-corpus `sft_*.jsonl` on the cluster leaked test patients** —
  found 2026-09-30, gated the same day. The files in `datasets/<DS>/` were
  built 2026-09-16, before the sampler screened test patients and
  test-identical images. `helpers/check-splits.py` on them gave test-patient /
  test-image overlaps of TUAB 44/62, TUAR 26/41, TUEP 12/149, TUEV 9/11,
  TUSL 16/60, TUSZ 50/437, plus train/val patient overlap on TUEV `001` and
  TUSZ `aaaaalqk`. No run trained on them: the first round used
  `datasets/pooled/` (built 09-22, 0 overlaps on every check). Now
  `helpers/build-sft-jsonl.py` runs the check on the in-memory split and
  writes nothing if any dataset overlaps (it replaced a train/val assert that
  could never fail), and `train/train.py` skips a pair whose jsonl fails the
  check. The old files must be rebuilt before the first per-dataset run.

- **`to_labels()` crashed on every TUAB/TUEP eval row since 2026-09-13** —
  fixed 2026-09-16. The config refactor (`de1c3b4`) trimmed `BINARY` from four
  fields to three (the prompt text moved to `config.yml`) but `to_labels` kept
  unpacking four, so any binary-dataset eval or retry submitted after that
  date raised `not enough values to unpack`. The sweep's existing binary rows
  predate it. Multi-label datasets were unaffected.
- A model added to `config.yml` after the samplers ran getting `sampled=0`
  rows, and therefore scope=none, for every path — fixed (`sync()` propagates
  a path's flag to new rows; see scope.md).
- Fine-tuned checkpoints having no scoring path — fixed (`train/run-eval.py`
  + `train/scripts/eval.py` HF-side, `eval/score.py` for the numbers; see
  tooling.md and operations.md 4e).

- Train windows flooding the fine-tune with one patient or with background —
  handled (`train/scripts/sample-train-split.py` flags a capped, balanced subset as
  `sampled=1`; scope derivation is split-aware; the test sampler resets only
  `split='test'`). `build-sft-jsonl.py` reads scope from `pipeline.db` and
  refuses to run if the sampler has not. Sampling logic is not in the trainer
  or the JSONL builder on purpose.

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
- Rationale references and eval rationales covering disjoint splits — fixed
  (`generate-rationales.py --split test` → `rationales-test.csv`; the train file
  is untouched and `run-judge.py check` verifies the join before any GPU time).
- **Image-deterministic parse failures never healing on resubmit** — fixed
  2026-09-21. `eval-retries` / `judge-retries` only retried transient errors,
  and at temperature 0 an invalid-JSON reply repeats exactly (gemma4:12b 19
  and minicpm-v4.6:1b 4 TUSZ windows, 21 and 11 attempts). `eval.py` and
  `judge.py` now make one sampled attempt (temperature 0.3) after the greedy
  ones fail to parse, like `train/scripts/eval.py` and the rationale generator; the
  judge's `reason` field defaults to "" (its two open failures were replies
  without it). If a window still fails after that, it stays pending and
  `score.py` reports the coverage; do not chase it further.
- Judge settings vanishing on the first eval retry — fixed. They now live under
  `config.yml`'s `judge:` key; the config-rewriting retry step whose renderer
  dropped unrendered keys has since been retired.
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
