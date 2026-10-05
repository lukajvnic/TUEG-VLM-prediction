# Fine-tune experiments

One entry per round of fine-tunes. `config.yml` `train.experiment` names the round currently being trained.
Every run of a round is named `<DS>-<experiment>`:
- checkpoints in `checkpoints/<key>/<DS>-<experiment>/`;
- Slurm job `eeg-vlm-train-<key>-<DS>-<experiment>`;
- wandb run `<key>-<DS>-<experiment>` in group `<experiment>`, tagged with the dataset and target;
- scored model `<key>-sft-<DS>-<experiment>`, with predictions in `datasets/<DS>/eval-<key>-sft-<DS>-<experiment>.csv`;
- results table from `python helpers/experiment-results.py <experiment>`, written to `experiments/<experiment>.csv`.

A new round gets a new name and never reuses one, so no round overwrites another. A finished round can be
archived under `archive/<name>/` with a README, as round 1 was. Each run's `manifest.json` records its
experiment, target, settings and commit.

## round1-pooled (archived)
- **When:** trained 2026-09-23 to 09-28; archived 2026-09-30.
- **Setup:**
  - one run per base over all six corpora pooled (6,563 train / 330 val windows);
  - target = gemma3:12b's label-conditioned rationale, then the booleans;
  - checkpoint chosen by a 64-window generation metric.
- **Code:** commits 0c93d76, 7c1b24a and e78e694 (from the manifests). 27 runs finished.
- **Where:** Narval `archive/round1-pooled/`, 46 GB:
  - adapters, trainer states, test predictions, the pooled jsonl, job logs, and snapshots of the summary tables and charts;
  - `README.md`, `results.csv` and `MANIFEST.txt`, which are also in git;
  - loss curves in `report/figures/loss-curves.png`.
- **Outcome:**
  - Over the 91 (base, dataset) rows, the fine-tuned recording balanced accuracy has a median of 0.422 and a median change of -0.078 versus zero-shot. Only 10 rows improve by more than 0.05.
  - Answers lean to one class: a median 75% of answers are the top answer, and 26 rows are at least 95% one answer.
  - Scoring never finished: median test coverage is 17%, and only 10 rows are fully scored.
  - Why: the loss was ~95% rationale prose (known-issues 2026-09-29).
- **Scratch purge:** Narval's scratch deletes old files. Copy the archive to project space if the adapters must survive.

## labels (running)
- **When:** started 2026-09-30.
- **Setup:**
  - one run per (base, dataset);
  - target `labels`: booleans-only JSON, prompt without the rationale instruction;
  - same windows and patient-disjoint split as the rationale data (`sft_labels_{train,val}.jsonl`);
  - checkpoint chosen by teacher-forced label balanced accuracy.
- **Code:** commit 0796505, with the experiment naming added after it.
- **First run:** qwen2.5vl:7b on TUAB, job 4342993 (12 h walltime). It is the lead's gate before any batch.
  - It waited ~2 days, then failed 12 min in, on 2026-10-03: the repo's `datasets/` folder shadowed the HF library
    (known-issues).
  - Resubmitted 2026-10-03 as job 4565667, which trains, then scores the TUAB test split in one allocation.
    GPU rehearsal 4565661 (2 h) checks it first; side run 4565671.
  - **Result (2026-10-04).** Job 4565667 took 5 h 15 m: 572 steps at 26.7 s/step, then 1,200 test windows scored at
    1.0 s/window with 0 failures. Measured with `eval/score.py TUAB` (bootstrap CI) on the TUAB test split, 300
    recordings and 222 patients at full coverage:

    | Model | Recording balanced accuracy | Recording macro-F1 [95% CI] | Window BA | Top answer |
    |---|---|---|---|---|
    | label-only fine-tune | **0.786** | 0.787 [0.739, 0.832] (clears the 0.346 floor) | 0.775 | normal 55% |
    | zero-shot qwen2.5vl:7b | 0.500 | 0.320 | | abnormal 100% (degenerate) |
    | round 1 pooled-rationale fine-tune | 0.48 | (33% coverage) | | abnormal 97% |

    Val label BA rose at every eval: 0.701, 0.729, 0.767, 0.789, 0.832 and 0.843 (best, step 572).
  - The first evidence that a VLM fine-tuned on these plots learns the task, not the class prior. One base on one
    dataset (the easiest binary one), so it does not generalise yet.
- **Full batch (submitted 2026-10-04).** The other 167 (base, dataset) pairs, one job each, train then score in
  one allocation.
  - Sized per pair by `helpers/plan-resources.py` into `train/resources.csv` (commit 50e6020): 113 jobs on 1
    GPU, 24 on 2, 30 on 3. Nice 0 for TUAB/TUEP, 100 for TUAR/TUEV/TUSL, 200 for TUSZ, 300 for the
    LLaVA-1.5 family.
  - The MIG memory probe (job 4653666) found the eval cache crash before any of them started. 120 of the
    167 would have died at their first eval. Fixed in `97d815f` and pulled on the cluster at 15:42, while all
    167 were still pending (known-issues). Re-probe of the crashed bases with the fix: job 4654374.
  - Probe peaks on an `a100_3g.20gb` slice (one step + one eval, one TUSZ window), with the fix: glm-ocr
    7.3 GB, minicpm-v4.6:1b 7.5, llava-phi3:3.8b 9.7, gemma3:4b 11.6, medgemma:4b 11.6, medgemma1.5:4b 11.6,
    qwen3-vl:2b 12.0, gemma4:e2b 12.9, granite3.2-vision:2b 15.3. qwen3-vl:4b peaked at 17.4 GB, over the 17 GB
    cut, so it stays on a full A100.
  - Those 9 bases' 54 jobs were cancelled while pending and resubmitted on MIG slices the same afternoon
    (operations.md, "Moving a pending fine-tune to a MIG slice"). They started within minutes. gemma3:4b
    trained at 8.6 s/step on the slice (the probe's single step said 10.6).
  - First finished pairs, 2026-10-04 evening, both TUEP, both scored 1084/1084 test windows:
    - gemma3:4b: val label BA 0.570 at step 100 (kept), then 0.507 and 0.492.
    - gemma4:e2b: val label BA 0.500 at every eval. It answers "epilepsy" for all 1084 test windows. Its train
      loss sat at 0.060-0.069 per answer token from step 30 on, which is ln 2 spread over the ~10 answer
      tokens: a coin flip on the one token that matters. Its TUAB run shows the same flat 0.069. Its training
      batch is well formed (256 image tokens, pixel values, image positions, all accepted by the forward).
      Job 4660614 (2026-10-04) fed the base model one TUAB prompt with three real windows and a blank image:
      the vision features are finite and the P(true)/P(false) ratio moves with the image (0.077, 0.035,
      0.032, and 0.000 for the blank). gemma3:4b, as a control, swings from 0.99 true to 0.98 false on the
      blank. So the image reaches gemma4:e2b; its fine-tune just never learned to use it. Not a pipeline bug.
      The image-token explanation was weakened on 2026-10-05: Gemma3/MedGemma see the same 256 tokens and
      do learn (TUAB 0.62-0.76). Every Gemma4 TUAB run trains at a loss of exactly ln 2 / 10 with gradient
      norms of 2e-5 to 7e-5 after ~step 50, against at least 3.8e-2 for the other 18 TUAB runs (trainer_state.json):
      the gradient dies. Gemma4's `final_logit_softcapping: 30.0` saturating is a suspect, unverified.
- **Multilabel collapsed; the rest of the multilabel batch was cancelled (2026-10-05).**
  - All 40 finished TUAR/TUEV/TUSL/TUSZ runs (10 MIG bases x 4) answer what the train label frequencies
    predict. Examples: TUSZ "bckg only" on ~100% of test windows, including all 2,018 fnsz windows; TUSL
    "seiz" on every window; TUEV "gped" on 64-100%. TUAR is mixed, with weak signal (fine-tuned BA 0.56-0.65
    for 6 of 10 bases). Counted from the eval CSVs; per-pair numbers are in `experiments/labels.csv` (run
    2026-10-05).
  - The checkpoint metric could not see it: a label-frequency predictor that sees the gold earlier fields
    scores 0.91 on TUSZ and TUSL. TUAR, TUEV and TUSL ran 8-94 steps with `eval-steps: 100`, so each got one
    eval.
  - TUSZ "any seizure" vs "bckg only" is 41/59 in train, about as balanced as TUAB, and still collapsed. So
    rare classes alone don't explain it; the fixed field order (bckg second) and greedy per-field decoding
    are suspects.
  - The 72 queued multilabel jobs (larger bases, 1-3 GPUs each, ~1,000 GPU-h estimated) were cancelled at the
    maintainer's request; list in `$SCRATCH/rehearsal/cancel-multilabel-20261005.txt`. Their 72 empty
    checkpoint dirs were removed (`rmdir`, empty dirs only) so `finetune_models()` stops deriving them. Their
    `pipeline.db` rows from the submit-time sync were left in place.
  - The 14 binary (TUAB/TUEP) jobs still queued were kept.
  - Signal check (2026-10-05, jobs 4709024/4709436/4709437 on MIG): P(true) at each field's boolean position,
    read during the same constrained greedy decode, for qwen3-vl:2b, gemma3:4b and granite3.2-vision:2b.
    TUSZ, 300 stratified test windows each (seed 0), per-class AUROC:
    - bckg 0.52-0.56, fnsz 0.45-0.50, gnsz 0.52-0.60, tcsz 0.44-0.46: no hidden signal. Seizure vs
      background is not ranked, so val-tuned thresholds cannot rescue TUSZ; training has to change.
    - absz 0.87-0.89 on all three, but its 30 positives come from 5 patients: possibly a patient cue.
      cpsz 0.71 (qwen only) and tnsz 0.69 (gemma only) rest on 4-7 patients and don't repeat.
    - TUEV, qwen3-vl:2b on 212 of 354 windows: artf 0.67, bckg 0.68, gped 0.61, pled 0.24. Modest signal,
      not yet confirmed on the full set. TUAR/TUSL not analysed: the SSH session dropped before the jobs
      finished. To finish: `cd $SCRATCH/rehearsal/research/signal && python3 analyze.py` (adds patient
      bootstrap CIs).
  - Next: a per-class format (one balanced yes/no question per class, an "any seizure" question for TUSZ),
    piloted on TUSZ in a side directory first.
- **Why this round:** see methodology-decisions.md: "The first fine-tune answers labels only", "One run per (base, dataset)" and "No token weighting".

## rationale-w10 (side run, outside the main pipeline)
- **When:** queued 2026-10-01 as job 4400514 (12 h), alongside the `labels` round. Pre-submit check on the
  login node: exactly one weighted token per TUAB example (the `false`/`true` after `":`), weight-1 loss equals
  the default CE, and transformers 5.14 skips the grad-accum divide (`model_accepts_loss_kwargs`).
- **Why:** the maintainer's "just to see" test. With the rationale kept in the target, does weighting the loss on
  the JSON's boolean value tokens 10x stop the collapse to the class prior? The lead rejected token weighting for
  the main track (methodology-decisions.md), so this lives only in `side/rationale-w10/`.
- **Setup:**
  - qwen2.5vl:7b on TUAB, rationale-first `sft_{train,val}.jsonl` (same windows and split as `labels`);
  - hyperparameters taken through `train/train.py`'s `init_trainer`;
  - weight 10 on the last true/false target token per field, 1 elsewhere;
  - loss `Σ w·CE / num_items_in_batch`.
  - The logged loss is weighted, so it is not comparable to the other rounds' curves.
- **Where:** `side/rationale-w10/` holds the checkpoints, logs and wandb offline runs, and `results/` (once
  `score.py` runs). wandb group `rationale-w10`. The main tree's `checkpoints/`, `logs/` and `summary.csv` never
  see it.
- **Result (2026-10-04).** Job 4565671 took 9 h 16 m: training at ~27.4 s/step, then scoring at ~14 s/window. It
  kept the final adapter. `results/summary.csv`, TUAB test, 300 recordings:
  - recording balanced accuracy **0.597**, recording macro-F1 0.588 [95% CI 0.529, 0.642] (clears the 0.346
    floor), window BA 0.589;
  - answers are mixed (abnormal 57%), not degenerate;
  - 1,195 of 1,200 windows scored. 5 replies hit the 512-token cap mid-rationale (`Unterminated string`).
- Against the same base and split: label-only 0.786 [0.739, 0.832], zero-shot 0.500 (always "abnormal"), round 1
  (rationale, no weight, pooled) 0.48 at 33% coverage and 97% "abnormal". So the 10x weight stops the collapse to
  the prior, but keeping the rationale in the target still costs about 0.19 BA against label-only.

## multilabel-pilot (side run, outside the main pipeline)
- **When:** built and submitted 2026-10-05, after the `labels` round's multilabel collapse (above).
- **Question:** does a different training format make a VLM read multilabel EEG plots? The maintainer asked for
  two runs on one base and one dataset, to compare:
  - `perclass`: one yes/no question per (window, class) plus "any seizure", each question 50/50 true/false.
    Scored by asking every question and reading P(true) at the answer token (teacher-forced, cutoff 0.5).
  - `joint-aux`: the labels round's joint JSON lines with seizure vs seizure-free cut to 50/50 (1,078 each),
    plus the same per-class lines as extra practice. Scored with the original joint prompt and constrained greedy
    decoding, so it is directly comparable with the zero-shot rows and needs no new baseline.
- **Setup:** qwen3-vl:2b-instruct (MIG slice; learned TUAB at 0.763 in the labels round) on TUSZ, the largest
  multilabel set. Windows and patient split are the labels round's (`sft_labels_*`); one epoch, eval/save at 5
  points, best checkpoint by the trainer's label balanced accuracy on the per-class val lines (balanced per
  question, so a frequency-only answer scores 0.5 there, unlike the labels round's 0.91). Other settings from
  `config.yml` `train:`.
  - perclass: 6,094 train lines (any_seizure 1,078/1,078, bckg 884/884, fnsz 595/595, gnsz 316/316, cpsz
    113/113, tcsz 27/27, absz 18/18, mysz 8/8, spsz 8/8; tnsz has no train positive). joint-aux: 8,250. Val:
    332 per-class lines from the 117 val windows. Counts from `build.py`.
  - Same data built locally and on Narval (md5 identical); every line's answer checked against its window's label.
- **Checks before submitting** (login node, no weights): the collator masks everything but the answer (1 boolean
  per per-class line, 9 per joint line); the scorer's boolean position and `logits_to_keep` work for Qwen3-VL;
  `summarise()` on the labels round's full qwen3-vl:2b TUSZ predictions reproduces its 0.500. Then a smoke job
  (4727011): both variants end to end, 2 steps, 24 test windows.
- **Where:** `side/multilabel-pilot/` (README there). Compare `results/summary-<variant>.csv` with the
  `qwen3-vl:2b-instruct-sft-TUSZ-labels` and zero-shot `qwen3-vl:2b-instruct` rows of `datasets/TUSZ/summary.csv`.
