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
- **Why this round:** see methodology-decisions.md: "The first fine-tune answers labels only", "One run per (base, dataset)" and "No token weighting".

## rationale-w10 (side run, outside the main pipeline)
- **When:** queued 2026-10-01, alongside the `labels` round.
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
