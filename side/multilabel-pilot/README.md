# multilabel pilot (side run, outside the main pipeline)

Why: every multilabel fine-tune of the `labels` round answered what the label frequencies predict (TUSZ "bckg only"
on ~100% of test windows), and per-class P(true) showed no hidden signal on TUSZ (knowledge/experiments.md). This
pilot tests two training formats on one base and one dataset, qwen3-vl:2b-instruct on TUSZ, before any rollout.

| variant | training lines (`build.py`) | scored with |
|---|---|---|
| `perclass` | one yes/no question per (window, class) plus "any seizure", each question 50/50 true/false | the same questions; P(true) at the answer token, present at >= 0.5 |
| `joint-aux` | the labels round's joint JSON lines, seizure vs seizure-free cut to 50/50, plus the perclass lines | the original joint prompt, constrained greedy decoding: what the labels round and the zero-shot rows ran |

Both: same windows and patient split as `datasets/TUSZ/sft_labels_{train,val}.jsonl`, one epoch, 5 evals, best
checkpoint by the trainer's label balanced accuracy on the perclass val lines (balanced per question, so a
frequency-only answer scores 0.5). Everything else is `config.yml`'s `train:` block via `train/train.py`.

```bash
python side/multilabel-pilot/build.py                 # data/ (deterministic, seed 0)
python side/multilabel-pilot/train.py smoke           # <=1 h job: both variants, 2 steps, 24 test windows, under smoke/
python side/multilabel-pilot/train.py perclass        # train + score in one job
python side/multilabel-pilot/train.py joint-aux
```

A resubmitted job skips training once `checkpoints/<variant>/manifest.json` exists and resumes scoring. Outputs:
`checkpoints/<variant>/`, `logs/` (Slurm + wandb offline, group `multilabel-pilot`), `results/eval-<variant>.csv`,
`summary-<variant>.csv`, `classes-<variant>.csv`, and `auroc-perclass.csv`. Compare with the labels round's row for
`qwen3-vl:2b-instruct-sft-TUSZ-labels` and the zero-shot `qwen3-vl:2b-instruct` row in `datasets/TUSZ/summary.csv`.
