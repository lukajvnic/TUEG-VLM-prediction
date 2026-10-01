# Side run: rationale target, boolean values weighted 10x

Not the main track. The lead rejected token weighting on 2026-09-30, and the main experiment (`labels`) trains on
the booleans alone. This side run asks: with the rationale kept in the target (rationale first, then the
booleans), does weighting the loss on the JSON's boolean value tokens 10x make the model decide from the plot,
rather than writing gemma's prose and falling back on the class prior?

Same pair as the main run's first job: qwen2.5vl:7b on TUAB. The data is `datasets/TUAB/sft_{train,val}.jsonl`,
with the same windows and patient-disjoint split as `sft_labels_*`. Hyperparameters come from `config.yml`
`train:`, taken through `train/train.py`'s own `init_trainer`. Only two things differ from that trainer:
- the collator adds `label_weights`: 10 on the last true/false token of each target (one per field), 1 on every
  other target token;
- `compute_loss` is `Σ weight·CE / num_items_in_batch`. That is the HF normalisation, so with weight 1 it
  equals the default loss.

The logged loss and eval_loss are weighted, so they are not comparable to the `labels` runs' curves. The
checkpoint is still picked on `eval_label_balanced_accuracy`.

## Commands (repo root, cluster)
```bash
python side/rationale-w10/train.py qwen2.5vl:7b TUAB --dry-run
python side/rationale-w10/train.py qwen2.5vl:7b TUAB          # job eeg-vlm-side-w10-qwen2.5vl-7b-TUAB
python side/rationale-w10/score.py qwen2.5vl:7b TUAB          # after training: test predictions + summary
wandb sync side/rationale-w10/logs/wandb/offline-run-*        # login node, after `wandb login --relogin`
```

## Where things go (nothing here is read by the main pipeline)
- `checkpoints/<key>/<DS>/`: adapter, trainer_state, `manifest.json` (with `label_weight`), `wandb-id`
- `logs/`: Slurm logs, `tasks.csv`, wandb offline runs
- `results/`: `eval-<key>-sft-<DS>-rationale-w10.csv` predictions, `summary.csv`, `failures.csv`
- wandb: project `eeg-vlm-finetune`, group `rationale-w10`, tags `TUAB`, `rationale`, `label-weight-10`
