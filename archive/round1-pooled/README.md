# Round 1: pooled rationale fine-tunes (2026-09-23 to 09-28)

Archived 2026-09-30 when the project moved to one fine-tune per (base, dataset) with named experiments
(knowledge/experiments.md). Nothing here is read by the pipeline any more.

- **What it was:** one LoRA run per base over all six corpora pooled (`datasets/pooled/sft_*.jsonl`, 6,563 train /
  330 val windows), target = gemma3:12b's label-conditioned rationale first, then the booleans. 2 epochs, rank 16,
  lr 2e-4. Checkpoint chosen by a generation metric on 64 val windows (noisy: 7 of 27 runs kept step 100).
- **Code:** commit(s) 0c93d76, 7c1b24a, e78e694 (from the manifests). 27 runs finished (manifest.json).
- **Outcome:** train loss 2.0 -> 0.25 in every run, but no real gain over zero-shot (table below;
  knowledge/known-issues.md, 2026-09-29 entry). Over the 91 (base, dataset) rows: fine-tuned recording balanced
  accuracy median 0.422 (min 0, max 0.723), median change vs zero-shot -0.078, better by more than 0.05 in 10.
  Answers lean to one class: top-answer share median 75%, at least 95% in 26 rows.
- **Scoring was never finished:** test coverage median 17%; only 10 rows are scored on at least 99% of the
  windows, so most numbers below are partial (see the coverage column).
- **Loss curves:** report/figures/loss-curves.png (from the trainer_state.json files kept here).

## Layout
- `checkpoints/<key>/pooled/`: adapters, trainer_state.json, manifest.json
- `datasets/<DS>/eval-<key>-sft-pooled.csv`: test predictions; `datasets/pooled/`: the training jsonl
- `datasets/<DS>/summary*.csv`, `chart-*.png`: snapshots of the tables as they were, zero-shot + round 1
- `logs/`: training and scoring job logs; `MANIFEST.txt`: every moved path; `results.csv`: the table below

## Results (recording balanced accuracy, test split)
| base | dataset | zero-shot | fine-tuned | coverage | top answer |
|---|---|---|---|---|---|
| bakllava:7b | TUAB | 0.500 | 0.500 | 1.00 | normal (100%) |
| gemma3:12b | TUAB | 0.553 | 0.512 | 0.21 | abnormal (78%) |
| gemma3:27b | TUAB | 0.500 | 0.635 | 1.00 | abnormal (67%) |
| gemma3:4b | TUAB | 0.500 | 0.000 | 0.04 | abnormal (81%) |
| gemma4:26b | TUAB | 0.503 | 0.611 | 0.17 | normal (62%) |
| gemma4:31b | TUAB | 0.500 | 0.549 | 0.47 | abnormal (68%) |
| gemma4:e2b | TUAB | 0.518 | 0.000 | 0.00 | abnormal (100%) |
| gemma4:e4b | TUAB | 0.496 | 0.505 | 0.25 | abnormal (77%) |
| glm-ocr:latest | TUAB | 0.500 | 0.500 | 1.00 | abnormal (100%) |
| granite3.2-vision:2b | TUAB | 0.492 | 0.000 | 0.02 | abnormal (64%) |
| mistral-small3.1:24b | TUAB | 0.500 | 0.592 | 0.18 | normal (51%) |
| mistral-small3.2:24b | TUAB | 0.478 | 0.499 | 0.22 | abnormal (71%) |
| qwen2.5vl:32b | TUAB | 0.475 | 0.491 | 0.25 | abnormal (94%) |
| qwen2.5vl:7b | TUAB | 0.500 | 0.484 | 0.33 | abnormal (97%) |
| qwen3-vl:30b-a3b-instruct | TUAB | 0.500 | 0.541 | 0.04 | normal (68%) |
| qwen3-vl:32b-instruct | TUAB | 0.496 | 0.000 | 0.01 | normal (56%) |
| bakllava:7b | TUAR | 0.500 | 0.504 | 1.00 | eyem, musc (75%) |
| gemma3:12b | TUAR | 0.502 | 0.000 | 0.01 | eyem, musc (38%) |
| gemma3:27b | TUAR | 0.515 | 0.000 | 0.09 | eyem (45%) |
| gemma3:4b | TUAR | 0.495 | 0.000 | 0.01 | elec, eyem, musc (37%) |
| gemma4:26b | TUAR | 0.581 | 0.000 | 0.04 | elec (73%) |
| gemma4:31b | TUAR | 0.515 | 0.000 | 0.01 | eyem (37%) |
| gemma4:e2b | TUAR | 0.500 | 0.000 | 0.00 | eyem (100%) |
| gemma4:e4b | TUAR | 0.539 | 0.000 | 0.07 | eyem (84%) |
| glm-ocr:latest | TUAR | 0.500 | 0.000 | 0.10 | eyem, musc (84%) |
| granite3.2-vision:2b | TUAR | 0.500 | 0.000 | 0.00 | eyem, musc (83%) |
| mistral-small3.1:24b | TUAR | 0.500 | 0.000 | 0.01 | musc (32%) |
| mistral-small3.2:24b | TUAR | 0.494 | 0.000 | 0.01 | eyem (46%) |
| qwen2.5vl:32b | TUAR | 0.497 | 0.000 | 0.05 | eyem, musc (45%) |
| qwen2.5vl:7b | TUAR | 0.500 | 0.000 | 0.03 | elec (43%) |
| qwen3-vl:30b-a3b-instruct | TUAR | 0.558 | 0.000 | 0.01 | eyem, musc (87%) |
| qwen3-vl:32b-instruct | TUAR | 0.513 | 0.000 | 0.00 | elec (33%) |
| bakllava:7b | TUEP | 0.506 | 0.506 | 0.98 | no_epilepsy (98%) |
| gemma3:12b | TUEP | 0.511 | 0.500 | 0.39 | epilepsy (99%) |
| gemma3:27b | TUEP | 0.499 | 0.501 | 0.97 | epilepsy (98%) |
| gemma3:4b | TUEP | 0.500 | 0.723 | 0.07 | epilepsy (62%) |
| gemma4:26b | TUEP | 0.489 | 0.500 | 0.26 | epilepsy (97%) |
| gemma4:31b | TUEP | 0.534 | 0.526 | 1.00 | epilepsy (52%) |
| gemma4:e2b | TUEP | 0.521 | 0.000 | 0.01 | no_epilepsy (100%) |
| gemma4:e4b | TUEP | 0.499 | 0.500 | 0.64 | epilepsy (100%) |
| glm-ocr:latest | TUEP | 0.500 | 0.512 | 1.00 | no_epilepsy (65%) |
| granite3.2-vision:2b | TUEP | 0.500 | 0.000 | 0.01 | epilepsy (75%) |
| mistral-small3.1:24b | TUEP | 0.482 | 0.462 | 0.09 | no_epilepsy (70%) |
| mistral-small3.2:24b | TUEP | 0.500 | 0.561 | 0.36 | no_epilepsy (70%) |
| qwen2.5vl:32b | TUEP | 0.452 | 0.488 | 0.09 | no_epilepsy (92%) |
| qwen2.5vl:7b | TUEP | 0.500 | 0.564 | 0.67 | no_epilepsy (82%) |
| qwen3-vl:30b-a3b-instruct | TUEP | 0.500 | 0.422 | 0.06 | epilepsy (84%) |
| qwen3-vl:32b-instruct | TUEP | 0.436 | 0.000 | 0.00 | epilepsy (100%) |
| bakllava:7b | TUEV | 0.500 | 0.502 | 0.98 | artf, bckg, eyem, gped, pled (89%) |
| gemma3:12b | TUEV | 0.502 | 0.512 | 0.64 | pled (96%) |
| gemma3:27b | TUEV | 0.524 | 0.501 | 1.00 | pled (99%) |
| gemma3:4b | TUEV | 0.501 | 0.000 | 0.08 | pled (48%) |
| gemma4:26b | TUEV | 0.556 | 0.557 | 0.95 | gped (54%) |
| gemma4:31b | TUEV | 0.529 | 0.500 | 0.17 | pled (78%) |
| gemma4:e2b | TUEV | 0.500 | 0.000 | 0.01 | artf, bckg, eyem (50%) |
| gemma4:e4b | TUEV | 0.515 | 0.542 | 1.00 | gped (43%) |
| glm-ocr:latest | TUEV | 0.500 | 0.562 | 1.00 | pled (50%) |
| granite3.2-vision:2b | TUEV | 0.499 | 0.000 | 0.02 | pled (33%) |
| mistral-small3.1:24b | TUEV | 0.485 | 0.503 | 0.29 | pled (84%) |
| mistral-small3.2:24b | TUEV | 0.494 | 0.000 | 0.18 | pled (73%) |
| qwen2.5vl:32b | TUEV | 0.530 | 0.000 | 0.08 | gped (43%) |
| qwen2.5vl:7b | TUEV | 0.500 | 0.476 | 0.33 | pled (31%) |
| qwen3-vl:30b-a3b-instruct | TUEV | 0.523 | 0.000 | 0.13 | pled (81%) |
| bakllava:7b | TUSL | 0.500 | 0.545 | 1.00 | seiz, slow (90%) |
| gemma3:12b | TUSL | 0.500 | 0.429 | 0.72 | seiz, slow (31%) |
| gemma3:27b | TUSL | 0.500 | 0.500 | 0.98 | bckg, seiz, slow (48%) |
| gemma3:4b | TUSL | 0.500 | 0.000 | 0.16 | bckg, slow (56%) |
| gemma4:26b | TUSL | 0.528 | 0.541 | 0.68 | bckg, seiz, slow (42%) |
| gemma4:31b | TUSL | 0.502 | 0.525 | 0.59 | seiz (86%) |
| gemma4:e2b | TUSL | 0.500 | 0.000 | 0.02 | bckg, seiz, slow (50%) |
| gemma4:e4b | TUSL | 0.559 | 0.000 | 0.41 | seiz, slow (78%) |
| glm-ocr:latest | TUSL | 0.500 | 0.640 | 0.99 | seiz (40%) |
| mistral-small3.1:24b | TUSL | 0.500 | 0.000 | 0.41 | bckg (75%) |
| mistral-small3.2:24b | TUSL | 0.472 | 0.000 | 0.38 | seiz (43%) |
| qwen2.5vl:32b | TUSL | 0.457 | 0.492 | 0.74 | seiz, slow (38%) |
| qwen2.5vl:7b | TUSL | 0.500 | 0.569 | 0.98 | seiz, slow (44%) |
| qwen3-vl:30b-a3b-instruct | TUSL | 0.500 | 0.000 | 0.28 | seiz (33%) |
| bakllava:7b | TUSZ | 0.500 | 0.580 | 0.19 | bckg (99%) |
| gemma3:12b | TUSZ | 0.578 | 0.000 | 0.01 | bckg (99%) |
| gemma3:27b | TUSZ | 0.500 | 0.500 | 0.23 | bckg (100%) |
| gemma3:4b | TUSZ | 0.619 | 0.000 | 0.00 | bckg (100%) |
| gemma4:26b | TUSZ | 0.703 | 0.000 | 0.00 | bckg (100%) |
| gemma4:31b | TUSZ | 0.500 | 0.000 | 0.00 | bckg (100%) |
| gemma4:e2b | TUSZ | 0.500 | 0.000 | 0.00 | bckg (100%) |
| gemma4:e4b | TUSZ | 0.500 | 0.000 | 0.01 | bckg (100%) |
| glm-ocr:latest | TUSZ | 0.500 | 0.500 | 0.72 | bckg (72%) |
| mistral-small3.1:24b | TUSZ | 0.583 | 0.000 | 0.02 | bckg (99%) |
| mistral-small3.2:24b | TUSZ | 0.549 | 0.000 | 0.01 | bckg (96%) |
| qwen2.5vl:32b | TUSZ | 0.500 | 0.000 | 0.03 | bckg (84%) |
| qwen2.5vl:7b | TUSZ | 0.500 | 0.000 | 0.06 | bckg (100%) |
| qwen3-vl:30b-a3b-instruct | TUSZ | 0.532 | 0.000 | 0.02 | bckg (100%) |
