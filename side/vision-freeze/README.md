# vision-freeze (side test, outside the main pipeline)

Why: frozen vision features of qwen3-vl:2b / qwen3-vl:4b / qwen2.5vl:7b already separate TUSZ seizure from
background windows at AUROC ~0.70 (linear probe, 2026-10-06), while every LoRA fine-tune reached 0.52-0.56. The
features carry the information; the fine-tune never learns to use it. NeuroCanvas (arXiv 2602.04769) got TUSZ
seizure detection to work with the vision encoder and merger trained and the language model frozen.

One question, the multilabel pilot's balanced "any seizure" lines (1,078 true / 1,078 false train, 52/52 val), on
qwen3-vl:2b-instruct, 2 epochs, 5 evals, best checkpoint by the trainer's label balanced accuracy:

| arm | what trains |
|---|---|
| `freeze` | `model.visual` (vision blocks, merger, deepstack mergers: 407M parameters) fully, fp32 master weights, lr 1e-5; language model frozen |
| `lora` | the labels round's LoRA recipe (`config.yml` `train:`), the control |
| `base` | nothing: the untrained model asked the same question |

Every arm scores P(true) at the answer's boolean token (teacher-forced through the training collator) on all
8,460 TUSZ test windows. `results/summary-<arm>.csv`: window AUROC with a patient-bootstrap 95% CI, AUROC within
recordings that have a seizure (TUSZ's scored background windows are mostly first/last windows of seizure-free
recordings), AUROC on the probe's 1,500 test windows (comparable with its 0.70), and balanced accuracy at 0.5.

```bash
python side/multilabel-pilot/build.py         # the training lines, if side/multilabel-pilot/data/ is missing
python side/vision-freeze/run.py smoke        # <=2 h job: every arm, 2 steps, 24 windows, under smoke/
python side/vision-freeze/run.py freeze       # train + score, one MIG job per arm
python side/vision-freeze/run.py lora
python side/vision-freeze/run.py base
```

A resubmitted job skips training once `checkpoints/<arm>/manifest.json` exists and resumes scoring.

## Round 2 (2026-10-07): longer training and a second rendering

Round 1's `freeze` reached window AUROC 0.690 (0.712 on the probe's windows, the frozen-feature probe's level), with
val accuracy still rising at the last eval. Round 2, all with the `freeze` recipe and 4 epochs:

| arm | images | training lines |
|---|---|---|
| `freeze-4ep` | shipped PNGs | the pilot's 1,078 / 1,078 |
| `freeze-4ep-render` | `rerender.py`'s images of the same windows | the pilot's 1,078 / 1,078 |
| `freeze-4ep-render-more` | `rerender.py`'s images | every seizure window of the training recordings, 4,664 / 4,664 |

`rerender.py` draws the same windows again, into `images/` (gitignored, ~3.9 GB, never mixed with the shipped
PNGs): the 18-row bipolar longitudinal montage (EEG electrodes only), one gain per recording shared by every row,
and soft compression (H*tanh(x/H)) instead of clipping. `rerender.py lists` writes `data/windows.csv` and
`data/more-train.jsonl`, and stops unless every shipped train label reproduces from the annotations (0 mismatches,
2026-10-07). Extra windows come only from recordings already in the fine-tune's train split. The images are
rendered locally (the EDFs are not on the cluster) and copied with rsync; `run.py` refuses to submit an arm if any
image it reads is missing.

```bash
python side/vision-freeze/rerender.py lists && python side/vision-freeze/rerender.py render 14   # local
rsync -a side/vision-freeze/images/ narval:/scratch/luka/TUEG-VLM-prediction/side/vision-freeze/images/
python side/vision-freeze/run.py freeze-4ep          # likewise freeze-4ep-render, freeze-4ep-render-more
```

## Background test (2026-10-14): realistic test background

97.8% of the scored TUSZ test background windows are the first or last window of their recording, so the AUROCs
above are mostly measured against easy background. `rerender.py background` picks up to 2 never-scored, non-edge,
background-only windows per test recording (one from each half of the candidates, seed 0), checks their labels
against the annotations, writes `data/background.csv` (4,471 windows, 2,262 recordings, 558 in seizure recordings)
and renders them. `run.py background-test ARM...` submits one job that scores them with each arm's own scorer into
`results/background-<arm>.csv`, then writes `results/summary-background-<arm>.csv`: AUROC and balanced accuracy of
the arm's seizure windows against the old and the interior background, and specificity by background type
(seizure-free recordings, 1-2 / 3-10 / >10 windows from a seizure).

`VF_RENDER_ROOT` overrides where rerendered images are read from. $SCRATCH took ~2-3.5 s to create one small file
on 2026-10-14, so the background images went to /project instead:

```bash
python side/vision-freeze/rerender.py background 14                       # local
rsync -a --files-from=<(tail -n +2 side/vision-freeze/data/background.csv | cut -d, -f1) \
  side/vision-freeze/images/ narval:/project/def-milad777/luka/vision-freeze-images/
rsync -a side/vision-freeze/data/background.csv narval:/scratch/luka/TUEG-VLM-prediction/side/vision-freeze/data/
VF_RENDER_ROOT=/project/def-milad777/luka/vision-freeze-images \
  python side/vision-freeze/run.py background-test r4-af-strip-more freeze-4ep-render-more freeze-4ep base-render
```

## Round 6 (2026-10-14): does the recipe transfer?

`freeze-4ep-render-more`'s recipe (vision trained, language frozen, answer only) on one yes/no question per other
corpus, and on qwen3-vl:4b. `rerender.py datasets` takes the fine-tune's vetted train/val windows
(`sft_labels_*.jsonl`, which carry the patient caps and the cross-corpus image screen) and the benchmark's scored test
windows, checks their labels against the source annotations (TUAR's `bckg` is left out of the check: the background
recovery set it on 517 of these windows), writes `data/<DS>-{train,val,test}.jsonl` and renders into `images/<DS>/`.

| arm | question (true when) | train / val / test windows (true) | epochs |
|---|---|---|---|
| `r6-tuab-abnormal` | abnormal | 2,283 (1,087) / 140 (52) / 1,200 (564) | 4 |
| `r6-tuep-epilepsy` | epilepsy | 834 (515) / 83 (35) / 1,084 (638) | 10 |
| `r6-tuar-eyem` | eyem | 371 (205) / 31 (15) / 3,540 (2,336) | 12 |
| `r6-tuev-epileptiform` | spsw, gped or pled | 366 (181) / 17 (6) / 354 (151) | 12 |
| `r6-4b-freeze-4ep-render` | TUSZ any seizure, qwen3-vl:4b, full A100 | the pilot lines | 4 |

More epochs on the small sets bring them toward TUSZ's ~1,000 steps (at most 12). TUEV's test "patients" are excerpt
ids, so its patient CI is optimistic.

```bash
python side/vision-freeze/rerender.py datasets 12      # local: lines, label check, images/<DS>/
VF_RENDER_ROOT=/project/def-milad777/luka/vision-freeze-images python side/vision-freeze/run.py r6-tuab-abnormal
```
