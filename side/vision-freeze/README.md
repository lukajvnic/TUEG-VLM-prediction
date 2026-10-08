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
