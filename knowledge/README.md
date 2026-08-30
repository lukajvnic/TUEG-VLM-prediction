# TUEG-VLM-prediction — project knowledge base

Documentation of the project's purpose, data pipeline, evaluation, and the
methodology decisions behind them. Written as a durable reference.

## What this project is

A **benchmark of vision-language models (VLMs) on EEG classification**. EEG
recordings from the Temple University EEG corpus (TUEG) are rendered as
**multi-channel waveform plots** (PNG images), and general-purpose VLMs (served
via **Ollama**) are asked to classify them zero-shot. A second track **fine-tunes**
a model on VLM-generated rationales.

It is a probe of *how well general multimodal models transfer to an
out-of-distribution scientific modality (EEG)* — not an attempt to beat
specialized 1D signal models. Keep that framing precise (see
[methodology-decisions.md](methodology-decisions.md)).

## Two goals, one dataset

- **(A) Zero-shot benchmark** — evaluate 36 off-the-shelf VLMs on a stratified
  sample of the **test** split (14,850 of 42,843 windows; scoring-unit class
  support unchanged — see [methodology-decisions.md](methodology-decisions.md)).
- **(B) Fine-tune** — generate ground-truth rationales on the **train** split, then
  LoRA-fine-tune a model (Qwen2.5-VL-7B).

The train/test split is **patient-level** (no patient in both), so B trains and
A evaluates on disjoint patients.

## The six datasets

| Dataset | Task | Type | Label level |
|---|---|---|---|
| TUAB | normal vs abnormal | binary | recording |
| TUEP | epilepsy vs not | binary | patient |
| TUEV | 6 event categories | multi-label | window (time-localized) |
| TUAR | 7 artifact categories | multi-label | window (time-localized) |
| TUSZ | 9 seizure/bckg categories | multi-label | window (time-localized) |
| TUSL | bckg/seiz/slow | multi-label | window (time-localized) |

See [datasets.md](datasets.md).

## Pipeline at a glance

```
EDF recordings  ──generate.py──▶  windowed waveform PNGs + labels.csv
                                   (train/ 70% + test/ 30%, patient-split)
                                          │
                    ┌─────────────────────┴─────────────────────┐
             train split                                    test split
                    │                                            │
  eval/scripts/generate-rationales.py                    eval/run-eval.py
      --split train                                 (35 VLMs classify, array job)
                    │                                            │
              LoRA fine-tune                            eval/summarize.py
                                                     (per-class + recording metrics)
                                                                 │
                                    eval/scripts/generate-rationales.py --split test
                                                (reference rationales, test split)
                                                                 │
                                                        eval/run-judge.py
                                          (does the model's rationale match? per model)
```

## Cluster context

Runs on **Digital Research Alliance of Canada** clusters (primarily **Narval**,
`narval.alliancecan.ca`, account `def-milad777`). The project was prompted in part
by a **resource-waste warning** — many efficiency choices (MIG GPU slices, request
batching, right-sized context/walltime) trace to that. See
[operations.md](operations.md).

## Files in this knowledge base

- [datasets.md](datasets.md) — the six TUEG datasets, sources, labels, quirks, rarity
- [data-generation.md](data-generation.md) — rendering, windowing, splitting, filenames
- [eval-pipeline.md](eval-pipeline.md) — config, dispatch, inference, scoring,
  and the rationale-agreement stage
- [rationale-generation.md](rationale-generation.md) — the fine-tune data track,
  and the reference rationales the agreement stage judges against
- [methodology-decisions.md](methodology-decisions.md) — why things are the way they are
- [operations.md](operations.md) — end-to-end commands and cluster workflow
- [tooling.md](tooling.md) — every script, what it does, and what is cruft
- [known-issues.md](known-issues.md) — limitations and open decisions

**This directory is gitignored** — it stays on the local machine and does not
reach the cluster. Keep it current anyway; it is the project's memory. See
`CLAUDE.md` at the repo root.
