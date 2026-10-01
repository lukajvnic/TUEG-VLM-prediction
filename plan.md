# Plan to CVPR (2026-09-30)

Deadline early November. The zero-shot benchmark (34 models, six datasets) is done and judged.
The fine-tune track ran once on 24 bases and collapsed: loss fell 2.0 -> 0.25, but every model
answers one class on the test set (bakllava TUAB "normal" 100%, gemma3:27b TUEP "epilepsy" 98%,
TUSZ "bckg" 100%, ...). The target JSON is 95% rationale tokens, so the model learned gemma's prose
and emits the class prior; the image was never needed to lower the loss. Details:
knowledge/known-issues.md, 2026-09-29 entry.

A full retrain of 29 bases is about a week of cluster time. The plan below fits in less.

## Changes, easiest and highest impact first

1. **Label-weighted loss. Dropped (project lead, 2026-09-30)** as overengineering; the loss stays
   the default cross-entropy over the whole target. See knowledge/methodology-decisions.md.
1b. **One run per (base, dataset), no pooling (project lead, 2026-09-30).** Each base is
   fine-tuned on each corpus separately with `python train/train.py <base> <DS>`. Training logs
   to wandb (offline on the nodes, `wandb sync` from a login node).
2. **Checkpoint selection.** Done 2026-09-29, free. Teacher-forced label accuracy over the whole
   val set (`eval_label_balanced_accuracy`) picks the checkpoint instead of a 64-window
   generation metric that returned 0 / 0.5 / 1 per corpus.
3. **Retrain only the bases that can see the plot.** Zero code; cuts the retrain from a week to
   ~2 days. The LLaVA-1.5 family (llava:7b, llava:13b, bakllava, llava-llama3, llava-phi3)
   resizes every image to 336 px, where a 22-channel trace is unreadable; they cannot improve.
   Keep their first-round results as the "insufficient resolution" row of the table. The ~20
   native-resolution bases (Qwen2.5-VL, Qwen3-VL, Gemma 3/4, MedGemma, GLM-OCR, MiniCPM-V 4.6,
   Mistral Small, granite) take 7-17 h each on one GPU and run in parallel.
4. **Grounded rationales. Not for this deadline.** gemma was given the label and asked to
   justify it; its rationales may be stories rather than observations, in which case no loss
   fixes the target. Regenerating the teacher data is a week before training starts. It goes in
   limitations.

## Schedule

- Day 0: train qwen2.5vl:7b on TUAB alone, label-only first (`train.target: labels`, maintainer
  2026-09-30: booleans-only JSON, rationales back only if this learns):
  `python train/train.py qwen2.5vl:7b TUAB`.
- Day 1: `python helpers/inspect-sft.py qwen2.5vl:7b` and the wandb curve. Gate: the fine-tune's
  answers on val/test are no longer one class and `eval_label_balanced_accuracy` moved. One pair
  before spending anything on the rest.
- If it passes: train the native-resolution bases per dataset (`python train/train.py all <DS>`), then
  `train/run-eval.py` (~2 days). Results by about Oct 7. Judges and `eval/score.py` as before.
- If it fails: the paper is the zero-shot benchmark plus "rationale SFT collapses to the class
  prior", with the first-round adapters and their answer distributions as the evidence. That is
  a finding for a capability probe, and it is already in hand.

Retraining a pair within a round means removing `checkpoints/<key>/<DS>-<experiment>/` first (the manifest is what
`train/train.py` reads as "finished"; a dir with checkpoints and no manifest resumes). The
first round is archived in `archive/round1-pooled/` (knowledge/experiments.md). A new round gets a
new `train.experiment` name instead of deleting anything.

## Not doing

- Hyperparameter sweeps until the first per-dataset run is read; tuning LR or rank while the objective is the
  prose optimises the wrong thing.
- llama4 (218 GB bf16, fused MoE experts cannot be quantised to fit a node), moondream (no local
  fine-tune path), gemma4:12b unless the venv's transformers stays >= 5.10.
