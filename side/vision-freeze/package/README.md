# EEG VLM fine-tuned models

The best fine-tuned model per dataset from the EEG waveform benchmark. Each takes a 20 s EEG window rendered as an
image and answers one yes/no question; `predict.py` returns P(true) and, with `--explain`, the model's rationale.

| model dir | base | question | test AUROC [95% patient CI] |
|---|---|---|---|
| `tusz-seizure-qwen3-vl-8b` | Qwen/Qwen3-VL-8B-Instruct | seizure (any type) | 0.917 [0.892, 0.935] |
| `tuab-abnormal-gemma4-e4b` | google/gemma-4-E4B-it | abnormal EEG | 0.851 [0.807, 0.891] |
| `tuar-eye-movement-medgemma1.5-4b` | google/medgemma-1.5-4b-it | eye movement artifact | 0.894 [0.859, 0.925] |
| `tuev-epileptiform-granite3.2-vision-2b` | ibm-granite/granite-vision-3.2-2b | epileptiform discharge | 0.981 [0.960, 0.993] |

Each `models/<name>/` has the LoRA adapter (with the fully trained vision tower inside it) and processor in
`adapter/`, the exact prompts in `model.json`, and 4 sample test images with the P(true) we got on the cluster in
`samples/expected.csv`. The base model downloads from Hugging Face on first run.

## Run

With Docker (needs an NVIDIA GPU and the NVIDIA container toolkit):

```bash
docker build -t eeg-vlm .
docker run --gpus all -e HF_TOKEN -v $PWD/hf-cache:/root/.cache/huggingface eeg-vlm \
  models/tuev-epileptiform-granite3.2-vision-2b models/tuev-epileptiform-granite3.2-vision-2b/samples/*.png
```

Without Docker, in Python 3.11:

```bash
pip install -r requirements.txt
python predict.py models/<name> image1.png image2.png [--explain] [--out predictions.csv]
```

Check an install by running a model on its `samples/*.png` and comparing with `samples/expected.csv`. Expect
differences of about 0.005, and up to 0.03 for windows near 0.5, where bf16 rounding decides the last digits (the
cluster ran on A100s; our check on a laptop CPU matched within these margins).

## Things to know

- **Gated bases.** Gemma 4 and MedGemma need their licence accepted on Hugging Face and an `HF_TOKEN` from that
  account.
- **Input format.** The models only know images in the paper's design: 2688x896 PNG, 20 s window, 25 rows (TUH
  TCP bipolar montage plus Fz-Cz, Cz-Pz and EKG), one gain per recording. Other renderings will not give meaningful
  scores. The renderer is `side/vision-freeze/rerender.py` (`draw_core`) in the project repo.
- **Memory.** The 8B model needs about 20 GB of GPU memory in bf16, the others 8-12 GB. CPU works but is slow
  (seconds to minutes per window).
- **Threshold.** `predicted` uses 0.5, which is not calibrated for every model; AUROC is the reported metric.
- **TUEV.** Most TUEV test recordings carry a single label, and most positives are periodic discharges (PLED/GPED).
  Within recordings that have both labels, AUROC is 0.85, and on spikes alone 0.92 (small samples).
- **Data licence.** The sample images come from the TUH EEG Corpus and fall under its data use agreement.
