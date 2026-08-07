#!/bin/bash
#SBATCH --job-name=generate-rationales
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=generate-rationales-%j.out
#SBATCH --error=generate-rationales-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the project directory}"
source .venv/bin/activate

export OLLAMA_HOST=127.0.0.1:11434
# Full-resolution EEG images can exceed Ollama's 4096-token default. This leaves
# ample room for both the image prompt and the 256-token report.
export OLLAMA_CONTEXT_LENGTH=16384
# Serve several rationale requests at once so the GPU stays busy.
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$SLURM_SUBMIT_DIR/ollama-models}"
mkdir -p "$OLLAMA_MODELS"

ollama serve &
ollama_pid=$!
trap 'kill "$ollama_pid" 2>/dev/null || true' EXIT

until ollama list >/dev/null 2>&1; do sleep 1; done
ollama show qwen2.5vl:3b >/dev/null 2>&1 || ollama pull qwen2.5vl:3b

python train/scripts/generate-rationales.py
