#!/bin/bash
#SBATCH --job-name=ollama-vlm
#SBATCH --account=def-milad777
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SCRATCH/TUEG-VLM-prediction"
mkdir -p logs

module load StdEnv/2023 apptainer/1.4.5
source .venv/bin/activate

export WANDB_MODE=offline
export OLLAMA_IMAGE="${PI_OLLAMA_IMAGE:-$SCRATCH/ollama/ollama.sif}"
export OLLAMA_MODELS="$SCRATCH/ollama/models"
export OLLAMA_LOAD_TIMEOUT=30m
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export APPTAINERENV_OLLAMA_MODELS="$OLLAMA_MODELS"
export APPTAINERENV_OLLAMA_LOAD_TIMEOUT="$OLLAMA_LOAD_TIMEOUT"
export APPTAINERENV_OLLAMA_MAX_LOADED_MODELS="$OLLAMA_MAX_LOADED_MODELS"
export APPTAINERENV_OLLAMA_NUM_PARALLEL="$OLLAMA_NUM_PARALLEL"
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_BASE_URL="http://$OLLAMA_HOST"
export APPTAINERENV_OLLAMA_HOST="$OLLAMA_HOST"

RUN_LOG_DIR="${PI_LOG_DIR:-logs}"
mkdir -p "$RUN_LOG_DIR"
OLLAMA_LOG="$RUN_LOG_DIR/ollama-${SLURM_JOB_ID}.log"
OLLAMA_PID=""
cleanup() {
    [[ -n "$OLLAMA_PID" ]] && kill "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "host=$(hostname) job=$SLURM_JOB_ID model=${1:-config.yml} ollama=$OLLAMA_HOST image=$OLLAMA_IMAGE"
nvidia-smi || true
apptainer exec --nv "$OLLAMA_IMAGE" ollama --version || true

apptainer exec --nv "$OLLAMA_IMAGE" ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

for _ in {1..60}; do
    kill -0 "$OLLAMA_PID" 2>/dev/null || { tail -100 "$OLLAMA_LOG"; exit 1; }
    apptainer exec --nv "$OLLAMA_IMAGE" ollama list >/dev/null 2>&1 && break
    sleep 2
done

apptainer exec --nv "$OLLAMA_IMAGE" ollama list >/dev/null 2>&1 || {
    echo "ERROR: Ollama did not become ready" >&2
    tail -100 "$OLLAMA_LOG" >&2 || true
    exit 1
}

if [[ $# -gt 0 ]]; then
    python scripts/eval.py --model "$1"
else
    python scripts/eval.py
fi
