#!/bin/bash
#SBATCH --job-name=tueg-vlm-array
#SBATCH --account=def-milad777
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

module load apptainer
source .venv/bin/activate

export WANDB_MODE=offline
export OLLAMA_MODELS="$SCRATCH/ollama/models"
export APPTAINERENV_OLLAMA_MODELS="$OLLAMA_MODELS"

MODEL=$(python -c 'import os, yaml; c=yaml.safe_load(open("config.yml")); models=list(c["models"]); selected=c.get("model"); run=models if selected == "all" else [selected]; print(run[int(os.environ["SLURM_ARRAY_TASK_ID"])])')

OLLAMA_PORT=$((11434 + SLURM_ARRAY_TASK_ID))
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
export OLLAMA_BASE_URL="http://$OLLAMA_HOST"
export APPTAINERENV_OLLAMA_HOST="$OLLAMA_HOST"

OLLAMA_LOG="logs/ollama-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
OLLAMA_PID=""
cleanup() {
    [[ -n "$OLLAMA_PID" ]] && kill "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "host=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID model=$MODEL ollama=$OLLAMA_HOST"
nvidia-smi || true

apptainer exec --nv "$SCRATCH/ollama/ollama.sif" ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

for _ in {1..60}; do
    kill -0 "$OLLAMA_PID" 2>/dev/null || { tail -100 "$OLLAMA_LOG"; exit 1; }
    apptainer exec --nv "$SCRATCH/ollama/ollama.sif" ollama list >/dev/null 2>&1 && break
    sleep 2
done

apptainer exec --nv "$SCRATCH/ollama/ollama.sif" ollama list >/dev/null 2>&1 || {
    echo "ERROR: Ollama did not become ready" >&2
    tail -100 "$OLLAMA_LOG" >&2 || true
    exit 1
}

python scripts/eval.py --model "$MODEL"
