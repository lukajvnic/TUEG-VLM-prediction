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

MODEL=$(python -c 'import base64, json, os, yaml; c=yaml.safe_load(open("config.yml")); override=os.environ.get("PI_MODEL_LIST_B64"); models=json.loads(base64.b64decode(override).decode("utf-8")) if override else list(c["models"]); selected=c.get("model"); run=models if override or selected == "all" else [selected]; print(run[int(os.environ["SLURM_ARRAY_TASK_ID"])])')

OLLAMA_PORT=$(python -c 'import os; job=int(os.environ.get("SLURM_JOB_ID", "0")); task=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")); print(20000 + ((job * 1009 + task) % 40000))')
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
export OLLAMA_BASE_URL="http://$OLLAMA_HOST"
export APPTAINERENV_OLLAMA_HOST="$OLLAMA_HOST"

RUN_LOG_DIR="${PI_LOG_DIR:-logs}"
mkdir -p "$RUN_LOG_DIR"
OLLAMA_LOG="$RUN_LOG_DIR/ollama-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
OLLAMA_PID=""
cleanup() {
    [[ -n "$OLLAMA_PID" ]] && kill "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "host=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID model=$MODEL ollama=$OLLAMA_HOST image=$OLLAMA_IMAGE"
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

python scripts/eval.py --model "$MODEL"
