#!/bin/bash
#SBATCH --job-name=tueg-vlm
#SBATCH --account=def-milad777
#SBATCH --cpus-per-task=4

set -euo pipefail

module load StdEnv/2023 apptainer/1.4.5
source ../.venv/bin/activate

IFS=$'\t' read -r MODEL DATASET < <(
    python -c 'import base64, json, os; tasks = json.loads(base64.b64decode(os.environ["RUNS_B64"])); print(*tasks[int(os.environ["SLURM_ARRAY_TASK_ID"])], sep="\t")'
)

export WANDB_MODE=offline
export OLLAMA_IMAGE="${OLLAMA_IMAGE:-$SCRATCH/ollama/ollama.sif}"
export OLLAMA_MODELS="$SCRATCH/ollama/models"
export OLLAMA_HOST="127.0.0.1:$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID) % 40000))"
export OLLAMA_BASE_URL="http://$OLLAMA_HOST"
# Serve exactly as many concurrent requests as eval.py will send. eval.py takes
# the model's own `parallel-requests` when set (a full A100 sustains more in
# flight than a 20 GB MIG slice) and falls back to the global setting, so the
# server and the client must read the same value from the same config.
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-$(
    MODEL="$MODEL" python -c '
import os, yaml
with open(os.path.join(os.environ["RUN_DIR"], "config.yml")) as f:
    config = yaml.safe_load(f)
settings = config["settings"]
model = config.get("models", {}).get(os.environ["MODEL"], {})
print(model.get("parallel-requests", settings["parallel-requests"]))
'
)}"
# A full-resolution EEG image + prompt measures ~4033 tokens; the 4096 default
# truncates once the JSON rationale is added. 8192 clears that with headroom
# while keeping KV-cache memory (context x OLLAMA_NUM_PARALLEL) modest on MIG.
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
export APPTAINERENV_OLLAMA_MODELS="$OLLAMA_MODELS"
export APPTAINERENV_OLLAMA_HOST="$OLLAMA_HOST"
export APPTAINERENV_OLLAMA_NUM_PARALLEL="$OLLAMA_NUM_PARALLEL"
export APPTAINERENV_OLLAMA_CONTEXT_LENGTH="$OLLAMA_CONTEXT_LENGTH"

OLLAMA_LOG="$RUN_DIR/logs/ollama-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
apptainer exec --nv "$OLLAMA_IMAGE" ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT

for _ in {1..60}; do
    apptainer exec --nv "$OLLAMA_IMAGE" ollama list >/dev/null 2>&1 && break
    sleep 2
done

apptainer exec --nv "$OLLAMA_IMAGE" ollama list >/dev/null
python scripts/eval.py --model "$MODEL" --dataset "$DATASET"
