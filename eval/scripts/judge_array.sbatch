#!/bin/bash
#SBATCH --job-name=tueg-judge
#SBATCH --account=def-milad777
#SBATCH --cpus-per-task=4

# One array task per benchmarked model: judges that model's rationales across all
# of its datasets, so the judge's weights load once per ~14.8k pairs rather than
# once per dataset. Submitted by eval/agreement.py; RUN_NAME and MODELS_B64 come
# in via --export.

set -euo pipefail

module load StdEnv/2023 apptainer/1.4.5
source ../.venv/bin/activate

MODEL=$(python -c 'import base64, json, os; print(json.loads(base64.b64decode(os.environ["MODELS_B64"]))[int(os.environ["SLURM_ARRAY_TASK_ID"])])')

read -r JUDGE_MODEL JUDGE_PARALLEL JUDGE_CTX < <(python -c '
import yaml
with open("judge.yml") as f:
    judge = yaml.safe_load(f)["judge"]
print(judge["model"], judge["parallel-requests"], judge["num-ctx"])
')

export WANDB_MODE=offline
export OLLAMA_IMAGE="${OLLAMA_IMAGE:-$SCRATCH/ollama/ollama.sif}"
export OLLAMA_MODELS="$SCRATCH/ollama/models"
export OLLAMA_HOST="127.0.0.1:$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID) % 40000))"
export OLLAMA_BASE_URL="http://$OLLAMA_HOST"
# Serve exactly as many concurrent requests as judge.py will send, from the same
# key in the same file, so server and client cannot drift apart.
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-$JUDGE_PARALLEL}"
# Text-only: two rationales plus the rubric measure ~700-1200 tokens, nothing
# like the ~4033 an image costs. A small context keeps KV cache
# (context x OLLAMA_NUM_PARALLEL) well inside the 20 GB MIG slice.
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-$JUDGE_CTX}"
export APPTAINERENV_OLLAMA_MODELS="$OLLAMA_MODELS"
export APPTAINERENV_OLLAMA_HOST="$OLLAMA_HOST"
export APPTAINERENV_OLLAMA_NUM_PARALLEL="$OLLAMA_NUM_PARALLEL"
export APPTAINERENV_OLLAMA_CONTEXT_LENGTH="$OLLAMA_CONTEXT_LENGTH"

OLLAMA_LOG="$RUN_DIR/logs/judge-ollama-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
apptainer exec --nv "$OLLAMA_IMAGE" ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT

for _ in {1..60}; do
    apptainer exec --nv "$OLLAMA_IMAGE" ollama list >/dev/null 2>&1 && break
    sleep 2
done
apptainer exec --nv "$OLLAMA_IMAGE" ollama list >/dev/null   # fail loudly if it never came up

apptainer exec --nv "$OLLAMA_IMAGE" ollama show "$JUDGE_MODEL" >/dev/null 2>&1 \
  || apptainer exec --nv "$OLLAMA_IMAGE" ollama pull "$JUDGE_MODEL"

# Record it with the run: the judge's verdicts are only comparable across tasks
# if the same Ollama build and the same model digest produced them.
apptainer exec --nv "$OLLAMA_IMAGE" ollama --version

python scripts/judge.py --run "$RUN_NAME" --model "$MODEL"
