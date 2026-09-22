#!/bin/bash
# Run the eval worker for one (model, dataset) on a GPU node, for the last few pending windows that are not
# worth an array job's queue time. Processes only rows pending in pipeline.db. One command from the login node:
#
#   bash helpers/eval-interactive.sh gemma4:12b TUSZ
#
# Outside an allocation it requests one (30 min, 1 GPU, 24G) and re-runs itself there; inside one (salloc) it
# just runs. Override the request with e.g. EVAL_TIME=1:00:00 EVAL_MEM=48G.
set -euo pipefail
model=${1:?usage: eval-interactive.sh MODEL DATASET}
dataset=${2:?usage: eval-interactive.sh MODEL DATASET}
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "requesting a GPU node..." >&2
  exec srun --account=def-milad777 --time="${EVAL_TIME:-0:30:00}" --mem="${EVAL_MEM:-24G}" --cpus-per-task=4 \
       --gres=gpu:1 --pty bash "$0" "$model" "$dataset"
fi
cd "$(dirname "$0")/.."

module load StdEnv/2023 apptainer/1.4.5
source .venv/bin/activate
port=$((20000 + RANDOM % 40000))
export OLLAMA_HOST=127.0.0.1:$port OLLAMA_BASE_URL=http://127.0.0.1:$port
export OLLAMA_MODELS=$SCRATCH/ollama/models OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_NUM_PARALLEL=1
export APPTAINERENV_OLLAMA_MODELS=$OLLAMA_MODELS APPTAINERENV_OLLAMA_HOST=$OLLAMA_HOST
export APPTAINERENV_OLLAMA_NUM_PARALLEL=1 APPTAINERENV_OLLAMA_CONTEXT_LENGTH=8192

apptainer exec --nv "$SCRATCH/ollama/ollama.sif" ollama serve > "logs/ollama-interactive-$$.log" 2>&1 &
trap 'kill %1 2>/dev/null || true' EXIT
for _ in $(seq 60); do curl -s "$OLLAMA_BASE_URL" > /dev/null && break; sleep 2; done
curl -s "$OLLAMA_BASE_URL" > /dev/null || { echo "ollama never became ready (logs/ollama-interactive-$$.log)" >&2; exit 1; }

export EVAL_TASKS=$(python -c "import base64,json;print(base64.b64encode(json.dumps([{'model':'$model','dataset':'$dataset','parallel':1}]).encode()).decode())")
SLURM_ARRAY_TASK_ID=0 python eval/models/eval.py
echo "pending now: $(sqlite3 pipeline.db "SELECT COUNT(*) FROM pipeline WHERE model='$model' AND dataset='$dataset' AND scope='full' AND evaled=0")"
