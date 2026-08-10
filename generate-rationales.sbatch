#!/bin/bash
#SBATCH --job-name=generate-rationales
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=generate-rationales-%j.out
#SBATCH --error=generate-rationales-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the project directory}"
module load StdEnv/2023 apptainer/1.4.5
source .venv/bin/activate

# Run Ollama from the Apptainer image -- native `ollama` is NOT on the compute
# node's PATH (that failure left the job spinning). Matches the eval array job.
export OLLAMA_IMAGE="${OLLAMA_IMAGE:-$SCRATCH/ollama/ollama.sif}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$SCRATCH/ollama/models}"
export OLLAMA_HOST="127.0.0.1:11434"
# A 1536x1536 image + prompt is ~3050 tokens; 8192 leaves ample room for the
# ~512-token report (init_model also pins num_ctx/num_predict per request).
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"
# Qwen2.5-VL in Ollama falls into infinite single-token repetition without flash
# attention -- the non-FA path mispredicts and the loop never emits EOS, so the
# reply is one character repeated for the whole num_predict budget.
# ollama/ollama#11452, #10767.
export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
export APPTAINERENV_OLLAMA_MODELS="$OLLAMA_MODELS"
export APPTAINERENV_OLLAMA_FLASH_ATTENTION="$OLLAMA_FLASH_ATTENTION"
export APPTAINERENV_OLLAMA_HOST="$OLLAMA_HOST"
export APPTAINERENV_OLLAMA_CONTEXT_LENGTH="$OLLAMA_CONTEXT_LENGTH"
export APPTAINERENV_OLLAMA_NUM_PARALLEL="$OLLAMA_NUM_PARALLEL"
mkdir -p "$OLLAMA_MODELS"

apptainer exec --nv "$OLLAMA_IMAGE" ollama serve > "ollama-${SLURM_JOB_ID}.log" 2>&1 &
ollama_pid=$!
trap 'kill "$ollama_pid" 2>/dev/null || true' EXIT

for _ in {1..60}; do
    apptainer exec --nv "$OLLAMA_IMAGE" ollama list >/dev/null 2>&1 && break
    sleep 2
done
apptainer exec --nv "$OLLAMA_IMAGE" ollama list >/dev/null   # fail loudly if Ollama never came up

# Read the tag out of the script so MODEL stays the single place to change it.
model=$(sed -n 's/^MODEL = "\(.*\)"$/\1/p' train/scripts/generate-rationales.py)
apptainer exec --nv "$OLLAMA_IMAGE" ollama show "$model" >/dev/null 2>&1 \
  || apptainer exec --nv "$OLLAMA_IMAGE" ollama pull "$model"

# Version matters as much as model choice here: vision-model regressions land often
# (ollama/ollama#13549 broke image input in 0.13.3+). Record it with the run.
apptainer exec --nv "$OLLAMA_IMAGE" ollama --version

python train/scripts/generate-rationales.py
