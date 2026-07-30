#!/bin/bash
#SBATCH --job-name=hf-install
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=hf-install-%j.out
#SBATCH --error=hf-install-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the project directory}"

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_CHECKPOINT_DIR="${HF_CHECKPOINT_DIR:-$SLURM_SUBMIT_DIR/hf-checkpoints}"
mkdir -p "$HF_HOME" "$HF_CHECKPOINT_DIR"

python hf-install.py --output-dir "$HF_CHECKPOINT_DIR"
