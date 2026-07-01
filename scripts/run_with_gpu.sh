#!/bin/bash
#SBATCH --job-name=ollama-vlm
#SBATCH --account=def-milad777
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd $SCRATCH/TUEG-VLM-prediction

module load apptainer

# Activate Python environment
source .venv/bin/activate

# W&B offline because GPU nodes may not have internet
export WANDB_MODE=offline

# Tell Ollama where the downloaded model files are
export OLLAMA_MODELS=$SCRATCH/ollama/models

# Start Ollama server in background
apptainer exec --nv $SCRATCH/ollama/ollama.sif ollama serve > $SCRATCH/ollama/server.log 2>&1 &

sleep 15

# Run script
python scripts/eval.py

# Clean up server
pkill -f "ollama serve"