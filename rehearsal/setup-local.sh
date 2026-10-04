#!/bin/bash
# one-time setup for the cpu rehearsal on this machine: a venv with Narval's exact package versions, the base's
# config / tokenizer / processor files (no weights) and the dataset's sft jsonl, both copied from Narval
#   bash rehearsal/setup-local.sh qwen2.5vl:7b TUAB
# then: HF_CHECKPOINTS=$PWD/hf-checkpoints-local .venv-rehearsal/bin/python rehearsal/rehearse.py cpu qwen2.5vl:7b TUAB --side
set -euo pipefail
cd "$(dirname "$0")/.."
key=$1; dataset=$2
remote=narval.alliancecan.ca:/scratch/luka

[ -x .venv-rehearsal/bin/python ] || python3.11 -m venv .venv-rehearsal
.venv-rehearsal/bin/pip install -q -r rehearsal/requirements-local.txt

repo=$(.venv-rehearsal/bin/python -c "import yaml,sys; print(yaml.safe_load(open('config.yml'))['bases'][sys.argv[1]]['repo'].replace('/', '--'))" "$key")
mkdir -p "hf-checkpoints-local/$repo"
rsync -a --exclude '*.safetensors' --exclude '*.bin' --exclude '*.pt' "$remote/hf-checkpoints/$repo/" "hf-checkpoints-local/$repo/"
rsync -a "$remote/TUEG-VLM-prediction/datasets/$dataset/sft_*.jsonl" "datasets/$dataset/"
echo "ready: HF_CHECKPOINTS=$PWD/hf-checkpoints-local .venv-rehearsal/bin/python rehearsal/rehearse.py cpu $key $dataset --side"
