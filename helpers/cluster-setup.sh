#!/bin/bash
# Build or repair the three cluster venvs and preflight each one. Login node (internet), from the project dir:
#   bash helpers/cluster-setup.sh            # all three
#   bash helpers/cluster-setup.sh unsloth    # one of: main llamafactory unsloth
# Idempotent: existing venvs are kept, pins re-applied, then helpers/preflight.py must pass or the script exits 1.
set -euo pipefail
cd "$(dirname "$0")/.."
module load StdEnv/2023 gcc python/3.11 cuda arrow   # arrow before any venv: pyarrow only exists through the module
LF_DIR=${LF_DIR:-$SCRATCH/LLaMA-Factory}
which=${1:-all}

venv() { [ -d "$1" ] || virtualenv --no-download "$1"; source "$1/bin/activate"; }  # Alliance's virtualenv sees module packages

if [ "$which" = all ] || [ "$which" = main ]; then
  echo "=== .venv: generic trainer + predict"
  venv .venv
  pip install --no-index -r requirements.txt || pip install -r requirements.txt
  python helpers/preflight.py main
  deactivate
fi

if [ "$which" = all ] || [ "$which" = llamafactory ]; then
  echo "=== .venv-llamafactory: MiniCPM-V 2.6 / 4.5"
  venv .venv-llamafactory
  [ -d "$LF_DIR" ] || git clone https://github.com/hiyouga/LLaMA-Factory.git "$LF_DIR"
  pip install --no-index torch torchvision || pip install torch torchvision
  pip install -e "$LF_DIR"
  # LLaMA-Factory bounds transformers <= 5.8.0 and datasets <= 4.0.0; the wheelhouse's +computecanada tags sort
  # above those, so pin below. --no-deps on datasets: its pyarrow requirement would hit the wheelhouse dummy wheel
  pip install "transformers==4.56.2" pydantic
  pip install --no-deps "datasets==3.6.0"
  python helpers/preflight.py llamafactory
  deactivate
fi

if [ "$which" = all ] || [ "$which" = unsloth ]; then
  echo "=== .venv-unsloth: DeepSeek-OCR"
  venv .venv-unsloth
  pip install --no-index torch torchvision || pip install torch torchvision
  pip install --no-index triton || true
  pip install --no-deps unsloth unsloth_zoo        # every unsloth release caps torch below the wheelhouse 2.14
  pip install "transformers==4.56.2" "trl==0.22.2" "peft>=0.18" accelerate bitsandbytes tyro protobuf sentencepiece \
      hf_transfer cut_cross_entropy einops addict easydict PyYAML matplotlib pydantic dill multiprocess xxhash pandas
  pip install --no-deps "datasets==3.6.0"
  python helpers/preflight.py unsloth
  deactivate
fi
echo "=== done"
