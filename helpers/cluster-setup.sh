#!/bin/bash
# Build or repair the three cluster venvs and preflight each one. Login node (internet), from the project dir:
#   bash helpers/cluster-setup.sh            # all three
#   bash helpers/cluster-setup.sh unsloth    # one of: main llamafactory unsloth
# Idempotent: existing venvs are kept, pins re-applied, then helpers/preflight.py must pass or the script exits 1.
# On the login node the preflight skips imports that load bitsandbytes (its wheelhouse CPU library faults on
# these EPYCs; the CUDA library is what a GPU node loads). PREFLIGHT_GPU=1 runs each preflight on a GPU node
# through a 15-minute srun instead, which covers those too (queue wait included).
set -euo pipefail
cd "$(dirname "$0")/.."
module load StdEnv/2023 gcc python/3.11 cuda arrow   # arrow before any venv: pyarrow only exists through the module
LF_DIR=${LF_DIR:-$SCRATCH/LLaMA-Factory}
which=${1:-all}
preflight() {
  if [ "${PREFLIGHT_GPU:-0}" = 1 ]; then
    srun --account=def-milad777 --gres=gpu:1 --mem=24G --cpus-per-task=2 --time=00:15:00 python helpers/preflight.py "$1"
  else
    python helpers/preflight.py "$1"
  fi
}

venv() { [ -d "$1" ] || virtualenv --no-download "$1"; source "$1/bin/activate"; }  # Alliance's virtualenv sees module packages

if [ "$which" = all ] || [ "$which" = main ]; then
  echo "=== .venv: generic trainer + predict"
  venv .venv
  pip install --no-index -r requirements.txt || pip install -r requirements.txt
  preflight main
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
  # its check_version list (extras/misc.py): transformers <=5.8.0, datasets <=4.0.0, accelerate <=1.15.0,
  # peft <=0.20.0, trl <=0.24.0; a wheelhouse build at exactly the upper bound (x.y.z+computecanada) fails it
  pip install "transformers==4.56.2" "accelerate==1.14.0" "peft==0.19.1" pydantic
  pip install --no-deps "trl==0.22.2" "datasets==3.6.0"   # --no-deps: their datasets/pyarrow requirements hit the wheelhouse dummy wheel
  preflight llamafactory
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
  preflight unsloth
  deactivate
fi
echo "=== done"
