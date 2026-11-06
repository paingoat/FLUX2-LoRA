#!/usr/bin/env bash
# Linux + existing Conda; independent of trainer/ai-toolkit setup.
set -euo pipefail
CORE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CORE_DIR/env.sh"
export TORCH_CUDA_ARCH_LIST=12.0
export HF_HUB_DISABLE_XET=0
export HF_HUB_ENABLE_HF_TRANSFER=0

if ! command -v conda >/dev/null 2>&1; then
  echo 'Conda must be installed and on PATH.' >&2
  exit 1
fi
command -v git >/dev/null
command -v nvidia-smi >/dev/null
nvidia-smi
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HF_ASSETS_CACHE"
probe="$(mktemp "$HF_HOME/.write-check.XXXXXX")"
rm -f -- "$probe"
df -h "$HF_HOME"
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Resolve source once and retain the checkout on reruns. SAM3_REVISION can be a reviewed commit/tag.
SAM3_SOURCE="${SAM3_SOURCE:-$HF_HOME/source/sam3}"
SAM3_REPOSITORY="${SAM3_REPOSITORY:-https://github.com/facebookresearch/sam3.git}"
SAM3_REVISION="${SAM3_REVISION:-660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7}"
if [[ ! -d "$SAM3_SOURCE/.git" ]]; then
  mkdir -p "$(dirname "$SAM3_SOURCE")"
  git clone "$SAM3_REPOSITORY" "$SAM3_SOURCE"
  git -C "$SAM3_SOURCE" checkout --detach "${SAM3_REVISION:-HEAD}"
fi
if [[ -n "$(git -C "$SAM3_SOURCE" status --porcelain)" ]]; then
  echo 'SAM3 checkout has local changes; use a clean source checkout.' >&2
  exit 1
fi
if [[ -n "${SAM3_REVISION:-}" ]]; then
  requested="$(git -C "$SAM3_SOURCE" rev-parse "$SAM3_REVISION^{commit}")"
  actual="$(git -C "$SAM3_SOURCE" rev-parse HEAD)"
  [[ "$requested" == "$actual" ]] || { echo 'Existing SAM3 checkout differs from SAM3_REVISION.' >&2; exit 1; }
fi

for kind in sam3 flux; do
  env_name="preservation-$kind"
  if ! conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    conda create -y -n "$env_name" python=3.12
  fi
  conda activate "$env_name"
  python -m pip install --upgrade pip
  python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
  if [[ "$kind" == sam3 ]]; then
    python -m pip install -r "$CORE_DIR/requirements-preservation-sam3.txt"
    python -m pip install -c "$CORE_DIR/requirements-preservation-sam3.txt" "$SAM3_SOURCE"
    python -m pip check
    python "$CORE_DIR/preservation_runtime.py" --kind sam3 --write-lock --sam3-source "$SAM3_SOURCE"
  else
    python -m pip install -r "$CORE_DIR/requirements-preservation.txt"
    python -m pip check
    python "$CORE_DIR/preservation_runtime.py" --kind flux --write-lock
  fi
  conda deactivate
done
echo 'Setup checked. See core/preservation.md for mask preparation and the 60-output pilot.'
