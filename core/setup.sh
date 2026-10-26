#!/usr/bin/env bash
# One-shot RunPod setup. Safe to re-run.
# Installs Miniconda, the trainer env, the inference env, ai-toolkit, and weights.
set -euo pipefail

CORE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CORE_DIR/.." && pwd)"

mkdir -p /workspace/data
# shellcheck source=env.sh
source "$CORE_DIR/env.sh"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "warning: HF_TOKEN is empty. Copy .env.example to .env if Hub downloads are rate-limited."
fi

MINICONDA="${MINICONDA:-/workspace/miniconda3}"
AITOOLKIT="${AITOOLKIT:-/workspace/ai-toolkit}"
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes

if [[ ! -x "$MINICONDA/bin/conda" ]]; then
  echo "installing Miniconda into $MINICONDA"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$MINICONDA"
  rm -f /tmp/miniconda.sh
fi
# shellcheck disable=SC1091
source "$MINICONDA/etc/profile.d/conda.sh"

if [[ ! -d "$AITOOLKIT/.git" ]]; then
  echo "cloning ai-toolkit into $AITOOLKIT"
  git clone --depth 1 https://github.com/ostris/ai-toolkit.git "$AITOOLKIT"
fi

install_torch() {
  pip install --upgrade pip
  pip install --no-cache-dir \
    torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130
}

if ! conda env list | awk '{print $1}' | grep -qx klein; then
  conda create -y -n klein python=3.12
fi
if [[ ! -f "$HF_HOME/.klein-requirements-ok" ]]; then
  conda activate klein
  install_torch
  pip install --no-cache-dir -r "$AITOOLKIT/requirements.txt"
  conda deactivate
  touch "$HF_HOME/.klein-requirements-ok"
else
  echo "klein env already installed ($HF_HOME/.klein-requirements-ok)"
fi

if ! conda env list | awk '{print $1}' | grep -qx klein-infer; then
  conda create -y -n klein-infer python=3.12
fi
if [[ ! -f "$HF_HOME/.klein-infer-requirements-ok" ]]; then
  conda activate klein-infer
  install_torch
  pip install --no-cache-dir -r "$CORE_DIR/requirements-infer.txt"
  conda deactivate
  touch "$HF_HOME/.klein-infer-requirements-ok"
else
  echo "klein-infer env already installed ($HF_HOME/.klein-infer-requirements-ok)"
fi

conda activate klein
download() {
  if command -v hf >/dev/null 2>&1; then
    hf download "$@"
  else
    huggingface-cli download "$@"
  fi
}

echo "downloading FLUX.2 [klein] base 4B transformer"
download black-forest-labs/FLUX.2-klein-base-4B flux-2-klein-base-4b.safetensors
echo "downloading FLUX.2 VAE"
download ai-toolkit/flux2_vae ae.safetensors
echo "downloading Qwen3-4B text encoder"
download Qwen/Qwen3-4B
echo "downloading FLUX.2 [klein] 4B distilled pipeline"
download black-forest-labs/FLUX.2-klein-4B
conda deactivate

echo "setup complete"
echo "  HF_HOME=$HF_HOME"
echo "  trainer: conda activate klein"
echo "  infer:   conda activate klein-infer"
