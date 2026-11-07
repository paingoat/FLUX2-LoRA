#!/usr/bin/env bash
# Dedicated training environment; leaves the PSNR/SAM3 environments untouched.
set -euo pipefail
CORE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CORE_DIR/env.sh"
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
REGION_ENV="${REGION_ENV:-klein-region-5090}"
export AITOOLKIT="${AITOOLKIT:-$HF_HOME/source/ai-toolkit-region}"
export HF_HUB_DISABLE_XET=0
export HF_HUB_ENABLE_HF_TRANSFER=0
REVISION=beb94824c51480a04d3c86c718603676fdef6938
mkdir -p "$HF_HOME/source"
if [[ ! -d "$AITOOLKIT/.git" ]]; then
  git clone https://github.com/ostris/ai-toolkit.git "$AITOOLKIT"
  git -C "$AITOOLKIT" checkout --detach "$REVISION"
fi
[[ "$(git -C "$AITOOLKIT" rev-parse HEAD)" == "$REVISION" ]] || {
  echo 'AI Toolkit revision differs; use a separate checkout at the pinned revision.' >&2; exit 1;
}
if ! conda env list | awk '{print $1}' | grep -qx "$REGION_ENV"; then
  conda create -y -n "$REGION_ENV" python=3.12
fi
conda activate "$REGION_ENV"
conda install -y -c conda-forge 'ffmpeg=7'
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# The pinned image trainer imports optional audio modules. Align torchcodec
# with torch 2.8 rather than its newer upstream audio default (0.15).
python - "$AITOOLKIT" "$CONDA_PREFIX" <<'PY'
from pathlib import Path
import sys
toolkit, prefix = map(Path, sys.argv[1:])
text = (toolkit / 'requirements_base.txt').read_text()
if text.count('torchcodec==0.15.0') != 1:
    raise SystemExit('unexpected toolkit requirements; inspect before installing')
(prefix / 'region-requirements.txt').write_text(text.replace('torchcodec==0.15.0', 'torchcodec==0.6.0') + '\nscipy==1.12.0\nscikit-image\n')
(prefix / 'region-constraints.txt').write_text('torch==2.8.0\ntorchvision==0.23.0\ntorchaudio==2.8.0\nnumpy==1.26.4\n')
PY
python -m pip install -c "$CONDA_PREFIX/region-constraints.txt" -r "$CONDA_PREFIX/region-requirements.txt"
python -m pip check
python "$CORE_DIR/patch_region_trainer.py" "$AITOOLKIT"
python -m pip freeze > "$CONDA_PREFIX/region-packages.txt"
export PYTHONPATH="$CORE_DIR:$AITOOLKIT${PYTHONPATH:+:$PYTHONPATH}"
python - "$CONDA_PREFIX" <<'PY'
import json, sys
from pathlib import Path
from patch_region_trainer import REVISION
# Import check only. The user runs GPU checks and training explicitly later.
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from diffusers import Flux2KleinPipeline
Path(sys.argv[1], 'region-runtime.json').write_text(json.dumps(dict(toolkit_revision=REVISION, torch='2.8.0', cuda='12.8'), indent=2))
PY
hf download black-forest-labs/FLUX.2-klein-base-4B flux-2-klein-base-4b.safetensors
hf download ai-toolkit/flux2_vae ae.safetensors
hf download Qwen/Qwen3-4B
echo "Ready: conda activate $REGION_ENV"
echo "export AITOOLKIT=$AITOOLKIT"
echo 'export KLEIN_PYTHON="$CONDA_PREFIX/bin/python"'
echo 'GPU support and VRAM remain to be checked on the actual RTX 5090.'
