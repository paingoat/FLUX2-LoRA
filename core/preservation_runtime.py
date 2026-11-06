"""Environment preflight and recorded model revisions for the preservation pilot."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from common import DISTILLED_REPO, load_runtime_env


def check_cuda() -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run in the RTX 5090 environment')
    capability = torch.cuda.get_device_capability()
    archs = torch.cuda.get_arch_list()
    if capability != (12, 0) or 'sm_120' not in archs:
        raise RuntimeError(f'expected RTX 5090 capability (12,0) and native sm_120; got {capability}, {archs}')
    # Exercise kernels rather than accepting a version string as proof of compatibility.
    a = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16)
    if not bool(torch.all(a @ a == 32)):
        raise RuntimeError('CUDA BF16 matrix multiplication failed')
    torch.cuda.synchronize()
    return dict(gpu=torch.cuda.get_device_name(), capability=list(capability), architectures=archs,
                torch=torch.__version__, cuda=torch.version.cuda, python=sys.version,
                total_vram_bytes=torch.cuda.get_device_properties(0).total_memory)


def lock_path(kind: str) -> Path:
    return Path(sys.prefix) / f'preservation-{kind}-lock.json'


def read_lock(kind: str) -> dict:
    path = lock_path(kind)
    if not path.is_file():
        raise ValueError(f'missing {path}; run core/setup_preservation.sh first')
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['sam3', 'flux'], required=True)
    parser.add_argument('--write-lock', action='store_true')
    parser.add_argument('--sam3-source', type=Path)
    args = parser.parse_args()
    load_runtime_env()
    info = check_cuda()
    info['cache'] = {key: os.environ[key] for key in ('HF_HOME', 'HF_HUB_CACHE', 'HF_XET_CACHE', 'HF_ASSETS_CACHE')}
    if args.write_lock:
        from huggingface_hub import HfApi
        path = lock_path(args.kind)
        previous = read_lock(args.kind) if path.exists() else {}
        repo = 'facebook/sam3' if args.kind == 'sam3' else DISTILLED_REPO
        revision_key = f'{args.kind}_model_revision'
        revision = previous.get(revision_key) or HfApi().model_info(repo).sha
        info[revision_key] = revision
        info['packages'] = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()
        if args.kind == 'sam3':
            if args.sam3_source is None:
                parser.error('--sam3-source required to record SAM3 code revision')
            info['sam3_code_revision'] = subprocess.check_output(
                ['git', '-C', str(args.sam3_source), 'rev-parse', 'HEAD'], text=True).strip()
            info['sam3_repository'] = subprocess.check_output(
                ['git', '-C', str(args.sam3_source), 'remote', 'get-url', 'origin'], text=True).strip()
            import sam3
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            import triton
            info['triton'] = triton.__version__
        else:
            from diffusers import Flux2KleinPipeline
            info['diffusers'] = importlib.metadata.version('diffusers')
        path.write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
        print(f'Environment lock: {path}')
    print(json.dumps(info, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
