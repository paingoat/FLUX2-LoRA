"""Explicit GPU preflight; no models downloaded and no fine-tuning started."""
import json
from pathlib import Path
import sys


def main():
    import torch
    import bitsandbytes as bnb
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; use the RTX 5090 environment')
    if torch.cuda.get_device_capability() != (12, 0):
        raise SystemExit('this profile expects RTX 5090 / compute capability 12.0')
    if 'sm_120' not in torch.cuda.get_arch_list():
        raise SystemExit('installed PyTorch wheel does not include sm_120')
    # Check actual BF16 kernels/backprop and the same optimizer used by training.
    parameter = torch.nn.Parameter(torch.randn(256, 256, device='cuda', dtype=torch.bfloat16))
    optimizer = bnb.optim.AdamW8bit([parameter], lr=1e-4)
    loss = (parameter @ parameter.T).float().square().mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    if not torch.isfinite(parameter).all():
        raise SystemExit('nonfinite BF16/8-bit optimizer check')
    info = dict(gpu=torch.cuda.get_device_name(), capability=[12, 0], torch=torch.__version__,
                cuda=torch.version.cuda, bitsandbytes=bnb.__version__,
                total_vram_gib=torch.cuda.get_device_properties(0).total_memory/1024**3,
                bf16_backward_adamw8bit='passed')
    destination = Path(sys.prefix) / 'region-gpu-check.json'
    destination.write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info, indent=2))
    print(f'Saved {destination}; full-model training VRAM still needs verification.')


if __name__ == '__main__':
    main()
