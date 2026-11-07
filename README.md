# FLUX.2 Klein two-input removal LoRA

Train with clean original + annotated reference, supervised by an edited target.

- [RunPod: template, screen, upload, setup and reconnect](core/runpod.md)
- [Training, validation, human scoring and VIBE evaluation](core/guide.md)
- [Single-image preservation pilot: SAM3 masks, 20 local Pico images, RTX 5090](core/preservation.md)
- [RTX 5090 fine-tuning: region-weighted loss, existing masks and A/B evaluation](core/region_5090.md)

Git contains source code, tests, mapping and text metadata. Images, imported
`samples/`, model weights, environments, caches and outputs are kept outside Git.
Upload the mapped dataset separately before running `core/prepare_dataset.py`.
