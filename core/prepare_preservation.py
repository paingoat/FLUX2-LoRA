"""Select local Pico removal samples, segment with SAM3 and make single-image cues."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image, ImageDraw

from common import REPO_ROOT, fit_klein_size, load_runtime_env

DEFAULT_OUTPUT = REPO_ROOT / 'output/preservation_data'
MODES = ('text', 'scribble', 'mask')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def select_samples(metadata: Path, count: int, seed: int) -> list[dict]:
    if count < 1:
        raise ValueError('count must be positive')
    candidates, seen = [], set()
    for line in metadata.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('operation') != 'remove' or row.get('review_status') not in {'auto_accepted', 'manual_approved'}:
            continue
        identity = row['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', identity):
            raise ValueError(f'invalid sample ID: {identity}')
        key = row.get('source_key') or row.get('source_hash') or identity
        if key in seen:
            continue
        original = (metadata.parent / row['original_file_name']).resolve()
        if not original.is_file() or not row.get('prompt_object', '').strip():
            continue
        seen.add(key)
        candidates.append(dict(id=identity, original=str(original), source_sha256=digest(original),
                               source_key=key, split=row.get('split'), object_prompt=row['prompt_object'].strip(),
                               instruction=f"Remove {row['prompt_object'].strip().rstrip('.')}. "
                               'Naturally restore the background and preserve all other content.'))
    candidates.sort(key=lambda row: row['id'])
    random.Random(seed).shuffle(candidates)
    if len(candidates) < count:
        raise ValueError(f'only {len(candidates)} unique usable removal samples, requested {count}')
    return candidates[:count]


def read_mask(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        if image.size != size:
            raise ValueError(f'{path}: mask size {image.size} != {size}; do not resize reviewed masks implicitly')
        array = np.asarray(image.convert('L'))
    if not np.isin(array, [0, 255]).all():
        raise ValueError(f'{path}: mask must be binary 0/255')
    fraction = float((array > 0).mean())
    if not 0 < fraction < 0.9:
        raise ValueError(f'{path}: empty or excessively broad mask ({fraction:.3f})')
    return Image.fromarray(array.copy())


def overlays(original: Image.Image, mask: Image.Image, alpha: float = 0.4):
    if original.size != mask.size or not 0 < alpha < 1:
        raise ValueError('overlay requires aligned images and alpha in (0,1)')
    region = np.asarray(mask) > 127
    if not region.any():
        raise ValueError('empty mask')
    source = np.asarray(original.convert('RGB'))
    tinted = source.copy()
    tinted[region] = np.rint((1 - alpha) * source[region] + alpha * np.array([255, 0, 0])).astype(np.uint8)
    ys, xs = np.where(region)
    # One padded, deterministic freehand-style loop; no sample-dependent style selection.
    rng = random.Random(42)
    width = max(2, round(max(original.size) / 256))
    pad = max(6, width * 3)
    left, right = max(0, xs.min() - pad), min(original.width - 1, xs.max() + pad)
    top, bottom = max(0, ys.min() - pad), min(original.height - 1, ys.max() + pad)
    points = []
    corners = [(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]
    for start, end in zip(corners, corners[1:]):
        for t in np.linspace(0, 1, 16, endpoint=False):
            x = round(start[0] + t * (end[0] - start[0]) + rng.uniform(-1, 1))
            y = round(start[1] + t * (end[1] - start[1]) + rng.uniform(-1, 1))
            points.append((max(0, min(original.width - 1, x)), max(0, min(original.height - 1, y))))
    stroke = Image.new('L', original.size)
    ImageDraw.Draw(stroke).line(points + [points[0]], fill=255, width=width, joint='curve')
    scribble = original.convert('RGB').copy()
    scribble.paste((255, 0, 0), mask=stroke)
    return scribble, Image.fromarray(tinted), stroke


def prompts(instruction: str) -> dict[str, str]:
    return {'text': instruction,
            'scribble': instruction + ' The red loop identifies the object to remove. Remove the annotation too.',
            'mask': instruction + ' The translucent red region identifies the object to remove. Remove the annotation too.'}


def segment_image(processor, image: Image.Image, object_prompt: str, box=None):
    """SAM3 grounding API; keep feature extraction and prompts in the same BF16 context."""
    import torch
    if box is not None:
        if (len(box) != 4 or not all(0 <= x <= 1 for x in box) or min(box[2:]) <= 0
                or box[0] - box[2] / 2 < 0 or box[0] + box[2] / 2 > 1
                or box[1] - box[3] / 2 < 0 or box[1] + box[3] / 2 > 1):
            raise ValueError('override box must be normalized cx,cy,w,h entirely inside the image')
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        state = processor.set_image(image)
        result = processor.set_text_prompt(state=state, prompt=object_prompt)
        if box is not None:
            result = processor.add_geometric_prompt(box=box, label=True, state=state)
    return result


def materialize(output: Path, selection: dict, candidates: dict) -> None:
    records, review = [], []
    for sample in selection['samples']:
        folder = output / sample['id']
        original = Image.open(folder / 'original.png').convert('RGB')
        mask = read_mask(folder / 'object_mask.png', original.size)
        scribble, tint, stroke = overlays(original, mask, selection['alpha'])
        scribble.save(folder / 'scribble.png')
        tint.save(folder / 'mask_overlay.png')
        stroke.save(folder / 'stroke_mask.png')
        preview = Image.new('RGB', (original.width * 3, original.height))
        for i, image in enumerate((original, scribble, tint)):
            preview.paste(image, (i * original.width, 0))
        preview.thumbnail((1500, 600))
        preview.save(folder / 'preview.jpg')
        files = {'original': 'original.png', 'object_mask': 'object_mask.png',
                 'scribble': 'scribble.png', 'mask': 'mask_overlay.png', 'stroke_mask': 'stroke_mask.png'}
        record = {**sample, 'files': {key: f"{sample['id']}/{name}" for key, name in files.items()},
                  'hashes': {key: digest(folder / name) for key, name in files.items()},
                  'prompts': prompts(sample['instruction']), 'sam3': candidates[sample['id']],
                  'mask_fraction': float((np.asarray(mask) > 0).mean())}
        records.append(record)
        review.append(dict(id=sample['id'], mask_sha256=record['hashes']['object_mask'], approved='', notes=''))
    write_json(output / 'manifest.json', {'protocol': 'single-image-preservation-v1',
               'selection_sha256': digest(output / 'selection.json'), 'config': {k: v for k, v in selection.items() if k != 'samples'},
               'samples': records})
    with (output / 'mask_review.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id', 'mask_sha256', 'approved', 'notes'])
        writer.writeheader()
        writer.writerows(review)
    cards = ''.join(f'<h2>{html.escape(r["id"])}</h2><p>{html.escape(r["object_prompt"])}</p>'
                    f'<img style="max-width:100%" src="{r["id"]}/preview.jpg">' for r in records)
    (output / 'mask_review.html').write_text('<meta charset="utf-8"><title>Mask review</title>'
        '<h1>Original | Scribble | SAM3 mask overlay</h1><p>Inspect all masks, then set approved=1 '
        'in mask_review.csv. Regenerating overlays resets approvals.</p>' + cards, encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=REPO_ROOT / 'metadata.jsonl')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--longest', type=int, default=1024)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--select-only', action='store_true')
    parser.add_argument('--rebuild-overlays', action='store_true', help='Use locally corrected binary masks; resets approvals')
    parser.add_argument('--overrides', type=Path, help='JSON: sample ID -> object_prompt, candidate_index, normalized cxcywh box')
    args = parser.parse_args()
    if args.longest < 64 or args.longest > 1024 or not 0 < args.alpha < 1:
        parser.error('longest must be 64..1024 and alpha must be in (0,1)')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / 'selection.json'
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
    else:
        if args.rebuild_overlays:
            parser.error('select and generate masks before rebuilding overlays')
        selection = dict(count=args.count, seed=args.seed, longest=args.longest, alpha=args.alpha,
                         metadata_sha256=digest(args.metadata),
                         samples=select_samples(args.metadata.resolve(), args.count, args.seed))
        write_json(selection_path, selection)
    # Existing selection is authoritative: reruns cannot silently change the 20 source images.
    if args.select_only:
        print(f"Selected {len(selection['samples'])} fixed samples: {selection_path}")
        return 0
    candidates_path = output / 'sam3_candidates.json'
    if args.rebuild_overlays:
        materialize(output, selection, json.loads(candidates_path.read_text(encoding='utf-8')))
        return 0
    if (output / 'manifest.json').exists():
        parser.error('prepared data exists; use a new --output, or --rebuild-overlays after correcting masks')
    load_runtime_env()
    import torch
    from huggingface_hub import hf_hub_download
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from preservation_runtime import check_cuda, read_lock
    check_cuda()
    lock = read_lock('sam3')
    revision = lock['sam3_model_revision']
    checkpoint = hf_hub_download('facebook/sam3', 'sam3.pt', revision=revision)
    torch.manual_seed(selection['seed'])
    processor = Sam3Processor(build_sam3_image_model(checkpoint_path=checkpoint, load_from_HF=False,
        device='cuda', eval_mode=True, enable_inst_interactivity=False), device='cuda', confidence_threshold=0.5)
    overrides = json.loads(args.overrides.read_text(encoding='utf-8')) if args.overrides else {}
    candidates = {}
    for sample in selection['samples']:
        folder = output / sample['id']
        folder.mkdir(exist_ok=True)
        source = Path(sample['original'])
        if digest(source) != sample['source_sha256']:
            raise ValueError(f'{source}: source changed since selection')
        with Image.open(source) as image:
            original = image.convert('RGB').resize(fit_klein_size(*image.size, selection['longest']), Image.Resampling.LANCZOS)
        original.save(folder / 'original.png')
        override = overrides.get(sample['id'], {})
        object_prompt = override.get('object_prompt', sample['object_prompt'])
        result = segment_image(processor, original, object_prompt, override.get('box'))
        scores = result['scores'].detach().float().cpu().numpy().reshape(-1)
        masks = result['masks'].detach().cpu().numpy().reshape(-1, original.height, original.width)
        if not len(scores):
            raise ValueError(f"{sample['id']}: SAM3 found no masks; retry with --overrides")
        index = int(override.get('candidate_index', int(scores.argmax())))
        if not 0 <= index < len(scores):
            raise ValueError('candidate_index out of range')
        for i, mask in enumerate(masks):
            Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f'candidate_{i}.png')
        Image.fromarray(masks[index].astype(np.uint8) * 255).save(folder / 'object_mask.png')
        candidates[sample['id']] = dict(model_revision=revision, code_revision=lock['sam3_code_revision'],
            repository=lock['sam3_repository'], confidence_threshold=0.5, environment=lock,
            object_prompt=object_prompt, scores=scores.tolist(), candidate_index=index, box=override.get('box'))
        write_json(candidates_path, candidates)
        print(f"{sample['id']}: {len(scores)} masks, selected {index}", flush=True)
    materialize(output, selection, candidates)
    print(f'Review masks at {output / "mask_review.html"}; approve in mask_review.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
