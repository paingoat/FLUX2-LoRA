"""Prepare single-image cues from existing Pico masks or reviewed SAM3 predictions."""
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
PROMPT_PROTOCOL = 'spatial-only-v2'


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
                               existing_mask=str((metadata.parent / row.get('mask_file_name',
                                   str(Path(row['original_file_name']).with_name('mask.png')))).resolve()),
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
            'scribble': 'Remove the object indicated by the red loop. Naturally restore the background, '
                        'remove the annotation, and preserve all other content.',
            'mask': 'Remove the object highlighted by the translucent red region. Naturally restore the background, '
                    'remove the annotation, and preserve all other content.'}


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
                  'prompts': prompts(sample['instruction']),
                  'mask_source': selection.get('mask_source', 'sam3'), 'mask_provenance': candidates[sample['id']],
                  'mask_fraction': float((np.asarray(mask) > 0).mean())}
        records.append(record)
        review.append(dict(id=sample['id'], mask_sha256=record['hashes']['object_mask'], approved='', notes=''))
    write_json(output / 'manifest.json', {'protocol': 'single-image-preservation-v1',
               'prompt_protocol': PROMPT_PROTOCOL,
               'selection_sha256': digest(output / 'selection.json'), 'config': {k: v for k, v in selection.items() if k != 'samples'},
               'samples': records})
    with (output / 'mask_review.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id', 'mask_sha256', 'approved', 'notes'])
        writer.writeheader()
        writer.writerows(review)
    cards = ''.join(f'<h2>{html.escape(r["id"])}</h2><p>{html.escape(r["object_prompt"])}</p>'
                    f'<img style="max-width:100%" src="{r["id"]}/preview.jpg">' for r in records)
    (output / 'mask_review.html').write_text('<meta charset="utf-8"><title>Mask review</title>'
        f'<h1>Original | Scribble | {selection.get("mask_source", "sam3")} mask overlay</h1><p>Inspect all masks, then set approved=1 '
        'in mask_review.csv. Regenerating overlays resets approvals.</p>' + cards, encoding='utf-8')


def prepare_masks(output: Path, selection: dict, overrides: dict, predictor=None, sam3_info=None) -> list[dict]:
    """Checkpoint each success; retry only changed/failed samples, never mix mask sources."""
    candidates_path = output / 'sam3_candidates.json'
    candidates = json.loads(candidates_path.read_text(encoding='utf-8')) if candidates_path.exists() else {}
    errors = []
    mode = selection.get('mask_source', 'sam3')
    for sample in selection['samples']:
        folder = output / sample['id']
        folder.mkdir(exist_ok=True)
        try:
            source = Path(sample['original'])
            if digest(source) != sample['source_sha256']:
                raise ValueError('source changed since selection')
            with Image.open(source) as image:
                source_size = image.size
                original = image.convert('RGB').resize(fit_klein_size(*image.size, selection['longest']), Image.Resampling.LANCZOS)
            override = overrides.get(sample['id'], {})
            object_prompt = override.get('object_prompt', sample['object_prompt'])
            mask_path = Path(sample.get('existing_mask', str(source.with_name('mask.png'))))
            spec = dict(source=mode, original_sha256=digest(source), size=list(original.size),
                        overrides=override, object_prompt=object_prompt,
                        model_revision=(sam3_info or {}).get('sam3_model_revision'),
                        code_revision=(sam3_info or {}).get('sam3_code_revision'),
                        input_mask_sha256=digest(mask_path) if mode == 'existing' else None)
            previous = candidates.get(sample['id'], {})
            saved_original, saved_mask = folder / 'original.png', folder / 'object_mask.png'
            if (previous.get('preparation') == spec and saved_mask.is_file() and saved_original.is_file()
                    and previous.get('mask_sha256') == digest(saved_mask)
                    and previous.get('original_sha256') == digest(saved_original)):
                read_mask(saved_mask, original.size)
                print(f"{sample['id']}: cached {mode} mask", flush=True)
                continue
            original.save(saved_original)
            if mode == 'existing':
                # Validate at source size, then use nearest-neighbor to keep the mask binary.
                mask = read_mask(mask_path, source_size).resize(original.size, Image.Resampling.NEAREST)
                provenance = dict(source='existing', input_mask=str(mask_path),
                                  input_mask_sha256=spec['input_mask_sha256'], method='repository cue/change mask; not SAM3')
            else:
                result = predictor(original, object_prompt, override.get('box'))
                scores = result['scores'].detach().float().cpu().numpy().reshape(-1)
                masks = result['masks'].detach().cpu().numpy().reshape(-1, original.height, original.width)
                if not len(scores):
                    raise ValueError('SAM3 found no masks; add text/box override and rerun')
                index = int(override.get('candidate_index', int(scores.argmax())))
                if not 0 <= index < len(scores):
                    raise ValueError('candidate_index out of range')
                for i, candidate in enumerate(masks):
                    Image.fromarray(candidate.astype(np.uint8) * 255).save(folder / f'candidate_{i}.png')
                mask = Image.fromarray(masks[index].astype(np.uint8) * 255)
                provenance = dict(source='sam3', environment=sam3_info, scores=scores.tolist(),
                                  candidate_index=index, object_prompt=object_prompt, box=override.get('box'), confidence_threshold=0.5)
                print(f"{sample['id']}: {len(scores)} masks, selected {index}", flush=True)
            mask.save(saved_mask)
            read_mask(saved_mask, original.size)
            candidates[sample['id']] = dict(**provenance, preparation=spec,
                mask_sha256=digest(saved_mask), original_sha256=digest(saved_original))
            write_json(candidates_path, candidates)
            if mode == 'existing':
                print(f"{sample['id']}: existing mask prepared", flush=True)
        except (ValueError, OSError, RuntimeError) as exc:
            errors.append(dict(id=sample['id'], error=str(exc)))
            print(f"{sample['id']}: ERROR {exc}", flush=True)
            if mode == 'sam3':
                import torch
                torch.cuda.empty_cache()
    with (output / 'prepare_errors.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id', 'error'])
        writer.writeheader()
        writer.writerows(errors)
    if not errors:
        materialize(output, selection, candidates)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=REPO_ROOT / 'metadata.jsonl')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--longest', type=int, default=1024)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--mask-source', choices=['sam3', 'existing'], default='sam3')
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
        if not args.metadata.is_file():
            parser.error(f'missing metadata: {args.metadata}. Download and extract the Pico input bundle first; see core/preservation.md')
        selection = dict(count=args.count, seed=args.seed, longest=args.longest, alpha=args.alpha, mask_source=args.mask_source,
                         metadata_sha256=digest(args.metadata),
                         samples=select_samples(args.metadata.resolve(), args.count, args.seed))
        write_json(selection_path, selection)
    if selection.get('mask_source', 'sam3') != args.mask_source:
        parser.error('mask source differs from saved selection; use a new --output to keep SAM3/existing experiments separate')
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
    overrides = json.loads(args.overrides.read_text(encoding='utf-8')) if args.overrides else {}
    if args.mask_source == 'existing':
        errors = prepare_masks(output, selection, overrides)
        print(f'{len(errors)} failures; see {output / "prepare_errors.csv"}' if errors else f'Ready: {output / "manifest.json"}')
        return int(bool(errors))
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
    errors = prepare_masks(output, selection, overrides,
        predictor=lambda image, prompt, box: segment_image(processor, image, prompt, box), sam3_info=lock)
    if errors:
        print(f'{len(errors)} failures recorded in {output / "prepare_errors.csv"}. Fix overrides and rerun; successful masks are reused.')
        return 1
    print(f'Review masks at {output / "mask_review.html"}; approve in mask_review.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
