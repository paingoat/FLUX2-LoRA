"""Two-input removal evaluation. Select on validation; test/VIBE stay separate."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw
from common import (DISTILLED_REPO, OUTPUT_DIR, PREPARED_DIR, PROMPT,
                    assert_not_old_removal, edit_prompt, load_runtime_env, resize_inputs)

GUIDANCE, STEPS, SEED = 4.0, 4, 42
METRICS = ('scribble_residue', 'preserve_psnr', 'preserve_ssim', 'target_lpips', 'edit_lpips')


def load_manifest(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(records, list) or not records:
        raise ValueError(f'{path}: expected nonempty manifest')
    identities = set()
    for record in records:
        record.setdefault('dataset', 'internal')
        identity = (record['dataset'], record['index'])
        if identity in identities or any(not re.fullmatch(r'[A-Za-z0-9_-]+', p) for p in identity):
            raise ValueError(f'duplicate or invalid sample identity: {identity}')
        identities.add(identity)
        if record['split'] not in {'validation', 'test', 'external'}:
            raise ValueError('benchmark accepts held-out data only')
        for key in ('original', 'reference', 'target', 'stroke_mask'):
            if record.get(key):
                candidate = Path(record[key])
                if not candidate.is_absolute():
                    candidate = path.parent / candidate
                candidate = assert_not_old_removal(candidate)
                if not candidate.is_file():
                    raise ValueError(f'missing {key}: {candidate}')
                record[key] = str(candidate)
        if not record.get('original') or not record.get('reference'):
            raise ValueError('manifest needs original and reference; prepare it again')
        expected = edit_prompt(record['instruction']) if record['dataset'] == 'vibe_removal' else PROMPT
        if record.get('prompt') != expected:
            raise ValueError(f'{identity}: prompt differs from two-input protocol')
    return records


def edit_region(stroke: np.ndarray) -> np.ndarray:
    """Approximate edit region from annotation geometry, not object ground truth."""
    import cv2
    size = max(3, round(max(stroke.shape) / 100) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(stroke.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    region = np.zeros_like(closed)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour) >= 3:
            # Enclose open/clipped circles; disconnected marks stay separate.
            cv2.drawContours(region, [cv2.convexHull(contour)], -1, 1, cv2.FILLED)
    return cv2.dilate(region | closed, kernel).astype(bool)


def psnr(left: np.ndarray, right: np.ndarray) -> float:
    mse = float(np.mean((left.astype(np.float32) - right.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def masked_ssim(original: np.ndarray, output: np.ndarray, edit: np.ndarray) -> float | None:
    import cv2
    from skimage.metrics import structural_similarity
    _, score_map = structural_similarity(original, output, channel_axis=2, data_range=255, full=True)
    # Only 7x7 windows lying wholly outside the edit and image border count.
    valid = cv2.erode((~edit).astype(np.uint8), np.ones((7, 7), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return float(score_map.mean(axis=2)[valid].mean()) if valid.any() else None


def score_pair(original: Image.Image, reference: Image.Image, output: Image.Image,
               target: Image.Image | None, lpips_fn=None, stroke_mask=None) -> dict:
    src, ref, out = [np.asarray(image.convert('RGB')) for image in (original, reference, output)]
    if src.shape != ref.shape or src.shape != out.shape:
        raise ValueError('metric images must have identical dimensions')
    difference = np.abs(ref.astype(np.int16) - src.astype(np.int16)).max(axis=2)
    stroke = difference > 8 if stroke_mask is None else np.asarray(stroke_mask) > 127
    if stroke.shape != src.shape[:2]:
        raise ValueError('stroke mask size differs from original')
    edit = edit_region(stroke)
    result = dict.fromkeys(METRICS)
    result['edit_fraction'] = float(edit.mean())
    if stroke.any():
        marker_distance = np.abs(out.astype(np.int16) - ref.astype(np.int16)).max(axis=2)
        strong_stroke = stroke & (difference >= 40)
        if strong_stroke.any():
            result['scribble_residue'] = float((marker_distance[strong_stroke] <= 30).mean())
    if stroke.any() and 0 < edit.mean() < 0.9:
        result['preserve_psnr'] = psnr(src[~edit], out[~edit])
        result['preserve_ssim'] = masked_ssim(src, out, edit)
    if target is not None and lpips_fn is not None:
        import torch
        device = next(lpips_fn.parameters()).device
        def distance(a, b):
            def tensor(array):
                image = Image.fromarray(array)
                if min(image.size) < 64:
                    factor = 64 / min(image.size)
                    image = image.resize(tuple(round(side * factor) for side in image.size))
                data = np.asarray(image).copy()
                return torch.from_numpy(data).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1
            with torch.inference_mode():
                return float(lpips_fn(tensor(a), tensor(b)).item())
        tgt = np.asarray(target.convert('RGB'))
        if tgt.shape != out.shape:
            raise ValueError('target size differs from output')
        result['target_lpips'] = distance(out, tgt)
        if stroke.any() and edit.any():
            ys, xs = np.where(edit)
            crop = np.s_[ys.min():ys.max()+1, xs.min():xs.max()+1]
            result['edit_lpips'] = distance(out[crop], tgt[crop])
    return result


def checkpoint_step(path: Path | None) -> int:
    if path is None:
        return -1
    match = re.search(r'_(\d{6,})$', path.stem)
    if match:
        return int(match.group(1))
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            return int(json.loads((handle.metadata() or {})['training_info'])['step'])
    except (OSError, ValueError, KeyError, ImportError):
        return -1


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['status'])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def contact_sheet(images: list[Image.Image], label: str, path: Path) -> None:
    thumbs = [image.copy() for image in images]
    canvas = Image.new('RGB', (sum(image.width for image in thumbs), max(image.height for image in thumbs) + 30), 'white')
    x = 0
    for image in thumbs:
        canvas.paste(image, (x, 30))
        x += image.width
    ImageDraw.Draw(canvas).text((5, 5), label, fill='black')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_report(rows: list[dict], destination: Path) -> None:
    grouped = {}
    for row in rows:
        key = tuple(row[field] for field in ('dataset', 'split', 'checkpoint', 'resolution'))
        grouped.setdefault(key, []).append(row)
    lines = ['# Two-input removal benchmark', '',
             'Original + annotated reference; identical prompts and seeds for baseline and LoRA.',
             'Choose checkpoints using internal validation only. Test and VIBE are final evaluation.',
             'Automatic metrics do not establish removal success. Complete the blind human review.',
             'Preservation uses clean original outside a region estimated from annotation geometry.',
             'LPIPS is advisory: targets can differ outside the edit. Missing targets stay blank.', '',
             '| Dataset | Split | Checkpoint | Resolution | OK/attempted | Residue ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Edit crop LPIPS ↓ |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key, items in sorted(grouped.items()):
        ok = [row for row in items if row['status'] == 'ok']
        values = []
        for metric in METRICS:
            numbers = [float(row[metric]) for row in ok if row.get(metric) not in (None, '')]
            values.append(f'{np.mean(numbers):.4f} (n={len(numbers)})' if numbers else '—')
        lines.append('| ' + ' | '.join(map(str, key)) + f' | {len(ok)}/{len(items)} | ' + ' | '.join(values) + ' |')
    lines += ['', 'Do not compare means if runs have different successful sample coverage.',
              'Residue counts marker-like colors at original stroke positions; displaced strokes are not detected.',
              'Edit LPIPS uses a bounding crop with context, not pixel-masked LPIPS.']
    destination.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def generate(pipe, images, prompt: str, seed: int) -> Image.Image:
    import torch
    generator = torch.Generator(device='cuda').manual_seed(seed)
    with torch.inference_mode():
        return pipe(prompt=prompt, image=images, num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE, width=images[0].width, height=images[0].height,
                    generator=generator).images[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, nargs='+', default=[PREPARED_DIR / 'eval/manifest.json'])
    parser.add_argument('--checkpoints', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--checkpoint', type=Path, nargs='+')
    parser.add_argument('--split', nargs='+', choices=['validation', 'test', 'external'], default=['validation'])
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR / 'benchmark')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--resolutions', type=int, nargs='+', default=[512, 1024])
    parser.add_argument('--seeds', '--seed', type=int, nargs='+', default=[SEED])
    args = parser.parse_args()
    records = [record for path in args.manifest for record in load_manifest(path) if record['split'] in args.split]
    if not records:
        parser.error('no samples match the requested splits')
    identities = [(r['dataset'], r['index']) for r in records]
    if len(set(identities)) != len(identities):
        parser.error('duplicate samples across manifests')
    if any(split != 'validation' for split in args.split) and not (args.checkpoint or args.baseline_only):
        parser.error('select --checkpoint explicitly for test/external evaluation')
    checkpoints = [] if args.baseline_only else (args.checkpoint or sorted(args.checkpoints.glob('*.safetensors'), key=checkpoint_step))
    if any(not path.is_file() for path in checkpoints):
        parser.error('checkpoint file not found')
    if len({p.stem for p in checkpoints}) != len(checkpoints):
        parser.error('checkpoint names must be unique')
    if not checkpoints and not (args.baseline or args.baseline_only):
        parser.error('no checkpoints found; use --baseline-only for an untrained baseline')
    if any(size < 64 for size in args.resolutions):
        parser.error('resolutions must be at least 64')
    if (args.output / 'results.csv').exists():
        parser.error('results exist; choose a new --output to preserve the previous run')
    load_runtime_env()
    import torch
    from diffusers import Flux2KleinPipeline
    from tqdm import tqdm
    if not torch.cuda.is_available():
        raise SystemExit('benchmark needs CUDA; run on the pod')
    pipe = Flux2KleinPipeline.from_pretrained(DISTILLED_REPO, torch_dtype=torch.bfloat16).to('cuda')
    lpips_fn = None
    if any(record.get('target') for record in records):
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to('cuda').eval()
        for parameter in lpips_fn.parameters():
            parameter.requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {'model': DISTILLED_REPO, 'steps': STEPS, 'guidance': GUIDANCE,
              'reference_order': ['original', 'reference'], 'seeds': args.seeds, 'resolutions': args.resolutions,
              'manifests': [{'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in args.manifest],
              'checkpoints': [str(p.resolve()) for p in checkpoints]}
    (args.output / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    jobs = ([None] if args.baseline or args.baseline_only else []) + checkpoints
    rows, loaded = [], False
    for checkpoint in jobs:
        if loaded:
            pipe.unload_lora_weights()
        loaded = checkpoint is not None
        tag = checkpoint.stem if checkpoint else 'baseline'
        if loaded:
            pipe.load_lora_weights(str(checkpoint), adapter_name='removal')
            pipe.set_adapters(['removal'], adapter_weights=[1.0])
        work = [(record, resolution, seed) for record in records
                for resolution in dict.fromkeys(args.resolutions) for seed in dict.fromkeys(args.seeds)]
        for record, resolution, seed in tqdm(work, desc=tag, unit='image'):
            key = f"{record['dataset']}/{record['index']}/{resolution}/{seed}/{tag}"
            review_id = hashlib.sha256(key.encode()).hexdigest()[:16]
            row = {'review_id': review_id, 'dataset': record['dataset'], 'split': record['split'],
                   'index': record['index'], 'checkpoint': tag, 'step': checkpoint_step(checkpoint),
                   'resolution': resolution, 'seed': seed, 'status': 'ok', 'error': '',
                   **dict.fromkeys(METRICS, ''), 'edit_fraction': '',
                   'original': record['original'], 'reference': record['reference'],
                   'prompt': record['prompt'], 'image': '', 'sheet': ''}
            try:
                with Image.open(record['original']) as src, Image.open(record['reference']) as ref:
                    original, reference = resize_inputs(src, ref, resolution)
                target = None
                if record.get('target'):
                    with Image.open(record['target']) as image:
                        target = image.convert('RGB').resize(original.size, Image.Resampling.LANCZOS)
                stroke_mask = None
                if record.get('stroke_mask'):
                    with Image.open(record['stroke_mask']) as image:
                        stroke_mask = np.asarray(image.convert('L').resize(original.size, Image.Resampling.NEAREST))
                output = generate(pipe, [original, reference], record['prompt'], seed)
                if output.size != original.size:
                    raise ValueError(f'unexpected output dimensions {output.size}')
                image_path = args.output / 'images' / f'{review_id}.png'
                image_path.parent.mkdir(parents=True, exist_ok=True)
                output.save(image_path)
                row['image'] = str(image_path.resolve())
                row.update({k: '' if v is None else v for k, v in
                            score_pair(original, reference, output, target, lpips_fn, stroke_mask).items()})
                sheet = args.output / 'sheets' / f'{review_id}.jpg'
                contact_sheet([original, reference, output] + ([target] if target else []),
                              'original | reference | output' + (' | target' if target else ''), sheet)
                row['sheet'] = str(sheet.resolve())
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row.update(status='error', error='CUDA OOM')
            except (ValueError, OSError) as exc:
                row.update(status='error', error=str(exc))
            rows.append(row)
            write_csv(args.output / 'results.csv', rows)
    write_report(rows, args.output / 'report.md')
    from review_benchmark import create_review
    create_review(args.output / 'results.csv')
    print(f'Results, report and blind review: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
