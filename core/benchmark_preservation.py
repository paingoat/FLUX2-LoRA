"""Compare text, scribble and filled-mask overlays using one PIL image per call."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image, ImageFilter

from benchmark import contact_sheet, write_csv
from common import DISTILLED_REPO, load_runtime_env
from prepare_preservation import DEFAULT_OUTPUT, MODES, PROMPT_PROTOCOL, digest, prompts, read_mask, write_json


def psnr_outside(original: Image.Image, output: Image.Image, mask: Image.Image, dilation: int = 0) -> float:
    if original.size != output.size or original.size != mask.size or dilation < 0:
        raise ValueError('PSNR requires identical image/mask sizes and nonnegative dilation')
    if dilation:
        mask = mask.filter(ImageFilter.MaxFilter(2 * dilation + 1))
    outside = np.asarray(mask.convert('L')) <= 127
    if not outside.any():
        raise ValueError('no pixels outside evaluation mask')
    difference = np.asarray(original.convert('RGB'), dtype=np.float64) - np.asarray(output.convert('RGB'), dtype=np.float64)
    mse = float(np.mean(difference[outside] ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def load_manifest(path: Path, allow_unreviewed: bool = False) -> tuple[dict, list[dict]]:
    manifest = json.loads(path.read_text(encoding='utf-8'))
    records = manifest.get('samples', [])
    if manifest.get('protocol') != 'single-image-preservation-v1' or not records:
        raise ValueError('expected nonempty single-image preservation manifest')
    if manifest.get('prompt_protocol') != PROMPT_PROTOCOL:
        raise ValueError('old prompt protocol; regenerate manifest with prepare_preservation.py '
                         '--rebuild-overlays and the original --mask-source/--output before rerunning')
    if len(records) != manifest['config']['count']:
        raise ValueError('sample count differs from the fixed selection')
    if digest(path.parent / 'selection.json') != manifest['selection_sha256']:
        raise ValueError('selection changed; prepare data again')
    with (path.parent / 'mask_review.csv').open(newline='', encoding='utf-8-sig') as handle:
        approvals = list(csv.DictReader(handle))
    if len({r['id'] for r in approvals}) != len(approvals):
        raise ValueError('duplicate mask review IDs')
    by_id = {row['id']: row for row in approvals}
    seen = set()
    for record in records:
        identity = record['id']
        if identity in seen or not identity.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f'duplicate/invalid sample ID: {identity}')
        seen.add(identity)
        approval = by_id.get(identity, {})
        if approval.get('mask_sha256') != record['hashes']['object_mask']:
            raise ValueError(f'{identity}: mask review hash differs; rebuild and review again')
        reviewed = approval.get('approved') == '1'
        provisional = (allow_unreviewed and record.get('mask_source') == 'existing'
                       and approval.get('approved') == '')
        if not reviewed and not provisional:
            raise ValueError(f'{identity}: mask pending, rejected or changed; inspect mask_review.html and approve CSV')
        record['mask_review_status'] = 'approved' if reviewed else 'unreviewed-existing-pilot'
        paths = {}
        for key, value in record['files'].items():
            candidate = (path.parent / value).resolve()
            if not candidate.is_relative_to(path.parent.resolve()) or not candidate.is_file():
                raise ValueError(f'invalid or missing prepared asset: {value}')
            if digest(candidate) != record['hashes'][key]:
                raise ValueError(f'{identity}/{key}: asset changed; rebuild overlays and review again')
            paths[key] = candidate
        with Image.open(paths['original']) as source:
            size = source.size
        read_mask(paths['object_mask'], size)
        for key in ('scribble', 'mask', 'stroke_mask'):
            with Image.open(paths[key]) as image:
                if image.size != size:
                    raise ValueError(f'{identity}/{key}: dimensions differ')
        if (not isinstance(record.get('prompt_original'), str) or not record['prompt_original'].strip()
                or record['instruction'] != record['prompt_original']):
            raise ValueError('text must use the unchanged dataset prompt_original; regenerate manifest')
        if record['prompts'] != prompts(record['prompt_original']):
            raise ValueError('prompts differ from spatial-only protocol; regenerate manifest with --rebuild-overlays')
        record['paths'] = paths
    return manifest, records


def generate(pipe, image: Image.Image, prompt: str, seed: int):
    import torch
    if not isinstance(image, Image.Image):
        raise ValueError('single-image experiment requires one PIL image, not a reference list')
    with torch.inference_mode():
        return pipe(image=image, prompt=prompt, width=image.width, height=image.height,
                    num_inference_steps=4, guidance_scale=1.0,
                    generator=torch.Generator(device='cuda').manual_seed(seed)).images[0]


def aggregate(values: list[float]) -> str:
    if not values:
        return 'n/a'
    if not all(math.isfinite(x) for x in values):
        return f'median={statistics.median(values):.3f}; +inf={sum(x == math.inf for x in values)}; n={len(values)}'
    return f'mean={statistics.mean(values):.3f}; median={statistics.median(values):.3f}; n={len(values)}'


def report(rows: list[dict], output: Path) -> None:
    grouped = {}
    for row in rows:
        grouped.setdefault((row['index'], row['seed']), {})[row['mode']] = row
    complete = [group for group in grouped.values()
                if set(group) == set(MODES) and all(row['status'] == 'ok' for row in group.values())]
    lines = ['# Single-image preservation pilot', '',
             'One clean/overlaid image per call. No LoRA, no post-compositing. '
             'PSNR compares raw output with the clean original outside the same fixed mask.',
             f'Prompt protocol: {PROMPT_PROTOCOL}. Text uses the unchanged dataset prompt_original; '
             'scribble/mask use fixed spatial instructions without object metadata.',
             'All three modes use the same sample/seed pairs below. '
             'PSNR alone does not establish removal success; use review.html.', '',
             f'Complete triples: {len(complete)}/{len(grouped)}. Failed attempts: '
             f'{sum(row["status"] != "ok" for row in rows)}.', '',
             '| Mode | PSNR outside object (dB) | PSNR outside mask + 8 px (dB) |',
             '|---|---|---|']
    sources = sorted({row.get('mask_source', 'sam3') for row in rows})
    lines[2:2] = [f'Mask source: {", ".join(sources)}. Existing masks are repository cue/change masks, '
                  'not precise SAM3 object masks. '
                  f'Unreviewed attempts: {sum(row.get("mask_review_status") == "unreviewed-existing-pilot" for row in rows)}.', '']
    for mode in MODES:
        lines.append(f'| {mode} | ' + ' | '.join(aggregate([g[mode][metric] for g in complete])
                    for metric in ('preserve_psnr', 'preserve_psnr_d8')) + ' |')
    lines += ['', 'Paired deltas (mode minus text), finite pairs only; +inf is never clipped:']
    for mode in ('scribble', 'mask'):
        deltas = [g[mode]['preserve_psnr'] - g['text']['preserve_psnr'] for g in complete
                  if math.isfinite(g[mode]['preserve_psnr']) and math.isfinite(g['text']['preserve_psnr'])]
        lines.append(f'- {mode}: {aggregate(deltas)}')
    lines += ['', 'This pilot uses existing Pico samples, including original train splits, with an '
              'untrained pretrained model. It is not a held-out LoRA benchmark.',
              'The 8 px dilation is measured at prepared inference resolution. Scribble pixels outside '
              'the object remain part of the primary preservation evaluation.',
              'If multiple seeds are used, n counts image/seed pairs, not independent source images.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def gallery(records: list[dict], rows: list[dict], output: Path) -> None:
    cards = []
    for record in records:
        for seed in sorted({row['seed'] for row in rows if row['index'] == record['id']}):
            matched = {row['mode']: row for row in rows if row['index'] == record['id'] and row['seed'] == seed}
            images = [Image.open(record['paths'][key]).convert('RGB') for key in ('original', 'object_mask')]
            for mode in MODES:
                row = matched.get(mode, {})
                images.append(Image.open(row['image']).convert('RGB') if row.get('status') == 'ok'
                              else Image.new('RGB', images[0].size, 'gray'))
            for image in images:
                image.thumbnail((320, 320))
            name = f"gallery/{record['id']}_{seed}.jpg"
            contact_sheet(images, 'original | evaluation mask | text | scribble | mask overlay', output / name)
            cards.append(f'<h2>{html.escape(record["id"])} seed={seed}</h2><img style="max-width:100%" src="{name}">')
    (output / 'gallery.html').write_text('<meta charset="utf-8"><title>Preservation pilot</title>' + ''.join(cards), encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_OUTPUT / 'manifest.json')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT.parent / 'preservation_run')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--limit', type=int, help='Smoke test: first N selected samples')
    parser.add_argument('--validate-only', action='store_true', help='Check assets and mask approvals without CUDA')
    parser.add_argument('--allow-unreviewed', action='store_true', help='Provisional existing-mask pilot only; never permits rejected or unreviewed SAM3 masks')
    parser.add_argument('--offload', action='store_true', help='Use model CPU offload for every mode')
    args = parser.parse_args()
    manifest, records = load_manifest(args.manifest.resolve(), allow_unreviewed=args.allow_unreviewed)
    if args.limit is not None:
        if args.limit < 1:
            parser.error('--limit must be positive')
        records = records[:args.limit]
    if args.validate_only:
        print(f'Validated {len(records)} samples; assets and dimensions verified; '
              f'{sum(r["mask_review_status"] != "approved" for r in records)} unreviewed existing masks')
        return 0
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('output is not empty; choose a new --output to preserve existing runs')
    load_runtime_env()
    import torch
    import diffusers
    from diffusers import Flux2KleinPipeline
    from preservation_runtime import check_cuda, read_lock
    runtime = check_cuda()
    lock = read_lock('flux')
    revision = lock['flux_model_revision']
    pipe = Flux2KleinPipeline.from_pretrained(DISTILLED_REPO, revision=revision, torch_dtype=torch.bfloat16)
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to('cuda')
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=DISTILLED_REPO, revision=revision, steps=4, guidance=1.0, dtype='bfloat16',
                  prompt_protocol=PROMPT_PROTOCOL,
                  text_prompt_source='dataset.prompt_original',
                  reference_count=1, offload=args.offload, seeds=list(dict.fromkeys(args.seeds)),
                  manifest_sha256=digest(args.manifest), sample_ids=[r['id'] for r in records],
                  mask_source=manifest['config'].get('mask_source', 'sam3'), allow_unreviewed=args.allow_unreviewed,
                  runtime=runtime, environment_lock=lock, diffusers=diffusers.__version__,
                  cache={key: value for key, value in os.environ.items()
                         if key in {'HF_HOME', 'HF_HUB_CACHE', 'HF_XET_CACHE', 'HF_ASSETS_CACHE'}})
    write_json(output / 'run_config.json', config)
    rows = []
    for record in records:
        original = Image.open(record['paths']['original']).convert('RGB')
        mask = read_mask(record['paths']['object_mask'], original.size)
        for seed in config['seeds']:
            for mode in MODES:
                identity = f"{record['id']}_{seed}_{mode}"
                row = dict(review_id=identity, dataset='pico_preservation', split=record.get('split'),
                           index=record['id'], seed=seed, mode=mode, status='error', error='',
                           preserve_psnr='', preserve_psnr_d8='', elapsed_sec='', peak_vram_allocated_mb='',
                           mask_fraction=record['mask_fraction'], prompt=record['prompts'][mode],
                           mask_source=record.get('mask_source', 'sam3'), mask_review_status=record['mask_review_status'],
                           image='', sheet='', checkpoint='pretrained', resolution=max(original.size))
                try:
                    image = original if mode == 'text' else Image.open(record['paths'][mode]).convert('RGB')
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    result = generate(pipe, image, row['prompt'], seed)
                    torch.cuda.synchronize()
                    row['elapsed_sec'] = round(time.perf_counter() - started, 3)
                    row['peak_vram_allocated_mb'] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
                    row['preserve_psnr'] = psnr_outside(original, result, mask)
                    row['preserve_psnr_d8'] = psnr_outside(original, result, mask, 8)
                    destination = output / 'images' / f'{identity}.png'
                    destination.parent.mkdir(exist_ok=True)
                    result.save(destination)
                    sheet = output / 'sheets' / f'{identity}.jpg'
                    thumbs = [im.copy() for im in (original, image, result)]
                    for thumb in thumbs:
                        thumb.thumbnail((512, 512))
                    contact_sheet(thumbs, 'original | input | output', sheet)
                    row.update(status='ok', image=str(destination), sheet=str(sheet))
                except (RuntimeError, ValueError, OSError) as exc:
                    row['error'] = str(exc)
                    torch.cuda.empty_cache()
                rows.append(row)
                write_csv(output / 'results.csv', rows)
                print(f"{identity}: {row['status']} {row['error']}", flush=True)
    report(rows, output)
    gallery(records, rows, output)
    from review_benchmark import create_review
    create_review(output / 'results.csv')
    print(f'Report, gallery and blind review: {output}')
    return int(any(row['status'] != 'ok' for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
