"""Reuse mapped masks; union annotation pixels and dilate for loss weighting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

from common import PREPARED_DIR
from prepare_dataset import load_rows, dataset_file


def supervision_masks(original, reference, object_mask, radius):
    if original.size != reference.size or original.size != object_mask.size:
        raise ValueError('original/reference/mask must have identical dimensions')
    if radius < 0:
        raise ValueError('dilation radius must be nonnegative')
    raw = np.asarray(object_mask.convert('L'))
    if not np.isin(raw, [0, 255]).all():
        raise ValueError('object mask must be binary 0/255, white = edit')
    obj = raw > 127
    if not 0 < obj.mean() < 0.9:
        raise ValueError('empty or excessive object mask')
    a, b = [np.asarray(i.convert('RGB')).astype(np.int16) for i in (original, reference)]
    stroke = np.abs(a - b).max(axis=2) > 8
    if not stroke.any():
        raise ValueError('no annotation found; verify original/reference alignment')
    region = Image.fromarray(((obj | stroke) * 255).astype(np.uint8))
    loss_mask = Image.fromarray(cv2.dilate(np.asarray(region), np.ones((2*radius+1, 2*radius+1), np.uint8))) if radius else region.copy()
    return region, loss_mask, Image.fromarray((stroke * 255).astype(np.uint8))


def prepare(prepared: Path, output: Path, mask_root: Path | None = None, radius_at_512: int = 25):
    if radius_at_512 < 0:
        raise ValueError('radius-at-512 must be nonnegative')
    card = json.loads((prepared / 'dataset_card.json').read_text(encoding='utf-8'))
    dataset = Path(card['source'])
    mask_root = mask_root or dataset / 'mask'
    output = output.resolve()
    if output.exists():
        raise ValueError(f'output exists; choose a fresh directory: {output}')
    # Validate every source before writing output. Never regenerate source masks.
    records = []
    for row in load_rows(dataset):
        paths = {key: dataset_file(dataset, row[key]) for key in ('original', 'reference', 'target_image')}
        paths['object_mask'] = mask_root / f"{row['index']}.png"
        with Image.open(paths['original']) as original, Image.open(paths['reference']) as reference, Image.open(paths['object_mask']) as mask, Image.open(paths['target_image']) as target:
            if target.size != original.size:
                raise ValueError(f"{row['index']}: target size mismatch")
            radius = round(radius_at_512 * max(original.size) / 512)
            supervision_masks(original, reference, mask, radius)
        records.append((row, paths, radius))
    train_ids = {row['index'] for row, *_ in records if row['split'] == 'train'}
    if train_ids != set(card['train_indexes']) or train_ids != {p.stem for p in (prepared / 'target').glob('*.png')}:
        raise ValueError('prepared train indexes do not match source; re-run prepare_dataset.py')
    for folder in ('train', 'region', 'stroke', 'preview'):
        (output / folder).mkdir(parents=True)
    provenance = []
    for row, paths, radius in records:
        with Image.open(paths['original']) as original, Image.open(paths['reference']) as reference, Image.open(paths['object_mask']) as mask:
            region, loss_mask, stroke = supervision_masks(original, reference, mask, radius)
            tinted = Image.new('RGB', original.size, (255, 0, 255))
            overlay = Image.composite(Image.blend(original.convert('RGB'), tinted, 0.35), original.convert('RGB'), loss_mask)
        name = f"{row['index']}.png"
        region.save(output / 'region' / name)
        stroke.save(output / 'stroke' / name)
        overlay.save(output / 'preview' / name)
        if row['split'] == 'train':
            loss_mask.save(output / 'train' / name)
        provenance.append(dict(index=row['index'], sample_id=row['sample_id'], split=row['split'], radius_source_px=radius,
                               loss_fraction=float((np.asarray(loss_mask) > 0).mean()),
                               loss_mask_sha256=hashlib.sha256((output / 'train' / name).read_bytes()).hexdigest() if row['split'] == 'train' else None,
                               source_sha256={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}))
    manifest = json.loads((prepared / 'eval/manifest.json').read_text(encoding='utf-8'))
    for row in manifest:
        row['region_mask'] = str(output / 'region' / f"{row['index']}.png")
        row['stroke_mask'] = str(output / 'stroke' / f"{row['index']}.png")
    (output / 'eval_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output / 'mask_provenance.json').write_text(json.dumps(dict(
        source='existing mapped masks; not newly generated SAM3 ground truth',
        radius_at_512=radius_at_512, kernel='square, side=2*radius+1; radius scales with source long side',
        annotation_threshold=8, records=provenance), indent=2), encoding='utf-8')
    print(f'{len(train_ids)} train masks; {len(manifest)} held-out masks; inspect {output / "preview"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=PREPARED_DIR)
    parser.add_argument('--output', type=Path, help='default: <prepared>/region_loss')
    parser.add_argument('--mask-root', type=Path, help='existing binary masks named by mapped index')
    parser.add_argument('--radius-at-512', type=int, default=25)
    args = parser.parse_args()
    prepare(args.prepared, args.output or args.prepared / 'region_loss', args.mask_root, args.radius_at_512)
