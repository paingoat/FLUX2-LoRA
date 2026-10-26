"""Prepare the official VIBE Removal subset as two ordered input images.

Only sources, visual instruction layers and Removal.json are downloaded.
VIBE remains an external evaluation set, never a training dataset.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.parse import quote
from urllib.request import urlopen
from urllib.error import URLError

import numpy as np
from PIL import Image, ImageOps

from common import DATASET_DIR, edit_prompt

REPO = "VIBE-Benchmark/VIBE-Benchmark"
TASK = Path("Tasks/Level-1-Deictic/Removal")


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes dataset root: {relative}")
    return path


def image_hash(path: Path) -> str:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        # Pixel hashes also detect identical images with different file metadata.
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def prepare(root: Path, output: Path, internal: Path = DATASET_DIR) -> list[dict]:
    task_dir = root / TASK
    records = json.loads((task_dir / "Removal.json").read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Removal.json must be a nonempty list")
    internal_paths = list((internal / "original").glob("*.png"))
    if not internal_paths:
        raise ValueError("map internal originals first, so overlap can be checked")
    hashes = {image_hash(path) for path in internal_paths}
    excluded, manifest, ids = [], [], set()
    for record in records:
        index = record["id"]
        if not index or Path(index).name != index or index in ids:
            raise ValueError(f"invalid or duplicate VIBE ID: {index}")
        ids.add(index)
        source = inside(root, record["file_paths"]["source"])
        layer_path = inside(task_dir, record["file_paths"]["visual_instruction"])
        if image_hash(source) in hashes:
            excluded.append(index)
            continue
        with Image.open(source) as image:
            original = ImageOps.exif_transpose(image).convert("RGBA")
        with Image.open(layer_path) as image:
            if "A" not in image.getbands():
                raise ValueError(f"{index}: expected a transparent annotation layer")
            layer = image.convert("RGBA")
        if layer.size != original.size:
            layer = layer.resize(original.size, Image.Resampling.LANCZOS)
        alpha = np.asarray(layer.getchannel("A"))
        stroke = alpha > 8
        if not stroke.any() or stroke.mean() > 0.5:
            raise ValueError(f"{index}: invalid annotation coverage {stroke.mean():.1%}")
        folder = output / "images" / index
        folder.mkdir(parents=True, exist_ok=True)
        original.convert("RGB").save(folder / "original.png")
        Image.alpha_composite(original, layer).convert("RGB").save(folder / "reference.png")
        Image.fromarray(stroke.astype(np.uint8) * 255).save(folder / "stroke.png")
        instruction = record["text_prompt"]["input_prompt"].strip()
        if not instruction:
            raise ValueError(f"{index}: missing instruction")
        manifest.append({
            "index": index, "dataset": "vibe_removal", "split": "external",
            "original": str((folder / "original.png").resolve()),
            "reference": str((folder / "reference.png").resolve()),
            "stroke_mask": str((folder / "stroke.png").resolve()),
            "target": None, "instruction": instruction, "prompt": edit_prompt(instruction),
            "meta_info": record.get("meta_info", {}),
        })
    if not manifest:
        raise ValueError("no non-overlapping VIBE samples remain")
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    revision_file = root / "removal_revision.txt"
    (output / "dataset_card.json").write_text(json.dumps({
        "source": REPO, "revision": revision_file.read_text().strip() if revision_file.exists() else "local-unpinned",
        "task": "Removal", "total": len(records), "evaluated": len(manifest),
        "excluded_exact_pixel_overlap": excluded, "reference_order": ["original", "reference"],
        "target_available": False,
        "overlap_check": "exact decoded RGB pixels against all internal originals; inspect near-duplicates manually",
    }, indent=2) + "\n", encoding="utf-8")
    return manifest


def download(root: Path, revision: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/datasets/{REPO}/revision/{quote(revision, safe='')}"
    with urlopen(api, timeout=60) as response:
        sha = json.load(response)["sha"]
    revision_file = root / "removal_revision.txt"
    same_revision = revision_file.exists() and revision_file.read_text().strip() == sha

    def fetch(filename: str):
        path = inside(root, filename)
        if same_revision and path.is_file():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{sha}/{quote(filename, safe='/')}"
        for attempt in range(3):
            try:
                with urlopen(url, timeout=60) as response, path.with_suffix(path.suffix + ".part").open("wb") as handle:
                    shutil.copyfileobj(response, handle)
                path.with_suffix(path.suffix + ".part").replace(path)
                return path
            except (URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    metadata = fetch((TASK / "Removal.json").as_posix())
    records = json.loads(Path(metadata).read_text(encoding="utf-8"))
    files = set()
    for record in records:
        source = record["file_paths"]["source"]
        layer = (TASK / record["file_paths"]["visual_instruction"]).as_posix()
        inside(root, source)
        inside(root, layer)
        files.update((source, layer))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, sorted(files)))
    revision_file.write_text(sha + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/data/VIBE-Benchmark"))
    parser.add_argument("--output", type=Path, default=Path("/workspace/datasets/vibe_removal"))
    parser.add_argument("--internal", type=Path, default=DATASET_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    if args.download:
        download(args.root, args.revision)
    records = prepare(args.root.resolve(), args.output.resolve(), args.internal)
    print(f"VIBE Removal: {len(records)} samples; manifest: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
