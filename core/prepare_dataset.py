"""Build the ai-toolkit folder from data/removal_dataset only.

Maps clean originals by sample_id, copies two ordered controls per training
target, and writes held-out manifests. The old data/removal tree is never opened.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
import re

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    DATASET_DIR,
    PREPARED_DIR,
    PROMPT,
    INSTRUCTION,
    REFERENCE_ORDER,
    SAMPLE_CONTROL_INDEXES,
    assert_not_old_removal,
    png_size,
)


def load_rows(dataset_dir: Path) -> list[dict[str, str]]:
    mapping_path = dataset_dir / "mapping.csv"
    meta_path = dataset_dir / "meta.jsonl"
    if not mapping_path.is_file():
        raise SystemExit(f"missing {mapping_path}")
    if not meta_path.is_file():
        raise SystemExit(f"missing {meta_path}")

    with mapping_path.open(newline="", encoding="utf-8") as handle:
        mapping = list(csv.DictReader(handle))
    splits: dict[str, str] = {}
    with meta_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            index = str(record.get("index", "")).strip()
            split = str(record.get("split", "")).strip()
            if not index or not split:
                raise SystemExit(f"{meta_path}:{line_number} needs index and split")
            splits[index] = split

    rows: list[dict[str, str]] = []
    indexes, sample_ids = set(), set()
    for row in mapping:
        index = row["index"].strip()
        sample_id = row["sample_id"].strip()
        if not re.fullmatch(r"[0-9]+", index) or not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
            raise SystemExit(f"invalid mapping identity: {row}")
        if index in indexes or sample_id in sample_ids:
            raise SystemExit(f"duplicate mapping index/sample_id: {index} {sample_id}")
        indexes.add(index)
        sample_ids.add(sample_id)
        if index not in splits:
            raise SystemExit(f"{index} has no split in {meta_path}")
        rows.append(
            {
                "index": index,
                "sample_id": row["sample_id"].strip(),
                "split": splits[index],
                "reference": row["reference"].strip(),
                "target_image": row["target_image"].strip(),
                "target_text": row["target_text"].strip(),
                "original": row.get("original", f"original/{index}.png").strip(),
            }
        )
    return rows


def dataset_file(dataset_dir: Path, relative: str) -> Path:
    path = assert_not_old_removal(dataset_dir / relative)
    if not path.is_relative_to(dataset_dir.resolve()):
        raise ValueError(f"path escapes dataset: {relative}")
    return path


def check_pair(dataset_dir: Path, row: dict[str, str]) -> tuple[Path, Path, Path]:
    reference = dataset_file(dataset_dir, row["reference"])
    target = dataset_file(dataset_dir, row["target_image"])
    caption = dataset_file(dataset_dir, row["target_text"])
    missing = [path for path in (reference, target, caption) if not path.is_file()]
    if missing:
        raise ValueError(f"{row['index']} missing files: {missing}")
    try:
        reference_size = png_size(reference)
        target_size = png_size(target)
    except ValueError as exc:
        raise ValueError(f"{row['index']}: {exc}") from exc
    if reference_size != target_size:
        raise ValueError(
            f"{row['index']}: reference {reference_size} != target {target_size}"
        )
    text = caption.read_text(encoding="utf-8").strip()
    if text not in {INSTRUCTION, PROMPT}:
        raise ValueError(
            f"{row['index']}: caption is not the shared prompt. got: {text!r}"
        )
    return reference, target, caption


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare(dataset_dir: Path, output_dir: Path, originals_root: Path | None = None) -> None:
    dataset_dir = assert_not_old_removal(dataset_dir)
    output_dir = output_dir.resolve()
    if output_dir == dataset_dir or dataset_dir in output_dir.parents or output_dir in dataset_dir.parents:
        raise SystemExit(f"output {output_dir} must be separate from {dataset_dir}")
    if "removal" in output_dir.parts:
        raise SystemExit(f"refusing output path that contains the old dataset name: {output_dir}")
    if originals_root:
        originals_root = originals_root.resolve()
        if output_dir == originals_root or output_dir in originals_root.parents or originals_root in output_dir.parents:
            raise SystemExit("prepared output must be separate from the originals source")

    rows = load_rows(dataset_dir)
    problems: list[str] = []
    checked = []
    alignment = []
    for row in rows:
        if row["split"] not in {"train", "validation", "test"}:
            problems.append(f"{row['index']}: unknown split {row['split']}")
            continue
        try:
            reference, target, caption = check_pair(dataset_dir, row)
            original = dataset_file(dataset_dir, row["original"])
            source = (originals_root / row["sample_id"] / "original.png") if originals_root else original
            if not source.is_file():
                raise ValueError(f"{row['index']}: missing original {source}")
            if png_size(source) != png_size(reference):
                raise ValueError(f"{row['index']}: original size does not match reference")
            with Image.open(source) as image, Image.open(reference) as overlay:
                a, b = np.asarray(image.convert("RGB")), np.asarray(overlay.convert("RGB"))
            fraction = float((np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2) > 8).mean())
            # A red ring should change only a small part of the clean source.
            if fraction <= 0 or fraction > 0.12:
                raise ValueError(f"{row['index']}: original/reference alignment suspect ({fraction:.1%} changed)")
            alignment.append({"index": row["index"], "changed_fraction": fraction})
            checked.append((row, reference, target, caption, source, original))
        except (ValueError, OSError) as exc:
            problems.append(str(exc))
    split_by_index = {row["index"]: row["split"] for row in rows}
    for index in SAMPLE_CONTROL_INDEXES:
        if split_by_index.get(index) != "validation":
            problems.append(f"sample control {index} must exist in validation")
    if problems:
        print(f"{len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        raise SystemExit(1)

    counts = Counter(row["split"] for row, *_ in checked)
    reference_dir = output_dir / "reference"
    original_dir = output_dir / "original"
    target_dir = output_dir / "target"
    sample_dir = output_dir / "sample_controls"
    eval_dir = output_dir / "eval"
    reset_dir(reference_dir)
    reset_dir(original_dir)
    reset_dir(target_dir)
    reset_dir(sample_dir)
    reset_dir(eval_dir)

    by_split: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    manifest: list[dict[str, str]] = []
    for row, reference, target, caption, source, original in checked:
        original.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != original:
            shutil.copyfile(source, original)
        by_split[row["split"]].append(row["index"])
        if row["split"] == "train":
            shutil.copyfile(original, original_dir / f"{row['index']}.png")
            shutil.copyfile(reference, reference_dir / f"{row['index']}.png")
            shutil.copyfile(target, target_dir / f"{row['index']}.png")
            (target_dir / f"{row['index']}.txt").write_text(PROMPT + "\n", encoding="utf-8")
        else:
            manifest.append(
                {
                    "index": row["index"],
                    "split": row["split"],
                    "reference": str(reference),
                    "original": str(original),
                    "dataset": "internal",
                    "target": str(target),
                    "prompt": PROMPT,
                }
            )
        if row["index"] in SAMPLE_CONTROL_INDEXES:
            if row["split"] != "validation":
                raise SystemExit(f"sample control {row['index']} is {row['split']}, expected validation")
            shutil.copyfile(reference, sample_dir / f"{row['index']}.png")
            shutil.copyfile(original, sample_dir / f"{row['index']}_original.png")

    missing_controls = [index for index in SAMPLE_CONTROL_INDEXES if not (sample_dir / f"{index}.png").is_file()]
    if missing_controls:
        raise SystemExit(f"validation sample controls missing: {missing_controls}")

    manifest.sort(key=lambda item: (item["split"], item["index"]))
    (eval_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    card = {
        "source": str(dataset_dir),
        "schema_version": 2,
        "reference_order": REFERENCE_ORDER,
        "alignment": alignment,
        "prompt": PROMPT,
        "counts": dict(counts),
        "train_indexes": by_split["train"],
        "validation_indexes": by_split["validation"],
        "test_indexes": by_split["test"],
        "sample_controls": [str(sample_dir / f"{index}.png") for index in SAMPLE_CONTROL_INDEXES],
    }
    (output_dir / "dataset_card.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"source: {dataset_dir}")
    print(f"prepared: {output_dir}")
    print(f"counts: {dict(counts)}")
    print(f"train pairs: {len(by_split['train'])}")
    print(f"held-out manifest: {len(manifest)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, default=PREPARED_DIR)
    parser.add_argument("--originals-root", type=Path, help="samples/remove directory; map by sample_id")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(args.dataset, args.output, args.originals_root)
