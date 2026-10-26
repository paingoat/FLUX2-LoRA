"""Copy the fields removal_dataset does not already have.

Reads data/removal and pico-scribble metadata once, then writes:

    data/removal_dataset/mask/<index>.png
    data/removal_dataset/meta.jsonl

Training and eval scripts should read only removal_dataset after this.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import struct
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data" / "removal_dataset"
DEFAULT_REMOVAL = REPO_ROOT / "data" / "removal"
DEFAULT_METADATA = REPO_ROOT / "data" / "pico-scribble-300-v1" / "metadata.jsonl"

META_FIELDS = (
    "index",
    "sample_id",
    "split",
    "scribble_style",
    "prompt_short",
    "prompt_object",
)


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def load_mapping(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"index", "sample_id", "reference"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"mapping.csv must have columns {sorted(required)}: {path}")
    return rows


def load_metadata(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = record.get("id")
            if not sample_id:
                raise SystemExit(f"{path}:{line_number} has no id")
            records[sample_id] = record
    return records


def is_complete(dataset_dir: Path, expected: int) -> bool:
    mask_dir = dataset_dir / "mask"
    meta_path = dataset_dir / "meta.jsonl"
    if not mask_dir.is_dir() or not meta_path.is_file():
        return False
    mask_count = len(list(mask_dir.glob("*.png")))
    with meta_path.open(encoding="utf-8") as handle:
        meta_count = sum(1 for line in handle if line.strip())
    return mask_count == expected and meta_count == expected


def collect_problems(
    rows: list[dict[str, str]],
    metadata: dict[str, dict],
    dataset_dir: Path,
    removal_dir: Path,
) -> list[str]:
    problems: list[str] = []
    seen_indexes: set[str] = set()
    for row in rows:
        index = row["index"].strip()
        sample_id = row["sample_id"].strip()
        if index in seen_indexes:
            problems.append(f"{index}: duplicate index in mapping.csv")
        seen_indexes.add(index)

        reference = dataset_dir / row["reference"].strip()
        source_dir = removal_dir / sample_id
        mask = source_dir / "mask.png"
        record = metadata.get(sample_id)

        if not reference.is_file():
            problems.append(f"{index} {sample_id}: missing reference {reference}")
        if not source_dir.is_dir():
            problems.append(f"{index} {sample_id}: missing source dir {source_dir}")
        elif not mask.is_file():
            problems.append(f"{index} {sample_id}: missing {mask}")
        if record is None:
            problems.append(f"{index} {sample_id}: not in metadata")
        else:
            missing_fields = [name for name in META_FIELDS if name not in ("index", "sample_id") and name not in record]
            if missing_fields:
                problems.append(f"{index} {sample_id}: metadata missing {missing_fields}")

        if reference.is_file() and mask.is_file():
            try:
                if png_size(reference) != png_size(mask):
                    problems.append(
                        f"{index} {sample_id}: mask {png_size(mask)} != reference {png_size(reference)}"
                    )
            except ValueError as exc:
                problems.append(f"{index} {sample_id}: {exc}")
    return problems


def write_dataset(
    rows: list[dict[str, str]],
    metadata: dict[str, dict],
    dataset_dir: Path,
    removal_dir: Path,
) -> None:
    mask_dir = dataset_dir / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    meta_path = dataset_dir / "meta.jsonl"
    split_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()

    with meta_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            index = row["index"].strip()
            sample_id = row["sample_id"].strip()
            record = metadata[sample_id]
            destination = mask_dir / f"{index}.png"
            shutil.copyfile(removal_dir / sample_id / "mask.png", destination)
            payload = {
                "index": index,
                "sample_id": sample_id,
                "split": record["split"],
                "scribble_style": record["scribble_style"],
                "prompt_short": record["prompt_short"],
                "prompt_object": record["prompt_object"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            split_counts[payload["split"]] += 1
            style_counts[payload["scribble_style"]] += 1

    print(f"wrote {len(rows)} masks -> {mask_dir}")
    print(f"wrote {meta_path}")
    print("splits:", dict(split_counts))
    print("styles:", dict(style_counts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--removal", type=Path, default=DEFAULT_REMOVAL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--force", action="store_true", help="rewrite mask/ and meta.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset.resolve()
    mapping_path = dataset_dir / "mapping.csv"
    if not mapping_path.is_file():
        print(f"missing {mapping_path}", file=sys.stderr)
        return 1
    if not args.metadata.is_file():
        print(f"missing {args.metadata}", file=sys.stderr)
        return 1

    rows = load_mapping(mapping_path)
    if not args.force and is_complete(dataset_dir, len(rows)):
        print(f"already complete: {len(rows)} masks and meta.jsonl in {dataset_dir}")
        return 0

    metadata = load_metadata(args.metadata.resolve())
    problems = collect_problems(rows, metadata, dataset_dir, args.removal.resolve())
    if problems:
        print(f"{len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    write_dataset(rows, metadata, dataset_dir, args.removal.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
