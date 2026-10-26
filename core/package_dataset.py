"""Package only mapped two-input removal data for upload outside Git."""
import argparse
import hashlib
from pathlib import Path
import tarfile

from common import DATASET_DIR, REPO_ROOT
from prepare_dataset import check_pair, dataset_file, load_rows


def package(dataset: Path, output: Path) -> Path:
    dataset = dataset.resolve()
    output = output.resolve()
    if output.is_relative_to(dataset):
        raise ValueError("archive must be outside the source dataset")
    rows = load_rows(dataset)
    paths = {dataset / "mapping.csv", dataset / "meta.jsonl"}
    for row in rows:
        paths.update(check_pair(dataset, row))
        original = dataset_file(dataset, row["original"])
        if not original.is_file():
            raise ValueError(f"missing {original}; map originals first")
        paths.add(original)
    if not rows:
        raise ValueError("empty mapping")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
        for path in sorted(paths):
            archive.add(path, arcname=(Path("data/removal_dataset") / path.relative_to(dataset)).as_posix())
    temporary.replace(output)
    with output.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{checksum}  {output.name}\n", encoding="utf-8")
    print(f"{len(rows)} samples / {len(paths)} files / {output.stat().st_size / 1024**2:.1f} MiB")
    print(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / ".local/removal_dataset.tar.gz")
    args = parser.parse_args()
    package(args.dataset, args.output)
