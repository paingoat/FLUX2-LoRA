"""Shared paths and checks for the red-circle removal experiment.

Training images come only from data/removal_dataset. data/removal is the old
scribble set and is rejected wherever a path is checked.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "data" / "removal_dataset"
PREPARED_DIR = Path(os.environ.get("REMOVAL_PREPARED", "/workspace/datasets/removal_scribble"))
OUTPUT_DIR = Path(os.environ.get("REMOVAL_OUTPUT", "/workspace/output/removal_scribble_4b"))
AITOOLKIT_DIR = Path(os.environ.get("AITOOLKIT", "/workspace/ai-toolkit"))
TRAIN_CONFIG = REPO_ROOT / "core" / "configs" / "removal_scribble_4b.yaml"

PROMPT = "Remove the object inside the red circle and naturally restore the background."

BASE_REPO = "black-forest-labs/FLUX.2-klein-base-4B"
DISTILLED_REPO = "black-forest-labs/FLUX.2-klein-4B"
VAE_REPO = "ai-toolkit/flux2_vae"
TEXT_ENCODER_REPO = "Qwen/Qwen3-4B"
TRANSFORMER_FILE = "flux-2-klein-base-4b.safetensors"
VAE_FILE = "ae.safetensors"

SAMPLE_CONTROL_INDEXES = ("0015", "0022", "0040", "0057")
AREA_TOKEN_LIMIT = 4096


def load_runtime_env() -> None:
    """Apply the same variables as core/env.sh when the shell did not source it."""
    env_file = REPO_ROOT / ".env"
    if env_file.is_file():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    os.environ.setdefault("HF_HOME", "/workspace/data")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def assert_not_old_removal(path: Path) -> Path:
    """Reject data/removal. removal_dataset does not contain that path segment."""
    resolved = path.resolve()
    if "removal" in resolved.parts:
        raise SystemExit(
            f"refusing {resolved}: data/removal is the old scribble set and is not a training source. "
            "Use data/removal_dataset."
        )
    return resolved


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def fit_klein_size(width: int, height: int, longest: int) -> tuple[int, int]:
    """Scale so the long side is `longest`, then satisfy klein's size rules.

    Width and height must be divisible by 16 and (width * height) / 256 <= 4096.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size {width}x{height}")
    scale = longest / max(width, height)
    width = max(1, int(round(width * scale)))
    height = max(1, int(round(height * scale)))
    area_limit = AREA_TOKEN_LIMIT * 256
    if width * height > area_limit:
        area_scale = (area_limit / (width * height)) ** 0.5
        width = max(1, int(width * area_scale))
        height = max(1, int(height * area_scale))
    width = max(16, int(round(width / 16.0)) * 16)
    height = max(16, int(round(height / 16.0)) * 16)
    while width * height / 256 > AREA_TOKEN_LIMIT and (width > 16 or height > 16):
        if width >= height and width > 16:
            width -= 16
        elif height > 16:
            height -= 16
    return width, height


def hub_snapshot_has(repo_id: str, filename: str | None = None) -> bool:
    """True when the Hub cache already contains this repo, or one file in it."""
    cache_root = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", Path(os.environ["HF_HOME"]) / "hub"))
    repo_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
    if not repo_dir.is_dir():
        return False
    if filename is None:
        return any(repo_dir.rglob("config.json")) or any(repo_dir.rglob("*.safetensors"))
    return any(path.name == filename for path in repo_dir.rglob(filename))
