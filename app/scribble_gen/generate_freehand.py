"""Draw a red free-hand ellipse around each removal mask and save scribble2.png.

For every sample folder, the script takes the foreground bounding box of
mask.png, pads it slightly, turns that box into a lightly wobbling ellipse,
and composites the stroke onto original.png.
"""

from __future__ import annotations

import argparse
import logging
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data" / "removal"

STROKE_COLOR = (220, 30, 30)
MIN_RADIUS = 4.0
N_POINTS = 128
PAD_RATIO = 0.08
OVERSHOOT = 0.12
SUPERSAMPLE = 3


def mask_bbox(mask_arr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the exclusive bounding box of foreground pixels, or None if empty."""
    if mask_arr.ndim != 2:
        raise ValueError(f"mask must be a 2D array, got shape {mask_arr.shape}")
    ys, xs = np.nonzero(mask_arr > 127)
    if xs.size == 0:
        return None
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def pad_bbox(
    bbox: tuple[int, int, int, int],
    pad_ratio: float,
    size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Expand each side by pad_ratio of the box size, then clamp to the image."""
    x0, y0, x1, y1 = bbox
    width, height = size
    box_w = x1 - x0
    box_h = y1 - y0
    pad_x = pad_ratio * box_w
    pad_y = pad_ratio * box_h
    x0f = max(0.0, x0 - pad_x)
    y0f = max(0.0, y0 - pad_y)
    x1f = min(float(width), x1 + pad_x)
    y1f = min(float(height), y1 + pad_y)
    if x1f - x0f < 2.0:
        mid = (x0f + x1f) / 2.0
        x0f = max(0.0, mid - 1.0)
        x1f = min(float(width), mid + 1.0)
    if y1f - y0f < 2.0:
        mid = (y0f + y1f) / 2.0
        y0f = max(0.0, mid - 1.0)
        y1f = min(float(height), mid + 1.0)
    return (x0f, y0f, x1f, y1f)


def _seed_for(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def freehand_ellipse(
    bbox: tuple[float, float, float, float],
    rng: np.random.Generator,
    n_points: int = N_POINTS,
    wobble: float = 0.055,
) -> list[tuple[float, float]]:
    """Inscribed ellipse with smooth Fourier wobble and a short crossing tail."""
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    rx = max((x1 - x0) / 2.0, MIN_RADIUS)
    ry = max((y1 - y0) / 2.0, MIN_RADIUS)

    harmonics = (1, 2, 3, 4)
    weights = rng.uniform(0.35, 1.0, size=len(harmonics))
    weights = weights / weights.sum() * wobble
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(harmonics))

    thetas = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    radial = np.ones(n_points, dtype=np.float64)
    for weight, phase, harmonic in zip(weights, phases, harmonics):
        radial += weight * np.cos(harmonic * thetas + phase)
    radial = np.clip(radial, 0.82, 1.18)

    xs = cx + rx * radial * np.cos(thetas)
    ys = cy + ry * radial * np.sin(thetas)
    points = [(float(x), float(y)) for x, y in zip(xs, ys)]
    points.append(points[0])
    points.extend(_crossing_tail(thetas, radial, cx, cy, rx, ry, rng))
    return points


def _crossing_tail(
    thetas: np.ndarray,
    radial: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Continue a short arc that starts on the loop and drifts inward across it."""
    n_tail = max(8, len(thetas) // 10)
    start = float(thetas[0])
    tail_theta = np.linspace(start, start + 2.0 * np.pi * OVERSHOOT, n_tail, endpoint=True)
    # Sample the periodic wobble so the tail begins on the existing stroke.
    base = np.interp(tail_theta % (2.0 * np.pi), thetas, radial, period=2.0 * np.pi)
    inward = float(rng.uniform(0.045, 0.07))
    drift = np.linspace(0.0, -inward, n_tail)
    scale = np.clip(base + drift, 0.75, 1.2)
    xs = cx + rx * scale * np.cos(tail_theta)
    ys = cy + ry * scale * np.sin(tail_theta)
    # Drop the first tail point; the loop already ends there.
    return [(float(x), float(y)) for x, y in zip(xs[1:], ys[1:])]


def clamp_points(
    points: list[tuple[float, float]],
    size: tuple[int, int],
) -> list[tuple[float, float]]:
    width, height = size
    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)
    return [
        (min(max(x, 0.0), float(max_x)), min(max(y, 0.0), float(max_y)))
        for x, y in points
    ]


def stroke_width(size: tuple[int, int]) -> int:
    return max(6, round(0.007 * min(size)))


def overlay_stroke(
    original: Image.Image,
    points: list[tuple[float, float]],
    color: tuple[int, int, int] = STROKE_COLOR,
    width: int | None = None,
) -> Image.Image:
    """Composite an antialiased rounded stroke onto a copy of the original."""
    if len(points) < 2:
        raise ValueError("need at least two points to draw a stroke")
    width = stroke_width(original.size) if width is None else width
    scale = SUPERSAMPLE
    canvas_size = (original.size[0] * scale, original.size[1] * scale)
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    scaled = [(x * scale, y * scale) for x, y in points]
    draw.line(scaled, fill=(*color, 255), width=width * scale, joint="curve")
    layer = layer.resize(original.size, Image.Resampling.LANCZOS)
    composed = Image.alpha_composite(original.convert("RGBA"), layer)
    return composed.convert("RGB")


def process_sample(
    folder: Path,
    *,
    pad_ratio: float = PAD_RATIO,
    overwrite: bool = False,
    n_points: int = N_POINTS,
) -> Path | None:
    """Write folder/scribble2.png. Return the path, or None if the sample is skipped."""
    mask_path = folder / "mask.png"
    original_path = folder / "original.png"
    out_path = folder / "scribble2.png"
    if not mask_path.is_file() or not original_path.is_file():
        logger.warning("skip %s: missing mask.png or original.png", folder.name)
        return None
    if out_path.exists() and not overwrite:
        logger.info("skip %s: scribble2.png already exists", folder.name)
        return out_path

    with Image.open(mask_path) as mask_image, Image.open(original_path) as original_image:
        mask = mask_image.convert("L")
        original = original_image.convert("RGB")
        if mask.size != original.size:
            logger.warning(
                "%s: resizing mask %s to original %s",
                folder.name,
                mask.size,
                original.size,
            )
            mask = mask.resize(original.size, Image.Resampling.NEAREST)
        bbox = mask_bbox(np.asarray(mask))
        if bbox is None:
            logger.warning("skip %s: mask has no foreground", folder.name)
            return None
        padded = pad_bbox(bbox, pad_ratio, original.size)
        rng = np.random.default_rng(_seed_for(folder.name))
        wobble = float(rng.uniform(0.04, 0.07))
        points = clamp_points(
            freehand_ellipse(padded, rng, n_points=n_points, wobble=wobble),
            original.size,
        )
        overlay_stroke(original, points).save(out_path)
    logger.info("wrote %s", out_path)
    return out_path


def _resolve_root(root: Path | None) -> Path:
    if root is None:
        return DEFAULT_ROOT
    if root.is_absolute():
        return root
    return (REPO_ROOT / root).resolve()


def iter_samples(root: Path, only: str | None) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"sample root does not exist: {root}")
    folders = sorted(path for path in root.iterdir() if path.is_dir())
    if only is None:
        return folders
    selected = [path for path in folders if path.name == only]
    if not selected:
        raise FileNotFoundError(f"no sample named {only!r} under {root}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay a red free-hand ellipse from each removal mask.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="sample root (default: data/removal, resolved from the repo root)",
    )
    parser.add_argument("--only", default=None, help="process a single sample folder name")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace scribble2.png when it already exists",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=PAD_RATIO,
        help="bbox padding ratio per side (default: 0.08, use 0 to hug the raw box)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    root = _resolve_root(args.root)
    folders = iter_samples(root, args.only)
    written = 0
    skipped = 0
    failed = 0
    for folder in folders:
        already_present = (folder / "scribble2.png").exists() and not args.overwrite
        result = process_sample(folder, pad_ratio=args.pad, overwrite=args.overwrite)
        if result is None:
            failed += 1
        elif already_present:
            skipped += 1
        else:
            written += 1
    logger.info(
        "done: %s written, %s skipped, %s failed, root=%s",
        written,
        skipped,
        failed,
        root,
    )


if __name__ == "__main__":
    main()
