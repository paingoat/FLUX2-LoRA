"""Score every saved EditLoRA on the held-out red-circle pairs.

Images and the prompt come from data/removal_dataset via the manifest written
by prepare_dataset.py. The copied mask/ folder is not read: the circle is
taken from red pixels on reference/.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    DISTILLED_REPO,
    OUTPUT_DIR,
    PREPARED_DIR,
    PROMPT,
    assert_not_old_removal,
    fit_klein_size,
    load_runtime_env,
)

GUIDANCE = 4.0
STEPS = 4
SEED = 42


def load_manifest(path: Path) -> list[dict[str, str]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise SystemExit(f"{path} has no held-out records")
    for record in records:
        for key in ("reference", "target"):
            record[key] = str(assert_not_old_removal(Path(record[key])))
        if record.get("prompt") != PROMPT:
            raise SystemExit(f"{record.get('index')} prompt is not the shared removal prompt")
    return records


def red_region(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the pure-red stroke and the circle it encloses.

    The marker is a narrow bright-red ring. A loose hue threshold also catches
    warm scenes, so the stroke is the pixels that are actually red.
    """
    import cv2

    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    stroke = (
        (red >= 200) & (green <= 70) & (blue <= 70) & ((red - np.maximum(green, blue)) >= 120)
    ).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(stroke, cv2.MORPH_CLOSE, kernel)
    flood = closed.copy()
    flood_mask = np.zeros((closed.shape[0] + 2, closed.shape[1] + 2), np.uint8)
    if closed[0, 0] == 0:
        cv2.floodFill(flood, flood_mask, (0, 0), 2)
    interior = flood == 0
    interior_fraction = float(interior.mean()) if interior.size else 1.0
    if interior_fraction < 0.002 or interior_fraction > 0.85:
        edit = cv2.dilate(stroke, kernel, iterations=2)
    else:
        edit = cv2.dilate((closed.astype(bool) | interior).astype(np.uint8), kernel, iterations=1)
    return stroke.astype(bool), edit.astype(bool)


def psnr(left: np.ndarray, right: np.ndarray) -> float:
    mse = float(np.mean((left.astype(np.float32) - right.astype(np.float32)) ** 2))
    if mse == 0:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def masked_ssim(reference: np.ndarray, output: np.ndarray, edit: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    matched = output.copy()
    matched[edit] = reference[edit]
    score = structural_similarity(reference, matched, channel_axis=2, data_range=255)
    return float(score)


def resize_pair(reference: Image.Image, target: Image.Image, longest: int) -> tuple[Image.Image, Image.Image, int, int]:
    width, height = fit_klein_size(reference.width, reference.height, longest)
    resampling = Image.Resampling.LANCZOS
    return (
        reference.convert("RGB").resize((width, height), resampling),
        target.convert("RGB").resize((width, height), resampling),
        width,
        height,
    )


def score_pair(reference: Image.Image, output: Image.Image, target: Image.Image, lpips_fn) -> dict[str, float | None]:
    import torch

    ref = np.asarray(reference)
    out = np.asarray(output.resize(reference.size, Image.Resampling.LANCZOS))
    tgt = np.asarray(target)
    stroke, edit = red_region(ref)
    outside = ~edit
    residue = None
    preserve_psnr = None
    preserve_ssim = None
    stroke_count = int(stroke.sum())
    if stroke_count > 0 and outside.any() and 0 < int(edit.sum()) < edit.size * 0.9:
        residue = float(np.logical_and(stroke, red_region(out)[0]).sum()) / stroke_count
        preserve_psnr = psnr(ref[outside], out[outside])
        preserve_ssim = masked_ssim(ref, out, edit)

    def tensor(image: np.ndarray) -> torch.Tensor:
        data = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        return data * 2.0 - 1.0

    device = next(lpips_fn.parameters()).device
    with torch.inference_mode():
        distance = lpips_fn(tensor(out).to(device), tensor(tgt).to(device))
    return {
        "scribble_residue": residue,
        "preserve_psnr": preserve_psnr,
        "preserve_ssim": preserve_ssim,
        "target_lpips": float(distance.item()),
    }


def contact_sheet(triples: list[tuple[str, Image.Image, Image.Image, Image.Image]], destination: Path) -> None:
    thumb_h = 240
    gap = 6
    rows = []
    width = 0
    for _label, reference, output, target in triples:
        thumbs = []
        for image in (reference, output, target):
            scale = thumb_h / image.height
            thumbs.append(image.resize((max(1, int(image.width * scale)), thumb_h), Image.Resampling.LANCZOS))
        rows.append(thumbs)
        width = max(width, gap + sum(image.width + gap for image in thumbs))
    canvas = Image.new("RGB", (width, (thumb_h + gap) * len(rows) + gap), "white")
    draw = ImageDraw.Draw(canvas)
    y = gap
    for (label, *_images), thumbs in zip(triples, rows):
        x = gap
        for image in thumbs:
            canvas.paste(image, (x, y))
            x += image.width + gap
        draw.text((gap, y + 4), label, fill=(255, 255, 0))
        y += thumb_h + gap
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=90)


def checkpoint_step(path: Path | None) -> int:
    """Step from a zero-padded filename, or from the final unsuffixed save."""
    if path is None:
        return -1
    match = re.search(r"_(\d{6,})$", path.stem)
    if match:
        return int(match.group(1))
    if not path.is_file():
        return -1
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            raw = (handle.metadata() or {}).get("training_info")
        if raw:
            return int(json.loads(raw)["step"])
    except Exception:
        return -1
    return -1


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def column_mean(items: list[dict], key: str) -> float | None:
    values = [float(item[key]) for item in items if item[key] != ""]
    return mean(values)


def write_report(rows: list[dict], destination: Path) -> None:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["checkpoint"], int(row["resolution"])), []).append(row)

    summary = []
    for (checkpoint, resolution), items in sorted(grouped.items(), key=lambda item: (item[0][1], checkpoint_step(Path(item[0][0])))):
        summary.append(
            {
                "checkpoint": checkpoint,
                "step": items[0]["step"],
                "resolution": resolution,
                "n": len(items),
                "scribble_residue": column_mean(items, "scribble_residue"),
                "preserve_psnr": column_mean(items, "preserve_psnr"),
                "preserve_ssim": column_mean(items, "preserve_ssim"),
                "target_lpips": column_mean(items, "target_lpips"),
            }
        )

    def fmt(value: float | None) -> str:
        return "" if value is None else f"{value:.4f}"

    lines = [
        "# EditLoRA removal benchmark",
        "",
        "Vùng đo là vòng tròn đỏ trên `reference/` của `removal_dataset`. `mask/` không được đọc.",
        "`scribble_residue` thấp hơn thì nét đỏ còn ít hơn. `preserve_psnr` và `preserve_ssim` cao hơn thì phần ngoài vòng tròn bị đụng ít hơn. `target_lpips` là tín hiệu mềm vì ảnh target không pixel-aligned.",
        "",
        "| Checkpoint | Step | Resolution | Residue | PSNR | SSIM | LPIPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary:
        lines.append(
            f"| {item['checkpoint']} | {item['step']} | {item['resolution']} | "
            f"{fmt(item['scribble_residue'])} | {fmt(item['preserve_psnr'])} | "
            f"{fmt(item['preserve_ssim'])} | {fmt(item['target_lpips'])} |"
        )

    for resolution in sorted({item["resolution"] for item in summary}):
        subset = [item for item in summary if item["resolution"] == resolution and item["checkpoint"] != "baseline"]
        usable = [item for item in subset if None not in (item["scribble_residue"], item["preserve_psnr"], item["preserve_ssim"], item["target_lpips"])]
        if len(usable) < 2:
            continue
        rank_sum = {id(item): 0 for item in usable}

        def add_ranks(key: str, higher_better: bool) -> None:
            ordered = sorted(usable, key=lambda item: item[key], reverse=higher_better)
            for place, item in enumerate(ordered, start=1):
                rank_sum[id(item)] += place

        add_ranks("scribble_residue", False)
        add_ranks("preserve_psnr", True)
        add_ranks("preserve_ssim", True)
        add_ranks("target_lpips", False)
        winner = min(usable, key=lambda item: (rank_sum[id(item)], -int(item["step"])))
        lines.append("")
        lines.append(
            f"Resolution {resolution}: checkpoint nên xem trước là step {winner['step']} "
            f"(`{winner['checkpoint']}`), theo hạng trung bình của 4 metric."
        )
        by_step = {int(item["step"]): item for item in usable}
        if 1500 in by_step and 1250 in by_step and int(winner["step"]) == 1500:
            latest = by_step[1500]
            previous = by_step[1250]
            if latest["preserve_psnr"] > previous["preserve_psnr"] or latest["scribble_residue"] < previous["scribble_residue"]:
                lines.append(
                    "Step 1500 vẫn tốt hơn step 1250. Pilot này có thể chưa chạm đỉnh; train tiếp bằng `python core/train.py --resume`."
                )

    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(pipe, reference: Image.Image, width: int, height: int, seed: int) -> Image.Image:
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.inference_mode():
        return pipe(
            prompt=PROMPT,
            image=reference,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            width=width,
            height=height,
            generator=generator,
        ).images[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PREPARED_DIR / "eval" / "manifest.json")
    parser.add_argument("--checkpoints", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "benchmark")
    parser.add_argument("--baseline", action="store_true", help="also score the distilled pipeline with no LoRA")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    load_runtime_env()
    import torch
    import lpips
    from diffusers import Flux2KleinPipeline

    if not torch.cuda.is_available():
        raise SystemExit("benchmark needs a CUDA GPU. Run it on the pod.")
    records = load_manifest(args.manifest)
    checkpoints = sorted(args.checkpoints.glob("*.safetensors"), key=lambda path: (checkpoint_step(path), path.name))
    if not checkpoints and not args.baseline:
        raise SystemExit(f"no checkpoints in {args.checkpoints}")

    pipe = Flux2KleinPipeline.from_pretrained(DISTILLED_REPO, torch_dtype=torch.bfloat16).to("cuda")
    lpips_fn = lpips.LPIPS(net="alex").to("cuda")
    lpips_fn.eval()
    for parameter in lpips_fn.parameters():
        parameter.requires_grad_(False)

    jobs: list[Path | None] = []
    if args.baseline:
        jobs.append(None)
    jobs.extend(checkpoints)

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    from tqdm import tqdm

    work = [(ckpt, record, resolution) for ckpt in jobs for record in records for resolution in args.resolutions]
    current_lora: Path | None = None
    lora_loaded = False
    sheets: dict[tuple[str, int], list] = {}
    started = False
    for ckpt, record, resolution in tqdm(work, unit="image"):
        tag = "baseline" if ckpt is None else ckpt.stem
        if not started or ckpt != current_lora:
            if lora_loaded:
                pipe.unload_lora_weights()
                lora_loaded = False
            if ckpt is not None:
                pipe.load_lora_weights(str(ckpt), adapter_name="removal")
                pipe.set_adapters(["removal"], adapter_weights=[1.0])
                lora_loaded = True
            current_lora = ckpt
            started = True

        reference, target, width, height = resize_pair(
            Image.open(record["reference"]),
            Image.open(record["target"]),
            resolution,
        )
        try:
            output = generate(pipe, reference, width, height, args.seed)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"OOM: {tag} {record['index']} @ {resolution}", file=sys.stderr)
            continue
        image_dir = args.output / "images" / tag / str(resolution)
        image_dir.mkdir(parents=True, exist_ok=True)
        output_path = image_dir / f"{record['index']}.png"
        output.save(output_path)
        metrics = score_pair(reference, output, target, lpips_fn)
        rows.append(
            {
                "checkpoint": tag,
                "step": checkpoint_step(ckpt),
                "resolution": resolution,
                "index": record["index"],
                "split": record["split"],
                "scribble_residue": "" if metrics["scribble_residue"] is None else f"{metrics['scribble_residue']:.6f}",
                "preserve_psnr": "" if metrics["preserve_psnr"] is None else f"{metrics['preserve_psnr']:.4f}",
                "preserve_ssim": "" if metrics["preserve_ssim"] is None else f"{metrics['preserve_ssim']:.4f}",
                "target_lpips": f"{metrics['target_lpips']:.6f}",
                "image": str(output_path),
            }
        )
        sheets.setdefault((tag, resolution), []).append((record["index"], reference, output, target))

    for (tag, resolution), triples in sheets.items():
        contact_sheet(triples, args.output / "sheets" / f"{tag}_{resolution}.jpg")

    csv_path = args.output / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["checkpoint"])
        writer.writeheader()
        writer.writerows(rows)
    report_path = args.output / "report.md"
    write_report(rows, report_path)
    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
