"""Gradio demo for the red-circle removal LoRA.

Loads the distilled klein pipeline by default. Training images are read only
from data/removal_dataset, through the manifest produced by prepare_dataset.py.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    BASE_REPO,
    DATASET_DIR,
    DISTILLED_REPO,
    OUTPUT_DIR,
    PREPARED_DIR,
    PROMPT,
    assert_not_old_removal,
    fit_klein_size,
    load_runtime_env,
)

MODELS = {
    "distilled (FLUX.2-klein-4B, 4 steps)": (DISTILLED_REPO, 4),
    "base (FLUX.2-klein-base-4B, 50 steps)": (BASE_REPO, 50),
}


class PipelineCache:
    def __init__(self) -> None:
        self.pipe = None
        self.repo: str | None = None
        self.lora: str | None = None
        self.scale: float | None = None

    def get(self, repo: str, lora_path: str | None, scale: float):
        import torch
        from diffusers import Flux2KleinPipeline

        if self.pipe is None or self.repo != repo:
            if self.pipe is not None:
                del self.pipe
                torch.cuda.empty_cache()
            self.pipe = Flux2KleinPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16).to("cuda")
            self.repo = repo
            self.lora = None
            self.scale = None
        if lora_path != self.lora:
            if self.lora:
                self.pipe.unload_lora_weights()
            if lora_path:
                self.pipe.load_lora_weights(lora_path, adapter_name="removal")
            self.lora = lora_path
            self.scale = None
        if lora_path and self.scale != scale:
            self.pipe.set_adapters(["removal"], adapter_weights=[float(scale)])
            self.scale = scale
        return self.pipe


CACHE = PipelineCache()


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"_(\d{6,})$", path.stem)
    # The final save has no step suffix and should be the default choice.
    step = int(match.group(1)) if match else 10**9
    return step, path.name


def checkpoint_map() -> dict[str, str | None]:
    choices: dict[str, str | None] = {"baseline (no LoRA)": None}
    if OUTPUT_DIR.is_dir():
        paths = sorted(OUTPUT_DIR.glob("*.safetensors"), key=checkpoint_sort_key)
        for path in paths:
            choices[path.name] = str(path)
    return choices


def test_records() -> list[dict[str, str]]:
    manifest = PREPARED_DIR / "eval" / "manifest.json"
    if manifest.is_file():
        records = json.loads(manifest.read_text(encoding="utf-8"))
        selected = [record for record in records if record.get("split") == "test"]
    else:
        import csv

        dataset = assert_not_old_removal(DATASET_DIR)
        splits = {}
        with (dataset / "meta.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                splits[record["index"]] = record["split"]
        selected = []
        with (dataset / "mapping.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if splits.get(row["index"]) != "test":
                    continue
                reference = assert_not_old_removal(dataset / row["reference"])
                selected.append(
                    {
                        "index": row["index"],
                        "split": "test",
                        "reference": str(reference),
                        "prompt": PROMPT,
                    }
                )
    return selected[:8]


def as_pil(value) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("composite") or value.get("background")
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    return Image.fromarray(value).convert("RGB")


def prepare_image(image: Image.Image) -> tuple[Image.Image, int, int]:
    longest = min(max(image.size), 1024)
    width, height = fit_klein_size(image.width, image.height, longest)
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    return resized, width, height


def run_pipe(
    image: Image.Image,
    repo: str,
    lora_path: str | None,
    scale: float,
    steps: int,
    guidance: float,
    seed: int,
) -> tuple[Image.Image, Image.Image]:
    import torch

    resized, width, height = prepare_image(image)
    pipe = CACHE.get(repo, lora_path, scale)
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    with torch.inference_mode():
        return pipe(
            prompt=PROMPT,
            image=resized,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=width,
            height=height,
            generator=generator,
        ).images[0], resized


def remove_object(editor_value, model_label: str, checkpoint: str, scale: float, steps: int, guidance: float, seed: int):
    image = as_pil(editor_value)
    if image is None:
        raise gr.Error("Upload an image and draw a red circle on the object to remove.")
    repo, _default_steps = MODELS[model_label]
    lora_path = checkpoint_map().get(checkpoint)
    if checkpoint != "baseline (no LoRA)" and lora_path is None:
        raise gr.Error(f"Checkpoint not found: {checkpoint}")
    try:
        output, model_input = run_pipe(image, repo, lora_path, scale, steps, guidance, seed)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    return (model_input, output)


def run_held_out(model_label: str, checkpoint: str, scale: float, steps: int, guidance: float, seed: int):
    records = test_records()
    if not records:
        raise gr.Error("No test split found in removal_dataset.")
    repo, _default_steps = MODELS[model_label]
    lora_path = checkpoint_map().get(checkpoint)
    gallery = []
    for record in records:
        reference = Image.open(assert_not_old_removal(Path(record["reference"]))).convert("RGB")
        output, _model_input = run_pipe(reference, repo, lora_path, scale, steps, guidance, seed)
        gallery.append((output, record["index"]))
    return gallery


def default_steps(model_label: str) -> int:
    return MODELS[model_label][1]


def build() -> gr.Blocks:
    load_runtime_env()
    choices = list(checkpoint_map())
    with gr.Blocks(title="Red-circle removal") as demo:
        gr.Markdown("Xóa vật thể nằm trong vòng tròn đỏ, rồi xóa luôn vòng tròn.")
        gr.Textbox(value=PROMPT, label="Prompt", interactive=False)
        with gr.Row():
            with gr.Column(scale=1):
                model_label = gr.Radio(list(MODELS), value=list(MODELS)[0], label="Pipeline")
                checkpoint = gr.Dropdown(choices, value=choices[-1], label="Checkpoint")
                refresh = gr.Button("Refresh checkpoints")
                scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA scale")
                steps = gr.Slider(1, 50, value=4, step=1, label="Inference steps")
                guidance = gr.Slider(1.0, 8.0, value=4.0, step=0.5, label="Guidance")
                seed = gr.Number(value=42, precision=0, label="Seed")
            with gr.Column(scale=2):
                with gr.Tab("Scribble & Remove"):
                    editor = gr.ImageEditor(
                        type="pil",
                        label="Draw a red circle",
                        brush=gr.Brush(
                            default_size=12,
                            colors=["#FF0000"],
                            default_color="#FF0000",
                            color_mode="fixed",
                        ),
                        sources=["upload", "clipboard"],
                        height=640,
                    )
                    run = gr.Button("Remove", variant="primary")
                    slider = gr.ImageSlider(label="Before / after")
                with gr.Tab("Held-out gallery"):
                    gallery_button = gr.Button("Run 8 test samples")
                    gallery = gr.Gallery(label="Test outputs", columns=4, height=640)

        refresh.click(lambda: gr.Dropdown(choices=list(checkpoint_map())), outputs=checkpoint)
        model_label.change(default_steps, inputs=model_label, outputs=steps)
        run.click(
            remove_object,
            inputs=[editor, model_label, checkpoint, scale, steps, guidance, seed],
            outputs=slider,
        )
        gallery_button.click(
            run_held_out,
            inputs=[model_label, checkpoint, scale, steps, guidance, seed],
            outputs=gallery,
        )
    return demo


if __name__ == "__main__":
    build().launch(server_name="0.0.0.0", server_port=7860, share=True)
