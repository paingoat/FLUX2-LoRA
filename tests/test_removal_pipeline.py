"""CPU regression checks; no weights, network, GPU or Gradio server required."""
import contextlib
import csv
import importlib
import json
from pathlib import Path
import sys
import tempfile
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2
import skimage.metrics
from PIL import Image, ImageDraw
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import benchmark
import common
import prepare_dataset
import package_dataset
import prepare_vibe
import review_benchmark
import train


def images(color="gray"):
    source = Image.new("RGB", (96, 96), color)
    ImageDraw.Draw(source).rectangle((40, 40, 50, 50), fill="black")
    overlay = source.copy()
    ImageDraw.Draw(overlay).ellipse((20, 20, 75, 75), outline="red", width=2)
    return source, overlay


class RemovalPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = Path(__file__).resolve().parents[1] / ".local"
        cls.temp_root.mkdir(exist_ok=True)

    def test_mapping_preserves_splits_and_writes_dual_controls(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            root = Path(temp)
            dataset, sources, prepared = root / "dataset", root / "sources", root / "prepared"
            mapping, meta = [], []
            source, reference = images()
            for index, split in (("0001", "train"), ("0002", "validation"), ("0003", "test")):
                sample = f"remove_{index}"
                original = sources / sample / "original.png"
                original.parent.mkdir(parents=True)
                source.save(original)
                for folder, image in (("reference", reference), ("target", source)):
                    path = dataset / folder / f"{index}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(path)
                (dataset / "target" / f"{index}.txt").write_text(common.INSTRUCTION)
                mapping.append(dict(index=index, sample_id=sample, reference=f"reference/{index}.png",
                                    target_image=f"target/{index}.png", target_text=f"target/{index}.txt"))
                meta.append(dict(index=index, split=split))
            benchmark.write_csv(dataset / "mapping.csv", mapping)
            (dataset / "meta.jsonl").write_text("\n".join(json.dumps(r) for r in meta))
            with patch.object(prepare_dataset, "SAMPLE_CONTROL_INDEXES", ("0002",)):
                prepare_dataset.prepare(dataset, prepared, sources)
            self.assertEqual([p.name for p in (prepared / "original").glob("*.png")], ["0001.png"])
            self.assertEqual((prepared / "target/0001.txt").read_text().strip(), common.PROMPT)
            records = benchmark.load_manifest(prepared / "eval/manifest.json")
            self.assertEqual({r["split"] for r in records}, {"validation", "test"})
            self.assertTrue((prepared / "sample_controls/0002_original.png").exists())
            self.assertEqual((dataset / "original/0001.png").read_bytes(), (sources / "remove_0001/original.png").read_bytes())
            archive = package_dataset.package(dataset, root / "upload.tar.gz")
            with tarfile.open(archive) as bundle:
                self.assertEqual(len(bundle.getnames()), 14)
                self.assertIn("data/removal_dataset/original/0001.png", bundle.getnames())
                self.assertFalse(any("mask/" in name for name in bundle.getnames()))
            self.assertTrue((root / "upload.tar.gz.sha256").is_file())
            # A duplicate mapping must fail before altering prepared data.
            benchmark.write_csv(dataset / "mapping.csv", mapping + [mapping[0]])
            with self.assertRaises(SystemExit):
                prepare_dataset.load_rows(dataset)

    def test_metrics_separate_preservation_from_removal(self):
        source, reference = images()
        unchanged = benchmark.score_pair(source, reference, source, None)
        self.assertEqual(unchanged["scribble_residue"], 0)
        self.assertEqual(unchanged["preserve_psnr"], 99)
        self.assertAlmostEqual(unchanged["preserve_ssim"], 1)
        self.assertIsNone(unchanged["target_lpips"])
        marked = benchmark.score_pair(source, reference, reference, None)
        self.assertEqual(marked["scribble_residue"], 1)
        damaged = source.copy()
        ImageDraw.Draw(damaged).rectangle((0, 0, 15, 95), fill="white")
        bad = benchmark.score_pair(source, reference, damaged, None)
        self.assertLess(bad["preserve_ssim"], unchanged["preserve_ssim"])
        self.assertLess(bad["preserve_psnr"], unchanged["preserve_psnr"])

    def test_ssim_excludes_windows_touching_edit(self):
        source, reference = images()
        difference = np.abs(np.asarray(source).astype(int) - np.asarray(reference).astype(int)).max(2) > 8
        region = benchmark.edit_region(difference)
        output = np.asarray(source).copy()
        output[region] = 255
        self.assertAlmostEqual(benchmark.masked_ssim(np.asarray(source), output, region), 1)

    def test_dual_input_order_in_app_and_benchmark(self):
        with patch.dict(sys.modules, {"gradio": SimpleNamespace(Error=ValueError)}):
            app = importlib.import_module("app")
        source, reference = images()
        prepared = app.editor_inputs({"background": source, "composite": reference})
        self.assertEqual(prepared[0].tobytes(), source.tobytes())
        self.assertEqual(prepared[1].tobytes(), reference.tobytes())
        with self.assertRaises(ValueError):
            app.editor_inputs({"composite": reference})
        generator = SimpleNamespace(manual_seed=lambda seed: seed)
        fake_torch = SimpleNamespace(Generator=lambda **kwargs: generator, inference_mode=contextlib.nullcontext)
        pipe = Mock(return_value=SimpleNamespace(images=[source]))
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(app.CACHE, "get", return_value=pipe):
            app.run_pipe(prepared, "repo", None, 1, 4, 4, 42)
            self.assertIs(pipe.call_args.kwargs["image"][0], prepared[0])
            self.assertIs(pipe.call_args.kwargs["image"][1], prepared[1])
            benchmark.generate(pipe, prepared, common.PROMPT, 42)
            self.assertEqual(pipe.call_args.kwargs["prompt"], common.PROMPT)

    def test_config_matches_training_prompt_and_order(self):
        text = common.TRAIN_CONFIG.read_text(encoding="utf-8")
        config = yaml.safe_load(text)
        process = config["config"]["process"][0]
        dataset = process["datasets"][0]
        self.assertTrue(dataset["control_path_1"].endswith("/original"))
        self.assertTrue(dataset["control_path_2"].endswith("/reference"))
        self.assertEqual(len(train.config_paths(text)), 11)
        self.assertEqual(process["sample"]["samples"][0]["prompt"], common.PROMPT)
        self.assertEqual(config["config"]["name"], common.OUTPUT_DIR.name)

    def test_vibe_schema_prompt_and_overlap(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            root = Path(temp)
            internal, vibe, output = root / "internal", root / "vibe", root / "output"
            (internal / "original").mkdir(parents=True)
            source, _ = images()
            source.save(internal / "original/0001.png")
            task = vibe / prepare_vibe.TASK
            (task / "annotated_images/layers").mkdir(parents=True)
            (vibe / "Common_Corpus").mkdir()
            records = []
            for index, color in (("Removal_0001", "gray"), ("Removal_0002", "green")):
                original, _ = images(color)
                original.save(vibe / f"Common_Corpus/{index}.png")
                layer = Image.new("RGBA", original.size)
                ImageDraw.Draw(layer).ellipse((20, 20, 75, 75), outline="red", width=2)
                layer.save(task / f"annotated_images/layers/{index}.png")
                records.append({"id": index, "file_paths": {"source": f"Common_Corpus/{index}.png",
                                "visual_instruction": f"annotated_images/layers/{index}.png", "target": ""},
                                "text_prompt": {"input_prompt": "Remove all the objects in the marked region."}})
            (task / "Removal.json").write_text(json.dumps(records))
            manifest = prepare_vibe.prepare(vibe, output, internal)
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["index"], "Removal_0002")
            self.assertIsNone(manifest[0]["target"])
            self.assertTrue(manifest[0]["prompt"].endswith(records[1]["text_prompt"]["input_prompt"]))
            self.assertEqual(benchmark.load_manifest(output / "manifest.json")[0]["split"], "external")
            with self.assertRaises(ValueError):
                prepare_vibe.inside(vibe, "../escape")

    def test_reports_keep_splits_separate_and_review_pending(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            root = Path(temp)
            rows = []
            for identity, split in (("a", "validation"), ("b", "test")):
                rows.append(dict(review_id=identity, dataset="internal", split=split, checkpoint="baseline",
                                 resolution=512, status="ok", prompt=common.PROMPT, sheet=str(root / "sheet.jpg"),
                                 **dict.fromkeys(benchmark.METRICS, 0.5)))
            benchmark.write_csv(root / "results.csv", rows)
            benchmark.write_report(rows, root / "report.md")
            review_benchmark.create_review(root / "results.csv")
            page = (root / "review.html").read_text(encoding="utf-8")
            self.assertNotIn("baseline", page)
            ratings = [dict(review_id="a", **dict.fromkeys(review_benchmark.CRITERIA, "1"), notes=""),
                       dict(review_id="b", **dict.fromkeys(review_benchmark.CRITERIA, ""), notes="")]
            benchmark.write_csv(root / "review.csv", ratings)
            report = review_benchmark.summarize(root / "results.csv", root / "review.csv").read_text()
            self.assertIn("1/1 (100.0%)", report)
            self.assertIn("pending", report)
            self.assertIn("| internal | validation |", report)
            self.assertIn("| internal | test |", report)
            ratings[0]["removal_success"] = "maybe"
            benchmark.write_csv(root / "review.csv", ratings)
            with self.assertRaises(ValueError):
                review_benchmark.summarize(root / "results.csv", root / "review.csv")

    def test_benchmark_driver_records_failed_generation_and_blinds_review(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            root = Path(temp)
            source, reference = images()
            source.save(root / "original.png")
            reference.save(root / "reference.png")
            record = dict(index="Removal_0001", dataset="vibe_removal", split="external",
                          original=str(root / "original.png"), reference=str(root / "reference.png"),
                          target=None, instruction="Remove all marked objects.",
                          prompt=common.edit_prompt("Remove all marked objects."))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps([record]))
            checkpoint = root / "removal_000000250.safetensors"
            checkpoint.write_bytes(b"mock-checkpoint")
            class OOM(RuntimeError):
                pass
            generator = SimpleNamespace(manual_seed=lambda seed: seed)
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, empty_cache=Mock(),
                                                         OutOfMemoryError=OOM), bfloat16="bf16",
                                    Generator=lambda **kwargs: generator, inference_mode=contextlib.nullcontext)
            pipe = Mock(side_effect=[SimpleNamespace(images=[source]), OOM()])
            pipe.to.return_value = pipe
            model = SimpleNamespace(from_pretrained=Mock(return_value=pipe))
            argv = ["benchmark.py", "--manifest", str(manifest), "--split", "external", "--baseline",
                    "--checkpoint", str(checkpoint), "--resolutions", "96", "--output", str(root / "results")]
            with patch.dict(sys.modules, {"torch": torch, "diffusers": SimpleNamespace(Flux2KleinPipeline=model)}), \
                    patch.object(sys, "argv", argv), patch.object(benchmark, "load_runtime_env"):
                self.assertEqual(benchmark.main(), 0)
            rows = review_benchmark.read_csv(root / "results/results.csv")
            self.assertEqual([row["status"] for row in rows], ["ok", "error"])
            self.assertEqual(rows[1]["error"], "CUDA OOM")
            page = (root / "results/review.html").read_text(encoding="utf-8")
            self.assertNotIn("removal_000000250", page)
            self.assertNotIn("file:///", page)  # downloadable folder stays portable
            self.assertEqual(len(review_benchmark.read_csv(root / "results/review.csv")), 1)
            self.assertEqual(len(pipe.call_args.kwargs["image"]), 2)


if __name__ == "__main__":
    unittest.main()
