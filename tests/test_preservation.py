"""CPU checks for scientific scoring, reviewed inputs and single-image wiring."""
import contextlib
import csv
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
import benchmark_preservation as bench
import common
import prepare_preservation as prep
import preservation_runtime as runtime


class PreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = Path(__file__).resolve().parents[1] / '.local'
        cls.temp_root.mkdir(exist_ok=True)

    def setUp(self):
        self.source = Image.new('RGB', (96, 96), (100, 120, 140))
        self.mask = Image.new('L', self.source.size)
        ImageDraw.Draw(self.mask).rectangle((35, 35, 60, 60), fill=255)

    def prepared(self, root):
        data = root / 'data'
        folder = data / 'remove_1'
        folder.mkdir(parents=True)
        self.source.save(folder / 'original.png')
        self.mask.save(folder / 'object_mask.png')
        sample = dict(id='remove_1', object_prompt='the cup', instruction='Remove the cup.', split='train')
        selection = dict(count=1, seed=42, longest=96, alpha=0.4, samples=[sample])
        prep.write_json(data / 'selection.json', selection)
        prep.materialize(data, selection, {'remove_1': {'model_revision': 'test', 'code_revision': 'test'}})
        return data

    def approve(self, data):
        path = data / 'mask_review.csv'
        with path.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        rows[0]['approved'] = '1'
        bench.write_csv(path, rows)

    def test_psnr_uses_outside_pixels_and_exact_denominator(self):
        changed = self.source.copy()
        changed.paste((255, 255, 255), mask=self.mask)
        self.assertEqual(bench.psnr_outside(self.source, changed, self.mask), math.inf)
        array = np.asarray(self.source).copy()
        array[np.asarray(self.mask) == 0] += 10
        actual = bench.psnr_outside(self.source, Image.fromarray(array), self.mask)
        self.assertAlmostEqual(actual, 20 * math.log10(255 / 10))
        self.assertTrue(math.isfinite(bench.psnr_outside(self.source, Image.fromarray(array), self.mask, 8)))
        with self.assertRaises(ValueError):
            bench.psnr_outside(self.source, self.source.resize((32, 32)), self.mask)
        with self.assertRaises(ValueError):
            bench.psnr_outside(self.source, self.source, Image.new('L', self.source.size, 255))

    def test_dilation_excludes_only_boundary_band(self):
        changed = self.source.copy()
        changed.putpixel((32, 40), (0, 0, 0))
        self.assertTrue(math.isfinite(bench.psnr_outside(self.source, changed, self.mask)))
        self.assertEqual(bench.psnr_outside(self.source, changed, self.mask, 8), math.inf)

    def test_overlay_geometry_and_determinism(self):
        scribble, tint, stroke = prep.overlays(self.source, self.mask)
        original = np.asarray(self.source)
        region = np.asarray(self.mask) > 0
        np.testing.assert_array_equal(np.asarray(tint)[~region], original[~region])
        np.testing.assert_array_equal(np.asarray(scribble)[np.asarray(stroke) == 0], original[np.asarray(stroke) == 0])
        np.testing.assert_array_equal(np.asarray(tint)[40, 40], [162, 72, 84])
        self.assertEqual(scribble.tobytes(), prep.overlays(self.source, self.mask)[0].tobytes())

    def test_overlay_prompts_never_include_object_metadata(self):
        cup = prep.prompts('Remove the distinctive purple cup.')
        person = prep.prompts('Remove the blonde-haired person on the left.')
        self.assertNotEqual(cup['text'], person['text'])
        for mode in ('scribble', 'mask'):
            self.assertEqual(cup[mode], person[mode])
            self.assertNotIn('purple', cup[mode])
            self.assertNotIn('blonde', person[mode])

    def test_legacy_or_leaking_prompt_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            data = self.prepared(Path(temp))
            self.approve(data)
            path = data / 'manifest.json'
            manifest = json.loads(path.read_text(encoding='utf-8'))
            del manifest['prompt_protocol']
            prep.write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, 'old prompt protocol'):
                bench.load_manifest(path)
            manifest['prompt_protocol'] = prep.PROMPT_PROTOCOL
            manifest['samples'][0]['prompts']['mask'] += ' Remove the cup.'
            prep.write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, 'prompts differ'):
                bench.load_manifest(path)

    def test_review_gate_and_hash_changes(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            data = self.prepared(Path(temp))
            with self.assertRaisesRegex(ValueError, 'mask pending'):
                bench.load_manifest(data / 'manifest.json')
            self.approve(data)
            _, records = bench.load_manifest(data / 'manifest.json')
            self.assertEqual(len(records), 1)
            self.source.save(data / 'remove_1/mask_overlay.png')
            with self.assertRaisesRegex(ValueError, 'asset changed'):
                bench.load_manifest(data / 'manifest.json')

    def test_selection_unique_deterministic_and_no_spatial_text(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            root = Path(temp)
            self.source.save(root / 'source.png')
            rows = [dict(id=f'remove_{i}', operation='remove', review_status='auto_accepted',
                         original_file_name='source.png', source_key=f'source_{i}',
                         prompt_object='the cup', prompt_short='Remove inside the oval.') for i in range(5)]
            rows.append({**rows[0], 'id': 'duplicate_source'})
            path = root / 'metadata.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            a, b = prep.select_samples(path, 5, 42), prep.select_samples(path, 5, 42)
            self.assertEqual(a, b)
            self.assertEqual(len({r['source_key'] for r in a}), 5)
            self.assertNotIn('oval', a[0]['instruction'])
            with self.assertRaises(ValueError):
                prep.select_samples(path, 6, 42)

    def test_existing_masks_run_without_sam3_and_resume_after_missing_mask(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            root = Path(temp)
            samples = []
            for i in range(2):
                folder = root / 'sources' / str(i)
                folder.mkdir(parents=True)
                self.source.save(folder / 'original.png')
                if i == 1:
                    self.mask.save(folder / 'mask.png')
                samples.append(dict(id=f'remove_{i}', original=str(folder / 'original.png'),
                    source_sha256=prep.digest(folder / 'original.png'), existing_mask=str(folder / 'mask.png'),
                    object_prompt='the cup', instruction='Remove the cup.', split='train'))
            selection = dict(count=2, seed=42, longest=64, alpha=0.4, mask_source='existing', samples=samples)
            out = root / 'prepared'
            out.mkdir()
            prep.write_json(out / 'selection.json', selection)
            with contextlib.redirect_stdout(io.StringIO()):
                errors = prep.prepare_masks(out, selection, {})
            self.assertEqual([r['id'] for r in errors], ['remove_0'])
            self.assertTrue((out / 'remove_1/object_mask.png').is_file())
            self.assertFalse((out / 'manifest.json').exists())
            self.mask.save(root / 'sources/0/mask.png')
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(prep.prepare_masks(out, selection, {}), [])
            self.assertIn('remove_1: cached existing mask', stdout.getvalue())
            _, records = bench.load_manifest(out / 'manifest.json', allow_unreviewed=True)
            self.assertTrue(all(r['mask_source'] == 'existing' for r in records))
            self.assertTrue(all(r['mask_review_status'] == 'unreviewed-existing-pilot' for r in records))
            with Image.open(records[0]['paths']['object_mask']) as image:
                self.assertEqual(image.size, (64, 64))
            with self.assertRaisesRegex(ValueError, 'mask pending'):
                bench.load_manifest(out / 'manifest.json')
            with (out / 'mask_review.csv').open(newline='', encoding='utf-8') as handle:
                review = list(csv.DictReader(handle))
            review[0]['approved'] = '0'
            bench.write_csv(out / 'mask_review.csv', review)
            with self.assertRaisesRegex(ValueError, 'mask pending'):
                bench.load_manifest(out / 'manifest.json', allow_unreviewed=True)

    def test_allow_unreviewed_never_bypasses_sam3_review(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            data = self.prepared(Path(temp))
            with self.assertRaisesRegex(ValueError, 'mask pending'):
                bench.load_manifest(data / 'manifest.json', allow_unreviewed=True)

    def test_sam3_no_detection_continues_and_success_is_reused(self):
        class Tensor:
            def __init__(self, value): self.value = np.asarray(value)
            def detach(self): return self
            def float(self): return self
            def cpu(self): return self
            def numpy(self): return self.value

        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            root = Path(temp)
            samples = []
            for i in range(2):
                image_path = root / f'source_{i}.png'
                self.source.save(image_path)
                samples.append(dict(id=f'remove_{i}', original=str(image_path),
                    source_sha256=prep.digest(image_path), object_prompt='the cup', instruction='Remove the cup.'))
            selection = dict(count=2, seed=42, longest=96, alpha=0.4, mask_source='sam3', samples=samples)
            out = root / 'prepared'
            out.mkdir()
            prep.write_json(out / 'selection.json', selection)
            success = dict(scores=Tensor([0.9]), masks=Tensor((np.asarray(self.mask) > 127)[None, None]))
            empty = dict(scores=Tensor([]), masks=Tensor(np.empty((0, 1, 96, 96))))
            predictor = Mock(side_effect=[empty, success])
            with patch.dict(sys.modules, torch=SimpleNamespace(cuda=Mock())), contextlib.redirect_stdout(io.StringIO()):
                errors = prep.prepare_masks(out, selection, {}, predictor, {'sam3_model_revision': 'test'})
                self.assertEqual(len(errors), 1)
                predictor.side_effect = [success]
                self.assertEqual(prep.prepare_masks(out, selection, {}, predictor, {'sam3_model_revision': 'test'}), [])
                self.assertEqual(predictor.call_count, 3)
            self.assertTrue((out / 'manifest.json').is_file())

    def test_generate_passes_one_image_and_fixed_distilled_settings(self):
        fake_torch = SimpleNamespace(inference_mode=contextlib.nullcontext,
                                     Generator=Mock(return_value=Mock()))
        pipe = Mock(return_value=SimpleNamespace(images=[self.source]))
        with patch.dict(sys.modules, torch=fake_torch):
            bench.generate(pipe, self.source, 'Remove the cup.', 42)
            self.assertIs(pipe.call_args.kwargs['image'], self.source)
            self.assertEqual(pipe.call_args.kwargs['num_inference_steps'], 4)
            self.assertEqual(pipe.call_args.kwargs['guidance_scale'], 1.0)
            with self.assertRaises(ValueError):
                bench.generate(pipe, [self.source], 'Remove.', 42)

    def test_sam3_grounding_api_keeps_features_and_prompts_under_autocast(self):
        active = []

        @contextlib.contextmanager
        def autocast(**kwargs):
            self.assertEqual(kwargs, dict(device_type='cuda', dtype='bf16'))
            active.append(True)
            yield
            active.pop()

        state = {}
        def check_image(image):
            self.assertTrue(active)
            self.assertIs(image, self.source)
            return state

        def check_text(*, state: dict, prompt: str):
            self.assertTrue(active)
            self.assertEqual(prompt, 'the cup')
            return state

        def check_box(*, state: dict, box: list, label: bool):
            self.assertTrue(active)
            self.assertEqual(box, [0.5, 0.5, 0.2, 0.2])
            self.assertTrue(label)
            return state

        processor = SimpleNamespace(set_image=check_image, set_text_prompt=check_text, add_geometric_prompt=check_box)
        fake_torch = SimpleNamespace(inference_mode=contextlib.nullcontext, autocast=autocast, bfloat16='bf16')
        with patch.dict(sys.modules, torch=fake_torch):
            self.assertIs(prep.segment_image(processor, self.source, 'the cup', [0.5, 0.5, 0.2, 0.2]), state)
            with self.assertRaises(ValueError):
                prep.segment_image(processor, self.source, 'the cup', [0, 0, 0.2, 0.2])

    def test_full_runner_records_failure_and_excludes_incomplete_triples(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            root = Path(temp)
            data = self.prepared(root)
            self.approve(data)
            output = root / 'run'
            fake_torch = SimpleNamespace(bfloat16='bf16', cuda=Mock())
            fake_torch.cuda.max_memory_allocated.return_value = 1024
            fake_diffusers = SimpleNamespace(Flux2KleinPipeline=Mock(), __version__='test')
            with patch.dict(sys.modules, torch=fake_torch, diffusers=fake_diffusers), \
                    patch.object(runtime, 'check_cuda', return_value={'gpu': 'fake'}), \
                    patch.object(runtime, 'read_lock', return_value={'flux_model_revision': 'abc'}), \
                    patch.object(bench, 'load_runtime_env'), \
                    patch.object(bench, 'generate', side_effect=[self.source, RuntimeError('test failure'), self.source]), \
                    patch.object(sys, 'argv', ['benchmark', '--manifest', str(data / 'manifest.json'), '--output', str(output)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(bench.main(), 1)
            with (output / 'results.csv').open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r['status'] for r in rows], ['ok', 'error', 'ok'])
            self.assertIn('Complete triples: 0/1', (output / 'report.md').read_text())
            self.assertTrue((output / 'gallery.html').is_file())
            self.assertTrue((output / 'review.html').is_file())

    def test_cache_aliases_are_consistent(self):
        with patch.dict(common.os.environ, {'HF_HOME': '/backup/data/art-gen', 'HF_HUB_CACHE': '/backup/data/art-gen/hub',
                                           'HUGGINGFACE_HUB_CACHE': '/old/cache'}, clear=True), \
                patch.object(common, 'REPO_ROOT', Path('nonexistent-root')):
            common.load_runtime_env()
            self.assertEqual(common.os.environ['HUGGINGFACE_HUB_CACHE'], '/backup/data/art-gen/hub')


if __name__ == '__main__':
    unittest.main()
