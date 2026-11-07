"""CPU checks for the objective, existing-mask preparation and stale-data guards."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))
from region_loss import region_weights
import prepare_region_masks as masks
import train
import patch_region_trainer as trainer_patch
import run_region_trainer


class RegionLossTests(unittest.TestCase):
    def test_text_cache_batch_is_one_before_entering_trainer(self):
        class CacheItem:
            def get_text_embedding_info_dict(self, caption_override=None, **kwargs):
                return {'caption': caption_override or 'remove object'}
        mixins = SimpleNamespace(TEXT_EMBED_CACHE_BATCH=16, TextEmbeddingFileItemDTOMixin=CacheItem)
        with patch.dict(sys.modules, {'toolkit': SimpleNamespace(dataloader_mixins=mixins)}), \
                patch.dict('os.environ', {'FLUX_REGION_SEED': '42', 'FLUX_REGION_TE_CACHE_TAG': 'bf16'}), \
                patch.object(sys, 'argv', ['run_region_trainer.py', 'config.yaml']), \
                patch.object(run_region_trainer.runpy, 'run_path') as run:
            run.side_effect = lambda *args, **kwargs: self.assertEqual(mixins.TEXT_EMBED_CACHE_BATCH, 1)
            run_region_trainer.main()
            run.assert_called_once_with('run.py', run_name='__main__')
            info = CacheItem().get_text_embedding_info_dict(caption_override='different caption', text_only=True)
            self.assertEqual(info, {'caption': 'different caption', 'region_text_encoder_config': 'bf16'})
            bf16_hash = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
            with patch.dict('os.environ', {'FLUX_REGION_TE_CACHE_TAG': 'fp8'}):
                run_region_trainer.configure_text_cache()
            fp8_info = CacheItem().get_text_embedding_info_dict(caption_override='different caption', text_only=True)
            self.assertNotEqual(bf16_hash, hashlib.sha256(json.dumps(fp8_info, sort_keys=True).encode()).hexdigest())

    def test_training_progress_does_not_count_preparation_bars(self):
        self.assertIsNone(train.parse_trainer_line('Caching text embeddings to disk: 60%|###| 51/85 [00:12]'))
        self.assertIsNone(train.parse_trainer_line('Caching latents: 100%|###| 85/85 [00:12]'))
        self.assertEqual(train.parse_trainer_line('removal_region: 2%|#| 1/50 [00:01, loss: 0.123]')['step'], 1)

    def test_weights_loss_and_gradient_match_hand_calculation(self):
        pred = torch.tensor([[[[1., 2.], [3., 4.]]]], requires_grad=True)
        mask = torch.tensor([[[[0., 1.], [0., 1.]]]])
        weights = region_weights(mask, 1.5)
        loss = (weights * pred.square()).mean()
        self.assertEqual(loss.item(), (1 + 2.5*4 + 9 + 2.5*16)/4)
        loss.backward()
        torch.testing.assert_close(pred.grad, 2 * pred.detach() * weights / 4)
        torch.testing.assert_close(region_weights(mask, 0), torch.ones_like(mask))
        torch.testing.assert_close(region_weights(torch.zeros_like(mask), 1.5), torch.ones_like(mask))
        torch.testing.assert_close(region_weights(torch.ones_like(mask), 1.5), torch.full_like(mask, 2.5))
        for strength in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                region_weights(mask, strength)
        with self.assertRaises(ValueError):
            region_weights(mask * 2, 1.5)

    def test_preparation_reuses_masks_keeps_splits_and_detects_stale_inputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.local') as temp:
            root = Path(temp)
            prepared, dataset, output = root/'prepared', root/'source', root/'region'
            for folder in ('original', 'reference', 'target', 'eval'):
                (prepared/folder).mkdir(parents=True)
            for folder in ('original', 'reference', 'target', 'mask'):
                (dataset/folder).mkdir(parents=True)
            rows = []
            source = Image.new('RGB', (96, 64), 'gray')
            reference = source.copy()
            ImageDraw.Draw(reference).ellipse((55, 15, 85, 45), outline='red', width=2)
            obj = Image.new('L', source.size)
            ImageDraw.Draw(obj).rectangle((10, 20, 20, 30), fill=255)
            for index, split in (('0001','train'), ('0002','validation')):
                row = dict(index=index, sample_id=f'id_{index}', split=split, original=f'original/{index}.png',
                           reference=f'reference/{index}.png', target_image=f'target/{index}.png')
                rows.append(row)
                for folder, image in (('original',source),('reference',reference),('target',source),('mask',obj)):
                    image.save(dataset/folder/f'{index}.png')
                    if split == 'train' and folder != 'mask':
                        image.save(prepared/folder/f'{index}.png')
            (prepared/'dataset_card.json').write_text(json.dumps(dict(source=str(dataset), train_indexes=['0001'])))
            (prepared/'eval/manifest.json').write_text(json.dumps([dict(index='0002',split='validation')]))
            before = hashlib.sha256((dataset/'mask/0001.png').read_bytes()).hexdigest()
            with patch.object(masks, 'load_rows', return_value=rows):
                masks.prepare(prepared, output, radius_at_512=5)
            self.assertEqual([p.name for p in (output/'train').glob('*.png')], ['0001.png'])
            region = np.asarray(Image.open(output/'region/0001.png'))
            self.assertEqual(region[25,15], 255)  # existing object, separate from annotation
            self.assertEqual(region[15,70], 255)  # annotation outside object
            self.assertEqual(region[0,0], 0)
            self.assertGreater(np.count_nonzero(np.asarray(Image.open(output/'train/0001.png'))), np.count_nonzero(region))
            self.assertEqual(before, hashlib.sha256((dataset/'mask/0001.png').read_bytes()).hexdigest())
            config = dict(folder_path=str(prepared/'target'), control_path_1=str(prepared/'original'),
                          control_path_2=str(prepared/'reference'), mask_path=str(output/'train'))
            train.validate_region_dataset(config)
            source.save(prepared/'reference/0001.png')
            with self.assertRaisesRegex(ValueError, 'stale'):
                train.validate_region_dataset(config)
            with patch.object(masks, 'load_rows', return_value=rows), self.assertRaises(ValueError):
                masks.prepare(prepared, output)

    def test_ab_configs_differ_only_in_name_and_weight(self):
        configs = [yaml.safe_load((ROOT/'core/configs'/name).read_text()) for name in
                   ('removal_uniform_5090.yaml','removal_region_5090.yaml')]
        a,b = copy.deepcopy(configs)
        self.assertEqual(a['region_loss']['strength'], 0)
        self.assertEqual(b['region_loss']['strength'], 1.5)
        self.assertIs(a['config']['process'][0]['model']['quantize_te'], False)
        self.assertEqual(a['config']['process'][0]['model']['te_dtype'], 'bf16')
        a['region_loss']['strength'] = b['region_loss']['strength']
        a['config']['name'] = b['config']['name']
        self.assertEqual(a,b)

    def test_patch_refuses_unrecognized_source(self):
        with self.assertRaises(ValueError):
            trainer_patch.patched_source('class SDTrainer: pass')

    def test_portable_config_uses_repo_paths_and_custom_output(self):
        original = yaml.safe_load((ROOT/'core/configs/removal_region_5090.yaml').read_text())
        with tempfile.TemporaryDirectory(dir=ROOT / '.local') as temp:
            root = Path(temp) / 'repo with spaces'
            prepared = root / '.local/removal_dual'
            with patch.object(train, 'REPO_ROOT', root), patch.object(train, 'PREPARED_DIR', prepared), \
                    patch.dict('os.environ', {'REMOVAL_TRAINING_FOLDER': str(root/'custom output')}):
                resolved = train.resolve_training_paths(copy.deepcopy(original))
                process = resolved['config']['process'][0]
                self.assertEqual(Path(process['training_folder']), root/'custom output')
                self.assertEqual(Path(process['datasets'][0]['mask_path']), prepared/'region_loss/train')
                self.assertEqual(Path(process['sample']['samples'][0]['ctrl_img_1']), prepared/'sample_controls/0015_original.png')
                self.assertNotIn('${', yaml.safe_dump(resolved))
                self.assertEqual(train.resolve_training_paths(copy.deepcopy(resolved)), resolved)
                process['training_folder'] = '${UNKNOWN_PATH}/output'
                with self.assertRaisesRegex(ValueError, 'unknown path variable'):
                    train.resolve_training_paths(resolved)

    def test_region_metric_excludes_edit_and_detects_background_damage(self):
        import benchmark
        class PerceptualStub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
            def forward(self, a, b):
                return (a-b).square().mean() + self.anchor * 0
        perceptual = PerceptualStub()
        source = Image.new('RGB', (96,96), 'gray')
        reference = source.copy()
        ImageDraw.Draw(reference).ellipse((30,30,65,65), outline='red', width=2)
        region = np.zeros((96,96), np.uint8)
        region[25:70,25:70] = 255
        edited = source.copy()
        ImageDraw.Draw(edited).rectangle((35,35,55,55), fill='white')
        score = benchmark.score_pair(source,reference,edited,source,perceptual,region_mask=region)
        self.assertEqual(score['preserve_psnr'], 99)
        self.assertEqual(score['preserve_psnr_d8'], 99)
        self.assertEqual(score['preserve_lpips_composite'], 0)
        self.assertGreater(score['target_lpips'], 0)
        self.assertGreater(score['edit_lpips'], 0)
        ImageDraw.Draw(edited).rectangle((0,0,15,95), fill='black')
        bad = benchmark.score_pair(source,reference,edited,source,perceptual,region_mask=region)
        self.assertLess(bad['preserve_psnr'], score['preserve_psnr'])
        self.assertLess(bad['preserve_ssim'], score['preserve_ssim'])
        self.assertGreater(bad['preserve_lpips_composite'], score['preserve_lpips_composite'])


if __name__ == '__main__':
    unittest.main()
