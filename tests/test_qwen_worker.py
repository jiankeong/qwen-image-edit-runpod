import base64
import io
import os
import errno
import tempfile
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

import handler


class QwenWorkerTests(unittest.TestCase):
    def test_nf4_auto_offload_selects_model_for_24gb_gpu(self):
        pipe = SimpleNamespace(
            enable_sequential_cpu_offload=unittest.mock.Mock(),
            enable_model_cpu_offload=unittest.mock.Mock(),
        )
        self.assertEqual(handler.configure_offload(pipe, 23.5), 'model')
        pipe.enable_model_cpu_offload.assert_called_once_with()
        pipe.enable_sequential_cpu_offload.assert_not_called()

    def test_sequential_offload_remains_explicit_option(self):
        pipe = SimpleNamespace(
            enable_sequential_cpu_offload=unittest.mock.Mock(),
            enable_model_cpu_offload=unittest.mock.Mock(),
        )
        self.assertEqual(handler.configure_offload(pipe, 23.5, 'sequential'), 'sequential')
        pipe.enable_sequential_cpu_offload.assert_called_once_with()

    def test_offload_selects_model_for_large_gpu(self):
        pipe = SimpleNamespace(
            enable_sequential_cpu_offload=unittest.mock.Mock(),
            enable_model_cpu_offload=unittest.mock.Mock(),
        )
        self.assertEqual(handler.configure_offload(pipe, 79.0), 'model')
        pipe.enable_model_cpu_offload.assert_called_once_with()

    def test_target_inference(self):
        self.assertEqual(handler.choose_target('把衣服换成蓝色'), 'clothes')
        self.assertEqual(handler.choose_target('换成海边风景'), 'background')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            handler.choose_target('make it beautiful')

    def test_masks_and_exact_composite(self):
        labels = np.zeros((16, 16), dtype=np.int64)
        labels[:8, :] = 4
        clothes = handler.make_mask(labels, 'clothes')
        background = handler.make_mask(labels, 'background')
        self.assertTrue(np.all(clothes[:8]))
        self.assertTrue(np.all(background[8:]))
        original = Image.new('RGB', (16, 16), (10, 20, 30))
        generated = Image.new('RGB', (16, 16), (200, 100, 50))
        result = handler.composite_exact(original, generated, clothes)
        self.assertEqual(result.getpixel((0, 0)), (200, 100, 50))
        self.assertEqual(result.getpixel((0, 15)), (10, 20, 30))

    def test_one_image_handler(self):
        image = Image.new('RGB', (16, 16), (10, 20, 30))
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        labels = np.zeros((16, 16), dtype=np.int64)
        labels[:8] = 4
        fake_pipeline = lambda **kwargs: SimpleNamespace(images=[Image.new('RGB', (16, 16), (200, 100, 50))])
        with patch.object(handler, 'parser_labels', return_value=labels), patch.object(handler, 'get_pipeline', return_value=fake_pipeline):
            response = handler.handler({'input': {
                'image': base64.b64encode(buffer.getvalue()).decode(),
                'prompt': 'Change the shirt to red', 'edit_target': 'clothes', 'output_mode': 'masked'
            }})
        result = Image.open(io.BytesIO(base64.b64decode(response['image'])))
        self.assertEqual(result.getpixel((0, 0)), (200, 100, 50))
        self.assertEqual(result.getpixel((0, 15)), (10, 20, 30))
        self.assertEqual(response['edit_target'], 'clothes')
        self.assertNotIn('raw_image', response)

    def test_raw_output_exposes_unmasked_lora_result(self):
        image = Image.new('RGB', (16, 16), (10, 20, 30))
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        labels = np.zeros((16, 16), dtype=np.int64)
        labels[:8] = 4
        generated = Image.new('RGB', (16, 16), (200, 100, 50))
        fake_pipeline = lambda **kwargs: SimpleNamespace(images=[generated])
        with patch.object(handler, 'parser_labels', return_value=labels), \
             patch.object(handler, 'get_pipeline', return_value=fake_pipeline):
            response = handler.handler({'input': {
                'image': base64.b64encode(buffer.getvalue()).decode(),
                'prompt': 'Change the shirt', 'edit_target': 'clothes', 'output_mode': 'masked', 'return_raw': True,
            }})
        final = Image.open(io.BytesIO(base64.b64decode(response['image'])))
        raw = Image.open(io.BytesIO(base64.b64decode(response['raw_image'])))
        self.assertEqual(final.getpixel((0, 15)), (10, 20, 30))
        self.assertEqual(raw.getpixel((0, 15)), (200, 100, 50))

    def test_raw_mode_returns_full_lora_output_without_parser(self):
        image = Image.new('RGB', (16, 16), (10, 20, 30))
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        generated = Image.new('RGB', (16, 16), (200, 100, 50))
        fake_pipeline = lambda **kwargs: SimpleNamespace(images=[generated])
        with patch.object(handler, 'parser_labels') as parser, \
             patch.object(handler, 'get_pipeline', return_value=fake_pipeline):
            response = handler.handler({'input': {
                'image': base64.b64encode(buffer.getvalue()).decode(),
                'prompt': 'Transform the image',
            }})
        parser.assert_not_called()
        output = Image.open(io.BytesIO(base64.b64decode(response['image'])))
        self.assertEqual(output.getpixel((0, 15)), (200, 100, 50))
        self.assertEqual(response['output_mode'], 'raw')
        self.assertEqual(response['edit_target'], 'full')

    def test_invalid_output_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'output_mode'):
            handler.handler({'input': {'prompt': 'edit', 'output_mode': 'invalid'}})

    def test_mock_pipeline_skips_download(self):
        with patch.dict(os.environ, {'USE_MOCK_PIPELINE': '1'}):
            result = handler.handler({'input': {'image': 'mock'}})
        self.assertEqual(result['mode'], 'mock')

    def test_quota_error_reports_volume_capacity(self):
        with tempfile.TemporaryDirectory() as volume:
            with patch.dict(os.environ, {'RUNPOD_VOLUME_PATH': volume}):
                message = handler.storage_quota_message(os.path.join(volume, 'hf-cache'))
        self.assertIn('GiB free', message)
        self.assertIn('35 GB free', message)
        self.assertIn('df -h', message)

    def test_pipeline_download_quota_has_actionable_error(self):
        def fail_download(*args, **kwargs):
            raise OSError(errno.EDQUOT, 'Disk quota exceeded')

        fake_diffusers = SimpleNamespace(QwenImageEditPlusPipeline=SimpleNamespace(from_pretrained=fail_download))
        fake_torch = SimpleNamespace(bfloat16='bf16')
        with tempfile.TemporaryDirectory() as volume:
            with patch.dict(os.environ, {'RUNPOD_VOLUME_PATH': volume}), \
                 patch.dict(sys.modules, {'torch': fake_torch, 'diffusers': fake_diffusers}), \
                 patch.object(handler, 'CACHE_DIR', __import__('pathlib').Path(volume) / 'hf-cache'), \
                 patch.object(handler, '_PIPELINE', None):
                with self.assertRaisesRegex(RuntimeError, 'Expand the Network Volume'):
                    handler.get_pipeline()

    def test_pipeline_loads_nf4_base_then_requested_lora(self):
        pipe = SimpleNamespace(
            load_lora_weights=unittest.mock.Mock(),
            enable_model_cpu_offload=unittest.mock.Mock(),
            enable_sequential_cpu_offload=unittest.mock.Mock(),
            vae=SimpleNamespace(enable_tiling=unittest.mock.Mock()),
        )
        load_base = unittest.mock.Mock(return_value=pipe)
        fake_diffusers = SimpleNamespace(QwenImageEditPlusPipeline=SimpleNamespace(from_pretrained=load_base))
        fake_torch = SimpleNamespace(
            bfloat16='bf16',
            cuda=SimpleNamespace(is_available=lambda: True, mem_get_info=lambda: (23 * 1024 ** 3, 24 * 1024 ** 3)),
        )
        with tempfile.TemporaryDirectory() as volume:
            with patch.dict(sys.modules, {'torch': fake_torch, 'diffusers': fake_diffusers}), \
                 patch.object(handler, 'CACHE_DIR', __import__('pathlib').Path(volume) / 'hf-cache'), \
                 patch.object(handler, '_PIPELINE', None), \
                 patch.dict(os.environ, {'QWEN_OFFLOAD_MODE': 'auto'}):
                self.assertIs(handler.get_pipeline(), pipe)
        self.assertEqual(load_base.call_args.args[0], 'seochan99/Qwen-Image-Edit-2511-bnb-nf4')
        self.assertEqual(pipe.load_lora_weights.call_args.args[0], handler.LORA_REPO)
        pipe.enable_model_cpu_offload.assert_called_once_with()
        pipe.vae.enable_tiling.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
