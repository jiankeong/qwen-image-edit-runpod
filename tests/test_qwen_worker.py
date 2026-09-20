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
    def test_offload_selects_sequential_for_24gb_gpu(self):
        pipe = SimpleNamespace(
            enable_sequential_cpu_offload=unittest.mock.Mock(),
            enable_model_cpu_offload=unittest.mock.Mock(),
        )
        self.assertEqual(handler.configure_offload(pipe, 23.5), 'sequential')
        pipe.enable_sequential_cpu_offload.assert_called_once_with()
        pipe.enable_model_cpu_offload.assert_not_called()

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
                'prompt': 'Change the shirt to red', 'edit_target': 'clothes'
            }})
        result = Image.open(io.BytesIO(base64.b64decode(response['image'])))
        self.assertEqual(result.getpixel((0, 0)), (200, 100, 50))
        self.assertEqual(result.getpixel((0, 15)), (10, 20, 30))
        self.assertEqual(response['edit_target'], 'clothes')

    def test_mock_pipeline_skips_download(self):
        with patch.dict(os.environ, {'USE_MOCK_PIPELINE': '1'}):
            result = handler.handler({'input': {'image': 'mock'}})
        self.assertEqual(result['mode'], 'mock')

    def test_quota_error_reports_volume_capacity(self):
        with tempfile.TemporaryDirectory() as volume:
            with patch.dict(os.environ, {'RUNPOD_VOLUME_PATH': volume}):
                message = handler.storage_quota_message(os.path.join(volume, 'hf-cache'))
        self.assertIn('GiB free', message)
        self.assertIn('100 GB free', message)
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


if __name__ == '__main__':
    unittest.main()
