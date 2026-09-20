import base64
import io
import os
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np
from PIL import Image

import handler
import bootstrap_models


def encoded_image(color=(10, 20, 30)):
    image = Image.new('RGB', (16, 16), color)
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return base64.b64encode(stream.getvalue()).decode()


class WorkerTests(unittest.TestCase):
    def test_quota_text_is_detected_even_without_errno(self):
        self.assertTrue(bootstrap_models.quota_error(
            OSError('I/O error: IO Error: Disk quota exceeded (os error 122)')))

    def test_cached_checkpoint_survives_low_volume(self):
        with __import__('tempfile').TemporaryDirectory() as directory:
            from pathlib import Path
            root = Path(directory)
            cache = root / 'volume' / 'hf-cache'
            ephemeral = root / 'ephemeral'
            cached_model = cache / 'v19-checkpoint.safetensors'
            cache.mkdir(parents=True)
            cached_model.write_bytes(b'model')
            calls = []

            def download(remote, location, *, ephemeral=False, local_only=False):
                calls.append((remote, ephemeral, local_only))
                if remote == handler.CHECKPOINT_PATH and local_only:
                    return cached_model
                if local_only:
                    raise RuntimeError('cache miss')
                self.fail('cached checkpoint should not be downloaded again')

            fake_usage = lambda path: SimpleNamespace(free=20 * bootstrap_models.GIB if Path(path) == ephemeral else 0)
            with patch.dict(os.environ, {'HF_HUB_CACHE': str(cache)}), \
                 patch.object(bootstrap_models, 'EPHEMERAL_CACHE', ephemeral), \
                 patch.object(bootstrap_models.shutil, 'disk_usage', side_effect=fake_usage):
                bootstrap_models.install_models(base=root / 'models', downloader=download)
            self.assertEqual((root / 'models/checkpoints' / handler.CHECKPOINT_FILE).resolve(), cached_model.resolve())
            self.assertEqual(calls, [(handler.CHECKPOINT_PATH, False, True)])

    def test_workflow_uses_v19_aio_checkpoint(self):
        graph = handler.build_workflow('source.png', 'red jacket', ' ', 4, 1.0, 1.0, 7)
        self.assertEqual(graph['2']['class_type'], 'CheckpointLoaderSimple')
        self.assertEqual(graph['2']['inputs']['ckpt_name'], 'Qwen-Rapid-AIO-NSFW-v19.safetensors')
        self.assertEqual(graph['5']['class_type'], 'TextEncodeQwenImageEditPlus')
        self.assertEqual(graph['5']['inputs']['clip'], ['2', 1])
        self.assertEqual(graph['5']['inputs']['vae'], ['2', 2])
        self.assertEqual(graph['1']['inputs']['image'], 'source.png')
        self.assertEqual(graph['8']['inputs']['model'], ['2', 0])
        self.assertEqual(graph['7']['class_type'], 'EmptyLatentImage')
        self.assertEqual(graph['7']['inputs']['width'], 1024)
        self.assertEqual(graph['8']['inputs']['sampler_name'], 'euler_ancestral')
        self.assertEqual(graph['8']['inputs']['scheduler'], 'beta')
        self.assertEqual(graph['8']['inputs']['steps'], 4)
        self.assertEqual(graph['8']['inputs']['cfg'], 1.0)
        self.assertEqual(graph['8']['inputs']['denoise'], 1.0)
        self.assertEqual(graph['8']['inputs']['seed'], 7)

    def test_lower_strength_uses_source_latent(self):
        graph = handler.build_workflow('source.png', 'red jacket', ' ', 8, 1.0, 0.5, 7, 768, 1024)
        self.assertEqual(graph['7']['class_type'], 'VAEEncode')
        self.assertEqual(graph['8']['inputs']['denoise'], 0.5)

    def test_defaults_and_single_image(self):
        pipe = Mock(return_value=SimpleNamespace(images=[Image.new('RGB', (16, 16), (200, 100, 50))]))
        with patch.object(handler, 'get_pipeline', return_value=pipe):
            answer = handler.handler({'input': {'image': encoded_image(), 'prompt': 'red jacket'}})
        self.assertEqual(answer['model'], handler.BASE_REPO)
        self.assertEqual(answer['output_mode'], 'raw')
        self.assertEqual(pipe.call_args.kwargs['num_inference_steps'], 4)
        self.assertEqual(pipe.call_args.kwargs['guidance_scale'], 1.0)
        self.assertEqual(pipe.call_args.kwargs['strength'], 1.0)
        self.assertEqual(pipe.call_args.kwargs['seed'], 0)
        self.assertEqual(pipe.call_args.kwargs['negative_prompt'], ' ')

    def test_custom_parameters(self):
        pipe = Mock(return_value=SimpleNamespace(images=[Image.new('RGB', (16, 16))]))
        with patch.object(handler, 'get_pipeline', return_value=pipe):
            handler.handler({'input': {'image': encoded_image(), 'prompt': 'edit', 'steps': 8,
                                      'true_cfg_scale': 2.5, 'strength': 0.5, 'seed': 42}})
        self.assertEqual(pipe.call_args.kwargs['guidance_scale'], 2.5)
        self.assertEqual(pipe.call_args.kwargs['seed'], 42)

    def test_invalid_parameters(self):
        for payload, message in [({'steps': 0}, 'steps'), ({'true_cfg_scale': 0}, 'guidance'),
                                 ({'strength': 0}, 'strength'), ({'seed': -1}, 'seed'),
                                 ({'output_mode': 'other'}, 'output_mode')]:
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, message):
                handler.handler({'input': {'prompt': 'edit', **payload}})

    def test_mask_composite_preserves_other_pixels(self):
        labels = np.zeros((16, 16), dtype=np.int64)
        labels[:8] = 4
        pipe = Mock(return_value=SimpleNamespace(images=[Image.new('RGB', (16, 16), (200, 100, 50))]))
        with patch.object(handler, 'get_pipeline', return_value=pipe), patch.object(handler, 'parser_labels', return_value=labels):
            response = handler.handler({'input': {'image': encoded_image(), 'prompt': 'red shirt',
                                                  'output_mode': 'masked', 'edit_target': 'clothes'}})
        output = Image.open(io.BytesIO(base64.b64decode(response['image'])))
        self.assertEqual(output.getpixel((0, 0)), (200, 100, 50))
        self.assertEqual(output.getpixel((0, 15)), (10, 20, 30))

    def test_mock_mode(self):
        with patch.dict(os.environ, {'USE_MOCK_PIPELINE': '1'}):
            result = handler.handler({'input': {'image': 'mock'}})
        self.assertEqual(result['model'], handler.BASE_REPO)
        self.assertEqual(result['mode'], 'mock')

    def test_comfy_client_errors_and_cleanup(self):
        with __import__('tempfile').TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'COMFY_INPUT_DIR': directory}), patch.object(handler, 'comfy_json', return_value={'error': 'bad node'}):
                with self.assertRaisesRegex(RuntimeError, 'workflow validation'):
                    handler.run_comfy_edit(image=Image.new('RGB', (16, 16)), prompt='edit', negative_prompt=' ',
                                           num_inference_steps=4, guidance_scale=1.0, strength=1.0, seed=0)
            self.assertEqual(os.listdir(directory), [])

    def test_comfy_prompt_ack_with_empty_node_errors_is_success(self):
        prompt_id = 'b3571018-2815-42cc-917f-073529861538'
        ack = {'prompt_id': prompt_id, 'number': 0, 'node_errors': {}}
        history = {prompt_id: {'status': {'status_str': 'success'},
                               'outputs': {'10': {'images': [{'filename': 'edited.png'}]}}}}
        output = io.BytesIO()
        Image.new('RGB', (16, 16), (200, 100, 50)).save(output, format='PNG')
        with __import__('tempfile').TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'COMFY_INPUT_DIR': directory}), \
                 patch.object(handler, 'comfy_json', side_effect=[ack, history]), \
                 patch.object(handler.urllib.request, 'urlopen', return_value=io.BytesIO(output.getvalue())):
                result = handler.run_comfy_edit(image=Image.new('RGB', (16, 16)),
                                                prompt='edit', negative_prompt=' ',
                                                num_inference_steps=4, guidance_scale=1.0,
                                                strength=1.0, seed=0)
            self.assertEqual(result.images[0].getpixel((0, 0)), (200, 100, 50))
            self.assertEqual(os.listdir(directory), [])
