import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'FreedomAISVR/Qwen-Image-Edit-Rapid-AIO-NSFW-v23-NVFP4-GGUF'


class DependencyContractTests(unittest.TestCase):
    def test_model_manifest_and_blackwell(self):
        hub = json.loads((ROOT / '.runpod/hub.json').read_text())
        smoke = json.loads((ROOT / '.runpod/tests.json').read_text())
        self.assertIn('Qwen Image Edit Rapid AIO', hub['title'])
        self.assertIn('ADA_32_PRO', hub['config']['gpuIds'])
        self.assertNotIn('ADA_24', hub['config']['gpuIds'])
        self.assertEqual(smoke['config']['gpuTypeId'], 'NVIDIA RTX PRO 6000 Blackwell Server Edition')
        self.assertEqual(hub['config']['allowedCudaVersions'], ['13.0', '13.1', '13.2', '13.3'])
        self.assertEqual(smoke['config']['allowedCudaVersions'], hub['config']['allowedCudaVersions'])
        self.assertIn(MODEL, (ROOT / 'handler.py').read_text())
        self.assertNotIn('stablediffusionapi/ultraepicairealism-v10', (ROOT / 'handler.py').read_text())

    def test_cuda13_comfy_gguf_stack(self):
        docker = (ROOT / 'Dockerfile').read_text()
        self.assertIn('nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04', docker)
        self.assertIn('https://download.pytorch.org/whl/cu130', docker)
        self.assertIn('city96/ComfyUI-GGUF', docker)
        self.assertIn('Comfy-Org/ComfyUI', docker)
        self.assertNotIn('AutoPipelineForImage2Image', docker)
        for setting in ('HF_HOME=/runpod-volume/hf-home', 'HF_HUB_CACHE=/runpod-volume/hf-cache', 'TMPDIR=/runpod-volume/tmp'):
            self.assertIn(setting, docker)
        self.assertGreater(docker.index('ENV TMPDIR=/runpod-volume/tmp'), docker.index('pip check'))

    def test_three_exact_assets(self):
        bootstrap = (ROOT / 'bootstrap_models.py').read_text()
        handler = (ROOT / 'handler.py').read_text()
        self.assertIn('qwen-v23-diffusion-NVFP4.gguf', handler)
        self.assertIn('text_encoder-NVFP4.gguf', handler)
        self.assertIn('vae.safetensors', handler)
        self.assertIn('hf_hub_download', bootstrap)
        self.assertIn('link.symlink_to(downloaded)', bootstrap)
        self.assertIn('diffusion_models', bootstrap)
        self.assertIn('text_encoders', bootstrap)
