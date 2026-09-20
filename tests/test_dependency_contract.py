import unittest
import json
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


class DependencyContractTests(unittest.TestCase):
    def test_default_lora_matches_runpod_manifest(self):
        manifest = json.loads((ROOT / '.runpod/hub.json').read_text())
        env = {item['key']: item['input'] for item in manifest['config']['env']}
        self.assertEqual(env['QWEN_LORA_REPO']['default'], 'ScottzillaSystems/qwen-image-edit-plus-nsfw-lora')
        self.assertIn('ScottzillaSystems/qwen-image-edit-plus-nsfw-lora', (ROOT / 'handler.py').read_text())
        self.assertIn('seochan99/Qwen-Image-Edit-2511-bnb-nf4', (ROOT / 'handler.py').read_text())
        self.assertIn('ADA_24', manifest['config']['gpuIds'])

    def test_single_compatible_hf_install(self):
        dockerfile = (ROOT / 'Dockerfile').read_text()
        self.assertEqual(dockerfile.count('pip install'), 1)
        self.assertIn("'diffusers==0.37.0'", dockerfile)
        self.assertIn("'transformers>=4.51,<5'", dockerfile)
        self.assertIn("'huggingface-hub>=0.34,<1.0'", dockerfile)
        self.assertIn("'bitsandbytes>=0.46,<0.49'", dockerfile)
        self.assertNotIn('git+https://github.com/huggingface/diffusers', dockerfile)
        self.assertIn('QwenImageEditPlusPipeline', dockerfile)
        self.assertIn('SegformerForSemanticSegmentation', dockerfile)
        self.assertIn('pip check', dockerfile)
        hub = Requirement('huggingface-hub>=0.34,<1.0')
        self.assertIn('0.36.0', hub.specifier)
        self.assertNotIn('1.32.0', hub.specifier)

    def test_hub_and_xet_caches_use_network_volume(self):
        dockerfile = (ROOT / 'Dockerfile').read_text()
        startup = (ROOT / 'startup.sh').read_text()
        for setting in (
            'HF_HOME=/runpod-volume/hf-home',
            'HF_HUB_CACHE=/runpod-volume/hf-cache',
            'HF_XET_CACHE=/runpod-volume/hf-home/xet',
            'TMPDIR=/runpod-volume/tmp',
        ):
            self.assertIn(setting, dockerfile)
        self.assertIn('mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TMPDIR"', startup)
        self.assertIn('COPY startup.sh /workspace/startup.sh', dockerfile)


if __name__ == '__main__':
    unittest.main()
