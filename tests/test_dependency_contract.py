import unittest
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


class DependencyContractTests(unittest.TestCase):
    def test_single_compatible_hf_install(self):
        dockerfile = (ROOT / 'Dockerfile').read_text()
        self.assertEqual(dockerfile.count('pip install'), 1)
        self.assertIn("'diffusers==0.37.0'", dockerfile)
        self.assertIn("'transformers>=4.51,<5'", dockerfile)
        self.assertIn("'huggingface-hub>=0.34,<1.0'", dockerfile)
        self.assertNotIn('git+https://github.com/huggingface/diffusers', dockerfile)
        self.assertIn('QwenImageEditPlusPipeline', dockerfile)
        self.assertIn('SegformerForSemanticSegmentation', dockerfile)
        self.assertIn('pip check', dockerfile)
        hub = Requirement('huggingface-hub>=0.34,<1.0')
        self.assertIn('0.36.0', hub.specifier)
        self.assertNotIn('1.32.0', hub.specifier)


if __name__ == '__main__':
    unittest.main()
