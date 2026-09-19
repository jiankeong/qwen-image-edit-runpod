import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("bootstrap_flux", Path(__file__).resolve().parents[1] / "scripts/bootstrap_flux.py")
flux = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(flux)
ROOT = Path(__file__).resolve().parents[1]


class FakeResponse(io.BytesIO):
    def __init__(self, data=b"", size=None, status=200):
        super().__init__(data)
        self.headers = {"Content-Length": str(size if size is not None else len(data))}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FluxTests(unittest.TestCase):
    def test_choose_lora(self):
        payload = json.dumps({"siblings": [{"rfilename": "README.md"}, {"rfilename": "lora.safetensors"}]}).encode()
        with patch.object(flux.urllib.request, "urlopen", return_value=FakeResponse(payload)):
            self.assertEqual(flux.choose_lora(), "lora.safetensors")

    def test_download_and_cached_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "models/loras/adapter.safetensors"
            calls = [FakeResponse(size=4), FakeResponse(b"DATA", size=4)]
            with patch.object(flux.urllib.request, "urlopen", side_effect=calls) as get:
                flux.download("https://example.test/adapter", target)
                self.assertEqual(target.read_bytes(), b"DATA")
                self.assertEqual(get.call_count, 2)
            with patch.object(flux.urllib.request, "urlopen", return_value=FakeResponse(size=4)) as get:
                flux.download("https://example.test/adapter", target)
                self.assertEqual(get.call_count, 1)

    def test_request_has_one_source_and_flux_nodes(self):
        request = json.loads((ROOT / "examples/request-template.json").read_text())["input"]
        graph = request["workflow"]
        self.assertEqual(request["images"][0]["name"], "input.png")
        self.assertEqual(len(request["images"]), 1)
        self.assertEqual(graph["1"]["class_type"], "CheckpointLoaderSimple")
        self.assertEqual(graph["2"]["class_type"], "LoraLoader")
        self.assertEqual(graph["4"]["class_type"], "VAEEncode")
        self.assertEqual(graph["8"]["inputs"]["latent_image"], ["4", 0])
        self.assertEqual(graph["8"]["inputs"]["cfg"], 1.0)
        self.assertEqual(graph["8"]["inputs"]["denoise"], 0.4)
        self.assertEqual(graph, json.loads((ROOT / "examples/flux-img2img-workflow.json").read_text()))
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    self.assertIn(value[0], graph)


if __name__ == "__main__":
    unittest.main()
