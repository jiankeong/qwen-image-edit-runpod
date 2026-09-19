import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("face_lock", ROOT / "custom_nodes/face_lock/__init__.py")
face_lock = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(face_lock)


class FaceLockTests(unittest.TestCase):
    def test_original_face_core_and_edited_outside(self):
        source = np.zeros((100, 100, 3), dtype=np.float32)
        source[:, :, 0] = 1.0
        edited = np.zeros_like(source)
        edited[:, :, 1] = 1.0
        result = face_lock.composite_original_faces(source, edited, [(30, 30, 40, 40)])
        np.testing.assert_array_equal(result[49, 50], source[49, 50])
        np.testing.assert_array_equal(result[0, 0], edited[0, 0])
        self.assertGreater(result[18, 50, 0], 0.0)
        self.assertLess(result[18, 50, 0], 1.0)

    def test_two_faces_are_preserved(self):
        source = np.ones((100, 160, 3), dtype=np.float32)
        edited = np.zeros_like(source)
        result = face_lock.composite_original_faces(source, edited, [(10, 20, 30, 30), (110, 20, 30, 30)])
        np.testing.assert_array_equal(result[34, 25], source[34, 25])
        np.testing.assert_array_equal(result[34, 125], source[34, 125])
        np.testing.assert_array_equal(result[90, 80], edited[90, 80])

    def test_no_face_and_size_mismatch_fail(self):
        a = np.zeros((20, 20, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "No face detected"):
            face_lock.composite_original_faces(a, a, [])
        with self.assertRaisesRegex(ValueError, "identical HWC"):
            face_lock.composite_original_faces(a, np.zeros((19, 20, 3)), [(2, 2, 10, 10)])

    def test_align_small_vae_crop_without_resampling(self):
        source = np.arange(103 * 105 * 3, dtype=np.float32).reshape(103, 105, 3)
        generated = np.zeros((96, 96, 3), dtype=np.float32)
        aligned = face_lock.align_original(source, generated)
        self.assertEqual(aligned.shape, generated.shape)
        np.testing.assert_array_equal(aligned[0, 0], source[3, 4])
        with self.assertRaisesRegex(ValueError, "normal VAE crop"):
            face_lock.align_original(source, np.zeros((64, 64, 3)))

    def test_workflow_saves_composited_face(self):
        request = json.loads((ROOT / "examples/request-template.json").read_text())["input"]
        graph = request["workflow"]
        self.assertEqual(len(request["images"]), 1)
        self.assertEqual(graph["11"]["class_type"], "PreserveOriginalFaces")
        self.assertEqual(graph["11"]["inputs"]["original"], ["3", 0])
        self.assertEqual(graph["11"]["inputs"]["edited"], ["9", 0])
        self.assertEqual(graph["10"]["inputs"]["images"], ["11", 0])


if __name__ == "__main__":
    unittest.main()
