"""ComfyUI node that restores original face pixels after image-to-image sampling."""

from pathlib import Path

import numpy as np


def face_mask(height, width, boxes, feather=0.18):
    """Return an alpha mask: 1 in each face core, 0 outside its feathered rim."""
    if not 0.0 <= feather < 1.0:
        raise ValueError("feather must be in [0, 1)")
    yy, xx = np.ogrid[:height, :width]
    result = np.zeros((height, width), dtype=np.float32)
    for x, y, w, h in boxes:
        if w <= 0 or h <= 0:
            continue
        # Enlarge the detector's rectangle to include the full face and some hair.
        cx, cy = x + 0.5 * w, y + 0.48 * h
        rx, ry = 0.77 * w, 0.83 * h
        distance = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
        alpha = np.clip((1.0 - distance) / max(feather, 1e-6), 0.0, 1.0)
        result = np.maximum(result, alpha.astype(np.float32))
    return result


def composite_original_faces(original, edited, boxes, feather=0.18):
    """Copy untouched original face pixels into an edited HWC image."""
    if original.shape != edited.shape or original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("Original and edited images must have identical HWC RGB dimensions")
    if not boxes:
        raise ValueError("No face detected; refusing to return an identity-changing result")
    alpha = face_mask(*original.shape[:2], boxes, feather)[..., None]
    return original * alpha + edited * (1.0 - alpha)


def align_original(original, edited):
    """Match ComfyUI VAE's small centered dimension crop without resampling faces."""
    if original.ndim != 3 or edited.ndim != 3 or original.shape[2:] != edited.shape[2:]:
        raise ValueError("Source and generated images must both be HWC RGB")
    dh = original.shape[0] - edited.shape[0]
    dw = original.shape[1] - edited.shape[1]
    if not (0 <= dh < 32 and 0 <= dw < 32):
        raise ValueError("Source and generated image sizes differ beyond a normal VAE crop")
    top, left = dh // 2, dw // 2
    return original[top:top + edited.shape[0], left:left + edited.shape[1]]


def detect_faces(image):
    import cv2

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Could not load OpenCV face detector: {cascade_path}")
    rgb = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(28, 28))
    return [tuple(map(int, box)) for box in boxes]


class PreserveOriginalFaces:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original": ("IMAGE",),
            "edited": ("IMAGE",),
            "minimum_faces": ("INT", {"default": 1, "min": 1, "max": 20}),
            "feather": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 0.5, "step": 0.01}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preserve"
    CATEGORY = "image/identity"

    def preserve(self, original, edited, minimum_faces, feather):
        import torch

        if original.shape[0] != edited.shape[0]:
            raise ValueError("Source and generated image batch sizes differ")
        restored = []
        for source_tensor, generated_tensor in zip(original, edited):
            source = source_tensor.detach().cpu().numpy().astype(np.float32)
            generated = generated_tensor.detach().cpu().numpy().astype(np.float32)
            source = align_original(source, generated)
            boxes = detect_faces(source)
            if len(boxes) < minimum_faces:
                raise ValueError(
                    f"Detected {len(boxes)} face(s), expected at least {minimum_faces}; "
                    "refusing to save an image with changed faces"
                )
            result = composite_original_faces(source, generated, boxes, feather)
            restored.append(torch.from_numpy(result).to(device=edited.device, dtype=edited.dtype))
        return (torch.stack(restored),)


NODE_CLASS_MAPPINGS = {"PreserveOriginalFaces": PreserveOriginalFaces}
NODE_DISPLAY_NAME_MAPPINGS = {"PreserveOriginalFaces": "Preserve Original Faces"}
