"""RunPod worker: one-image Qwen Rapid AIO v19 NSFW editing via ComfyUI."""

import base64
import errno
import io
import math
import os
import re
import shutil
import json
import time
import urllib.request
import urllib.parse
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

BASE_REPO = "Phr00t/Qwen-Image-Edit-Rapid-AIO"
CHECKPOINT_PATH = "v19/Qwen-Rapid-AIO-NSFW-v19.safetensors"
CHECKPOINT_FILE = "Qwen-Rapid-AIO-NSFW-v19.safetensors"
COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188")
PARSER_REPO = "mattmdjaga/segformer_b2_clothes"
CACHE_DIR = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume")) / "hf-cache"
CLOTHES_LABELS = (4, 5, 6, 7, 8, 17)
_PIPELINE = None
_PARSER = None
_PROCESSOR = None


def storage_quota_message(path):
    """Report the actual filesystem capacity rather than hiding errno 122."""
    volume = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
    usage = shutil.disk_usage(volume)
    gib = 1024 ** 3
    return (
        f"Model download exhausted storage at {path}; Network Volume "
        f"{volume}: {usage.free / gib:.1f} GiB free / "
        f"{usage.total / gib:.1f} GiB total. "
        "The v19 AIO checkpoint is 28.4 GB; reserve at least 40 GB free "
        "for the model, parser and download overhead. "
        "Expand the Network Volume or remove old "
        "model/cache files after checking what they contain. "
        f"Inspect with: df -h {volume}; du -sh {volume}/hf-cache "
        f"{volume}/hf-home {volume}/models 2>/dev/null"
    )


def choose_target(prompt, target="auto"):
    if target in ("clothes", "background"):
        return target
    if target != "auto":
        raise ValueError("edit_target must be auto, clothes, or background")
    lower = prompt.lower()
    clothes = bool(re.search(r"衣|服装|裙|裤|外套|衬衫|上衣|dress|shirt|jacket|coat|pants|skirt|clothes|outfit|garment", lower))
    background = bool(re.search(r"背景|风景|场景|天空|海边|森林|background|scenery|landscape|beach|forest|sky", lower))
    if clothes == background:
        raise ValueError("Prompt target is ambiguous; set edit_target to clothes or background")
    return "clothes" if clothes else "background"


def decode_image(value):
    if not isinstance(value, str) or not value:
        raise ValueError("input.image must contain one base64 image")
    if value.startswith("data:"):
        value = value.partition(",")[2]
    raw = base64.b64decode(value, validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        image = ImageOps.exif_transpose(image)
        if image.width * image.height > 20_000_000:
            raise ValueError("Input image exceeds 20 megapixels")
        return image.convert("RGB")


def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def prepare_image_for_pipeline(image):
    """Bound Qwen inference size without changing the source used for compositing."""
    prepared = image.copy()
    prepared.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
    if prepared.width * prepared.height > 1_500_000:
        scale = math.sqrt(1_500_000 / (prepared.width * prepared.height))
        prepared = prepared.resize(
            (max(8, int(prepared.width * scale) // 8 * 8), max(8, int(prepared.height * scale) // 8 * 8)),
            Image.Resampling.LANCZOS,
        )
    return prepared


def parser_labels(image):
    global _PARSER, _PROCESSOR
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    if _PARSER is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        token = os.getenv("HF_TOKEN") or None
        _PROCESSOR = AutoImageProcessor.from_pretrained(PARSER_REPO, cache_dir=str(CACHE_DIR), token=token)
        _PARSER = SegformerForSemanticSegmentation.from_pretrained(PARSER_REPO, cache_dir=str(CACHE_DIR), token=token).eval().to("cpu")
    inputs = _PROCESSOR(images=image, return_tensors="pt")
    with torch.inference_mode():
        logits = _PARSER(**inputs).logits
        logits = torch.nn.functional.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)[0].cpu().numpy()


def make_mask(labels, target):
    if target == "clothes":
        selected = np.isin(labels, CLOTHES_LABELS)
    elif target == "background":
        selected = labels == 0
    else:
        raise ValueError("Unsupported edit target")
    if selected.sum() < 64:
        raise ValueError(f"No usable {target} region found in input image")
    return selected


def composite_exact(original, edited, mask):
    if edited.size != original.size:
        edited = edited.resize(original.size, Image.Resampling.LANCZOS)
    if mask.shape != (original.height, original.width):
        raise ValueError("Mask dimensions do not match source image")
    source_pixels = np.asarray(original.convert("RGB"))
    edited_pixels = np.asarray(edited.convert("RGB"))
    output_pixels = np.where(mask[..., None], edited_pixels, source_pixels).astype(np.uint8)
    if not np.array_equal(output_pixels[~mask], source_pixels[~mask]):
        raise RuntimeError("Unselected source pixels changed")
    return Image.fromarray(output_pixels, mode="RGB")


def comfy_json(path, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(COMFY_URL + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=30) as reply:
        return json.load(reply)


def build_workflow(image_name, prompt, negative_prompt, steps, cfg, strength, seed,
                   width=1024, height=1024):
    """ComfyUI API graph using the v19 all-in-one checkpoint."""
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT_FILE}},
        "5": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["2", 1], "vae": ["2", 2], "image1": ["1", 0], "prompt": prompt}},
        "6": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["2", 1], "vae": ["2", 2], "image1": ["1", 0], "prompt": negative_prompt}},
        "7": ({"class_type": "EmptyLatentImage", "inputs": {
            "width": width, "height": height, "batch_size": 1}}
              if strength == 1.0 else
              {"class_type": "VAEEncode", "inputs": {"pixels": ["1", 0], "vae": ["2", 2]}}),
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["2", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "euler_ancestral", "scheduler": "beta", "positive": ["5", 0],
            "negative": ["6", 0], "latent_image": ["7", 0], "denoise": strength}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["2", 2]}},
        "10": {"class_type": "SaveImage", "inputs": {
            "images": ["9", 0], "filename_prefix": "qwen_rapid_edit"}},
    }


def get_pipeline():
    """Keep a callable interface for local tests and the one-image handler."""
    return run_comfy_edit


def run_comfy_edit(*, image, prompt, negative_prompt, num_inference_steps,
                   guidance_scale, strength, seed):
    input_dir = Path(os.getenv("COMFY_INPUT_DIR", "/opt/ComfyUI/input"))
    name = f"runpod_{uuid.uuid4().hex}.png"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / name
    image.save(input_path, format="PNG")
    try:
        workflow = build_workflow(name, prompt, negative_prompt, num_inference_steps,
                                  guidance_scale, strength, seed, image.width, image.height)
        queued = comfy_json("/prompt", {"prompt": workflow, "client_id": uuid.uuid4().hex})
        if queued.get("error") or queued.get("node_errors"):
            raise RuntimeError(f"ComfyUI workflow validation: {queued}")
        prompt_id = queued["prompt_id"]
        deadline = time.monotonic() + int(os.getenv("COMFY_TIMEOUT_SECONDS", "900"))
        while time.monotonic() < deadline:
            history = comfy_json("/history/" + urllib.parse.quote(prompt_id, safe=""))
            if prompt_id in history:
                entry = history[prompt_id]
                if entry.get("status", {}).get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI execution failed: {entry.get('status')}")
                images = entry.get("outputs", {}).get("10", {}).get("images", [])
                if images:
                    info = images[0]
                    query = urllib.parse.urlencode({"filename": info["filename"],
                                                    "subfolder": info.get("subfolder", ""),
                                                    "type": info.get("type", "output")})
                    with urllib.request.urlopen(COMFY_URL + "/view?" + query, timeout=30) as reply:
                        with Image.open(io.BytesIO(reply.read())) as result:
                            return type("Output", (), {"images": [result.convert("RGB")]})()
            time.sleep(1)
        raise TimeoutError(f"ComfyUI prompt {prompt_id} exceeded timeout")
    finally:
        input_path.unlink(missing_ok=True)


def handler(event):
    payload = event.get("input", event)
    if os.getenv("USE_MOCK_PIPELINE") == "1":
        return {"ok": True, "mode": "mock", "model": BASE_REPO}
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("input.prompt must be a nonempty string")
    output_mode = payload.get("output_mode", "raw")
    if output_mode not in ("raw", "masked"):
        raise ValueError("output_mode must be raw or masked")
    guidance_scale = payload.get("guidance_scale", payload.get("true_cfg_scale", 1.0))
    strength = payload.get("strength", 1.0)
    steps = payload.get("steps", 4)
    if isinstance(guidance_scale, bool) or not isinstance(guidance_scale, (int, float)) or not math.isfinite(guidance_scale) or guidance_scale <= 0:
        raise ValueError("input.guidance_scale must be a positive finite number")
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 < strength <= 1:
        raise ValueError("input.strength must be between 0 and 1")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100:
        raise ValueError("input.steps must be an integer between 1 and 100")
    if steps * strength < 1:
        raise ValueError("input.steps * input.strength must be at least 1")
    negative_prompt = payload.get("negative_prompt", " ")
    if not isinstance(negative_prompt, str):
        raise ValueError("input.negative_prompt must be a string")
    seed = payload.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 64:
        raise ValueError("input.seed must be an unsigned 64-bit integer")
    original = decode_image(payload.get("image"))
    if output_mode == "masked":
        target = choose_target(prompt, payload.get("edit_target", "auto"))
        labels = parser_labels(original)
        mask = make_mask(labels, target)
    else:
        target = "full"
    generated = get_pipeline()(
        image=prepare_image_for_pipeline(original), prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps, guidance_scale=float(guidance_scale), strength=float(strength),
        seed=seed,
    ).images[0]
    result = composite_exact(original, generated, mask) if output_mode == "masked" else generated
    response = {"image": encode_image(result), "format": "png", "output_mode": output_mode, "edit_target": target, "model": BASE_REPO}
    if output_mode == "masked" and payload.get("return_raw") is True:
        response["raw_image"] = encode_image(generated)
    return response


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
