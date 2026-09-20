"""RunPod worker: Qwen-Image-Edit-2511 + NSFW LoRA, with target-only pixels."""

import base64
import errno
import io
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

BASE_REPO = "seochan99/Qwen-Image-Edit-2511-bnb-nf4"
LORA_REPO = os.getenv("QWEN_LORA_REPO", "ScottzillaSystems/qwen-image-edit-plus-nsfw-lora")
PARSER_REPO = "mattmdjaga/segformer_b2_clothes"
CACHE_DIR = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume")) / "hf-cache"
CLOTHES_LABELS = (4, 5, 6, 7, 8, 17)
_PIPELINE = None
_PARSER = None
_PROCESSOR = None


def configure_offload(pipe, free_vram_gib, mode="auto"):
    """The NF4 transformer fits in 24 GB; offload whole components for speed."""
    if mode not in ("auto", "model", "sequential"):
        raise ValueError("QWEN_OFFLOAD_MODE must be auto, model, or sequential")
    selected = "model" if mode == "auto" else mode
    if selected == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()
    print(f"[qwen-worker] GPU free={free_vram_gib:.1f} GiB; offload={selected}", flush=True)
    return selected


def storage_quota_message(path):
    """Report the actual filesystem capacity rather than hiding errno 122."""
    volume = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
    usage = shutil.disk_usage(volume)
    gib = 1024 ** 3
    return (
        f"Model download exhausted storage at {path}; Network Volume "
        f"{volume}: {usage.free / gib:.1f} GiB free / "
        f"{usage.total / gib:.1f} GiB total. "
        "The NF4 base is about 18 GB on disk; reserve at least 35 GB free "
        "for the base, LoRA, parser and download overhead. "
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


def get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        import torch
        from diffusers import QwenImageEditPlusPipeline

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        token = os.getenv("HF_TOKEN") or None
        try:
            pipe = QwenImageEditPlusPipeline.from_pretrained(BASE_REPO, torch_dtype=torch.bfloat16, cache_dir=str(CACHE_DIR), token=token)
            print(f"[qwen-worker] loading LoRA={LORA_REPO}", flush=True)
            pipe.load_lora_weights(LORA_REPO, cache_dir=str(CACHE_DIR), token=token)
        except OSError as exc:
            if exc.errno not in (errno.ENOSPC, errno.EDQUOT, 122):
                raise
            raise RuntimeError(storage_quota_message(CACHE_DIR)) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for Qwen-Image-Edit-2511 inference")
        free_vram_gib = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        pipe.vae.enable_tiling()
        configure_offload(pipe, free_vram_gib, os.getenv("QWEN_OFFLOAD_MODE", "auto"))
        _PIPELINE = pipe
    return _PIPELINE


def handler(event):
    payload = event.get("input", event)
    if os.getenv("USE_MOCK_PIPELINE") == "1":
        return {"ok": True, "mode": "mock", "model": BASE_REPO, "lora": LORA_REPO}
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("input.prompt must be a nonempty string")
    output_mode = payload.get("output_mode", "raw")
    if output_mode not in ("raw", "masked"):
        raise ValueError("output_mode must be raw or masked")
    original = decode_image(payload.get("image"))
    if output_mode == "masked":
        target = choose_target(prompt, payload.get("edit_target", "auto"))
        labels = parser_labels(original)
        mask = make_mask(labels, target)
    else:
        target = "full"
    generated = get_pipeline()(
        image=[original], prompt=prompt, negative_prompt=" ",
        num_inference_steps=int(payload.get("steps", 40)),
        true_cfg_scale=4.0, guidance_scale=1.0, num_images_per_prompt=1,
    ).images[0]
    result = composite_exact(original, generated, mask) if output_mode == "masked" else generated
    response = {"image": encode_image(result), "format": "png", "output_mode": output_mode, "edit_target": target, "model": BASE_REPO, "lora": LORA_REPO}
    if output_mode == "masked" and payload.get("return_raw") is True:
        response["raw_image"] = encode_image(generated)
    return response


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
