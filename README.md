# Qwen-Image-Edit-2511 NF4 + ScottzillaSystems LoRA — RunPod

This worker uses the [seochan99/Qwen-Image-Edit-2511-bnb-nf4](https://huggingface.co/seochan99/Qwen-Image-Edit-2511-bnb-nf4) quantized base and loads [ScottzillaSystems/qwen-image-edit-plus-nsfw-lora](https://huggingface.co/ScottzillaSystems/qwen-image-edit-plus-nsfw-lora). The base is a BitsAndBytes NF4 version of Qwen-Image-Edit-2511, not a separate editing architecture. If `QWEN_LORA_REPO` is already set in the RunPod endpoint, update it to the ScottzillaSystems repository or remove the override before redeploying.

## One uploaded image, targeted output

Send [examples/request-template.json](examples/request-template.json). Replace `YOUR_BASE64_IMAGE_HERE` with one PNG/JPEG image encoded as base64. Set `prompt` and `edit_target` to `clothes` or `background`; `auto` infers the target from Chinese/English keywords and fails if ambiguous.

The worker runs Qwen image editing on the source image, segments the source with [mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes), then **copies every original pixel outside the selected region** into the final PNG. Clothes mode selects garment labels; background mode selects the background label. The target region is still synthesized and the border may be imperfect. Segmentation errors can select the wrong region. A pixel-equality check verifies that all unselected pixels remain unchanged.

## Deployment

Attach a Network Volume at `/runpod-volume`; the Hub model cache, Xet cache and temporary downloads are all directed there. Allocate at least **35 GB free** for the 18 GB NF4 base, LoRA, parser and downloads; previous BF16 files are not removed automatically. Start testing on a **24 GB GPU** with one request at a time. The model author reports approximately 17 GB VRAM for the quantized base on an RTX 4090; LoRA, image resolution and runtime overhead may push the total higher. `USE_MOCK_PIPELINE=1` skips model loading in Hub smoke tests. `HF_TOKEN` can help with download rate limits.

If a worker reports `Disk quota exceeded`, check its **Network Volume**, not only container-disk settings:

```bash
df -h /runpod-volume
du -sh /runpod-volume/hf-cache /runpod-volume/hf-home /runpod-volume/models 2>/dev/null
```

Expand the volume or deliberately remove obsolete checkpoints/caches until at least 35 GB is free, then retry. A partial Hub download may resume. The old BF16 Qwen cache may still occupy the volume; check before deleting any cache because deleting the active NF4 cache forces a redownload.

Local tests cover request validation and exact outside-mask pixel preservation. Container build, GPU model loading, mask quality and RunPod image output still require a deployed test.

### GPU memory

The NF4 base uses **model CPU offload** by default, including on 24 GB GPUs, and enables VAE tiling. This avoids the very slow sequential offload used for the previous BF16 base. `QWEN_OFFLOAD_MODE=model|sequential|auto` can override it; `auto` selects `model`. If a large image still causes OOM, try `sequential` or a higher-VRAM GPU. The specific NF4 + LoRA pair and image quality still need live RunPod verification.

The Docker build pins Diffusers 0.37.0, Transformers 4.x, `huggingface-hub<1.0` and installs BitsAndBytes in one resolver transaction. It imports the Qwen and SegFormer classes and runs `pip check` before the image is published, preventing the earlier `huggingface-hub==1.32.0` runtime import error.
