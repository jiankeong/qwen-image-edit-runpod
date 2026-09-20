# Qwen-Image-Edit-2511 mixed NF4 + ScottzillaSystems LoRA — RunPod

This worker uses [toandev/Qwen-Image-Edit-2511-4bit](https://huggingface.co/toandev/Qwen-Image-Edit-2511-4bit), a mixed-precision NF4 quantization of the same 2511 base with critical image layers retained in BF16, and loads [ScottzillaSystems/qwen-image-edit-plus-nsfw-lora](https://huggingface.co/ScottzillaSystems/qwen-image-edit-plus-nsfw-lora). It retains the model card's named safetensors loading: `weight_name="qwen-image-edit-plus-nsfw-lora.safetensors"`, `adapter_name="mcnl-nsfw-v1"`, then `set_adapters(["mcnl-nsfw-v1"])`. This is not the LoRA author's exact BF16 setup; output quality and LoRA compatibility require a live GPU test. The quantized base lists a CC BY-NC-SA 4.0 license. If `QWEN_LORA_REPO` is already set in the RunPod endpoint, update it to the ScottzillaSystems repository or remove the override before redeploying.

## One uploaded image, targeted output

Send [examples/raw-edit-request.json](examples/raw-edit-request.json) when the full LoRA-generated image is the goal. Replace `YOUR_BASE64_IMAGE_HERE` with one PNG/JPEG image encoded as base64 and set the prompt to the desired transformation. Include a trigger term such as `nsfw` from the LoRA model card when testing its effect. The default `output_mode` is `raw`: `image` is the complete Qwen + LoRA output, without garment/background masking. This can also change identity and other image details. This edit pipeline still requires one input image.

Set `input.true_cfg_scale` to a positive number to override classifier-free guidance; omit it for the Scottzilla model-card default of `4.0`. At `1.0`, the worker omits the negative prompt and disables true CFG for a faster but potentially less prompt-faithful run. `input.steps` remains independently configurable (default `40`).

For exact preservation outside garments or background, send [examples/request-template.json](examples/request-template.json) with `"output_mode": "masked"`. Set `edit_target` to `clothes` or `background`; `auto` infers the target from Chinese/English keywords and fails if ambiguous.

In `masked` mode, the worker runs Qwen image editing on the source image, segments the source with [mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes), then **copies every original pixel outside the selected region** into the final PNG. Clothes mode selects garment labels; background mode selects the background label. The target region is still synthesized and the border may be imperfect. Segmentation errors can select the wrong region. A pixel-equality check verifies that all unselected pixels remain unchanged.

If the expected LoRA effect seems absent in `masked` mode, send the same request with `"return_raw": true`. The response then includes `raw_image` (the unmasked Qwen + LoRA result) alongside `image` (the target-only composite), without a second inference. If `raw_image` shows the effect and `image` does not, the garment/background mask removed it; if both lack it, inspect the prompt and loaded LoRA repository rather than changing the mask.

## Deployment

Attach a Network Volume at `/runpod-volume`; the Hub model cache, Xet cache and temporary downloads are all directed there. Allocate at least **60 GB free** for the ~21.6 GB quantized base, LoRA, parser and download overhead. Start testing on a **24 GB GPU** with one request at a time; the model publisher describes 16–24 GB operation, but this worker's LoRA path is not yet GPU-verified. The earlier 24 GB OOM involved the full BF16 transformer, not this quantized base. `USE_MOCK_PIPELINE=1` skips model loading in Hub smoke tests. `HF_TOKEN` can help with download rate limits.

If a worker reports `Disk quota exceeded`, check its **Network Volume**, not only container-disk settings:

```bash
df -h /runpod-volume
du -sh /runpod-volume/hf-cache /runpod-volume/hf-home /runpod-volume/models 2>/dev/null
```

Expand the volume or deliberately remove obsolete checkpoints/caches until at least 60 GB is free, then retry. A partial Hub download may resume. The previous BF16 or NF4 cache may still occupy the volume; check before deleting any cache because deleting the active quantized cache forces a redownload.

Local tests cover request validation and exact outside-mask pixel preservation. Container build, GPU model loading, mask quality and RunPod image output still require a deployed test.

### GPU memory

The mixed-precision NF4 base uses **model CPU offload** by default, including on 24 GB GPUs. `QWEN_OFFLOAD_MODE=model|sequential|auto` can override it; `auto` selects `model`, while `sequential` uses less VRAM at a large speed cost. The VAE tiling call remains disabled while diagnosing the earlier distorted output. The exact image quality still needs live RunPod verification.

The Docker build pins Diffusers 0.37.0, Transformers 4.x and `huggingface-hub<1.0` in one resolver transaction. It imports the Qwen and SegFormer classes and runs `pip check` before the image is published, preventing the earlier `huggingface-hub==1.32.0` runtime import error.
