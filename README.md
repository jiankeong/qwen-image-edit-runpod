# Qwen-Image-Edit-2511 + ScottzillaSystems NSFW LoRA — RunPod

This worker uses the full [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) editing pipeline and loads [ScottzillaSystems/qwen-image-edit-plus-nsfw-lora](https://huggingface.co/ScottzillaSystems/qwen-image-edit-plus-nsfw-lora). It no longer uses the Qwen Turbo GGUF or FLUX checkpoint.

## One uploaded image, targeted output

Send [examples/request-template.json](examples/request-template.json). Replace `YOUR_BASE64_IMAGE_HERE` with one PNG/JPEG image encoded as base64. Set `prompt` and `edit_target` to `clothes` or `background`; `auto` infers the target from Chinese/English keywords and fails if ambiguous.

The worker runs Qwen image editing on the source image, segments the source with [mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes), then **copies every original pixel outside the selected region** into the final PNG. Clothes mode selects garment labels; background mode selects the background label. The target region is still synthesized and the border may be imperfect. Segmentation errors can select the wrong region. A pixel-equality check verifies that all unselected pixels remain unchanged.

## Deployment

Attach a Network Volume at `/runpod-volume`; the Hub model cache, Xet cache and temporary downloads are all directed there. Allocate at least **100 GB free** for the BF16 base, LoRA, parser and downloads; previous model files are not removed automatically. The base repository alone contains a 40.9 GB transformer and 16.6 GB text encoder. Start testing on an **80 GB GPU**. The full Qwen-Image-Edit-2511 is a 20B BF16 model and cold starts may be long. `USE_MOCK_PIPELINE=1` skips model loading in Hub smoke tests. `HF_TOKEN` can help with download rate limits.

If a worker reports `Disk quota exceeded`, check its **Network Volume**, not only container-disk settings:

```bash
df -h /runpod-volume
du -sh /runpod-volume/hf-cache /runpod-volume/hf-home /runpod-volume/models 2>/dev/null
```

Expand the volume or deliberately remove obsolete checkpoints/caches until at least 100 GB is free, then retry. A partial Hub download may resume. Do not delete the active `/runpod-volume/hf-cache` just to make space unless you intend to redownload the base model.

Local tests cover request validation and exact outside-mask pixel preservation. Container build, GPU model loading, mask quality and RunPod image output still require a deployed test.

### GPU memory

`enable_model_cpu_offload()` still moves the entire 40.9 GB transformer onto the GPU during denoising, so a 24 GB GPU fails even if model loading succeeds. The worker now selects **sequential CPU offload** when less than 48 GiB of GPU memory is free; it also enables VAE tiling. This avoids the whole-transformer transfer but is much slower and may exceed the endpoint timeout. For dependable throughput, use a GPU with around **80 GB VRAM**. `QWEN_OFFLOAD_MODE=model|sequential|auto` can override the automatic choice; `auto` is the default. Allocator fragmentation settings cannot make a 40.9 GB component fit in 24 GB VRAM.

The Docker build pins Diffusers 0.37.0, Transformers 4.x and `huggingface-hub<1.0` in one resolver transaction. It imports the Qwen and SegFormer classes and runs `pip check` before the image is published, preventing the earlier `huggingface-hub==1.32.0` runtime import error.
