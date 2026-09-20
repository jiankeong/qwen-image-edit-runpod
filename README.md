# Qwen-Image-Edit-2511 + ScottzillaSystems NSFW LoRA — RunPod

This worker uses the full [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) editing pipeline and loads [ScottzillaSystems/qwen-image-edit-plus-nsfw-lora](https://huggingface.co/ScottzillaSystems/qwen-image-edit-plus-nsfw-lora). It no longer uses the Qwen Turbo GGUF or FLUX checkpoint.

## One uploaded image, targeted output

Send [examples/request-template.json](examples/request-template.json). Replace `YOUR_BASE64_IMAGE_HERE` with one PNG/JPEG image encoded as base64. Set `prompt` and `edit_target` to `clothes` or `background`; `auto` infers the target from Chinese/English keywords and fails if ambiguous.

The worker runs Qwen image editing on the source image, segments the source with [mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes), then **copies every original pixel outside the selected region** into the final PNG. Clothes mode selects garment labels; background mode selects the background label. The target region is still synthesized and the border may be imperfect. Segmentation errors can select the wrong region. A pixel-equality check verifies that all unselected pixels remain unchanged.

## Deployment

Attach a Network Volume at `/runpod-volume`; the model cache is `/runpod-volume/hf-cache`. Allocate at least **100 GB free** for the BF16 base, LoRA, parser and download cache; previous model files are not removed automatically. Start testing on an **80 GB GPU**. The full Qwen-Image-Edit-2511 is a 20B BF16 model and cold starts may be long. `USE_MOCK_PIPELINE=1` skips model loading in Hub smoke tests. `HF_TOKEN` can help with download rate limits.

Local tests cover request validation and exact outside-mask pixel preservation. Container build, GPU model loading, mask quality and RunPod image output still require a deployed test.

The Docker build pins Diffusers 0.37.0, Transformers 4.x and `huggingface-hub<1.0` in one resolver transaction. It imports the Qwen and SegFormer classes and runs `pip check` before the image is published, preventing the earlier `huggingface-hub==1.32.0` runtime import error.
