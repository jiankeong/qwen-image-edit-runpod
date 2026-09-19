# FLUX.1-dev + Shar514 Flux-Uncensored-V2 — RunPod Serverless

The requested [Shar514/Flux-Uncensored-V2](https://huggingface.co/Shar514/Flux-Uncensored-V2) is a **LoRA adapter**, not a standalone checkpoint or Qwen GGUF. This worker loads the [Comfy-Org FLUX.1-dev FP8 checkpoint](https://huggingface.co/Comfy-Org/flux1-dev/blob/main/flux1-dev-fp8.safetensors), then applies the requested LoRA in ComfyUI. The base checkpoint is about 17.2 GB. The startup script discovers the adapter's `.safetensors` filename from Hugging Face instead of relying on the model card's inconsistent repository references.

## Deploy

- Base Docker image: `runpod/worker-comfyui:5.8.6-base-cuda12.8.1`.
- Attach a RunPod Network Volume at `/runpod-volume`; allocate **at least 30 GB free** for the new checkpoint, LoRA, partial downloads and reserve. Old Qwen files are not deleted. If they remain on the same volume, increase capacity accordingly or remove them manually after backup.
- On first real startup, the worker downloads `/runpod-volume/models/checkpoints/flux1-dev-fp8.safetensors` and `/runpod-volume/models/loras/Shar514_Flux-Uncensored-V2.safetensors`. Interrupted downloads resume from `.part` files.
- `HF_TOKEN` is optional for the Comfy-Org checkpoint and this LoRA, but can help with Hugging Face rate limits. The official `black-forest-labs/FLUX.1-dev` repository is gated; this deployment instead uses Comfy-Org's FP8 ComfyUI checkpoint. Observe the FLUX.1-dev non-commercial license.
- Hub smoke tests set `USE_MOCK_PIPELINE=1`, so they test the worker without downloading 17+ GB.

## One-image clothes edit

Send [examples/request-template.json](examples/request-template.json) as the RunPod request. Replace `YOUR_BASE64_IMAGE_HERE` with base64 of **one** input image, and edit node `5`'s `text` prompt. The workflow alone is in [examples/flux-img2img-workflow.json](examples/flux-img2img-workflow.json).

The graph loads the source image with `LoadImage`, encodes it with the FLUX VAE, applies the Shar514 LoRA, and samples from the source latent (`denoise: 0.4`). Keep `cfg: 1.0`; FLUX prompt guidance is node `6`. Lower `denoise` toward 0.25 for closer likeness, or raise toward 0.55 for more substantial clothing changes. This is **img2img**, not a dedicated masked garment editor: facial identity cannot be guaranteed by a prompt or denoise setting alone. For exact face preservation, use a garment-only mask and composite the untouched original face, or use a dedicated edit model/workflow.

Use approximately 1024-pixel input images to control memory and latency. The first request may take longer because the worker must download and load the models. The 24 GB GPU entries in the Hub config are capacity candidates, not a verified throughput guarantee.
