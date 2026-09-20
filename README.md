# Qwen Image Edit Rapid AIO v23 NVFP4 GGUF RunPod worker

This worker uses the exact three components in [FreedomAISVR/Qwen-Image-Edit-Rapid-AIO-NSFW-v23-NVFP4-GGUF](https://huggingface.co/FreedomAISVR/Qwen-Image-Edit-Rapid-AIO-NSFW-v23-NVFP4-GGUF): `qwen-v23-diffusion-NVFP4.gguf`, `text_encoder/text_encoder-NVFP4.gguf`, and `vae/vae.safetensors`. The prior SDXL checkpoint and Qwen LoRA are not loaded. The GGUF files run through [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF), not Diffusers `from_pretrained`.

## Request

Submit one base64 PNG/JPEG in `input.image`. See [raw example](examples/raw-edit-request.json) or [masked example](examples/request-template.json). `prompt` says what to change. Defaults: `steps=4`, `true_cfg_scale=1.0` (alias `guidance_scale`), `strength=1.0`, `seed=0`, `negative_prompt=" "`, `output_mode="raw"`. `steps` controls the rapid sampler; `strength` maps to KSampler denoise. `output_mode="masked"` uses a clothing/background parser and composites unchanged original pixels outside the selected region. It is not inpainting; the edited region can still vary. `return_raw=true` adds the uncomposited output. PNG response uses base64.

## Deployment

The model card specifies NVIDIA Blackwell RTX 50-series, CUDA 13.0+; this image uses CUDA 13.2 and PyTorch cu132. Select RTX 5090 (32 GB) or Blackwell 96 GB, not L4/4090/A100. The model files total about 16 GB; give the Network Volume at `/runpod-volume` at least 35 GB free for downloads, cache and outputs. The bootstrap downloads directly to the volume and symlinks those files into ComfyUI model directories without duplicate copies. It starts ComfyUI locally and submits an API workflow using `UnetLoaderGGUF`, `CLIPLoaderGGUF`, and `TextEncodeQwenImageEditPlus`.

`USE_MOCK_PIPELINE=1` is only for Hub smoke tests; it skips GPU inference/download. Local unit tests verify workflow construction, input validation, and mask compositing. Docker build and live RunPod GPU image-quality test remain to be performed.
