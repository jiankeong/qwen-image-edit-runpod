# Qwen Image Edit Rapid AIO v19 NSFW RunPod worker

This worker loads the [Phr00t/Qwen-Image-Edit-Rapid-AIO](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO) `v19/Qwen-Rapid-AIO-NSFW-v19.safetensors` FP8 all-in-one checkpoint (28.4 GB). The checkpoint bundles the diffusion model, text encoder, and VAE. ComfyUI loads it with `CheckpointLoaderSimple`; the former v23 NVFP4 GGUF components are no longer loaded. The author describes v19 as the best release for edit consistency and recommends 4–8 steps, CFG 1, and `euler_ancestral`/`beta` sampling.

## Request

Submit one base64 PNG/JPEG in `input.image`. See [raw example](examples/raw-edit-request.json) or [masked example](examples/request-template.json). `prompt` says what to change. Defaults: `steps=4`, `true_cfg_scale=1.0` (alias `guidance_scale`), `strength=1.0`, `seed=0`, `negative_prompt=" "`, `output_mode="raw"`. At the default `strength=1`, the workflow uses an empty latent sized to the input, as recommended for Rapid AIO. At lower strengths, it starts from a VAE-encoded input and maps `strength` to KSampler denoise. `output_mode="masked"` uses a clothing/background parser and composites unchanged original pixels outside the selected region. It is not inpainting; the edited region can still vary. `return_raw=true` adds the uncomposited output. PNG response uses base64.

## Deployment

Use a high-memory GPU such as RTX PRO 6000 Blackwell (96 GB). The 28.4 GB checkpoint plus the model runtime will not reliably fit a 32 GB GPU. Reserve at least 40 GB free on the Network Volume at `/runpod-volume`; a 60 GB or larger volume is preferable for downloads, cache, parser, and outputs. Bootstrap reuses a complete cached checkpoint. If the Network Volume lacks 35 GiB free for a cold download, it can use the container disk under `/tmp/qwen-hf-cache`, which must also have at least 35 GiB free. Temporary cache downloads again on a cold worker, so persistent Network Volume space is preferable.

`USE_MOCK_PIPELINE=1` is only for Hub smoke tests; it skips GPU inference/download. Local unit tests verify workflow construction, input validation, and mask compositing. Docker build and live RunPod GPU image-quality test remain to be performed.
