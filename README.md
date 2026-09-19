# Qwen Image Edit Uncensored GGUF — RunPod Serverless

Target model repository:

`ChrisColeTech/qwen-image-edit-uncensored-GGUF`

This deployment keeps the large model files on the RunPod Network Volume (`/runpod-volume`) instead of baking them into the Docker image.

## What it does

- Base image: `runpod/worker-comfyui:5.8.6-base-cuda12.8.1` (CUDA 12.8 / PyTorch cu128)
- Installs `city96/ComfyUI-GGUF`
- Discovers the actual `.gguf` file from the requested Hugging Face repository at worker startup
- Prefers a Q4_K_M / Q4 quant when the repository contains one
- Stores the selected transformer as `/runpod-volume/models/diffusion_models/qwen_image_edit_uncensored_q4.gguf`
- Downloads Qwen2.5-VL text encoder, mmproj, and Qwen Image VAE to the same Network Volume
- Keeps `.runpod/tests.json`; Hub smoke tests set `USE_MOCK_PIPELINE=1` so they do not download model weights

## Network Volume

Attach your RunPod Network Volume to the Serverless endpoint; RunPod mounts it at
`/runpod-volume`. Allocate **at least 30 GB** for the Q4 transformer (about
11.7 GB), text encoder (about 4.4 GB), mmproj (about 1.4 GB), VAE, and spare
space. `containerDiskInGb` does not increase Network Volume capacity. Check
`df -h /runpod-volume` and `du -sh /runpod-volume/models` on a worker to
confirm the mount and available space. Increase the volume size if the worker
reports `Disk quota exceeded`. An interrupted `.part` download resumes after
you expand the volume.

The first real worker downloads the model files. Later workers reuse them.

## Important workflow note

The repository deliberately does **not** invent a ComfyUI API workflow for this GGUF build. Custom-node names and Qwen Image Edit conditioning nodes change across ComfyUI / ComfyUI-GGUF versions. After deployment, create or import a Qwen Image Edit GGUF workflow in the exact deployed ComfyUI version and export **API Format**. Then send it through worker-comfyui with `input.png` in the `images` array.

`examples/request-template.json` shows the outer RunPod request shape.
