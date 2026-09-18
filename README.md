# Qwen Image Edit Uncensored v1.1 GGUF — RunPod Serverless

Target diffusion model:
`ChrisColeTech/qwen-image-edit-uncensored-v1.1-GGUF`

This project uses:
- RunPod `worker-comfyui` 5.8.6 base
- ComfyUI-GGUF
- the repo's Q4_K_M quant when available (fallbacks are automatic)
- official Qwen Image FP8 text encoder
- official Qwen Image VAE

## Deploy — easiest way

1. Create a new GitHub repository.
2. Upload everything in this folder.
3. In RunPod: **Serverless → New Endpoint → Start from GitHub Repo**.
4. Select the repository.
5. Context path: `/`
6. Dockerfile path: `Dockerfile`
7. Start with a GPU with **24 GB+ VRAM**. If you hit OOM, switch to 48 GB.
8. Active Workers: `0`; Max Workers: `1`; Flash Boot: ON.
9. Deploy.

The Docker build downloads the models, so the built image is large. This reduces model-download work when a worker starts.

## Test

After deployment:

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  --data @examples/test_input.json
```

By default worker-comfyui returns generated images as base64. Configure S3-compatible storage in RunPod if you want URLs instead.

## Use your own image + prompt

The worker-comfyui API expects a ComfyUI API workflow. In `examples/test_input.json`:

- Change node `6` → `inputs.prompt`
- Change `input.images[0].image` to your image URL or supported image input
- Keep the uploaded image name as `input.png` because workflow node `1` loads `input.png`
- Change node `9` for seed/steps/CFG if needed

## Important

The diffusion GGUF repository can change its filenames. `scripts/download_models.py` queries the repository at Docker build time and automatically selects Q4_K_M when present, then renames it to:

`qwen_image_edit_uncensored_q4.gguf`

This keeps the workflow filename stable.

If the model repository later changes architecture incompatibly, the Docker build may still succeed while the workflow needs updating.
