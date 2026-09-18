import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

COMFY = Path("/comfyui")
UNET = COMFY / "models" / "unet"
TEXT = COMFY / "models" / "text_encoders"
VAE = COMFY / "models" / "vae"
for p in (UNET, TEXT, VAE):
    p.mkdir(parents=True, exist_ok=True)

token = os.getenv("HF_TOKEN") or None
api = HfApi(token=token)

MODEL_REPO = "ChrisColeTech/qwen-image-edit-uncensored-v1.1-GGUF"

def repo_files(repo):
    return api.list_repo_files(repo_id=repo, repo_type="model")

def choose_diffusion_file():
    files = repo_files(MODEL_REPO)
    ggufs = [f for f in files if f.lower().endswith(".gguf") and "mmproj" not in f.lower()]
    if not ggufs:
        raise RuntimeError(f"No GGUF found in {MODEL_REPO}")

    # Prefer Q4_K_M for a good quality/VRAM compromise.
    preferences = ["q4_k_m", "q4_k_s", "q4_0", "q5_k_m", "q6_k", "q8_0"]
    for pref in preferences:
        for f in ggufs:
            if pref in f.lower():
                return f
    return ggufs[0]

def fetch(repo, filename, dest_dir, dest_name=None):
    print(f"Downloading {repo}/{filename}")
    cached = hf_hub_download(repo_id=repo, filename=filename, token=token)
    dest = Path(dest_dir) / (dest_name or Path(filename).name)
    if dest.exists():
        dest.unlink()
    # Copy rather than symlink so the Docker layer is self-contained.
    import shutil
    shutil.copy2(cached, dest)
    print(" ->", dest)
    return dest

diffusion = choose_diffusion_file()
fetch(MODEL_REPO, diffusion, UNET, "qwen_image_edit_uncensored_q4.gguf")

# Stable, standard Qwen Image Edit components.
fetch(
    "Comfy-Org/Qwen-Image_ComfyUI",
    "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    TEXT,
    "qwen_2.5_vl_7b_fp8_scaled.safetensors",
)
fetch(
    "Comfy-Org/Qwen-Image_ComfyUI",
    "split_files/vae/qwen_image_vae.safetensors",
    VAE,
    "qwen_image_vae.safetensors",
)

print("All models downloaded.")
