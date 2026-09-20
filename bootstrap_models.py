"""Download the exact three model components once to the mounted volume."""

import errno
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

from handler import BASE_REPO, CLIP_FILE, DIFFUSION_FILE, VAE_FILE, storage_quota_message


def install_models(base=Path("/opt/ComfyUI/models")):
    assets = (
        (DIFFUSION_FILE, base / "diffusion_models" / DIFFUSION_FILE),
        ("text_encoder/" + CLIP_FILE, base / "text_encoders" / CLIP_FILE),
        ("vae/" + VAE_FILE, base / "vae" / VAE_FILE),
    )
    cache = Path(os.environ["HF_HUB_CACHE"])
    for remote, link in assets:
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            downloaded = Path(hf_hub_download(BASE_REPO, remote, cache_dir=cache,
                                              token=os.getenv("HF_TOKEN") or None))
        except OSError as exc:
            if exc.errno in (errno.ENOSPC, errno.EDQUOT, 122):
                raise RuntimeError(storage_quota_message(cache)) from exc
            raise
        if not link.is_symlink() or link.resolve() != downloaded.resolve():
            link.unlink(missing_ok=True)
            link.symlink_to(downloaded)
        print(f"[qwen-bootstrap] {remote} -> {link}", flush=True)


if __name__ == "__main__":
    install_models()
