"""Install the v19 AIO checkpoint from persistent cache or container disk."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from handler import BASE_REPO, CHECKPOINT_FILE, CHECKPOINT_PATH, storage_quota_message

GIB = 1024 ** 3
# The 28.4 GB checkpoint needs room for the download and filesystem overhead.
MIN_FREE_GIB = {CHECKPOINT_PATH: 35}
EPHEMERAL_CACHE = Path(os.getenv("QWEN_EPHEMERAL_CACHE", "/tmp/qwen-hf-cache"))


def quota_error(error):
    message = str(error).lower()
    return "disk quota exceeded" in message or "no space left on device" in message or "os error 122" in message


def run_download(remote, cache, *, ephemeral=False, local_only=False):
    """Use a child process because huggingface_hub reads cache env at import time."""
    cache = Path(cache)
    home = cache.parent / ("qwen-hf-home" if ephemeral else "hf-home")
    xet = home / "xet"
    temp = cache.parent / ("qwen-tmp" if ephemeral else "tmp")
    for directory in (cache, home, xet, temp):
        directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(HF_HOME=str(home), HF_HUB_CACHE=str(cache), HF_XET_CACHE=str(xet),
               TMPDIR=str(temp), HF_XET_CHUNK_CACHE_SIZE_BYTES="0")
    code = (
        "import os\n"
        "from huggingface_hub import hf_hub_download\n"
        "print(hf_hub_download(os.environ['QWEN_REPO'], os.environ['QWEN_FILE'], "
        "cache_dir=os.environ['HF_HUB_CACHE'], token=os.getenv('HF_TOKEN') or None, "
        "local_files_only=os.environ['QWEN_LOCAL_ONLY'] == '1'))\n"
    )
    env.update(QWEN_REPO=BASE_REPO, QWEN_FILE=remote, QWEN_LOCAL_ONLY="1" if local_only else "0")
    result = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"hf_hub_download exited {result.returncode}")
    return Path(result.stdout.strip().splitlines()[-1])


def install_models(base=Path("/opt/ComfyUI/models"), downloader=run_download):
    assets = (
        (CHECKPOINT_PATH, base / "checkpoints" / CHECKPOINT_FILE),
    )
    volume_cache = Path(os.environ["HF_HUB_CACHE"])
    volume_cache.mkdir(parents=True, exist_ok=True)
    for remote, link in assets:
        link.parent.mkdir(parents=True, exist_ok=True)
        required = MIN_FREE_GIB[remote] * GIB
        try:
            downloaded = Path(downloader(remote, volume_cache, ephemeral=False, local_only=True))
            cached = downloaded.is_file()
        except RuntimeError:
            cached = False
        try_volume = not cached and shutil.disk_usage(volume_cache).free >= required
        if try_volume:
            try:
                downloaded = Path(downloader(remote, volume_cache, ephemeral=False, local_only=False))
            except Exception as exc:
                if not quota_error(exc):
                    raise
                print(f"[qwen-bootstrap] Network Volume quota reached while downloading {remote}; trying container disk", flush=True)
                try_volume = False
        if not cached and not try_volume:
            EPHEMERAL_CACHE.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(EPHEMERAL_CACHE).free
            if free < required:
                raise RuntimeError(
                    storage_quota_message(volume_cache)
                    + f" Container disk also has only {free / GIB:.1f} GiB free; "
                    f"{required / GIB:.0f} GiB required for {remote}."
                )
            print(f"[qwen-bootstrap] using temporary container cache for {remote}; increase Network Volume for persistence", flush=True)
            try:
                downloaded = Path(downloader(remote, EPHEMERAL_CACHE, ephemeral=True, local_only=False))
            except Exception as exc:
                if quota_error(exc):
                    raise RuntimeError(storage_quota_message(volume_cache) + " Container disk also exhausted.") from exc
                raise
        if not downloaded.is_file():
            raise RuntimeError(f"Model download did not produce a file: {downloaded}")
        if not link.is_symlink() or link.resolve() != downloaded.resolve():
            link.unlink(missing_ok=True)
            link.symlink_to(downloaded)
        print(f"[qwen-bootstrap] {remote} -> {link}", flush=True)


if __name__ == "__main__":
    install_models()
