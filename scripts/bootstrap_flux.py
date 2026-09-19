"""Cache a FLUX.1-dev FP8 checkpoint plus Shar514's LoRA on the Network Volume."""

import errno
import json
import os
import sys
import urllib.request
from pathlib import Path
from urllib.parse import quote

VOLUME = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
CHECKPOINT = VOLUME / "models/checkpoints/flux1-dev-fp8.safetensors"
LORA = VOLUME / "models/loras/Shar514_Flux-Uncensored-V2.safetensors"
LORA_REPO = os.getenv("FLUX_LORA_REPO", "Shar514/Flux-Uncensored-V2")
BASE_REPO = "Comfy-Org/flux1-dev"
TOKEN = os.getenv("HF_TOKEN", "")
HEADERS = {"User-Agent": "flux-image-edit-runpod/1.0"}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"


def resolve(repo, name):
    return f"https://huggingface.co/{repo}/resolve/main/{quote(name, safe='/')}?download=true"


def choose_lora():
    url = f"https://huggingface.co/api/models/{LORA_REPO}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
        info = json.load(response)
    files = [item.get("rfilename", "") for item in info.get("siblings", [])]
    candidates = [name for name in files if name.lower().endswith(".safetensors")]
    if "lora.safetensors" in candidates:
        return "lora.safetensors"
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(f"Expected one LoRA .safetensors in {LORA_REPO}; found {candidates}")


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=HEADERS, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        size = int(response.headers.get("Content-Length") or 0)
    if not size:
        raise RuntimeError(f"No Content-Length for {url}; refusing an unverifiable download")
    if dest.exists() and dest.stat().st_size == size:
        print(f"[flux-bootstrap] cached: {dest} ({size} bytes)", flush=True)
        return
    partial = dest.with_suffix(dest.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        partial.unlink()
        offset = 0
    if offset == size:
        partial.replace(dest)
        return
    headers = dict(HEADERS)
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as src:
        if offset and src.status != 206:
            offset = 0
        available = os.statvfs(partial.parent)
        free_bytes = available.f_bavail * available.f_frsize
        needed = size - offset + 1024 ** 3
        if free_bytes < needed:
            raise RuntimeError(
                f"Insufficient Network Volume space at {partial.parent}: need {needed:,} bytes "
                f"including reserve, available {free_bytes:,}; partial file: {partial}"
            )
        try:
            with open(partial, "ab" if offset else "wb") as out:
                while chunk := src.read(16 * 1024 * 1024):
                    out.write(chunk)
        except OSError as error:
            if error.errno in (errno.EDQUOT, errno.ENOSPC):
                raise RuntimeError(f"Network Volume quota exhausted; resumable download retained at {partial}") from error
            raise
    if partial.stat().st_size != size:
        raise RuntimeError(f"Incomplete download {partial}: {partial.stat().st_size:,}/{size:,} bytes")
    partial.replace(dest)
    print(f"[flux-bootstrap] downloaded: {dest} ({size} bytes)", flush=True)


def main():
    import fcntl
    VOLUME.mkdir(parents=True, exist_ok=True)
    with open(VOLUME / ".flux_uncensored_bootstrap.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lora_name = choose_lora()
        print(f"[flux-bootstrap] base={BASE_REPO}/flux1-dev-fp8.safetensors", flush=True)
        print(f"[flux-bootstrap] lora={LORA_REPO}/{lora_name}", flush=True)
        download(resolve(BASE_REPO, "flux1-dev-fp8.safetensors"), CHECKPOINT)
        download(resolve(LORA_REPO, lora_name), LORA)
        print(f"[flux-bootstrap] ready: {CHECKPOINT}, {LORA}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[flux-bootstrap] ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
