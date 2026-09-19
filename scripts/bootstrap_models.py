import json
import errno
import os
import sys
import time
import urllib.request
from pathlib import Path

HF_REPO = os.getenv("QWEN_HF_REPO", "ChrisColeTech/qwen-image-edit-uncensored-GGUF")
VOL = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
MODEL_ROOT = VOL / "models"
DIFF = MODEL_ROOT / "diffusion_models"
TEXT = MODEL_ROOT / "text_encoders"
VAE = MODEL_ROOT / "vae"
LOCK = VOL / ".qwen_image_edit_bootstrap.lock"
READY = VOL / ".qwen_image_edit_ready_v1"
HF_TOKEN = os.getenv("HF_TOKEN", "")

for d in (DIFF, TEXT, VAE):
    d.mkdir(parents=True, exist_ok=True)

headers = {"User-Agent": "qwen-image-edit-runpod/1.0"}
if HF_TOKEN:
    headers["Authorization"] = f"Bearer {HF_TOKEN}"


def api_json(url):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def download(url, dest: Path):
    if dest.exists() and dest.stat().st_size > 10_000_000:
        print(f"[qwen-bootstrap] exists: {dest} ({dest.stat().st_size} bytes)", flush=True)
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    head = urllib.request.Request(url, headers=headers, method="HEAD")
    with urllib.request.urlopen(head, timeout=60) as response:
        size = int(response.headers.get("Content-Length") or 0)
    offset = tmp.stat().st_size if tmp.exists() else 0
    if size and offset > size:
        tmp.unlink()
        offset = 0
    if size and offset == size:
        tmp.replace(dest)
        return

    request_headers = dict(headers)
    if offset:
        request_headers["Range"] = f"bytes={offset}-"
    req = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(req, timeout=120) as src:
        # Some mirrors ignore Range; restart the partial file only in that case.
        if offset and src.status != 206:
            offset = 0
        free = os.statvfs(tmp.parent)
        available = free.f_bavail * free.f_frsize
        remaining = max(0, size - offset) if size else 0
        reserve = 1024 ** 3
        if size and available < remaining + reserve:
            raise RuntimeError(
                f"Insufficient space on {tmp.parent}: need {remaining + reserve:,} bytes "
                f"(including 1 GiB reserve), available {available:,}. "
                "Attach or expand the RunPod Network Volume at /runpod-volume "
                "(30 GB or more recommended); container disk size is separate. "
                f"Partial download remains at {tmp}."
            )
        try:
            with open(tmp, "ab" if offset else "wb") as out:
                while True:
                    chunk = src.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
        except OSError as e:
            if e.errno in (errno.EDQUOT, errno.ENOSPC):
                raise RuntimeError(
                    f"Storage quota exhausted at {tmp}; expand the RunPod Network Volume "
                    "at /runpod-volume to at least 30 GB. The .part file is retained "
                    "for a resumed download after capacity is increased."
                ) from e
            raise
    if size and tmp.stat().st_size != size:
        raise RuntimeError(f"Incomplete download {tmp}: {tmp.stat().st_size:,}/{size:,} bytes")
    tmp.replace(dest)
    print(f"[qwen-bootstrap] downloaded: {dest} ({dest.stat().st_size} bytes)", flush=True)


def choose_uncensored_gguf():
    info = api_json(f"https://huggingface.co/api/models/{HF_REPO}")
    files = [s.get("rfilename", "") for s in info.get("siblings", [])]
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    if not ggufs:
        raise RuntimeError(f"No .gguf found in {HF_REPO}")
    prefs = ["q4_k_m", "q4_k", "q5_k_m", "q5_k", "q6_k", "q4_0", "q8_0"]
    for pref in prefs:
        for f in ggufs:
            low = f.lower()
            if pref in low and "mmproj" not in low and "qwen2.5-vl" not in low:
                return f
    for f in ggufs:
        low = f.lower()
        if "mmproj" not in low and "qwen2.5-vl" not in low:
            return f
    raise RuntimeError(f"Only encoder/mmproj GGUFs found in {HF_REPO}: {ggufs}")


def hf_resolve(repo, path):
    from urllib.parse import quote
    return f"https://huggingface.co/{repo}/resolve/main/{quote(path, safe='/')}?download=true"


def ensure_support_files():
    # Known-good Qwen Image Edit components used by ChrisColeTech's current GGUF builds.
    support_repo = os.getenv("QWEN_SUPPORT_REPO", "ChrisColeTech/qwen-image-edit-turbo-GGUF")
    enc = "split/text_encoders/Qwen2.5-VL-7B-Instruct-q4_0.gguf"
    mm = "split/text_encoders/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf"
    vae_candidates = ["split/vae/qwen_image_vae.safetensors", "split/qwen_image_vae.safetensors"]
    download(hf_resolve(support_repo, enc), TEXT / "Qwen2.5-VL-7B-Instruct-q4_0.gguf")
    download(hf_resolve(support_repo, mm), TEXT / "Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf")
    last = None
    for path in vae_candidates:
        try:
            download(hf_resolve(support_repo, path), VAE / "qwen_image_vae.safetensors")
            last = None
            break
        except Exception as e:
            last = e
    if last:
        raise last


def write_ready(model_source):
    payload = {
        "repo": HF_REPO,
        "source_model": model_source,
        "diffusion": str(DIFF / "qwen_image_edit_uncensored_q4.gguf"),
        "text_encoder": str(TEXT / "Qwen2.5-VL-7B-Instruct-q4_0.gguf"),
        "mmproj": str(TEXT / "Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf"),
        "vae": str(VAE / "qwen_image_vae.safetensors"),
        "time": time.time(),
    }
    READY.write_text(json.dumps(payload, indent=2))


def main():
    print(f"[qwen-bootstrap] repo={HF_REPO}", flush=True)
    print(f"[qwen-bootstrap] volume={VOL}", flush=True)
    try:
        import fcntl
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        with open(LOCK, "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            model_path = choose_uncensored_gguf()
            print(f"[qwen-bootstrap] selected transformer={model_path}", flush=True)
            download(hf_resolve(HF_REPO, model_path), DIFF / "qwen_image_edit_uncensored_q4.gguf")
            ensure_support_files()
            write_ready(model_path)
    except Exception as e:
        print(f"[qwen-bootstrap] ERROR: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise
    print(f"[qwen-bootstrap] ready: {READY}", flush=True)


if __name__ == "__main__":
    main()
