#!/bin/sh
set -eu

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TMPDIR"
if [ "${USE_MOCK_PIPELINE:-0}" != "1" ]; then
  python /workspace/bootstrap_models.py
  mkdir -p /runpod-volume/comfy-input /runpod-volume/comfy-output
  python /opt/ComfyUI/main.py --listen 127.0.0.1 --port 8188 \
    --input-directory /runpod-volume/comfy-input \
    --output-directory /runpod-volume/comfy-output &
  export COMFY_INPUT_DIR=/runpod-volume/comfy-input
  python - <<'PY'
import time, urllib.request
for _ in range(120):
    try:
        urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=2).close()
        break
    except Exception:
        time.sleep(2)
else:
    raise RuntimeError('ComfyUI did not become ready within 240 seconds')
import json
with urllib.request.urlopen('http://127.0.0.1:8188/object_info', timeout=30) as response:
    info = json.load(response)
for node in ('UnetLoaderGGUF', 'CLIPLoaderGGUF', 'TextEncodeQwenImageEditPlus'):
    if node not in info:
        raise RuntimeError(f'ComfyUI missing required node: {node}')
PY
fi
exec python /workspace/handler.py
