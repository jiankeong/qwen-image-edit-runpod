FROM runpod/worker-comfyui:5.8.6

USER root

# Install the GGUF custom node without touching the base torch/CUDA stack.
RUN git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git /comfyui/custom_nodes/ComfyUI-GGUF \
    && if [ -f /comfyui/custom_nodes/ComfyUI-GGUF/requirements.txt ]; then \
         pip install --no-cache-dir --no-deps -r /comfyui/custom_nodes/ComfyUI-GGUF/requirements.txt || true; \
       fi

COPY scripts/bootstrap_models.py /opt/qwen/bootstrap_models.py
COPY handler.py /workspace/handler.py

# Import hook: for real workers, synchronously prepare model files on the Network Volume
# before the official worker-comfyui handler starts. Hub smoke tests skip downloads.
RUN python - <<'PY'
from pathlib import Path
p = Path('/usr/local/lib/python3.11/site-packages/sitecustomize.py')
old = p.read_text() if p.exists() else ''
block = r'''
import os, subprocess, sys
if os.getenv("USE_MOCK_PIPELINE", "0") != "1":
    script = "/opt/qwen/bootstrap_models.py"
    if os.path.exists(script):
        subprocess.run([sys.executable, script], check=True)
'''
p.write_text(old + '\n' + block)
PY
