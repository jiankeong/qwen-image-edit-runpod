FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf-home \
    HF_HUB_CACHE=/runpod-volume/hf-cache \
    HF_XET_CACHE=/runpod-volume/hf-home/xet \
    HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
    CC=/usr/bin/gcc

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip python3-dev build-essential git ca-certificates \
      libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# CUDA 13.0 supports Blackwell and allows a wider RunPod host-driver pool.
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 \
      --index-url https://download.pytorch.org/whl/cu130 \
    && git clone --depth 1 https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI \
    && pip install --no-cache-dir -r /opt/ComfyUI/requirements.txt \
      'runpod>=1.7,<2' 'huggingface-hub>=0.34,<1' \
      'transformers>=4.51,<5' 'hf-xet>=1.1' \
    && pip check

# The Network Volume is mounted only at runtime; dpkg/pip need /tmp while building.
ENV TMPDIR=/runpod-volume/tmp

COPY handler.py /workspace/handler.py
COPY bootstrap_models.py /workspace/bootstrap_models.py
COPY startup.sh /workspace/startup.sh
CMD ["sh", "/workspace/startup.sh"]
