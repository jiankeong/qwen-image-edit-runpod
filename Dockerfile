FROM runpod/worker-comfyui:5.8.6-base

USER root

# Qwen Image Edit GGUF loader
RUN git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git /comfyui/custom_nodes/ComfyUI-GGUF \
    && pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-GGUF/requirements.txt

# Used only during image build to fetch models from Hugging Face
RUN pip install --no-cache-dir huggingface_hub

COPY scripts/download_models.py /tmp/download_models.py

# HF_TOKEN is optional for these public repos. If Hugging Face rate-limits the build,
# add it as a build secret/environment variable in your build environment.
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

RUN python /tmp/download_models.py

# worker-comfyui base image already contains the RunPod handler/startup.
