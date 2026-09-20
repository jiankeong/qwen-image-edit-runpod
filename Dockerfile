FROM runpod/worker-comfyui:5.8.6-base-cuda12.8.1

USER root

# Keep the tested CUDA/PyTorch base and use the official Qwen 2511 Diffusers pipeline.
RUN pip install --no-cache-dir 'runpod>=1.7' 'transformers>=4.51' 'accelerate>=1.5' 'peft>=0.15' \
    && pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git

COPY handler.py /workspace/handler.py
CMD ["python", "/workspace/handler.py"]
