FROM runpod/worker-comfyui:5.8.6-base-cuda12.8.1

USER root

# Install the GGUF custom node without touching the base torch/CUDA stack.
RUN git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git /comfyui/custom_nodes/ComfyUI-GGUF \
    && if [ -f /comfyui/custom_nodes/ComfyUI-GGUF/requirements.txt ]; then \
         pip install --no-cache-dir --no-deps -r /comfyui/custom_nodes/ComfyUI-GGUF/requirements.txt || true; \
       fi

COPY scripts/bootstrap_models.py /opt/qwen/bootstrap_models.py
COPY scripts/start_qwen.sh /opt/qwen/start_qwen.sh
COPY config/extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY handler.py /workspace/handler.py

# Prepare Network Volume models before starting the inherited ComfyUI worker.
# The Hub smoke test skips the downloads.
RUN chmod +x /opt/qwen/start_qwen.sh
CMD ["/opt/qwen/start_qwen.sh"]
