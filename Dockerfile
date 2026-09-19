FROM runpod/worker-comfyui:5.8.6-base-cuda12.8.1

USER root

COPY scripts/bootstrap_flux.py /opt/flux/bootstrap_flux.py
COPY scripts/start_flux.sh /opt/flux/start_flux.sh
COPY config/extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY handler.py /workspace/handler.py

# Prepare Network Volume models before starting the inherited ComfyUI worker.
# The Hub smoke test skips the downloads.
RUN chmod +x /opt/flux/start_flux.sh
CMD ["/opt/flux/start_flux.sh"]
