FROM runpod/worker-comfyui:5.8.6-base-cuda12.8.1

USER root

# Resolve the full HF stack together. Transformers 4.x requires hub<1.0;
# installing Diffusers main separately previously upgraded hub to 1.x.
RUN pip install --no-cache-dir \
      'runpod>=1.7' \
      'diffusers==0.37.0' \
      'transformers>=4.51,<5' \
      'huggingface-hub>=0.34,<1.0' \
      'accelerate>=1.5' \
      'peft>=0.17' \
    && python -c "import huggingface_hub, transformers, diffusers; from transformers import AutoImageProcessor, SegformerForSemanticSegmentation; from diffusers import QwenImageEditPlusPipeline; assert int(huggingface_hub.__version__.split('.')[0]) == 0; print('HF_IMPORT_OK hub=' + huggingface_hub.__version__ + ' transformers=' + transformers.__version__ + ' diffusers=' + diffusers.__version__)" \
    && pip check

COPY handler.py /workspace/handler.py
CMD ["python", "/workspace/handler.py"]
