# RunPod vLLM Worker for Qwen 32B
# Deploys via GitHub integration

FROM runpod/base:0.6.2-cuda12.2.0

WORKDIR /

# Install vLLM and dependencies
RUN pip install --no-cache-dir \
    vllm==0.6.4.post1 \
    runpod>=1.7.0 \
    transformers>=4.45.0 \
    accelerate \
    sentencepiece

# Copy handler
COPY src/handler.py /handler.py

# Environment defaults (can be overridden in RunPod console)
ENV MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
ENV MAX_MODEL_LEN="32768"
ENV GPU_MEMORY_UTILIZATION="0.95"
ENV DTYPE="auto"
ENV QUANTIZATION="awq"

CMD ["python", "-u", "/handler.py"]
