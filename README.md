# RunPod vLLM Worker for Jack Heavy-Hitter

Custom vLLM worker for deploying Qwen 32B on RunPod Serverless.

## Deployment

### Option 1: Deploy from GitHub (Recommended)

1. Push this folder to a GitHub repository
2. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
3. Click **New Endpoint** → **Import Git Repository**
4. Select your repository
5. Configure:
   - **GPUs:** L40, L40S, A6000, A40 (48GB+ VRAM)
   - **Workers Min:** 0
   - **Workers Max:** 1
   - **Workers Standby:** 0
   - **Idle Timeout:** 60-120 seconds

### Option 2: Build and Push to Docker Hub

```bash
docker build -t yourusername/jack-vllm-worker:latest .
docker push yourusername/jack-vllm-worker:latest
```

Then use the Docker image in RunPod console.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` | HuggingFace model ID |
| `MAX_MODEL_LEN` | `32768` | Maximum context length |
| `GPU_MEMORY_UTILIZATION` | `0.95` | GPU memory fraction |
| `DTYPE` | `auto` | Data type (auto, float16, bfloat16) |
| `QUANTIZATION` | `awq` | Quantization method (awq, gptq, none) |
| `HF_TOKEN` | - | HuggingFace token for gated models |

## API Usage

### Messages Format (OpenAI-compatible)

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [
        {"role": "system", "content": "You are Jack, a helpful AI assistant."},
        {"role": "user", "content": "Hello!"}
      ],
      "max_tokens": 500,
      "temperature": 0.7
    }
  }'
```

### Simple Prompt Format

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Write a haiku about coding.",
      "max_tokens": 100
    }
  }'
```

## Response Format

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Generated text here..."
    },
    "finish_reason": "stop",
    "index": 0
  }],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 100,
    "total_tokens": 150
  },
  "model": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
}
```

## Cost Settings (CRITICAL)

Always set these to avoid burning money:

```
Workers Standby: 0   ← NO standby workers
Workers Min: 0       ← Scale to zero
Workers Max: 1       ← Only 1 worker unless parallel needed
Idle Timeout: 60     ← Shut down after 60s idle
```

---

*Part of the 008 Assistant project*

