"""
RunPod Serverless Handler for vLLM
Supports OpenAI-compatible chat completions format
"""
import os
import runpod
from vllm import LLM, SamplingParams
from vllm.entrypoints.chat_utils import apply_chat_template

# Configuration from environment
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32768"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.95"))
DTYPE = os.environ.get("DTYPE", "auto")
QUANTIZATION = os.environ.get("QUANTIZATION", "awq")

print(f"Loading model: {MODEL_NAME}")
print(f"Max model length: {MAX_MODEL_LEN}")
print(f"GPU memory utilization: {GPU_MEMORY_UTILIZATION}")

# Initialize vLLM engine
llm = LLM(
    model=MODEL_NAME,
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    dtype=DTYPE,
    quantization=QUANTIZATION if QUANTIZATION != "none" else None,
    trust_remote_code=True,
)

# Get tokenizer for chat template
tokenizer = llm.get_tokenizer()


def handler(job):
    """
    RunPod handler for vLLM inference.
    
    Supports two input formats:
    1. messages format (OpenAI-compatible):
       {"messages": [{"role": "user", "content": "..."}], "max_tokens": 100, ...}
    
    2. prompt format (simple):
       {"prompt": "...", "max_tokens": 100, ...}
    """
    job_input = job.get("input", {})
    
    # Extract parameters
    messages = job_input.get("messages")
    prompt = job_input.get("prompt")
    
    # Sampling parameters (with defaults)
    sampling_params_input = job_input.get("sampling_params", {})
    max_tokens = job_input.get("max_tokens") or sampling_params_input.get("max_tokens", 2048)
    temperature = job_input.get("temperature") or sampling_params_input.get("temperature", 0.7)
    top_p = job_input.get("top_p") or sampling_params_input.get("top_p", 0.95)
    top_k = job_input.get("top_k") or sampling_params_input.get("top_k", -1)
    stop = job_input.get("stop") or sampling_params_input.get("stop")
    
    # Build the prompt
    if messages:
        # Apply chat template for messages format
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    elif prompt:
        prompt_text = prompt
    else:
        return {"error": "No 'messages' or 'prompt' provided in input"}
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop=stop if stop else None,
    )
    
    # Generate
    try:
        outputs = llm.generate([prompt_text], sampling_params)
        output = outputs[0]
        
        generated_text = output.outputs[0].text
        
        # Return in OpenAI-compatible format
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": output.outputs[0].finish_reason,
                "index": 0
            }],
            "usage": {
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
                "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
            },
            "model": MODEL_NAME
        }
    except Exception as e:
        return {"error": str(e)}


# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})
