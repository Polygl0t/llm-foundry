"""
Long-Context Text Generation using vLLM

This script generates text continuations using base language models with vLLM for
efficient inference, designed for testing long-context capabilities of extended models.

OneRuler Integration:
    Uses "The Book of Disquiet" by Fernando Pessoa in Portuguese
    from the OneRuler benchmark (https://arxiv.org/abs/2503.01996)
    as a realistic long-context test case (~200k tokens).

Usage:
    python generate_continuation.py \
        --model_name username/extended-model \
        --output_file ./output.txt \
        --max_context_tokens 16384 \
        --max_tokens 512 \
        --temperature 0.8 \
        --top_p 0.9 \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.9
"""
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch

import subprocess
import argparse
import time
import os
import requests

# Constants
TRITON_CACHE_CLEANUP_AGE = 3600  # 1 hour in seconds
VRAM_MB_TO_GB = 1024
# Using the book from the `https://github.com/mungg/OneRuler` repo.
# Learn more about OneRuler here: https://arxiv.org/abs/2503.01996
# Approximatly 200k tokens in Portuguese (The Book of Disquiet, by Fernando Pessoa)
URL = "https://raw.githubusercontent.com/mungg/OneRuler/refs/heads/main/OneRuler/data/books/pt/the_book_of_disquietude_pt.txt"

def setup_triton_cache():
    """Setup Triton cache directory with proper permissions and cleanup"""
    cache_dir = os.environ.get('TRITON_CACHE_DIR', './.cache/triton_cache')
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
    cuda_visible_device = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    rank_cache_dir = f"{cache_dir}/{slurm_job_id}/rank_{cuda_visible_device}"
    
    print(f"[CACHE] {rank_cache_dir}")
    os.makedirs(rank_cache_dir, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = rank_cache_dir
    
    # Clean up stale cache files
    cleanup_stale_cache_files(rank_cache_dir)

def cleanup_stale_cache_files(cache_dir):
    """Remove cache files older than specified age"""
    try:
        current_time = time.time()
        for root, _, files in os.walk(cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if os.path.getmtime(file_path) < current_time - TRITON_CACHE_CLEANUP_AGE:
                        os.remove(file_path)
                except (OSError, IOError):
                    pass  # Ignore errors when cleaning up
    except Exception:
        pass

def load_model_and_tokenizer(
    model_name, 
    cache_dir, 
    tensor_parallel_size, 
    gpu_memory_utilization
):
    """Load the model and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=True,
        cache_dir=cache_dir,
    )

    model = LLM(
        model=model_name,
        dtype=torch.float16 if "AWQ" in model_name else torch.bfloat16,
        download_dir=cache_dir,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    return tokenizer, model

def get_nvidia_smi_vram():
    """Get the current VRAM usage of NVIDIA GPUs in GB."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        vram_list = result.decode("utf-8").strip().split("\n")
        return [float(v) / VRAM_MB_TO_GB for v in vram_list]
    except Exception:
        return [0.0]  # Return 0 instead of error string

def load_context_from_url(url):
    """Load text content from a URL to use as context."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch content from URL: {e}")

def truncate_context_to_tokens(
    context, 
    tokenizer, 
    max_tokens
):
    """Truncate context to a maximum number of tokens."""
    tokens = tokenizer.encode(context)
    if len(tokens) <= max_tokens:
        return context
    
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    print(f"[INFO] Context truncated from {len(tokens)} to {max_tokens} tokens")
    return truncated_text

def generate_continuation(
    model, 
    tokenizer, 
    context, 
    sampling_params
):
    """Generate text continuation using the model."""
    t0 = time.time()
    outputs = model.generate([context], sampling_params, use_tqdm=False)
    elapsed_time = time.time() - t0
    
    nvidia_smi_vram = get_nvidia_smi_vram()[0]
    generated_text = outputs[0].outputs[0].text
    tokens_generated = len(tokenizer(generated_text).input_ids)
    
    print(f"[STATS] Time taken: {elapsed_time:.2f}s | VRAM: {nvidia_smi_vram:.2f} GB | Tokens generated: {tokens_generated}")

    # Return context and generation
    return f"[CONTEXT START]\n\n{context}\n\n[CONTEXT END]\n\n[GENERATED TEXT START]\n\n{generated_text}\n\n[GENERATED TEXT END]"


def save_generated_text(text, output_path):
    """Save the generated text to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[SAVED] Output written to {output_path}")

def main(args):
    """Main execution function"""
    # Setup
    setup_triton_cache()

    # Load model and tokenizer
    print(f"[INFO] Loading model: {args.model_name}")
    tokenizer, model = load_model_and_tokenizer(
        args.model_name, 
        args.cache_dir, 
        args.tensor_parallel_size, 
        args.gpu_memory_utilization
    )

    # Load context from URL
    print(f"[INFO] Loading context from: {URL}")
    context = load_context_from_url(URL)
    context_tokens = len(tokenizer(context).input_ids)
    print(f"[INFO] Context loaded: {context_tokens} tokens")
    
    # Truncate context if max_context_tokens is specified
    if args.max_context_tokens and context_tokens > args.max_context_tokens:
        context = truncate_context_to_tokens(context, tokenizer, args.max_context_tokens)
        context_tokens = len(tokenizer(context).input_ids)
        print(f"[INFO] Using truncated context: {context_tokens} tokens")

    # Define sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k,
        top_p=args.top_p
    )

    # Generate continuation
    print(f"[INFO] Generating {args.max_tokens} tokens...")
    full_text = generate_continuation(
        model=model, 
        tokenizer=tokenizer,
        context=context,
        sampling_params=sampling_params,
    )

    # Save output
    save_generated_text(full_text, args.output_file)
    
    total_tokens = len(tokenizer(full_text).input_ids)
    print(f"[INFO] Total tokens in output: {total_tokens}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text continuations using base language models.")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face base model name.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for model loading.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for model loading.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated text.")
    parser.add_argument("--max_context_tokens", type=int, default=None, help="Maximum number of tokens to use from context (truncates if exceeded).")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty.")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory to cache the model and tokenizer.")
    
    args = parser.parse_args()

    print("Starting text generation! 🚀")
    main(args)
    print("Text generation completed successfully! 🎉")
