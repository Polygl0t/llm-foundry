"""
General-Purpose Synthetic Text Generation Pipeline with vLLM

This script generates synthetic text samples using Hugging Face language models and vLLM for
high-throughput inference.

Example usage:
    python generate_base.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --dataset_path data/seed_texts.jsonl \
        --column_name text \
        --system "You are a helpful assistant." \
        --output_dir outputs/synthetic \
        --output_file generated.jsonl \
        --max_length 512 \
        --num_return_sequences 3 \
        --temperature 0.8 \
        --tensor_parallel_size 2
"""
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import datasets
import random
import torch

import subprocess
import argparse
import glob
import time
import json
import os

TRITON_CACHE_CLEANUP_AGE = 3600
VRAM_MB_TO_GB = 1024

def setup_triton_cache():
    """Setup Triton cache directory with proper permissions and cleanup of stale files."""
    cache_dir = os.environ.get('TRITON_CACHE_DIR', './.cache/triton_cache')
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
    cuda_visible_device = os.environ.get('CUDA_VISIBLE_DEVICES', '0').replace(',', '-')
    rank_cache_dir = f"{cache_dir}/{slurm_job_id}/rank_{cuda_visible_device}"
    
    print(rank_cache_dir)
    os.makedirs(rank_cache_dir, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = rank_cache_dir
    
    # Clean up stale cache files
    try:
        current_time = time.time()
        for root, _, files in os.walk(rank_cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if os.path.getmtime(file_path) < current_time - TRITON_CACHE_CLEANUP_AGE:
                        os.remove(file_path)
                except (OSError, IOError):
                    pass  # Ignore errors when cleaning up
    except Exception:
        pass

def load_model_and_tokenizer(model_name, cache_dir, tensor_parallel_size, gpu_memory_utilization):
    """Load the model and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=True,
        cache_dir=cache_dir,
    )

    # [`vllm.LLM`](https://docs.vllm.ai/en/latest/api/vllm/#vllm.LLM)
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

def generate_samples(model, tokenizer, input_string, system, sampling_params):
    """Generate text samples using the model."""

    # [`apply_chat_template`](https://huggingface.co/docs/transformers/main/chat_templating#using-applychattemplate)
    raw_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": input_string}
        ],
        tokenize=False,
        add_generation_prompt=True
    )

    t0 = time.time()
    outputs = model.generate([raw_text], sampling_params, use_tqdm=False)
    elapsed_time = time.time() - t0
    
    nvidia_smi_vram = get_nvidia_smi_vram()[0]
    tokens_generated = len(tokenizer(outputs[0].outputs[0].text).input_ids)
    
    print(f"[STATS] Time taken: {elapsed_time:.2f}s | VRAM: {nvidia_smi_vram:.2f} GB | Tokens: {tokens_generated}")

    return [output.outputs[0].text for output in outputs]

def save_samples(samples, output_file, file_prefix):
    """Save generated samples to a file."""
    with open(output_file, "a", encoding="utf-8") as f:
        for idx, sample in enumerate(samples):
            json_line = json.dumps(
                {
                    "idx": file_prefix if len(samples) == 1 else f"{file_prefix}_{idx+1}", 
                    "text": sample
                }
            )
            f.write(json_line + "\n")

def load_dataset_from_directory(dataset_path, cache_dir, seed):
    """Load dataset from a directory of JSONL or Parquet files"""
    dataset_files = glob.glob(os.path.join(dataset_path, "*.jsonl"))
    dataset_type = "json"
    
    if not dataset_files:
        dataset_files = glob.glob(os.path.join(dataset_path, "*.parquet"))
        dataset_type = "parquet"
        if not dataset_files:
            raise ValueError(f"No JSONL or Parquet files found in {dataset_path}")

    dataset = datasets.load_dataset(
        dataset_type,
        data_files=dataset_files,
        split='train',
        num_proc=len(dataset_files),
        cache_dir=cache_dir,
    )

    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    
    return dataset

def load_dataset_from_jsonl(dataset_path, seed):
    """Load dataset from a single JSONL file"""
    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            try:
                dataset.append(json.loads(line))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    
    if seed is not None:
        random.seed(seed)
        random.shuffle(dataset)
    
    return dataset

def load_dataset_from_hf(dataset_path, dataset_split, dataset_subset, cache_dir, seed):
    """Load dataset from Hugging Face"""
    load_args = {
        "path": dataset_path,
        "split": dataset_split,
        "cache_dir": cache_dir,
    }

    if dataset_subset is not None:
        load_args["name"] = dataset_subset

    dataset = datasets.load_dataset(**load_args)

    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    
    return dataset

def load_dataset(args):
    """Load dataset based on the provided path"""
    if os.path.isdir(args.dataset_path):
        return load_dataset_from_directory(args.dataset_path, args.cache_dir, args.seed)
    elif args.dataset_path.endswith(".jsonl"):
        return load_dataset_from_jsonl(args.dataset_path, args.seed)
    else:
        return load_dataset_from_hf(
            args.dataset_path, 
            args.dataset_split, 
            args.dataset_subset, 
            args.cache_dir, 
            args.seed
        )

def get_starting_row(file_path, row_start):
    """Determine the starting row for processing"""
    if row_start is not None:
        return row_start
    
    if not os.path.exists(file_path):
        return 0
    
    max_idx = 0
    with open(file_path, "r") as f:
        for line in f:
            try:
                json_object = json.loads(line)
                idx_value = int(json_object['idx'].split("_")[1])
                max_idx = max(max_idx, idx_value)
            except (json.JSONDecodeError, KeyError, ValueError, IndexError):
                continue
    
    return max_idx + 1

def chunk_text(text, tokenizer, max_chunk_size, chunk_once):
    """Chunk text into smaller pieces if it exceeds max_chunk_size"""
    tokenized_text = tokenizer(text).input_ids
    chunks = [
        tokenized_text[i:i + max_chunk_size] 
        for i in range(0, len(tokenized_text), max_chunk_size)
    ]
    decoded_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
    
    return [decoded_chunks[0]] if chunk_once else decoded_chunks

def process_sample(sample, counter, args, model, tokenizer, sampling_params, file_path):
    """Process a single sample from the dataset"""
    text_content = sample[args.column_name]
    token_count = len(tokenizer(text_content).input_ids)

    # Handle chunking if necessary
    if token_count > args.max_chunk_size:
        print(f"[CHUNKING] Row {counter} ({token_count} tokens) into {args.max_chunk_size}-token chunks...")
        text_samples = chunk_text(text_content, tokenizer, args.max_chunk_size, args.chunk_once)
    else:
        text_samples = [text_content]

    # Process each chunk
    for i, text in enumerate(text_samples):
        full_prompt = f"{args.prompt_prefix}{text}{args.prompt_suffix}"
        
        print(f"[GENERATING] Samples for row {counter}. Chunk {i + 1}/{len(text_samples)}...")
        
        generated_samples = generate_samples(
            model=model, 
            tokenizer=tokenizer,
            input_string=full_prompt,
            system=args.system,
            sampling_params=sampling_params,
        )

        file_prefix = f"row_{counter}" if len(text_samples) == 1 else f"row_{counter}_chunk_{i}"
        save_samples(generated_samples, file_path, file_prefix)

def main(args):
    """Main execution function"""
    # Setup
    setup_triton_cache()

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(
        args.model_name, 
        args.cache_dir, 
        args.tensor_parallel_size, 
        args.gpu_memory_utilization
    )

    # Define sampling parameters
    # [`SamplingParams`](https://nm-vllm.readthedocs.io/en/latest/dev/sampling_params.html)
    sampling_params = SamplingParams(
        max_tokens=args.max_length,
        stop=[tokenizer.eos_token],
        stop_token_ids=[tokenizer.eos_token_id],
        n=args.num_return_sequences,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k,
        top_p=args.top_p
    )

    # Setup output
    os.makedirs(args.output_dir, exist_ok=True)
    file_path = os.path.join(args.output_dir, args.output_file)

    # Determine starting row
    row_start = get_starting_row(file_path, args.row_start)

    # Initialize output file if needed
    if not os.path.exists(file_path):
        open(file_path, "w").close()

    print("[INFO] Starting synthesis process...")
    print(f"[INFO] Generator: {args.model_name}")
    print(f"[INFO] Dataset: {args.dataset_path}")
    print(f"[INFO] Starting from row: {row_start}")

    # Load dataset
    dataset = load_dataset(args)
    print(f"[INFO] Loaded dataset with {len(dataset)} samples.")

    # Process each sample
    for counter, sample in enumerate(dataset):
        if counter < row_start:
            continue
        
        process_sample(sample, counter, args, model, tokenizer, sampling_params, file_path)
        
    print("[INFO] Iteration completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic samples using a Hugging Face models and vLLM.")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for model loading.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for model loading.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--dataset_subset", type=str, default=None, help="Subset of the dataset to use.")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. If set to an integer, the dataset will be shuffled.")
    parser.add_argument("--column_name", type=str, required=True, help="Column in the dataset where the seed text is located.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated samples.")
    parser.add_argument("--output_file", type=str, default="output.jsonl", help="Output file name.")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum length of generated text.")
    parser.add_argument("--max_chunk_size", type=int, default=5000, help="Maximum chunk size (in tokens) for the model.")
    parser.add_argument("--chunk_once", action="store_true", help="Chunk the text and only use the first chunk.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling.")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to return.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty.")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory to cache the model and tokenizer.")
    parser.add_argument("--system", type=str, default="", help="System message to prepend to the input.")
    parser.add_argument("--prompt_prefix", type=str, default="", help="Prompt to prepend to the input.")
    parser.add_argument("--prompt_suffix", type=str, default="", help="Prompt to append to the input.")
    parser.add_argument("--row_start", type=int, default=None, help="Row index to start generating samples.")
    
    args = parser.parse_args()

    print("Starting synthesis! 🚀")
    main(args)
    print("Synthesis completed successfully! 🎉")
