"""
General-Purpose Synthetic Text Generation Pipeline with vLLM

This script generates synthetic text samples using Hugging Face language models and vLLM for
high-throughput inference.

Example usage:
    python generate.py \
        --model_name_or_path Qwen/Qwen3-0.6B \
        --dataset_path data/seed_texts.jsonl \
        --text_column text \
        --system "Summarize the following text." \
        --output_dir outputs/synthetic \
        --output_file generated.jsonl \
        --max_length 512
"""
from vllm import SamplingParams
import argparse
import os

from utils import (
    DatasetLoader,
    get_logger,
    run_rollouts,
    load_model_and_tokenizer,
    setup_triton_cache,
    get_starting_row,
)


def main(args):

    # Get a logger for our script.
    logger = get_logger("SyntheticGenerator")
    logger.info("Starting synthesis!")

    # Setup the Triton cache.
    setup_triton_cache()

    # Load model and tokenizer.
    tokenizer, model = load_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path, 
        cache_dir=args.cache_dir, 
        tensor_parallel_size=args.tensor_parallel_size, 
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    # Define sampling parameters.
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

    # Setup output directory and file.
    os.makedirs(args.output_dir, exist_ok=True)
    file_path = os.path.join(args.output_dir, args.output_file)

    # Determine the starting row.
    row_start = get_starting_row(file_path, args.row_start)

    # Initialize output file if needed.
    if not os.path.exists(file_path):
        open(file_path, "w").close()

    logger.info("Starting synthesis process...")
    logger.info(f"Generator: {args.model_name_or_path}")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Starting from row: {row_start}")

    # Load dataset
    dataset = DatasetLoader(
        path=args.dataset_path,
        cache_dir=args.cache_dir,
        seed=args.seed,
        split=args.dataset_split,
        subset=args.dataset_subset,
    ).load()
    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    # Process each sample
    for counter, sample in enumerate(dataset):
        if counter < row_start:
            continue
        
        run_rollouts(
            sample,
            counter,
            text_column=args.text_column,
            metadata_columns=args.metadata_columns,
            model=model,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            file_path=file_path,
            system=args.system,
            prompt_prefix=args.prompt_prefix,
            prompt_suffix=args.prompt_suffix,
            max_chunk_size=args.max_chunk_size,
            chunk_once=args.chunk_once,
            track_vram=args.track_vram,
            enable_thinking=args.enable_thinking,
        )
        
    logger.info("Synthesis completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic samples using a Hugging Face models and vLLM.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Hugging Face model name or path.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for model loading.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for model loading.")
    parser.add_argument("--track_vram", action="store_true", help="Whether to track VRAM usage during generation.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--dataset_subset", type=str, default=None, help="Subset of the dataset to use.")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. If set to an integer, the dataset will be shuffled.")
    parser.add_argument("--text_column", type=str, required=True, help="Column in the dataset containing the seed text.")
    parser.add_argument("--metadata_columns", type=str, nargs="*", default=[], help="Additional dataset columns to include in the metadata field of each output record.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated samples.")
    parser.add_argument("--output_file", type=str, default="output.jsonl", help="Output file name.")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum length of generated text.")
    parser.add_argument("--max_chunk_size", type=int, default=8192, help="Maximum chunk size (in tokens) for the model.")
    parser.add_argument("--chunk_once", action="store_true", help="Chunk the text and only use the first chunk.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p sampling.")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to return.")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty.")
    parser.add_argument("--cache_dir", type=str, default="./.cache", help="Directory to cache the model and tokenizer.")
    parser.add_argument("--system", type=str, default="", help="System message to prepend to the input.")
    parser.add_argument("--prompt_prefix", type=str, default="", help="Prompt to prepend to the input.")
    parser.add_argument("--prompt_suffix", type=str, default="", help="Prompt to append to the input.")
    parser.add_argument("--row_start", type=int, default=None, help="Row index to start generating samples.")
    parser.add_argument("--enable_thinking", action="store_true", help="Whether to enable thinking mode during generation.")
    args = parser.parse_args()

    main(args)
