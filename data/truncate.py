"""
Dataset Truncation by Token Count

Creates a subset of a large dataset by selecting samples up to a target token limit.
Useful for creating smaller training datasets from larger corpora.

Requirements:
- Input dataset must have a token count column (default: "token_count")

Output structure:
- train-00000-of-NNNNN.{jsonl|parquet}: Chunked dataset files
- .metadata: Dataset statistics (sample count, tokens, chunks, columns)

Usage:
    # Create 4B token subset from JSONL files
    python truncate.py --input_dir data/full --output_dir data/4B \
        --target_tokens 4000000000 --input_type jsonl
    
    # Create 1B token subset with custom chunk size
    python truncate.py --input_dir data/parquet --output_dir data/1B \
        --target_tokens 1000000000 --input_type parquet --output_type parquet \
        --max_tokens_per_chunk 500000000
"""
import datasets
import numpy as np
import argparse
import glob
import os


def load_dataset_with_token_count(input_dir, input_type="jsonl", cache_dir=None):
    """Load dataset from JSONL or Parquet files."""
    data_files = glob.glob(f"{input_dir}/*.{input_type}")
    if not data_files:
        raise ValueError(f"No {input_type.upper()} files found in '{input_dir}'.")
    
    dataset = datasets.load_dataset(
        "json" if input_type == "jsonl" else "parquet",
        data_files=data_files,
        split="train",
        cache_dir=cache_dir,
        num_proc=len(data_files),
    )
    
    return dataset


def create_subset_with_token_limit(dataset, target_tokens, token_column="token_count"):
    """Create a subset of the dataset with up to target_tokens tokens."""
    accumulated_tokens = 0
    selected_indices = []
    
    print(f"[INFO] Selecting samples to reach {target_tokens:,} tokens...")
    
    for idx, example in enumerate(dataset):
        token_count = example[token_column]
        
        # Check if adding this sample would exceed the target
        if accumulated_tokens + token_count > target_tokens:
            print(f"[INFO] Stopping at index {idx} to avoid exceeding target.")
            break
        
        accumulated_tokens += token_count
        selected_indices.append(idx)
        
        # Progress update every 10k samples
        if (idx + 1) % 10000 == 0:
            print(f"[PROGRESS] Processed {idx + 1:,} samples | Accumulated {accumulated_tokens:,} tokens")
    
    print(f"\n[INFO] Selected {len(selected_indices):,} samples with {accumulated_tokens:,} tokens")
    
    # Create subset
    subset = dataset.select(selected_indices)
    return subset, accumulated_tokens


def save_dataset_in_chunks(dataset, output_dir, total_tokens, output_type="jsonl", max_tokens_per_chunk=300_000_000):
    """Save dataset in chunks based on token limit per chunk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate number of chunks
    n_chunks = max(1, (total_tokens + max_tokens_per_chunk - 1) // max_tokens_per_chunk)
    print(f"[INFO] Splitting dataset into {n_chunks} chunks (~{max_tokens_per_chunk:,} tokens per chunk)")
    
    sample_count = len(dataset)
    
    # Split into chunks
    indices = np.array_split(np.arange(sample_count), n_chunks)
    chunks = [dataset.select(idx) for idx in indices]
    
    tokens_per_chunk = total_tokens // n_chunks
    print(f"[INFO] Expected ~{tokens_per_chunk:,} tokens per chunk.")
    
    # Save chunks
    extension = output_type if output_type == "parquet" else "jsonl"
    save_fn = lambda chunk, path: (
        chunk.to_parquet(path) if output_type == "parquet" else chunk.to_json(path)
    )
    
    for i, chunk in enumerate(chunks):
        filename = f"{output_dir}/train-{i:05d}-of-{n_chunks:05d}.{extension}"
        print(f"[INFO] Saving chunk {i+1}/{n_chunks}: {filename}")
        save_fn(chunk, filename)
    
    # Save metadata
    metadata_path = f"{output_dir}/.metadata"
    with open(metadata_path, "w") as meta_file:
        meta_file.write(f"Samples: {sample_count}\n")
        meta_file.write(f"Tokens: {total_tokens}\n")
        meta_file.write(f"Chunks: {n_chunks}\n")
        meta_file.write(f"Columns: {dataset.column_names}\n")
    
    print(f"[INFO] Metadata saved to {metadata_path}")

def main(args):
    # Validate arguments
    assert args.input_type in ["jsonl", "parquet"], "Input type must be either 'jsonl' or 'parquet'."
    assert args.output_type in ["jsonl", "parquet"], "Output type must be either 'jsonl' or 'parquet'."
    
    # Load dataset
    print(f"[INFO] Loading dataset from {args.input_dir}")
    dataset = load_dataset_with_token_count(args.input_dir, args.input_type, args.cache_dir)
    print(f"[INFO] Loaded dataset with {len(dataset):,} examples\n{dataset}")
    
    # Verify token_column exists
    if args.token_column not in dataset.column_names:
        raise ValueError(f"Column '{args.token_column}' not found in dataset. Available columns: {dataset.column_names}")
    
    # Create subset with target tokens
    subset, total_tokens = create_subset_with_token_limit(
        dataset,
        args.target_tokens,
        args.token_column
    )
    
    # Save dataset in chunks
    save_dataset_in_chunks(
        subset,
        args.output_dir,
        total_tokens,
        args.output_type,
        args.max_tokens_per_chunk
    )
    
    print(f"\n[SUCCESS] Dataset subset saved to {args.output_dir}")
    print(f"[SUCCESS] Total samples: {len(subset):,}")
    print(f"[SUCCESS] Total tokens: {total_tokens:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a subset of a dataset with a specific token limit.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing the dataset files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save the subset")
    parser.add_argument("--target_tokens", type=int, required=True, help="Target number of tokens for the subset (e.g., 4000000000 for 4B)")
    parser.add_argument("--token_column", type=str, default="token_count", help="Name of the column containing token counts")
    parser.add_argument("--input_type", choices=["jsonl", "parquet"], default="jsonl", help="Type of the input files")
    parser.add_argument("--output_type", choices=["jsonl", "parquet"], default="jsonl", help="Type of the output files")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for datasets")
    parser.add_argument("--max_tokens_per_chunk", type=int, default=300_000_000, help="Maximum tokens per chunk (default: 300M)")
    
    args = parser.parse_args()
    
    print(f"Creating a subset of the dataset with a token limit of {args.target_tokens:,} tokens... 🚀")
    main(args)
    print("Subset creation complete! 🎆")
