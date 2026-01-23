"""
Dataset Shuffling and Rechunking Tool

This script loads pre-tokenized datasets, shuffles them thoroughly, and re-splits them into
a specified number of chunks. Useful for mixing multiple dataset sources, randomizing training
order, and controlling dataset size before training.

Workflow:
1. Load all files from specified directories
2. Shuffle file order and dataset samples
3. Optionally truncate to max_tokens
4. Split into n_chunks (or same as input file count)
5. Save with updated metadata

Example usage:
    python shuffle.py \
        --dataset_dir data/tokenized/source1 data/tokenized/source2 \
        --dataset_type parquet \
        --output_dir data/shuffled_mixed \
        --max_tokens 5000000000 \
        --tokens_per_chunk 300000000 \
        --seed 42
"""
import numpy as np
import datasets
import argparse
import glob
import os

def read_metadata(metadata_path):
    """Read metadata file and return a dictionary of key-value pairs."""
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()
    return metadata


def main(args):
    
    assert args.dataset_type in ["jsonl", "parquet"], f"Dataset type must be either 'jsonl' or 'parquet', got {args.dataset_type}."

    dataset_files = []
    dataset_dir = args.dataset_dir
    if isinstance(dataset_dir, str):
        dataset_dir = [dataset_dir]

    # Below, we loop over all training directories and collect the dataset files that
    # have the correct file extension.
    for train_dir in dataset_dir:
        if os.path.isfile(train_dir) and train_dir.endswith(f".{args.dataset_type}"):
            dataset_files.append(train_dir)
        elif os.path.isdir(train_dir):
            dataset_files += glob.glob(f"{train_dir}/*.{args.dataset_type}")

    # Try to read tokenizer name and block_size from source metadata files
    tokenizer_name = ""
    block_size = None
    for train_dir in dataset_dir:
        if os.path.isdir(train_dir):
            metadata_path = os.path.join(train_dir, ".metadata")
        elif os.path.isfile(train_dir):
            # If it's a file, check the parent directory for metadata
            parent_dir = os.path.dirname(train_dir)
            metadata_path = os.path.join(parent_dir, ".metadata")
        else:
            continue
        
        metadata = read_metadata(metadata_path)
        if not tokenizer_name and metadata.get("Tokenizer"):
            tokenizer_name = metadata["Tokenizer"]
            print(f"Found tokenizer name in metadata: {tokenizer_name}")
        if block_size is None and metadata.get("Block size"):
            block_size = int(metadata["Block size"])
            print(f"Found block size in metadata: {block_size}")
        if tokenizer_name and block_size is not None:
            break

    if block_size is None:
        raise ValueError("Block size not found in any metadata file. Please ensure source directories have a .metadata file with 'Block size' specified.")

    np.random.seed(args.seed)
    np.random.shuffle(dataset_files)

    # Load the datasets from disk
    # [datasets.load_dataset](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset)
    dataset = datasets.load_dataset(
        "json" if args.dataset_type == "jsonl" else args.dataset_type,
        data_files=dataset_files,
        split='train',
        num_proc=len(dataset_files),
        cache_dir=args.cache_dir,
    )
    dataset = dataset.shuffle(seed=args.seed)

    print(f"Loaded {len(dataset):,} samples from {len(dataset_files)} files.\n{dataset}")
    total_sample_count = len(dataset)
    total_token_count = total_sample_count * block_size

    # Limit the amount of tokens if required
    if args.max_tokens is not None:
        if total_token_count > args.max_tokens:
            max_rows = args.max_tokens // block_size
            print(f"Subset has more than {args.max_tokens} tokens. Truncating to {max_rows:,} rows (~{args.max_tokens} tokens).")
            dataset = dataset.select(range(max_rows))
            total_sample_count = len(dataset)
            total_token_count = total_sample_count * block_size
        
    print(f"Number of samples: {total_sample_count:,}")
    print(f"Number of tokens: {total_token_count:,}")

    # Calculate number of chunks based on tokens_per_chunk limit
    n_chunks = max(1, (total_token_count + args.tokens_per_chunk - 1) // args.tokens_per_chunk)  # Ceiling division
    print(f"Splitting dataset into {n_chunks} chunks (max {args.tokens_per_chunk:,} tokens per chunk).")
    
    indices = np.array_split(np.arange(len(dataset)), n_chunks)
    chunks = [dataset.select(idx) for idx in indices]

    tokens_per_chunk = total_token_count // n_chunks
    print(f"Expecting {tokens_per_chunk:,} tokens per chunk.")
        
    os.makedirs(args.output_dir, exist_ok=True)
    if args.dataset_type == "parquet":
        for i, chunk in enumerate(chunks):
            chunk.to_parquet(f"{args.output_dir}/train-{i:05d}-of-{n_chunks:05d}.parquet")
    else:
        for i, chunk in enumerate(chunks):
            chunk.to_json(f"{args.output_dir}/train-{i:05d}-of-{n_chunks:05d}.jsonl")

    # Save a .metadata file with dataset statistics
    with open(f"{args.output_dir}/.metadata", "w") as meta_file:
        meta_file.write(f"Samples: {total_sample_count}\n")
        meta_file.write(f"Tokens: {total_token_count}\n")
        meta_file.write(f"Tokens per chunk: {tokens_per_chunk}\n")
        meta_file.write(f"Block size: {block_size}\n")
        meta_file.write(f"Chunks: {n_chunks}\n")
        meta_file.write(f"Tokenizer: {tokenizer_name}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle a dataset.")
    parser.add_argument("--dataset_dir", type=str, nargs='+', required=True, help="Path(s) to the dataset directory or file.")
    parser.add_argument("--dataset_type", type=str, choices=["jsonl", "parquet"], required=True, help="Type of the dataset files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output chunks.")
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum number of tokens to include in the dataset.")
    parser.add_argument("--tokens_per_chunk", type=int, default=300_000_000, help="Maximum number of tokens per chunk.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for `datasets`.")
    args = parser.parse_args()

    print("Starting dataset shuffling ...")
    main(args)
    print("Dataset shuffling completed. 🎉")