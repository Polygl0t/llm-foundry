"""
Filter and Chunk Long-Context Dataset Samples

This script filters datasets for long-context training samples based on token counts
and quality scores, then chunks them into manageable shards for efficient training.

Filtering Workflow:
    1. Load dataset from JSONL or Parquet files
    2. Apply subset filter if specified (e.g., specific language)
    3. Filter samples by minimum token count
    4. Optionally split by quality score into separate folders
    5. Calculate optimal chunk count based on max_tokens_per_chunk
    6. Save chunks as train-{i:05d}-of-{n:05d}.{extension}
    7. Generate .metadata file with statistics


Output Structure:
    When using --split_by_score:
    output_dir/
    ├── score_1/
    │   ├── train-00000-of-00010.parquet
    │   ├── train-00001-of-00010.parquet
    │   ├── ...
    │   └── .metadata
    ├── score_2/
    │   └── ...
    └── score_3/
        └── ...

Usage:
    # Filter for long samples (>8K tokens) from a specific language
    python filter_long_context_dataset.py \
        --datasets_dir ./raw_data \
        --output_dir ./filtered_data \
        --min_tokens 8192 \
        --subset_filter 'finepdfs_por_Latn' \
        --input_type parquet \
        --output_type parquet \
        --split_by_score \
        --max_tokens_per_chunk 300000000 \
        --num_proc 8
"""
import datasets
import argparse
import glob
import os
import numpy as np


def process_and_save_dataset(dataset, args, output_dir, subset_name=None):
    """Process and save the filtered dataset."""
    sample_count = len(dataset)
    
    # Calculate total token count
    if args.token_count_column in dataset.column_names:
        token_count = sum(dataset[args.token_count_column])
    else:
        token_count = 0
        print(f"Warning: '{args.token_count_column}' column not found. Token count will be 0.")
    
    print(f"Number of samples: {sample_count:,}")
    print(f"Number of tokens: {token_count:,}")
    
    # Calculate number of chunks based on max tokens per chunk
    max_tokens_per_chunk = args.max_tokens_per_chunk
    n_chunks = max(1, int(np.ceil(token_count / max_tokens_per_chunk)))
    print(f"Splitting dataset into {n_chunks} chunks (max {max_tokens_per_chunk:,} tokens per chunk).")
    
    # Split dataset into chunks
    indices = np.array_split(np.arange(sample_count), n_chunks)
    chunks = [dataset.select(idx.tolist()) for idx in indices]
    
    # Calculate actual tokens per chunk
    tokens_per_chunk = token_count // n_chunks if n_chunks > 0 else 0
    print(f"Expecting approximately {tokens_per_chunk:,} tokens per chunk.")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save chunks
    extension = args.output_type if args.output_type == "parquet" else "jsonl"
    save_fn = lambda chunk, path: (
        chunk.to_parquet(path) if args.output_type == "parquet" else chunk.to_json(path)
    )
    
    for i, chunk in enumerate(chunks):
        filename = f"{output_dir}/train-{i:05d}-of-{n_chunks:05d}.{extension}"
        save_fn(chunk, filename)
        print(f"Saved chunk {i+1}/{n_chunks} to: {filename}")
    
    # Save metadata
    with open(f"{output_dir}/.metadata", "w") as meta_file:
        meta_file.write(f"Samples: {sample_count}\n")
        meta_file.write(f"Tokens: {token_count}\n")
        meta_file.write(f"Tokens per chunk: {tokens_per_chunk}\n")
        meta_file.write(f"Chunks: {n_chunks}\n")
        meta_file.write(f"Max tokens per chunk: {max_tokens_per_chunk}\n")
        meta_file.write(f"Minimum token length: {args.min_tokens}\n")
        if subset_name:
            meta_file.write(f"Subset: {subset_name}\n")
        if args.subset_filter:
            meta_file.write(f"Subset filter: {args.subset_filter}\n")
    
    print(f"Saved metadata to: {output_dir}/.metadata")


def main(args):
    # Validate arguments
    assert args.input_type in ["jsonl", "parquet"], "Dataset type must be either 'jsonl' or 'parquet'."
    assert args.output_type in ["jsonl", "parquet"], "Output type must be either 'jsonl' or 'parquet'."
    
    # Load dataset
    data_files = glob.glob(f"{args.datasets_dir}/*.{args.input_type}")
    if not data_files:
        raise ValueError(f"No {args.input_type.upper()} files found in '{args.datasets_dir}'.")
    
    dataset = datasets.load_dataset(
        "json" if args.input_type == "jsonl" else "parquet",
        data_files=data_files,
        split="train",
        cache_dir=args.cache_dir,
        num_proc=len(data_files),
    )
    print(f"Loaded dataset with {len(dataset):,} examples from {args.input_type.upper()} files.\n{dataset}")
    
    # Filter for long context samples
    print(f"\nFiltering dataset...")
    
    # Apply subset filter if specified
    if args.subset_filter and args.subset_column in dataset.column_names:
        print(f"Filtering for subset == '{args.subset_filter}'")
        dataset = dataset.filter(
            lambda x: x[args.subset_column] == args.subset_filter,
            num_proc=args.num_proc if args.num_proc else 1,
            desc=f"Filtering dataset for {args.subset_column} == {args.subset_filter}",
        )
        print(f"After subset filter: {len(dataset):,} examples")
    
    # Apply minimum token count filter
    if args.token_count_column in dataset.column_names:
        print(f"Filtering for {args.token_count_column} > {args.min_tokens}")
        dataset = dataset.filter(
            lambda x: x[args.token_count_column] > args.min_tokens,
            num_proc=args.num_proc or 1,
            desc=f"Filtering dataset for {args.token_count_column} > {args.min_tokens}",
        )
        print(f"After token count filter: {len(dataset):,} examples")
    else:
        print(f"Warning: '{args.token_count_column}' column not found. Skipping token count filter.")
    
    if len(dataset) == 0:
        print("No examples remaining after filtering. Exiting.")
        return
    
    # Process by score or as whole dataset
    if args.split_by_score and args.score_column in dataset.column_names:
        unique_scores = dataset.unique(args.score_column)
        print(f"\nFound {len(unique_scores)} unique scores in '{args.score_column}': {unique_scores}")
        
        for score in unique_scores:
            # Filter by score
            ds = dataset.filter(
                lambda x: x[args.score_column] == score,
                num_proc=args.num_proc or 1,
                desc=f"Filtering dataset for {args.score_column} == {score}",
            )
            
            if len(ds) == 0:
                print(f"\nNo examples found for `{args.score_column}` {score}, skipping...")
                continue
            
            print(f"\nProcessing `{args.score_column}` subset: '{score}' | {len(ds):,} examples")
            
            # Process and save
            output_dir = os.path.join(args.output_dir, str(score))
            process_and_save_dataset(ds, args, output_dir, subset_name=str(score))
    
    else:
        # Process entire dataset
        if args.split_by_score:
            print(f"\nWarning: No `{args.score_column}` column found. Processing the entire dataset without splitting by score.")
        else:
            print(f"\nProcessing the entire dataset...")
        
        process_and_save_dataset(dataset, args, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and save long-context dataset samples")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save the filtered dataset")
    parser.add_argument("--datasets_dir", type=str, required=True, help="Directory containing the datasets to process")
    parser.add_argument("--min_tokens", type=int, required=True, help="Minimum number of tokens required for a sample to be included")
    parser.add_argument("--input_type", choices=["jsonl", "parquet"], default="jsonl", help="Type of the input files")
    parser.add_argument("--output_type", choices=["jsonl", "parquet"], default="parquet", help="Type of the output files")
    parser.add_argument("--cache_dir", type=str, help="Cache directory for datasets")
    parser.add_argument("--subset_column", type=str, default="subset", help="Name of the subset column in the dataset")
    parser.add_argument("--subset_filter", type=str, default=None, help="Value to filter the subset column by (e.g., 'finepdfs_por_Latn')")
    parser.add_argument("--token_count_column", type=str, default="token_count", help="Name of the token count column in the dataset")
    parser.add_argument("--score_column", type=str, default="edu_int_score", help="Name of the score column in the dataset")
    parser.add_argument("--split_by_score", action='store_true', help="Whether to split the output by score into separate folders")
    parser.add_argument("--max_tokens_per_chunk", type=int, default=300_000_000, help="Maximum number of tokens per chunk")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes to use")
    
    args = parser.parse_args()

    print("Filtering long-context dataset samples...")
    main(args)
    print("Filtering complete! 🎆")
