"""
Validation Split Creation Tool for Tokenized Datasets

This script creates validation splits by extracting a specified number of samples from multiple
training data files and consolidating them into a separate validation file. Useful for preparing
train/validation splits from pre-tokenized datasets.

Output:
- Single validation file with extracted samples
- Updated source files with remaining training samples
- .metadata file containing validation split statistics

Example usage:
    python make_validation_split.py \
        --input_dir data/train_chunks \
        --output_dir data/validation \
        --input_type parquet \
        --output_file validation_split \
        --n_samples 20000 \
        --n_files 10
"""
import os
import datasets
import math
import argparse
import random
import glob


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


def get_files_from_folder(input_dir, input_type, n_files=None):
    """Get list of files from a folder, optionally randomly selecting n_files."""
    extension = ".parquet" if input_type == "parquet" else ".jsonl"
    pattern = os.path.join(input_dir, f"*{extension}")
    all_files = sorted(glob.glob(pattern))
    
    if not all_files:
        raise FileNotFoundError(f"No {extension} files found in {input_dir}")
    
    if n_files is not None and n_files < len(all_files):
        files = random.sample(all_files, n_files)
    else:
        files = all_files
    
    return sorted(files)


def main(input_dir, output_dir, input_type, output_file, n_samples, n_files=None):
    """
    Removes n_samples rows (evenly as possible) from randomly selected files in `input_dir`,
    saves the removed rows to a single file, and overwrites the source files
    with the remaining rows.
    """
    # Get files from folder
    files = get_files_from_folder(input_dir, input_type, n_files)
    print(f"Selected {len(files)} files for sampling")
    
    # Read tokenizer name from source folder metadata
    source_metadata_path = os.path.join(input_dir, ".metadata")
    source_metadata = read_metadata(source_metadata_path)
    tokenizer_name = source_metadata.get("Tokenizer", "")
    
    # Read existing metadata from output dir if it exists
    output_metadata_path = os.path.join(output_dir, ".metadata")
    existing_metadata = read_metadata(output_metadata_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load all datasets using datasets.load_dataset
    datasets_list = [
        datasets.load_dataset(input_type, data_files=f, split="train") for f in files
    ]
    lengths = [len(ds) for ds in datasets_list]
    total_rows = sum(lengths)

    if n_samples > total_rows:
        raise ValueError("n_samples is greater than total number of rows in all files.")

    # Compute how many samples to remove from each file (as even as possible)
    samples_to_remove = []
    remaining = n_samples
    for i, l in enumerate(lengths):
        # For the last file, take all remaining samples needed (but not more than available)
        if i == len(lengths) - 1:
            take = min(remaining, l)
        else:
            # Proportionally assign samples to remove, but not more than available or needed
            take = min(math.floor(n_samples * l / total_rows), l, remaining)
        samples_to_remove.append(take)
        remaining -= take

    # Remove samples and collect them for validation split
    removed_tables = []
    for i, (ds, n_remove, path) in enumerate(zip(datasets_list, samples_to_remove, files)):
        if n_remove == 0:
            # If nothing to remove, just save the dataset back to its original file
            if input_type == "parquet":
                ds.to_parquet(path)
            else:
                ds.to_json(path)
            continue
        # Select the first n_remove rows for validation
        removed = ds.select(range(n_remove))
        # Keep the rest for training
        kept = ds.select(range(n_remove, len(ds)))
        # Overwrite the source file with the remaining rows
        if input_type == "parquet":
            kept.to_parquet(path)
        else:
            kept.to_json(path)
        # Collect removed rows for validation split
        removed_tables.append(removed)

    # Get the block size from the first dataset. Calculate the length of the first entry in 'input_ids'
    block_size = len(datasets_list[0][0]['input_ids']) if 'input_ids' in datasets_list[0].features else 0
    # Concatenate all removed rows and save to a single file
    if removed_tables:
        concat = datasets.concatenate_datasets(removed_tables)
        print(concat)
        sample_count = len(concat)
        token_count = sample_count * block_size
        print(f"Number of samples: {sample_count:,}")
        print(f"Number of tokens: {token_count:,}")
        if input_type == "parquet":
            concat.to_parquet(os.path.join(output_dir, output_file + ".parquet"))
        else:
            concat.to_json(os.path.join(output_dir, output_file + ".jsonl"))

        # Combine with existing metadata if present
        if existing_metadata:
            prev_samples = int(existing_metadata.get("Samples", 0))
            prev_tokens = int(existing_metadata.get("Tokens", 0))
            prev_chunks = int(existing_metadata.get("Chunks", 0))
            total_samples = prev_samples + sample_count
            total_tokens = prev_tokens + token_count
            total_chunks = prev_chunks + 1
        else:
            total_samples = sample_count
            total_tokens = token_count
            total_chunks = 1

        # Write metadata about the validation split
        with open(output_metadata_path, "w") as meta_file:
            meta_file.write(f"Samples: {total_samples}\n")
            meta_file.write(f"Tokens: {total_tokens}\n")
            meta_file.write(f"Tokens per chunk: {token_count}\n")
            meta_file.write(f"Block size: {block_size}\n")
            meta_file.write(f"Chunks: {total_chunks}\n")
            meta_file.write(f"Tokenizer: {tokenizer_name}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a validation split by removing samples from files in a folder.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input files to sample from.")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save the validation split and metadata.")
    parser.add_argument("--input_type", type=str, default="parquet", choices=["parquet", "json"], help="Input file type.")
    parser.add_argument("--output_file", type=str, default="validation_split", help="Filename for the validation split file.")
    parser.add_argument("--n_samples", type=int, default=20000, help="Total number of samples to remove for validation split.")
    parser.add_argument("--n_files", type=int, default=None, help="Number of files to randomly select from the folder (default: use all files).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for file selection.")
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print("Sampling validation split...")
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        input_type=args.input_type,
        output_file=args.output_file,
        n_samples=args.n_samples,
        n_files=args.n_files
    )
    print("Validation split created! 🎉")