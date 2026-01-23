"""
Dataset Format Converter

Converts datasets between JSONL and Parquet formats.

Usage:
    # Convert JSONL to Parquet
    python save_as.py --input_dir data/jsonl --output_dir data/parquet \\
        --input_type jsonl --output_type parquet
    
    # Convert Parquet to JSONL
    python save_as.py --input_dir data/parquet --output_dir data/jsonl \\
        --input_type parquet --output_type jsonl
    
    # Convert with multiple input files (outputs same number of chunks)
    python save_as.py --input_dir data/input --output_dir data/output \\
        --input_type jsonl --output_type parquet
"""
import datasets
import numpy as np
import argparse
import glob
import os


def load_dataset(input_dir, input_type="jsonl", cache_dir=None):
    """Load dataset from JSONL or Parquet files."""
    data_files = glob.glob(f"{input_dir}/*.{input_type}")
    if not data_files:
        raise ValueError(f"No {input_type.upper()} files found in '{input_dir}'.")
    
    print(f"[INFO] Found {len(data_files)} {input_type} file(s)")
    
    dataset = datasets.load_dataset(
        "json" if input_type == "jsonl" else "parquet",
        data_files=data_files,
        split="train",
        cache_dir=cache_dir,
        num_proc=len(data_files),
    )
    
    return dataset, len(data_files)


def save_dataset(dataset, output_dir, output_type="jsonl", num_chunks=1):
    """Save dataset in the specified format, optionally splitting into chunks."""
    os.makedirs(output_dir, exist_ok=True)
    
    sample_count = len(dataset)
    print(f"[INFO] Saving {sample_count:,} samples in {num_chunks} chunk(s)")
    
    extension = output_type if output_type == "parquet" else "jsonl"
    
    if num_chunks > 1:
        # Split into chunks
        indices = np.array_split(np.arange(sample_count), num_chunks)
        chunks = [dataset.select(idx) for idx in indices]
        
        # Save each chunk
        for i, chunk in enumerate(chunks):
            filename = f"{output_dir}/train-{i:05d}-of-{num_chunks:05d}.{extension}"
            print(f"[INFO] Saving chunk {i+1}/{num_chunks}: {filename} ({len(chunk):,} samples)")
            
            if output_type == "parquet":
                chunk.to_parquet(filename)
            else:
                chunk.to_json(filename)
    else:
        # Save as single file
        filename = f"{output_dir}/train.{extension}"
        print(f"[INFO] Saving to {filename}")
        
        if output_type == "parquet":
            dataset.to_parquet(filename)
        else:
            dataset.to_json(filename)
    
    # Save metadata
    metadata_path = f"{output_dir}/.metadata"
    with open(metadata_path, "w") as meta_file:
        meta_file.write(f"Samples: {sample_count}\n")
        meta_file.write(f"Chunks: {num_chunks}\n")
        meta_file.write(f"Output type: {output_type}\n")
        meta_file.write(f"Columns: {dataset.column_names}\n")
    
    print(f"[INFO] Metadata saved to {metadata_path}")


def main(args):
    # Validate arguments
    assert args.input_type in ["jsonl", "parquet"], "Input type must be either 'jsonl' or 'parquet'."
    assert args.output_type in ["jsonl", "parquet"], "Output type must be either 'jsonl' or 'parquet'."
    
    # Load dataset
    print(f"[INFO] Loading dataset from {args.input_dir}")
    dataset, num_input_files = load_dataset(args.input_dir, args.input_type, args.cache_dir)
    print(f"[INFO] Loaded dataset with {len(dataset):,} examples")
    print(f"[INFO] Columns: {dataset.column_names}\n")
    
    # Determine number of chunks
    if args.input_type == "jsonl" and args.output_type == "parquet" and "token_count" in dataset.column_names:
        # Calculate total tokens
        total_tokens = sum(dataset["token_count"])
        print(f"[INFO] Total tokens in dataset: {total_tokens:,}")
        
        # Set max tokens per chunk
        num_chunks = max(1, (total_tokens + args.max_tokens_per_chunk - 1) // args.max_tokens_per_chunk)
        print(f"[INFO] Calculating chunks based on token count (~{args.max_tokens_per_chunk:,} tokens per chunk)")
    else:
        # Use number of input files as number of chunks
        num_chunks = num_input_files
    
    # Save dataset
    save_dataset(
        dataset,
        args.output_dir,
        args.output_type,
        num_chunks
    )
    
    print(f"\n[SUCCESS] Dataset saved to {args.output_dir}")
    print(f"[SUCCESS] Total samples: {len(dataset):,}")
    print(f"[SUCCESS] Format: {args.input_type} → {args.output_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset between JSONL and Parquet formats.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing the dataset files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save the converted dataset")
    parser.add_argument("--input_type", choices=["jsonl", "parquet"], required=True, help="Type of the input files")
    parser.add_argument("--output_type", choices=["jsonl", "parquet"], required=True, help="Type of the output files")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for datasets")
    parser.add_argument("--max_tokens_per_chunk", type=int, default=300_000_000, help="Maximum tokens per chunk when converting from JSONL to Parquet (default: 300M)")
    
    args = parser.parse_args()
    
    print(f"Converting dataset from {args.input_type} to {args.output_type}... 🚀")
    main(args)
    print("Conversion complete! 🎆")
