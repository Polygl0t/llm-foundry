"""
Shared utilities for language filter scripts.
"""
import glob
import os
import numpy as np
import datasets


def load_dataset(input_dir, input_type, cache_dir=None):
    assert input_type in ["jsonl", "parquet"], "input_type must be 'jsonl' or 'parquet'."
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
    print(f"[INFO] Loaded dataset with {len(dataset):,} examples from {input_type.upper()} files.")
    print(f"[INFO] Columns: {dataset.column_names}")
    return dataset, data_files


def save_dataset(dataset, output_dir, output_type, n_chunks):
    assert output_type in ["jsonl", "parquet"], "output_type must be 'jsonl' or 'parquet'."
    os.makedirs(output_dir, exist_ok=True)
    count = len(dataset)
    indices = np.array_split(np.arange(count), n_chunks)
    chunks = [dataset.select(idx.tolist()) for idx in indices if len(idx) > 0]
    extension = output_type if output_type == "parquet" else "jsonl"
    save_fn = lambda chunk, path: (
        chunk.to_parquet(path) if output_type == "parquet" else chunk.to_json(path)
    )
    print(f"\n[INFO] Splitting dataset into {n_chunks} chunks (matching input file count)")
    for i, chunk in enumerate(chunks):
        filename = f"{output_dir}/train-{i:05d}-of-{n_chunks:05d}.{extension}"
        save_fn(chunk, filename)
        print(f"[INFO] Saved chunk {i+1}/{n_chunks} with {len(chunk):,} examples to {filename}")
    with open(f"{output_dir}/.metadata", "w") as meta_file:
        meta_file.write(f"Number of samples: {count}\n")
        if 'token_count' in dataset.column_names:
            meta_file.write(f"Number of tokens: {sum(dataset['token_count'])}\n")
        meta_file.write(f"Chunks: {n_chunks}\n")
        meta_file.write(f"Columns: {dataset.column_names}\n")
    print("\n[INFO] Metadata saved to .metadata file")


def is_messages_column(dataset, column_name):
    if column_name not in dataset.column_names:
        return False
    for example in dataset:
        value = example.get(column_name)
        if value is None:
            continue
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], dict) and 'content' in value[0]:
                return True
        break
    return False


def flatten_messages(messages):
    if not messages:
        return ""
    contents = []
    for msg in messages:
        if isinstance(msg, dict) and 'content' in msg:
            content = msg['content']
            if content:
                contents.append(str(content))
    return '\n'.join(contents)
