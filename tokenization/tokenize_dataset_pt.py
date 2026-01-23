"""
Causal Language Modeling (CLM) Dataset Tokenization Pipeline

This script tokenizes text datasets for standard causal language modeling (pre-training) by
concatenating documents and chunking them into fixed-size training blocks. Supports both
raw text and chat template formatting.

Processing Pipeline:
1. Tokenize raw text or apply chat template
2. Add BOS/EOS tokens as configured
3. Concatenate all tokens across documents
4. Split into fixed-size blocks (block_size)
5. Filter to exact block_size
6. Apply token limits if specified
7. Split into chunks and save

Example usage:
    python tokenize_dataset_pt.py \
        --datasets_dir data/pretrain_raw \
        --output_dir data/pretrain_tokenized \
        --tokenizer_name checkpoints/tokenizer \
        --block_size 2048 \
        --add_bos_token \
        --score_column edu_score \
        --max_tokens 1000000000 \
        --tokens_per_chunk 300000000 \
        --num_proc 16 \
        --seed 42
"""
from transformers import AutoTokenizer
import numpy as np
import datasets
import argparse
import random
import glob
import os

def load_tokenizer(args):
    """Load and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        cache_dir=args.cache_dir or "./.cache",
        token=args.token,
        use_fast=True,
    )
    
    # Validate required tokens
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id. Please provide a tokenizer that defines an end-of-sequence token.")
    
    if args.add_bos_token and tokenizer.bos_token_id is None:
        raise ValueError("Tokenizer has no bos_token_id. Please provide a tokenizer that defines a beginning-of-sequence token.")
    
    # Load chat template if needed
    if args.apply_chat_template:
        if args.chat_template_path:
            with open(args.chat_template_path) as f:
                tokenizer.chat_template = f.read()
        elif tokenizer.chat_template is None:
            raise ValueError("The tokenizer does not have a chat template. Please provide --chat_template_path or use a tokenizer that supports chat templates.")
        print("Using chat template for tokenization.")
    
    return tokenizer


def create_tokenize_function(tokenizer, args):
    """Create a tokenization function with the given configuration."""
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id if args.add_bos_token else None
    
    def tokenize(examples):
        """Tokenize the text in the examples."""
        if args.apply_chat_template:
            ids = tokenizer.apply_chat_template(
                examples[args.text_column],
                return_assistant_tokens_mask=False,
                return_dict=True,
                add_generation_prompt=False,
            )
            
            # Ensure EOS token is present
            for i, tokens in enumerate(ids['input_ids']):
                if tokens[-1] != eos_id:
                    tokens.append(eos_id)
                if args.add_bos_token and tokens[0] != bos_id:
                    ids['input_ids'][i] = [bos_id] + tokens
            
            result = {'input_ids': ids['input_ids']}
        else:
            # Standard tokenization
            input_ids = tokenizer(
                examples[args.text_column],
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )
            
            # Add BOS and EOS tokens
            if args.add_bos_token:
                input_ids['input_ids'] = [[bos_id] + tokens + [eos_id] for tokens in input_ids['input_ids']]
            else:
                input_ids['input_ids'] = [[eos_id] + tokens for tokens in input_ids['input_ids']]
            
            result = input_ids
        
        # Add attention_mask and labels if requested
        if args.include_attention_mask_and_labels:
            result['attention_mask'] = [[1] * len(tokens) for tokens in result['input_ids']]
            result['labels'] = [tokens[:] for tokens in result['input_ids']]
        
        return result
    
    return tokenize


def create_group_texts_function(block_size):
    """Create a function to group texts into fixed-size blocks."""
    def group_texts(examples):
        """Group texts together so that we have chunks of `block_size`."""
        # Concatenate all tokens
        concatenated_examples = {
            k: [token for example in examples[k] for token in example]
            for k in examples.keys()
        }
        
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Truncate to multiple of block_size
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        
        # Split into blocks
        result = {
            k: [tokens[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, tokens in concatenated_examples.items()
        }
        
        return result
    
    return group_texts


def process_and_save_dataset(dataset, args, tokenize_fn, group_texts_fn, output_dir, subset_name=None):
    """Process a dataset through tokenization, grouping, filtering, and saving."""
    # Tokenize
    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Running tokenizer on every text in dataset",
        num_proc=args.num_proc or 1,
        load_from_cache_file=True,
    )
    
    # Group texts
    dataset = dataset.map(
        group_texts_fn,
        batched=True,
        desc=f"Grouping texts in chunks of {args.block_size}",
        num_proc=args.num_proc or 1,
        load_from_cache_file=True,
    )
    
    # Filter to exact block_size
    dataset = dataset.filter(
        lambda x: len(x['input_ids']) == args.block_size,
        num_proc=args.num_proc or 1,
        desc="Filtering samples with input_ids length not equal to block_size",
    )
    
    sample_count = len(dataset)
    token_count = sample_count * args.block_size
    
    # Apply max_tokens limit if specified
    if args.max_tokens and token_count > args.max_tokens:
        max_rows = args.max_tokens // args.block_size
        print(f"Subset has more than {args.max_tokens:,} tokens. Truncating to {max_rows:,} rows (~{args.max_tokens:,} tokens).")
        dataset = dataset.select(range(max_rows))
        sample_count = len(dataset)
        token_count = sample_count * args.block_size
    
    print(f"Number of samples: {sample_count:,}")
    print(f"Number of tokens: {token_count:,}")
    
    # Calculate number of chunks based on `args.tokens_per_chunk`
    n_chunks = max(1, (token_count + args.tokens_per_chunk - 1) // args.tokens_per_chunk)  # Ceiling division
    print(f"Splitting dataset into {n_chunks} chunks (max {args.tokens_per_chunk:,} tokens per chunk).")
    indices = np.array_split(np.arange(sample_count), n_chunks)
    chunks = [dataset.select(idx) for idx in indices]
    
    tokens_per_chunk = token_count // n_chunks
    print(f"Expecting {tokens_per_chunk:,} tokens per chunk.")
    
    # Save chunks
    os.makedirs(output_dir, exist_ok=True)
    save_fn = lambda chunk, path: (
        chunk.to_parquet(path) if args.output_type == "parquet" else chunk.to_json(path)
    )
    extension = args.output_type if args.output_type == "parquet" else "jsonl"
    
    for i, chunk in enumerate(chunks):
        filename = f"{output_dir}/train-{i:05d}-of-{n_chunks:05d}.{extension}"
        save_fn(chunk, filename)
    
    # Save metadata
    with open(f"{output_dir}/.metadata", "w") as meta_file:
        meta_file.write(f"Samples: {sample_count}\n")
        meta_file.write(f"Tokens: {token_count}\n")
        meta_file.write(f"Tokens per chunk: {tokens_per_chunk}\n")
        meta_file.write(f"Block size: {args.block_size}\n")
        meta_file.write(f"Chunks: {n_chunks}\n")
        meta_file.write(f"Tokenizer: {args.tokenizer_name}\n")
        if subset_name:
            meta_file.write(f"Subset: {subset_name}\n")


def main(args):
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Validate arguments
    assert args.input_type in ["jsonl", "parquet"], "Dataset type must be either 'jsonl' or 'parquet'."
    assert args.output_type in ["jsonl", "parquet"], "Output type must be either 'jsonl' or 'parquet'."
    
    # Load tokenizer
    tokenizer = load_tokenizer(args)
    
    # Create processing functions
    tokenize_fn = create_tokenize_function(tokenizer, args)
    group_texts_fn = create_group_texts_function(args.block_size)
    
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
    
    # Process by score or as whole dataset
    if args.score_column in dataset.column_names:
        # Process each unique score
        unique_scores = dataset.unique(args.score_column)
        
        for score in unique_scores:
            # Filter by score
            ds = dataset.filter(
                lambda x: x[args.score_column] == score,
                num_proc=args.num_proc or 1,
                desc=f"Filtering dataset for score == {score}",
            )
            
            if len(ds) == 0:
                print(f"No examples found for `{args.score_column}` {score}, skipping...")
                continue
            
            print(f"Loaded `{args.score_column}` subset: '{score}' | {len(ds):,} examples\n{ds}")
            
            # Process and save
            output_dir = os.path.join(args.output_dir, str(score))
            process_and_save_dataset(ds, args, tokenize_fn, group_texts_fn, output_dir, subset_name=str(score))
    
    else:
        # Process entire dataset
        print(f"No `{args.score_column}` column found, processing the entire dataset...")
        process_and_save_dataset(dataset, args, tokenize_fn, group_texts_fn, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize a dataset for Causal Language Modeling")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save the tokenized dataset")
    parser.add_argument("--datasets_dir", type=str, required=True, help="Directory containing the datasets to tokenize")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name/path of the tokenizer to use")
    parser.add_argument("--block_size", type=int, required=True, help="Block size to use")
    parser.add_argument("--cache_dir", type=str, help="Cache directory to store the tokenizer")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column in the dataset")
    parser.add_argument("--input_type", choices=["jsonl", "parquet"], default="jsonl", help="Type of the input files")
    parser.add_argument("--output_type", choices=["jsonl", "parquet"], default="parquet", help="Type of the output files")
    parser.add_argument("--score_column", type=str, default="edu_int_score", help="Name of the score column in the dataset")
    parser.add_argument("--apply_chat_template", action='store_true', help="Whether to apply a chat template to the text column")
    parser.add_argument("--chat_template_path", type=str, help="Path to chat template file")
    parser.add_argument("--add_bos_token", action='store_true', help="Whether to add a beginning-of-sequence token at the start of each input")
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum number of tokens in the dataset")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes to use")
    parser.add_argument("--include_attention_mask_and_labels", action='store_true', help="Whether to include attention_mask (all 1's) and labels (copy of input_ids) in the output")
    parser.add_argument("--tokens_per_chunk", type=int, default=300_000_000, help="Maximum number of tokens per chunk")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()

    print("Tokenizing dataset for Causal Language Modeling...")
    main(args)
    print("Tokenization complete! 🎆")
