"""
Supervised Fine-Tuning (SFT) Dataset Tokenization with optional Sequence Packing.

This script tokenizes a dataset for supervised fine-tuning (SFT) using a chat template.
It supports optional sequence packing using the Best-Fit Decreasing algorithm to create fixed-size chunks.

Best-Fit Decreasing Strategy:
- Sorts sequences by length (descending)
- Fits each sequence into the chunk with least leftover space
- Creates new chunks only when necessary
- Pads remaining space with pad tokens
- Masks non-assistant tokens with -100 for loss calculation

The returned dataset contains:
- input_ids: Tokenized input sequences
- seq_lengths: Lengths of each sequence
Optionally:
    - assistant_masks: Binary masks indicating assistant tokens (1 for assistant, 0 otherwise)
    - labels: Token IDs for loss calculation (-100 for non-assistant tokens)

Example usage:
    python tokenize_dataset_sft.py \
        --datasets_dir data/sft_raw \
        --output_dir data/sft_tokenized \
        --tokenizer_name checkpoints/tokenizer \
        --chat_template_path templates/chat_template.jinja \
        --block_size 4096 \
        --enable_packing \
        --save_by_task_type \
        --task_type_column task_type \
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
    
    # Load chat template if not already present
    if tokenizer.chat_template is None:
        if args.chat_template_path is None:
            raise ValueError("Chat template path must be provided if the tokenizer does not have a chat template.")
        else:
            with open(args.chat_template_path) as f:
                tokenizer.chat_template = f.read()
    
    # Validate pad token
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        raise ValueError("The tokenizer does not have a pad_token_id. Please set a pad token for proper padding.")
    
    if pad_id == tokenizer.eos_token_id:
        raise ValueError("The pad token id cannot be the same as the eos token id.")
    
    return tokenizer


def create_tokenize_function(tokenizer, message_column, return_assistant_tokens_mask=False, return_labels=False):
    """Create a tokenization function for SFT datasets."""
    def tokenize(examples):
        """Tokenize the text in the examples."""
        ids = tokenizer.apply_chat_template(
            examples[message_column],
            return_assistant_tokens_mask=return_assistant_tokens_mask,
            return_dict=True,
            add_generation_prompt=False,
        )
        
        result = {
            'input_ids': ids['input_ids'],
            'seq_lengths': [len(seq) for seq in ids['input_ids']]
        }
        
        if return_assistant_tokens_mask:
            # Add assistant_masks when it is requested
            result['assistant_masks'] = ids['assistant_masks']
        if return_labels:
            # Add labels when it is requested
            # If assistant_masks are provided, use them to mask out non-assistant tokens
            if 'assistant_masks' in ids:
                result['labels'] = [
                    [token if mask == 1 else -100 for token, mask in zip(input_seq, mask_seq)]
                    for input_seq, mask_seq in zip(ids['input_ids'], ids['assistant_masks'])
                ]
            else:
                # If no assistant_masks, set labels to input_ids
                result['labels'] = [input_seq.copy() for input_seq in result['input_ids']]
        
        return result
    
    return tokenize


def create_pack_messages_function(tokenizer, block_size, return_assistant_tokens_mask=False, return_labels=False):
    """Create a message packing function using Best-Fit Decreasing algorithm."""
    pad_id = tokenizer.pad_token_id
    
    def pack_messages(examples):
        """
        Group texts together so that we have chunks of `block_size`.
        
        Strategy:
        - Discard any sequence whose length > block_size.
        - Sort sequences by length descending (best-fit decreasing).
        - Place each sequence into the existing chunk that will leave the least leftover space (but still fits).
        - If none fits, create a new chunk.
        - Finalize full chunks immediately and pad remaining partial chunks at the end.
        """
        out_inputs = []
        out_seq_lengths = []
        out_labels = [] if return_labels else None
        out_masks = [] if return_assistant_tokens_mask else None
        
        # Collect valid sequences using seq_lengths
        sequences = []
        for i, seq_len in enumerate(examples["seq_lengths"]):
            if 0 < seq_len <= block_size:
                seq_data = {
                    "len": seq_len,
                    "input": list(examples["input_ids"][i]),
                }
                if return_labels:
                    seq_data["label"] = list(examples["labels"][i])
                if return_assistant_tokens_mask:
                    seq_data["mask"] = list(examples["assistant_masks"][i])
                sequences.append(seq_data)
        
        # Sort by length descending
        sequences.sort(key=lambda x: x["len"], reverse=True)
        
        # Active partial chunks
        partial_chunks = []
        
        for seq_data in sequences:
            L = seq_data["len"]
            
            # Find best-fit chunk
            best_idx = None
            best_leftover = block_size + 1
            
            for i, ch in enumerate(partial_chunks):
                space_left = block_size - ch["len"]
                if L <= space_left:
                    leftover = space_left - L
                    if leftover < best_leftover:
                        best_leftover = leftover
                        best_idx = i
            
            if best_idx is None:
                # Start new chunk or finalize if exact fit
                if L == block_size:
                    out_inputs.append(seq_data["input"])
                    out_seq_lengths.append(block_size)
                    if return_labels:
                        out_labels.append(seq_data["label"])
                    if return_assistant_tokens_mask:
                        out_masks.append(seq_data["mask"])
                else:
                    # Create new partial chunk with copies of lists to avoid mutation issues
                    new_chunk = {"len": seq_data["len"], "input": seq_data["input"][:]}
                    if return_labels:
                        new_chunk["label"] = seq_data["label"][:]
                    if return_assistant_tokens_mask:
                        new_chunk["mask"] = seq_data["mask"][:]
                    partial_chunks.append(new_chunk)
            else:
                # Add to best-fit chunk
                ch = partial_chunks[best_idx]
                ch["input"].extend(seq_data["input"])
                if return_labels:
                    ch["label"].extend(seq_data["label"])
                if return_assistant_tokens_mask:
                    ch["mask"].extend(seq_data["mask"])
                ch["len"] += L
                
                # Finalize if full
                if ch["len"] == block_size:
                    out_inputs.append(ch["input"])
                    out_seq_lengths.append(block_size)
                    if return_labels:
                        out_labels.append(ch["label"])
                    if return_assistant_tokens_mask:
                        out_masks.append(ch["mask"])
                    partial_chunks.pop(best_idx)
        
        # Pad and finalize remaining chunks
        for ch in partial_chunks:
            pad_len = block_size - ch["len"]
            ch["input"].extend([pad_id] * pad_len)
            out_inputs.append(ch["input"])
            out_seq_lengths.append(block_size)
            if return_labels:
                ch["label"].extend([-100] * pad_len)
                out_labels.append(ch["label"])
            if return_assistant_tokens_mask:
                ch["mask"].extend([0] * pad_len)
                out_masks.append(ch["mask"])
        
        # Build result fields
        result = {"input_ids": out_inputs, "seq_lengths": out_seq_lengths}
        
        # Add optional fields
        if return_labels:
            result["labels"] = out_labels
        if return_assistant_tokens_mask:
            result["assistant_masks"] = out_masks
        
        return result
    
    return pack_messages

def process_and_save_dataset(dataset, args, tokenize_fn, pack_fn, output_dir, subset_name=None):
    """Process a dataset through tokenization, packing, filtering, and saving."""
    # Tokenize
    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Running tokenizer on every text in dataset",
        num_proc=args.num_proc or 1,
        load_from_cache_file=True,
    )

    if args.enable_packing:
        # Pack messages (internally filters out sequences > block_size)
        dataset = dataset.map(
            pack_fn,
            batched=True,
            desc=f"Packing messages into chunks of {args.block_size}",
            num_proc=args.num_proc or 1,
            load_from_cache_file=True,
        )
        
        sample_count = len(dataset)
        token_count = sample_count * args.block_size
        print(f"[INFO] Number of samples after packing: {sample_count:,}")
        print(f"[INFO] Number of tokens: {token_count:,}")

    else:
        # No packing - keep variable length sequences
        # Filter to make sure all sequences are <= block_size
        def filter_block_size(examples):
            return examples['seq_lengths'] <= args.block_size
        
        dataset = dataset.filter(
            filter_block_size,
            num_proc=args.num_proc or 1,
            desc="Filtering samples with field lengths exceeding block_size",
        )
        sample_count = len(dataset)
        token_count = sum(dataset['seq_lengths'])
        print(f"[INFO] Number of samples (no packing): {sample_count:,}")
        print(f"[INFO] Number of tokens: {token_count:,}")

    # Calculate number of chunks based on `args.tokens_per_chunk`
    n_chunks = max(1, (token_count + args.tokens_per_chunk - 1) // args.tokens_per_chunk)  # Ceiling division
    print(f"[INFO] Splitting dataset into {n_chunks} chunks (max {args.tokens_per_chunk:,} tokens per chunk).")
    
    if sample_count == 0:
        print("[WARNING] No samples after processing. Skipping save.")
        return
    
    # Split into chunks
    print(f"[INFO] Splitting dataset into {n_chunks} chunks.")
    indices = np.array_split(np.arange(sample_count), n_chunks)
    chunks = [dataset.select(idx) for idx in indices]
    
    tokens_per_chunk = token_count // n_chunks
    print(f"[INFO] Expecting {tokens_per_chunk:,} tokens per chunk.")
    
    # Save chunks
    os.makedirs(output_dir, exist_ok=True)
    extension = args.output_type if args.output_type == "parquet" else "jsonl"
    save_fn = lambda chunk, path: (
        chunk.to_parquet(path) if args.output_type == "parquet" else chunk.to_json(path)
    )
    
    for i, chunk in enumerate(chunks):
        filename = f"{output_dir}/train-{i:05d}-of-{n_chunks:05d}.{extension}"
        save_fn(chunk, filename)
    
    # Save metadata
    with open(f"{output_dir}/.metadata", "w") as meta_file:
        meta_file.write(f"Samples: {sample_count}\n")
        meta_file.write(f"Tokens: {token_count}\n")
        meta_file.write(f"Tokens per chunk: {tokens_per_chunk}\n")
        meta_file.write(f"Packing: {'enabled' if args.enable_packing else 'disabled'}\n")
        if args.enable_packing:
            meta_file.write(f"Block size: {args.block_size}\n")
        meta_file.write(f"Chunks: {n_chunks}\n")
        meta_file.write(f"Tokenizer: {args.tokenizer_name}\n")
        if subset_name:
            meta_file.write(f"Task type: {subset_name}\n")


def main(args):
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Validate arguments
    assert args.input_type in ["jsonl", "parquet"], "Dataset type must be either 'jsonl' or 'parquet'."
    assert args.output_type in ["jsonl", "parquet"], "Output type must be either 'jsonl' or 'parquet'."
    
    # Load tokenizer
    tokenizer = load_tokenizer(args)
    
    # Create processing functions - pass through user's explicit choices
    tokenize_fn = create_tokenize_function(
        tokenizer, 
        args.message_column, 
        return_assistant_tokens_mask=args.return_assistant_tokens_mask,
        return_labels=args.return_labels
    )
    
    pack_fn = None
    if args.enable_packing:
        pack_fn = create_pack_messages_function(
            tokenizer, 
            args.block_size, 
            return_assistant_tokens_mask=args.return_assistant_tokens_mask,
            return_labels=args.return_labels
        )
    
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
    print(f"[INFO] Loaded dataset with {len(dataset):,} examples from {args.input_type.upper()} files.\n{dataset}")
    
    # Process by task_type or as whole dataset
    if args.save_by_task_type and args.task_type_column in dataset.column_names:
        # Get unique task types
        unique_task_types = dataset.unique(args.task_type_column)
        print(f"[INFO] Found {len(unique_task_types)} unique task types: {unique_task_types}")
        
        for task_type in unique_task_types:
            # Filter by task_type
            ds = dataset.filter(
                lambda x: x[args.task_type_column] == task_type,
                num_proc=args.num_proc or 1,
                desc=f"Filtering dataset for task_type == {task_type}",
            )
            
            if len(ds) == 0:
                print(f"[WARNING] No examples found for task_type '{task_type}', skipping...")
                continue
            
            print(f"\n[INFO] Processing task_type: '{task_type}' | {len(ds):,} examples")

            # Process and save
            output_dir = os.path.join(args.output_dir, str(task_type))
            process_and_save_dataset(ds, args, tokenize_fn, pack_fn, output_dir, subset_name=str(task_type))
    
    else:
        # Process entire dataset
        if args.save_by_task_type:
            print(f"[WARNING] --save_by_task_type specified but column '{args.task_type_column}' not found. Processing entire dataset.")
        
        process_and_save_dataset(dataset, args, tokenize_fn, pack_fn, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize a dataset for supervised fine-tuning (SFT) using a chat template and Best-Fit Decreasing packing.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save the tokenized dataset")
    parser.add_argument("--datasets_dir", type=str, required=True, help="Directory containing the datasets to tokenize")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name/path of the tokenizer to use")
    parser.add_argument("--block_size", type=int, required=True, help="Max block size to use")
    parser.add_argument("--chat_template_path", type=str, default=None, help="Path to chat template file")
    parser.add_argument("--enable_packing", action='store_true', help="Enable sequence packing (outputs fixed-length chunks of block_size)")
    parser.add_argument("--return_assistant_tokens_mask", action='store_true', help="Include assistant_masks field in the final output")
    parser.add_argument("--return_labels", action='store_true', help="Include labels field in the final output (requires assistant_masks for proper masking)")
    parser.add_argument("--cache_dir", type=str, help="Cache directory to store the tokenizer")
    parser.add_argument("--message_column", type=str, default="messages", help="Name of the message column in the dataset")
    parser.add_argument("--input_type", choices=["jsonl", "parquet"], default="jsonl", help="Type of the input files")
    parser.add_argument("--output_type", choices=["jsonl", "parquet"], default="parquet", help="Type of the output files")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes to use")
    parser.add_argument("--tokens_per_chunk", type=int, default=300_000_000, help="Maximum number of tokens per chunk")
    parser.add_argument("--save_by_task_type", action='store_true', help="Save dataset split by task_type column")
    parser.add_argument("--task_type_column", type=str, default="task_type", help="Name of the task_type column in the dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()

    print("Tokenizing dataset for Supervised Fine-Tuning...")
    main(args)
    print("Tokenization complete! 🎆")