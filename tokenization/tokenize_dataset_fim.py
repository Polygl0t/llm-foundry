"""
Fill-In-the-Middle (FIM) Dataset Tokenization for Code Generation

This script tokenizes code datasets for Fill-In-the-Middle (FIM) training, a technique that
improves code completion by training models to predict middle sections given prefix and suffix
context. Essential for training code-capable models.

FIM Format:
    Original:  <bos> [PREFIX] [MIDDLE] [SUFFIX] <eos>
    FIM:       <bos> <fim_prefix> [PREFIX] <fim_suffix> [SUFFIX] <fim_middle> [MIDDLE] <eos>

Processing Pipeline:
1. Tokenize code samples with BOS/EOS tokens
2. Concatenate and chunk into block_size
3. Split on EOS tokens to preserve document boundaries
4. Apply FIM transformation probabilistically
5. Recombine subchunks respecting block_size
6. Ensure proper token placement and padding
7. Filter to exact block_size and save

Example usage:
    python tokenize_dataset_fim.py \
        --datasets_dir data/code_raw \
        --output_dir data/code_fim_tokenized \
        --tokenizer_name checkpoints/code_tokenizer \
        --block_size 2048 \
        --fim_ratio 0.5 \
        --fim_prefix_token "<|fim_prefix|>" \
        --fim_suffix_token "<|fim_suffix|>" \
        --fim_middle_token "<|fim_middle|>" \
        --add_bos_token \
        --tokens_per_chunk 300000000 \
        --num_proc 16 \
        --seed 42
"""
from transformers import AutoTokenizer
import numpy as np
import datasets
import argparse
import glob
import random
import os

def main(args):

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the tokenizer we will use to tokenize the text
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        cache_dir=args.cache_dir if args.cache_dir else "./.cache",
        token=args.token if args.token else None,
        use_fast=True,
    )

    # Assert that the tokenizer has the required special tokens (eos and FIM tokens)
    assert tokenizer.eos_token_id is not None, "Tokenizer has no eos_token_id. Please provide a tokenizer that defines an end-of-sequence token."
    assert args.fim_prefix_token in tokenizer.get_vocab(), f"Tokenizer does not contain the FIM prefix token: {args.fim_prefix_token}"
    assert args.fim_suffix_token in tokenizer.get_vocab(), f"Tokenizer does not contain the FIM suffix token: {args.fim_suffix_token}"
    assert args.fim_middle_token in tokenizer.get_vocab(), f"Tokenizer does not contain the FIM middle token: {args.fim_middle_token}"
    if args.add_bos_token:
        assert tokenizer.bos_token_id is not None, "Tokenizer has no bos_token_id. Please provide a tokenizer that defines a beginning-of-sequence token."

    PRE_TOKEN_ID = tokenizer.convert_tokens_to_ids(args.fim_prefix_token) # "<|fim_prefix|>"
    SUF_TOKEN_ID = tokenizer.convert_tokens_to_ids(args.fim_suffix_token) # "<|fim_suffix|>"
    MID_TOKEN_ID = tokenizer.convert_tokens_to_ids(args.fim_middle_token) # "<|fim_middle|>"
    EOS_TOKEN_ID = tokenizer.eos_token_id # "<|im_end|>"
    if args.add_bos_token:
        BOS_TOKEN_ID = tokenizer.bos_token_id # "<|im_start|>"

    print(f"Loaded tokenizer {args.tokenizer_name}")
    print(f"FIM tokens - Prefix: {PRE_TOKEN_ID}, Suffix: {SUF_TOKEN_ID}, Middle: {MID_TOKEN_ID}")

    # Define a function to tokenize the text
    def tokenize(examples, add_bos_token=args.add_bos_token):

        # Tokenize a sample and return them as a list of token ids, 
        # we don't need the attention mask or token type ids for our model.
        # Also, we don't add special tokens here because we will add them manually.
        input_ids = tokenizer(
            examples[args.text_column], 
            return_attention_mask=False, 
            return_token_type_ids=False,
            add_special_tokens=False
        )

        if add_bos_token:
            # Enclose each example with beginning-of-sequence and end-of-sequence tokens
            # Having fixed tokens always present at the beginning of the context (such as <|im_start|>) 
            # have been shown to improve model quality and training stability, serve as attention sinks, 
            # and allow to store global knowledge.
            # - Raffel et al. [2020](https://jmlr.org/papers/volume21/20-074/20-074.pdf)
            # - Dong et al. [2024](https://arxiv.org/abs/2411.13676)
            # - Xiao et al. [2024](https://arxiv.org/abs/2309.17453)
            # - OpenAI et al. [2025](https://arxiv.org/abs/2508.10925)
            input_ids['input_ids'] = [[BOS_TOKEN_ID] + sublist + [EOS_TOKEN_ID] for sublist in input_ids['input_ids']]
        else:
            # Add only the end-of-sequence token at the end (was incorrectly added at the beginning)
            input_ids['input_ids'] = [sublist + [EOS_TOKEN_ID] for sublist in input_ids['input_ids']]
        
        return input_ids

    # Define a function to group the texts together so that we have chunks of `block_size`
    def group_texts(examples):
        """ Group texts together so that we have chunks of `block_size` """

        # Concatenate tokens from examples for each key in the examples dictionary
        concatenated_examples = {
            k: [t for example in examples[k] for t in example ] for k in examples.keys()
        }

        # Calculate the total length of the concatenated tokens for the first key in examples.
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # If the total length is greater than or equal to the block size, adjust it to be a multiple of the block size.
        if total_length >= args.block_size:
            total_length = (total_length // args.block_size) * args.block_size

        # Split the concatenated tokens into blocks of size `block_size`.
        result = {
            # For each key and token list in concatenated examples
            k: [
                # Create a sublist for each block of size `block_size`
                t[i : i + args.block_size]
                for i in range(0, total_length, args.block_size)
            ]
            for k, t in concatenated_examples.items()
        }

        # Return the processed blocks of tokens.
        return result

    # Format the dataset to be a fill-in-the-middle dataset
    def apply_fim_after_chunking(examples, fim_ratio=args.fim_ratio, block_size=args.block_size, add_bos_token=args.add_bos_token):
        """
        Applies FIM formatting after chunking, respecting block size limits.

        Args:
            examples (dict): Dictionary with 'input_ids' as key and list of token ID blocks as values.
            fim_ratio (float): Probability of applying FIM formatting.
            block_size (int): Maximum number of tokens per final chunk.
            add_bos_token (bool): Whether to ensure the presence of a beginning-of-sequence token.

        Returns:
            dict: Transformed examples with 'input_ids' key.
        """

        new_chunks = []

        # Helper to ensure BOS presence (if requested), respect block_size, and end with EOS
        def finalize_block(block):
            # We will use EOS as padding token
            pad_id = EOS_TOKEN_ID

            # Ensure block is non-empty (create minimal block if needed)
            if not block:
                if add_bos_token:
                    blk = [BOS_TOKEN_ID]
                else:
                    blk = []
                # ensure at least one EOS if block_size allows
                if block_size > 0:
                    if add_bos_token and block_size > 1:
                        blk.append(EOS_TOKEN_ID)
                    elif not add_bos_token:
                        blk = [EOS_TOKEN_ID]  # minimal safe block
                # pad to block_size
                while len(blk) < block_size:
                    blk.append(pad_id)
                return blk[:block_size]

            # If BOS requested, ensure it is present at start
            if add_bos_token and block[0] != BOS_TOKEN_ID:
                block = [BOS_TOKEN_ID] + block

            # Ensure final token is EOS (replace or append)
            if block[-1] != EOS_TOKEN_ID:
                if len(block) < block_size:
                    block = block + [EOS_TOKEN_ID]
                else:
                    block[-1] = EOS_TOKEN_ID

            # Trim if too long
            if len(block) > block_size:
                block = block[:block_size]
                # ensure last token is EOS after trimming
                block[-1] = EOS_TOKEN_ID

            # Pad if shorter than block_size
            while len(block) < block_size:
                block.append(pad_id)

            return block

        for chunk in examples["input_ids"]:
            # 1. Split on eos_token_id to get subchunks (more efficient using numpy)
            chunk_array = np.array(chunk)
            eos_indices = np.where(chunk_array == EOS_TOKEN_ID)[0]
            
            subchunks = []
            start = 0
            for eos_idx in eos_indices:
                if eos_idx > start:
                    subchunks.append(chunk[start:eos_idx])
                start = eos_idx + 1
            # Add remaining tokens after last EOS
            if start < len(chunk):
                subchunks.append(chunk[start:])

            processed_subchunks = []
            for subchunk in subchunks:
                if random.random() < fim_ratio and len(subchunk) > 8:
                    # Convert to string and try to apply FIM
                    try:
                        decoded = tokenizer.decode(subchunk, skip_special_tokens=True)
                        n = len(decoded)
                        
                        # Ensure minimum length for meaningful splits
                        if n < 12:  # Minimum length to allow meaningful prefix/middle/suffix
                            processed_subchunks.append(subchunk)
                            continue
                            
                        # Calculate split points with minimum lengths
                        min_section = max(1, n // 10)  # At least 10% or 1 char for each section
                        i = random.randint(min_section, max(min_section + 1, n // 3))
                        j = random.randint(max(i + min_section, n // 3), max(i + min_section + 1, 2 * n // 3))

                        prefix, middle, suffix = decoded[:i], decoded[i:j], decoded[j:]

                        # Skip if any section is empty
                        if not prefix or not middle or not suffix:
                            processed_subchunks.append(subchunk)
                            continue

                        # Tokenize parts
                        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                        middle_ids = tokenizer.encode(middle, add_special_tokens=False)
                        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)

                        # Reserve space for control tokens
                        total_len = (
                            len(prefix_ids)
                            + len(middle_ids)
                            + len(suffix_ids)
                            + 4  # FIM tokens
                        )

                        # Truncate middle if necessary
                        if total_len > block_size:
                            excess = total_len - block_size
                            middle_ids = middle_ids[:-excess] if excess < len(middle_ids) else []

                        if random.random() < 0.5:
                            # PSM: prefix + suffix + middle
                            processed = (
                                [PRE_TOKEN_ID]
                                + prefix_ids
                                + [SUF_TOKEN_ID]
                                + suffix_ids
                                + [MID_TOKEN_ID]
                                + middle_ids
                                + [EOS_TOKEN_ID]
                            )
                        else:
                            # SPM: suffix + prefix+middle
                            processed = (
                                [PRE_TOKEN_ID]
                                + [SUF_TOKEN_ID]
                                + suffix_ids
                                + [MID_TOKEN_ID]
                                + prefix_ids
                                + middle_ids
                                + [EOS_TOKEN_ID]
                            )

                        processed_subchunks.append(processed)
                    except Exception:
                        # Fallback to original if FIM transformation fails
                        processed_subchunks.append(subchunk)
                else:
                    processed_subchunks.append(subchunk)

            # Recombine processed subchunks up to block_size
            current_block = []
            for sub in processed_subchunks:
                # Ensure individual subchunks don't exceed block_size
                if len(sub) > block_size:
                    sub = sub[:block_size - 1] + [EOS_TOKEN_ID]  # Ensure EOS token is at the end

                # Skip empty subchunks
                if not sub:
                    continue
                    
                if len(current_block) + len(sub) <= block_size:
                    current_block.extend(sub)
                else:
                    # Add current_block if it's not empty (finalize before storing)
                    if current_block:
                        new_chunks.append(finalize_block(current_block))
                    # Start new block with current subchunk, ensuring it fits
                    if len(sub) <= block_size:
                        current_block = sub
                    else:
                        current_block = sub[:block_size - 1] + [EOS_TOKEN_ID]

            # Add final block if it exists (finalize before storing)
            if current_block:
                new_chunks.append(finalize_block(current_block))

        return {"input_ids": new_chunks}


    # Load the datasets from disk
    assert args.input_type in ["jsonl", "parquet"], "Dataset type must be either 'jsonl' or 'parquet'."
    assert args.output_type in ["jsonl", "parquet"], "Output type must be either 'jsonl' or 'parquet'."

    data_files = glob.glob(f"{args.datasets_dir}/*.{args.input_type}")
    assert len(data_files) > 0, f"No {args.input_type.upper()} files found in '{args.datasets_dir}'."

    dataset = datasets.load_dataset(
        "json" if args.input_type == "jsonl" else "parquet",
        data_files=data_files, 
        split="train",
        cache_dir=args.cache_dir,
        num_proc=len(data_files),
    )
    print(f"Loaded dataset with {len(dataset):,} examples from {args.input_type.upper()} files.\n{dataset}")

    # Tokenize the dataset
    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Running tokenizer on every text in dataset",
        num_proc=args.num_proc if args.num_proc else 1,
        load_from_cache_file=True,
    )

    # Group the texts in chunks of `block_size`
    dataset = dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {args.block_size}",
        num_proc=args.num_proc if args.num_proc else 1,
        load_from_cache_file=True,
    )

    # Apply FIM formatting after chunking
    dataset = dataset.map(
        apply_fim_after_chunking,
        batched=True,
        desc=f"Applying FIM formatting after chunking",
        num_proc=args.num_proc if args.num_proc else 1,
        load_from_cache_file=True,
    )

    # Remove any sample that the length of `input_ids` is not equal to `block_size`
    dataset = dataset.filter(
        lambda x: len(x['input_ids']) == args.block_size,
        num_proc=args.num_proc if args.num_proc else 1,
        desc="Filtering samples with input_ids length not equal to block_size",
    )

    sample_count = len(dataset)
    token_count = sample_count * args.block_size

    # Limit the amount of tokens if required
    if args.max_tokens is not None:
        if token_count > args.max_tokens:
            max_rows = args.max_tokens // args.block_size
            print(f"Subset has more than {args.max_tokens} tokens. Truncating to {max_rows:,} rows (~{args.max_tokens} tokens).")
            dataset = dataset.select(range(max_rows))
            sample_count = len(dataset)
            token_count = sample_count * args.block_size

    print(f"Number of samples: {sample_count:,}")
    print(f"Number of tokens: {token_count:,}")

    # Calculate number of chunks based on `args.tokens_per_chunk`
    n_chunks = max(1, (token_count + args.tokens_per_chunk - 1) // args.tokens_per_chunk)  # Ceiling division
    print(f"Splitting dataset into {n_chunks} chunks (max {args.tokens_per_chunk:,} tokens per chunk).")
    indices = np.array_split(np.arange(len(dataset)), n_chunks)
    chunks = [dataset.select(idx) for idx in indices]

    tokens_per_chunk = token_count // n_chunks
    print(f"Expecting {tokens_per_chunk:,} tokens per chunk.")

    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.output_type == "parquet":
        for i, chunk in enumerate(chunks):
            chunk.to_parquet(f"{args.output_dir}/train-{i:05d}-of-{n_chunks:05d}.parquet")
    else:
        for i, chunk in enumerate(chunks):
            chunk.to_json(f"{args.output_dir}/train-{i:05d}-of-{n_chunks:05d}.jsonl")
            
    # Save a .metadata file with dataset statistics
    with open(f"{args.output_dir}/.metadata", "w") as meta_file:
        meta_file.write(f"Samples: {sample_count}\n")
        meta_file.write(f"Tokens: {token_count}\n")
        meta_file.write(f"Tokens per chunk: {tokens_per_chunk}\n")
        meta_file.write(f"Block size: {args.block_size}\n")
        meta_file.write(f"Chunks: {n_chunks}\n")
        meta_file.write(f"Tokenizer: {args.tokenizer_name}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize a dataset for Fill-In-the-Middle (FIM) training.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save the tokenized dataset")
    parser.add_argument("--cache_dir", type=str, help="Cache directory to store the tokenizer")
    parser.add_argument("--datasets_dir", type=str, required=True, help="Directory containing the datasets to tokenize")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column in the dataset")
    parser.add_argument("--input_type", choices=["jsonl", "parquet"], default="jsonl", help="Type of the input files (e.g., 'jsonl', 'parquet')")
    parser.add_argument("--output_type", choices=["jsonl", "parquet"], default="parquet", help="Type of the output files (e.g., 'jsonl', 'parquet')")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name/path of the tokenizer to use")
    parser.add_argument("--block_size", type=int, required=True, help="Block size to use")
    parser.add_argument("--fim_prefix_token", type=str, default="<|fim_prefix|>", help="Prefix token for FIM")
    parser.add_argument("--fim_suffix_token", type=str, default="<|fim_suffix|>", help="Suffix token for FIM")
    parser.add_argument("--fim_middle_token", type=str, default="<|fim_middle|>", help="Middle token for FIM") 
    parser.add_argument("--fim_ratio", type=float, default=0.5, help="Probability of applying FIM formatting")
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum number of tokens in the dataset")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes to use")
    parser.add_argument("--tokens_per_chunk", type=int, default=300_000_000, help="Maximum number of tokens per chunk")
    parser.add_argument("--add_bos_token", action="store_true", help="Whether to add a beginning-of-sequence token at the start of each example")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    print("Starting dataset tokenization for FIM...")
    main(args)
    print("Dataset tokenization for FIM completed. 🎉")