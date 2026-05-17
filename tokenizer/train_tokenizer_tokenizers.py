"""
BPE Tokenizer Training with Hugging Face Tokenizers Library

This script trains a Byte-Pair Encoding (BPE) tokenizer from scratch using the Hugging Face
tokenizers library. Designed for creating custom tokenizers optimized for specific languages,
domains, or mixed content (code, natural language, etc.).

Example usage:
    python train_tokenizer_tokenizers.py \
        --data_path data/training_corpus \
        --data_type jsonl \
        --text_column text \
        --vocab_size 32000 \
        --output_dir checkpoints/my_tokenizer \
        --add_bos_token \
        --byte_fallback \
        --hub_repo_id username/my-tokenizer \
        --token hf_xxx

Only BOS, EOS, UNK, and PAD are registered as special tokens. Chat/control markers
such as "<think>" and "<tool_call>" are added to the vocabulary as regular tokens
so they remain visible when decoding with skip_special_tokens=True.
"""
from transformers import PreTrainedTokenizerFast
import os
import argparse
import unicodedata
from utils import (
    get_logger,
    EXTRA_TOKENS,
    load_text_dataset,
    update_tokenizer_config,
    validate_saved_tokenizer,
    write_special_tokens_map,
    push_tokenizer_to_hub,
)

logger = get_logger("BPE-Tokenizer-Trainer")

from tokenizers import (
    AddedToken,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)



def main(args):
    
    dataset = load_text_dataset(args.data_path, args.data_type, cache_dir=args.cache_dir, num_proc=args.num_proc)

    # Pre-normalize the text to NFKC (helps to prevent mojibake issues)
    def normalize_to_nfkc(example):
        example[args.text_column] = unicodedata.normalize("NFKC", example[args.text_column])
        return example

    dataset = dataset.map(normalize_to_nfkc, num_proc=args.num_proc)
    logger.info(f"Loaded dataset with {len(dataset):,} samples")

    # Define a Model
    # See https://huggingface.co/docs/tokenizers/api/models#tokenizers.models.BPE
    tokenizer = Tokenizer(models.BPE(byte_fallback=args.byte_fallback))
    
    # Define a normalizer.
    # See https://huggingface.co/docs/tokenizers/api/normalizers#tokenizers.normalizers.NFC
    tokenizer.normalizer = normalizers.NFC()

    # Define a pre-tokenizer.
    # See https://huggingface.co/docs/tokenizers/api/pre-tokenizers#tokenizers.pre_tokenizers.ByteLevel
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)
    
    core_special_tokens = [args.bos_token, args.eos_token, args.pad_token, args.unk_token]

    # Define the model trainer
    # See https://huggingface.co/docs/tokenizers/api/trainers#tokenizers.trainers.BpeTrainer
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size - len(EXTRA_TOKENS),
        special_tokens=core_special_tokens,
        show_progress=True,
    )

    # Define a generator dor the BPE trainer
    def get_training_corpus(bs=args.batch_size):
        """
        Just a generator that will yield batches of text data.
        """
        for i in range(0, len(dataset), bs):
            yield dataset[i : i + bs][args.text_column]

    # Train the tokenizer
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    regular_added_tokens = [AddedToken(token, special=False, normalized=False) for token in EXTRA_TOKENS]
    num_added_tokens = tokenizer.add_tokens(regular_added_tokens)
    if num_added_tokens != len(EXTRA_TOKENS):
        raise ValueError(f"Expected to add {len(EXTRA_TOKENS)} regular tokens, but added {num_added_tokens}.")

    # Get the token IDs for the main special tokens (will use them to hardcode them in the config)
    bos_token_id = tokenizer.token_to_id(args.bos_token)
    eos_token_id = tokenizer.token_to_id(args.eos_token)
    pad_token_id = tokenizer.token_to_id(args.pad_token)
    unk_token_id = tokenizer.token_to_id(args.unk_token)

    # Define a post-processor
    if args.add_bos_token:
        # Apparently, this is the "correct" way to add a BOS token using the `tokenizers` library.
        # See: https://github.com/huggingface/tokenizers/issues/1643
        # See https://huggingface.co/docs/tokenizers/en/api/post-processors#tokenizers.processors.TemplateProcessing
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{args.bos_token} $A",
            special_tokens=[(args.bos_token, bos_token_id)],
        )
    
    else:
        # See https://huggingface.co/docs/tokenizers/api/post-processors#tokenizers.processors.ByteLevel
        tokenizer.post_processor = processors.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)
        
    # Define a decoder
    # See https://huggingface.co/docs/tokenizers/api/decoders#tokenizers.decoders.ByteLevel
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)

    logger.info("Wrapping the tokenizer with PreTrainedTokenizerFast")
    # Wrap the tokenizer
    # See https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        pad_token=args.pad_token,
        unk_token=args.unk_token,
        padding_side=args.padding_side,
        truncation_side=args.truncation_side,
        model_max_length=args.model_max_length,
        clean_up_tokenization_spaces=False,
    )

    # Assert that the size of the tokenizer matches the expected vocabulary size
    assert len(wrapped_tokenizer) == args.vocab_size, f"Expected vocab size {args.vocab_size}, but got {len(wrapped_tokenizer)}"

    # Save the tokenizer
    if not os.path.exists(args.output_dir):
         os.makedirs(args.output_dir)
    wrapped_tokenizer.save_pretrained(args.output_dir)
    write_special_tokens_map(
        args.output_dir,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        unk_token=args.unk_token,
        pad_token=args.pad_token,
    )

    update_tokenizer_config(
        args.output_dir,
        legacy=False,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        unk_token_id=unk_token_id,
        add_bos_token=args.add_bos_token,
        add_eos_token=args.add_eos_token,
    )

    validate_saved_tokenizer(args.output_dir)

    # Upload your new tokenizer to the Hub!
    if args.hub_repo_id is not None and args.token is not None:
        push_tokenizer_to_hub(args.output_dir, args.hub_repo_id, args.token, private=args.private)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    
    # Required arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data file or directory containing data files")
    
    # Optional arguments with defaults
    parser.add_argument("--data_type", type=str, default="txt", choices=["txt", "jsonl", "parquet", "csv"],
                        help="Type of data files (txt, jsonl, parquet, csv)")
    parser.add_argument("--cache_dir", type=str, default="./.cache",
                        help="Directory to cache datasets")
    parser.add_argument("--num_proc", type=int, default=8,
                        help="Number of processes for dataset loading")
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="Batch size for training")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column name containing text data")
    parser.add_argument("--bos_token", type=str, default="<|im_start|>",
                        help="Beginning of sequence token")
    parser.add_argument("--eos_token", type=str, default="<|im_end|>",
                        help="End of sequence token")
    parser.add_argument("--pad_token", type=str, default="<|pad|>",
                        help="Padding token")
    parser.add_argument("--unk_token", type=str, default="<|unk|>",
                        help="Unknown token")
    parser.add_argument("--padding_side", type=str, default="right", choices=["left", "right"],
                        help="Side to add padding (left or right)")
    parser.add_argument("--truncation_side", type=str, default="right", choices=["left", "right"],
                        help="Side to truncate (left or right)")
    parser.add_argument("--add_bos_token", action="store_true",
                        help="Whether to add BOS token by default")
    parser.add_argument("--add_eos_token", action="store_true",
                        help="Whether to add EOS token by default")
    parser.add_argument("--model_max_length", type=int, default=1000000000000000019884624838656,
                        help="Maximum sequence length")
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="Vocabulary size")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="Directory to save tokenizer")
    parser.add_argument("--hub_repo_id", type=str, default=None,
                        help="Hugging Face Hub repo ID to upload tokenizer")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face token for uploading to Hub")
    parser.add_argument("--private", action="store_true", default=True,
                        help="Whether Hub repo should be private")
    parser.add_argument("--byte_fallback", action="store_true", default=True,
                        help="Whether to use byte fallback")
    
    args = parser.parse_args()

    logger.info("Training a tokenizer!")
    main(args)
    logger.info("Tokenizer trained successfully!")
