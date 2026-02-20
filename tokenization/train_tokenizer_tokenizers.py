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

BUG: Tokens like "<think>" or "<tool_call>" should not be special tokens in the tokenizer.
        This is because, if they are special tokens, evals from the harness will not be able
        to strip them from the input (bad for reasoning models). Also, the tokenizer will always
        skip them when decoding, which means that they will not appear in the decoded output (bad for interpretability
        and the probable reason why the eval harness is not parsing them correctly).
        Only the BOS, EOS, UNK, and PAD tokens should be special tokens in the tokenizer. 
        The rest of the tokens should just be normal tokens in the tokenizer.

TODO: For now, I'm manually removing the special tokens from the tokenizer after training it. This is a very hacky, but it works. 
        In the future, we should consider implementing the addition of special tokens in a different way.

        - `special_tokens_map.json` should only contain the BOS, EOS, UNK, and PAD tokens.
        - `special_tokens_map.json` should have no add_special_tokens field. Also, all the non special tokens should be set with `"special": false`.
        - `tokenizer.json` all non special tokens should be set with `"special": false`. Only the BOS, EOS, UNK, and PAD tokens should be set with `"special": true`.
"""
from transformers import PreTrainedTokenizerFast, AutoTokenizer
import datasets
import json
import glob
import os
import argparse
import unicodedata

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

# A general list of special tokens
SPECIAL_TOKENS = [
        "<|im_start|>",
        "<|im_end|>",
        "<|pad|>",
        "<|unk|>",
        "<tools>",
        "</tools>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "<think>",
        "</think>",
        "<answer>",
        "</answer>",
        "<context>",
        "</context>",
        "<|fim_prefix|>",
        "<|fim_suffix|>",
        "<|fim_middle|>",
        "<|repo_name|>",
        "<|image|>",
        "<|image_pad|>",
        "<|image_placeholder|>",
        # The indented special tokens are a trick from the Olmo tokenizer.
        # (note: I think Pythia or GPT-neoX also did something like this) 
        # This helps make the tokenizer more efficient when dealing with code data.
        "                        ",
        "                       ",
        "                      ",
        "                     ",
        "                    ",
        "                   ",
        "                  ",
        "                 ",
        "                ",
        "               ",
        "              ",
        "             ",
        "            ",
        "           ",
        "          ",
        "         ",
        "        ",
        "       ",
        "      ",
        "     ",
        "    ",
        "   ",
        "  ",
    ]


def main(args):
    
    assert args.data_type in ['txt', 'jsonl', 'parquet', 'csv'], f"Invalid data type: {args.data_type}. Needs to be one of ['txt', 'jsonl', 'parquet', 'csv']"

    # check if the path is a directory
    if os.path.isdir(args.data_path):

        # Get all files of a specific type in the data directory
        data_files = glob.glob(os.path.join(args.data_path, f"*.{args.data_type}"))

        if not data_files:
            raise ValueError(f"No data files found in {args.data_path} with extension {args.data_type}")

    elif os.path.isfile(args.data_path):
        data_files = [args.data_path]

    else:
        raise ValueError(f"Invalid data path: {args.data_path}")

    data_type = "text" if args.data_type == "txt" else "json" if args.data_type == "jsonl" else args.data_type

    # Load your training data (supports txt, parquet, csv, and jsonl files.)
    dataset = datasets.load_dataset(
        data_type, 
        data_files=data_files, 
        split="train",
        cache_dir=args.cache_dir,
        num_proc=args.num_proc
    )

    # Pre-normalize the text to NFKC (helps to prevent mojibake issues)
    def normalize_to_nfkc(example):
        example[args.text_column] = unicodedata.normalize("NFKC", example[args.text_column])
        return example

    dataset = dataset.map(
         normalize_to_nfkc,
         num_proc=args.num_proc
    )

    print(f"Loaded dataset with {len(dataset)} samples from {data_files}")

    # Define a Model
    # [tokenizers.models.BPE](https://huggingface.co/docs/tokenizers/api/models#tokenizers.models.BPE)
    tokenizer = Tokenizer(models.BPE(byte_fallback=args.byte_fallback))
    
    # Define a normalizer.
    # [tokenizers.normalizers.NFC](https://huggingface.co/docs/tokenizers/api/normalizers#tokenizers.normalizers.NFC)
    tokenizer.normalizer = normalizers.NFC()

    # Define a pre-tokenizer.
    # [tokenizers.pre_tokenizers.ByteLevel](https://huggingface.co/docs/tokenizers/api/pre-tokenizers#tokenizers.pre_tokenizers.ByteLevel)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)
    
    # Define the model trainer
    # [tokenizers.trainers.BpeTrainer](https://huggingface.co/docs/tokenizers/api/trainers#tokenizers.trainers.BpeTrainer)
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=SPECIAL_TOKENS, show_progress=True)

    # Define a generator dor the BPE trainer
    def get_training_corpus(bs=args.batch_size):
        """
        Just a generator that will yield batches of text data.
        """
        for i in range(0, len(dataset), bs):
            yield dataset[i : i + bs][args.text_column]

    # Train !!! 🚀
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # Get the token IDs for the main special tokens (will use them to hardcode them in the config)
    bos_token_id = tokenizer.token_to_id(args.bos_token)
    eos_token_id = tokenizer.token_to_id(args.eos_token)
    pad_token_id = tokenizer.token_to_id(args.pad_token)
    unk_token_id = tokenizer.token_to_id(args.unk_token)

    # Define a post-processor
    if args.add_bos_token:
        # Apparently, this is the "correct" way to add a BOS token using the `tokenizers` library.
        # See: https://github.com/huggingface/tokenizers/issues/1643
        # [TemplateProcessing](https://huggingface.co/docs/tokenizers/en/api/post-processors#tokenizers.processors.TemplateProcessing)
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{args.bos_token} $A",
            special_tokens=[(args.bos_token, bos_token_id)],
        )
    
    else:
        # [tokenizers.processors.ByteLevel](https://huggingface.co/docs/tokenizers/api/post-processors#tokenizers.processors.ByteLevel)
        tokenizer.post_processor = processors.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)
        
    # Define a decoder
    # [tokenizers.decoders.ByteLevel](https://huggingface.co/docs/tokenizers/api/decoders#tokenizers.decoders.ByteLevel)
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)

    print("Wrapping the tokenizer with PreTrainedTokenizerFast")
    # Wrap the tokenizer
    # [PreTrainedTokenizerFast](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast)
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

    wrapped_tokenizer.additional_special_tokens = [
        token for token in SPECIAL_TOKENS if token not in 
        [
            args.bos_token, args.eos_token, args.pad_token, args.unk_token
        ]
    ]

    # Assert that the size of the tokenizer matches the expected vocabulary size
    assert wrapped_tokenizer.vocab_size == args.vocab_size, f"Expected vocab size {args.vocab_size}, but got {wrapped_tokenizer.vocab_size}"

    # Save the tokenizer
    if not os.path.exists(args.output_dir):
         os.makedirs(args.output_dir)
    wrapped_tokenizer.save_pretrained(args.output_dir)

    # Add the special tokens to the config
    with open(os.path.join(args.output_dir, "tokenizer_config.json"), "r") as f:
            tokenizer_config = json.load(f)

    tokenizer_config['legacy'] = False
    tokenizer_config['bos_token_id'] = bos_token_id
    tokenizer_config['eos_token_id'] = eos_token_id
    tokenizer_config['pad_token_id'] = pad_token_id
    tokenizer_config['unk_token_id'] = unk_token_id
    tokenizer_config['add_bos_token'] = args.add_bos_token
    tokenizer_config['add_eos_token'] = args.add_eos_token

    with open(os.path.join(args.output_dir, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f, indent=4)

    # Make sure you can load the tokenizer in any setting.
    assert AutoTokenizer.from_pretrained(args.output_dir, use_fast=False)
    assert AutoTokenizer.from_pretrained(args.output_dir, use_fast=True)

    # Upload your new tokenizer to the Hub!
    if args.hub_repo_id is not None and args.token is not None:
        print(f"Pushing tokenizer to Hugging Face Hub at {args.hub_repo_id}...")
        wrapped_tokenizer.push_to_hub(args.hub_repo_id, token=args.token, private=args.private)


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

    print("Training a tokenizer! 🚀")
    main(args)
    print("Tokenizer trained successfully! 🎉")
