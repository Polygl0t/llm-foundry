"""
Llama-Compatible Tokenizer Training with SentencePiece

This script trains a SentencePiece-based tokenizer compatible with Llama architectures.
Supports both BPE and Unigram algorithms with extensive normalization options for
multilingual and code-aware tokenization.

Algorithm Options:
- BPE: Byte-Pair Encoding (good for code and multilingual)
- Unigram: Probabilistic subword segmentation (better for Asian languages)

Example usage:
    python train_tokenizer_sentencepiece.py \
        --train_dataset_dir data/corpus \
        --dataset_type jsonl \
        --text_column text \
        --vocab_size 32000 \
        --model_type bpe \
        --output_dir checkpoints/llama_tokenizer \
        --add_bos_token --add_eos_token \
        --num_samples 10000000 \
        --tokenizer_name username/llama-tokenizer \
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
import os
import json
import glob
import argparse
from tqdm import tqdm
import sentencepiece as spm
from datasets import load_dataset
from huggingface_hub import create_repo, HfApi
from transformers import LlamaTokenizerFast, LlamaTokenizer, AutoTokenizer

# A general list of special tokens
SPECIAL_TOKENS = [
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

    # Check if the dataset file exists. Path to the dataset file. The SentencePieceTrainer requires a .txt file as input.
    if not os.path.exists(args.dataset_file):

        # Load the dataset from the huggingface Hub and prepare it for training.
        if args.train_dataset_dir is not None:

            # Load the datasets from disk
            assert args.dataset_type in ['txt', 'jsonl', 'parquet', 'csv'], f"Invalid data type: {args.dataset_type}. Needs to be one of ['txt', 'jsonl', 'parquet', 'csv']"
            data_files = glob.glob(f"{args.train_dataset_dir}/*{args.dataset_type}")
            assert len(data_files) > 0, f"No {args.dataset_type.upper()} files found in '{args.train_dataset_dir}'."

            args.dataset_type = "text" if args.dataset_type == "txt" else "json" if args.dataset_type == "jsonl" else args.dataset_type

            dataset = load_dataset(
                args.dataset_type,
                data_files=data_files, 
                split="train",
                cache_dir=args.cache_dir,
                num_proc=len(data_files),
            )
            print(f"Loaded dataset with {len(dataset):,} examples from {args.dataset_type.upper()} files.\n{dataset}")
        else:
            raise ValueError("No dataset directory provided. Please provide a dataset directory to train the tokenizer.")

        dataset = dataset.remove_columns([col for col in dataset.column_names if col != args.text_column])

        dataset = dataset.shuffle(seed=args.seed)
        if args.num_samples is not None:
            dataset = dataset.select(range(args.num_samples))
        
        print("Number of samples selected from the dataset:", len(dataset))

        with open(args.dataset_file, "w", encoding="utf-8") as f:
            for example in tqdm(dataset):
                f.write(example["text"] + "\n")
        print("Dataset file created:", args.dataset_file)
        
    else:
        print("Dataset file already exists. Skipping the dataset preparation step.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print("Training the tokenizer...")

    # Learn more about the arguments of `SentencePieceTrainer` in [here](https://github.com/google/sentencepiece/blob/master/doc/options.md).
    spm.SentencePieceTrainer.Train(
        input=args.dataset_file,
        input_format="text",                                    # This script is designed to work with plain text files. However, the SentencePieceTrainer also supports other formats, like `tsv`
        num_threads=args.num_threads,                           # Speed up your training by using more threads.
        model_prefix=f'{args.output_dir}/spm_tokenizer',        # You can use `model_prefix` to specify where the model files will be saved.
        vocab_size=args.vocab_size - (len(SPECIAL_TOKENS) + 1), # Ideally, your tokenizer should have a vocab size of `vocab_size + len(SPECIAL_TOKENS) + 1` (for the "<|pad|>" token) that results in a vocab size that is a multiple of 2.
        unk_id=0,                                               # ID for unknown token
        unk_piece=args.unk_token,                               # Unknown token
        bos_id=1,                                               # ID for beginning-of-sentence token
        bos_piece=args.bos_token,                               # Beginning-of-sentence token
        eos_id=2,                                               # ID for end-of-sentence token
        eos_piece=args.eos_token,                               # End-of-sentence token
        model_type=args.model_type,                             # The type of the tokenizer model. You can choose between `bpe`, `unigram`, `word`, and `char`.
        normalization_rule_name="nmt_nfkc",                     # Learn more about the normalization options [here](https://github.com/google/sentencepiece/blob/master/doc/normalization.md) 
        byte_fallback=True,                                     # decompose unknown pieces into UTF-8 byte pieces
        split_by_unicode_script=True,                           # Use Unicode script to split sentence pieces
        split_by_number=True,                                   # Split tokens by numbers (0-9)
        split_digits=True,                                      # Split all digits (0-9) into separate pieces
        split_by_whitespace=True,                               # Use a white space to split sentence pieces
        add_dummy_prefix=True,                                  # Whether to add a space to the first word if there isn't already one. This lets us treat "hello" exactly like "say hello".
        allow_whitespace_only_pieces=False,                     # We are setting this to false because we have already manually handled whitespace with our `SPECIAL_TOKENS`
        remove_extra_whitespaces=True,                          # Removes leading, trailing, and duplicate internal whitespace
        train_extremely_large_corpus=True if args.model_type == "unigram" else False, # Set this to True when training a unigram tokenizer on a large corpus to avoid SEGFAULTs
    )

    # Get a slow Llama tokenizer
    tokenizer = LlamaTokenizer(
        os.path.join(args.output_dir, "spm_tokenizer.model"),
        bos_token=args.bos_token,
        unk_token=args.unk_token,
        eos_token=args.eos_token,
        # Chosing padding and truncation sides dependes on your use case.
        # For example, padding from the left means that the model will see the end of the sequence 
        # while the beginning is padded. This is useful for causal language modeling tasks. On the other hand,
        # padding from the right means that the model will see the beginning of the sequence while the end is padded.
        # This is useful for sequence classification tasks. In terms of truncation, a similar logic applies.
        # See more details [here](https://discuss.huggingface.co/t/the-effect-of-padding-side/67188).
        padding_side=args.padding_side,
        truncation_side=args.truncation_side,
        add_bos_token=args.add_bos_token,
        add_eos_token=args.add_eos_token,
        clean_up_tokenization_spaces=args.clean_up_tokenization_spaces,
        add_prefix_space=args.add_prefix_space,
        legacy=False,
    )

    # Add the "<|pad|>" token. Why? Read [this](https://huggingface.co/docs/transformers/main/model_doc/llama2#usage-tips).
    tokenizer.add_special_tokens({"pad_token":args.pad_token})
    tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})

    print("LlamaTokenizer vocab size:", len(tokenizer))
    assert len(tokenizer) == args.vocab_size, f"Tokenizer vocab size {len(tokenizer)} does not match the expected vocab size {args.vocab_size}."

    tokenizer.save_pretrained(args.output_dir)

    # Get a fast Llama tokenizer
    tokenizer = LlamaTokenizerFast(
        os.path.join(args.output_dir, "tokenizer.model"),
        bos_token=args.bos_token,
        unk_token=args.unk_token,
        eos_token=args.eos_token,
        padding_side=args.padding_side,
        truncation_side=args.truncation_side,
        add_bos_token=args.add_bos_token,
        add_eos_token=args.add_eos_token,
        clean_up_tokenization_spaces=args.clean_up_tokenization_spaces,
        add_prefix_space=args.add_prefix_space,
        legacy=False,
    )
    tokenizer.add_special_tokens({"pad_token":args.pad_token})
    tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
    print("LlamaTokenizerFast vocab size:", len(tokenizer))
    assert len(tokenizer) == args.vocab_size, f"Tokenizer vocab size {len(tokenizer)} does not match the expected vocab size {args.vocab_size}."

    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "tokenizer_config.json"), "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)
    
    tokenizer_config['legacy'] = False
    tokenizer_config['bos_token_id'] = tokenizer.bos_token_id
    tokenizer_config['eos_token_id'] = tokenizer.eos_token_id
    tokenizer_config['pad_token_id'] = tokenizer.pad_token_id
    tokenizer_config['unk_token_id'] = tokenizer.unk_token_id
    tokenizer_config['padding_side'] = args.padding_side
    tokenizer_config['truncation_side'] = args.truncation_side
    tokenizer_config['add_prefix_space'] = args.add_prefix_space

    with open(os.path.join(args.output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=4)

    assert AutoTokenizer.from_pretrained(args.output_dir, use_fast=False)
    assert AutoTokenizer.from_pretrained(args.output_dir, use_fast=True)

    print("Tokenizer trained and saved to:", args.output_dir)

    # Push the folder to the hub.
    if args.tokenizer_name is not None and args.token is not None:
        print("Pushing the tokenizer to the hub...")
        create_repo(
            repo_id=args.tokenizer_name, 
            token=args.token,
            repo_type="model",
            exist_ok=True,
            private=True
        )

        api = HfApi(token=args.token)

        api.upload_folder(
            repo_id=args.tokenizer_name,
            folder_path=args.output_dir,
        )

        print("Tokenizer uploaded to the hub.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Llama tokenizer using SentencePiece")
    parser.add_argument("--dataset_file", type=str, default=None, help="The path to the dataset file")
    parser.add_argument("--train_dataset_dir", type=str, default=None, help="The path to the dataset directory")
    parser.add_argument("--dataset_type", type=str, default="jsonl", help="The type of the dataset files (e.g., 'jsonl', 'parquet', 'csv', 'txt')")
    parser.add_argument("--text_column", type=str, default=None, help="The name of the text column in the dataset")
    parser.add_argument("--num_threads", type=int, default=16, help="The number of threads to use for training the tokenizer")
    parser.add_argument("--output_dir", type=str, default=None, help="The output directory to save the tokenizer")
    parser.add_argument("--seed", type=int, default=None, help="The random seed used when shuffling the dataset")
    parser.add_argument("--token", type=str, default=None, help="The token to access the dataset on the hub")
    parser.add_argument("--cache_dir", type=str, default=None, help="The directory to cache the dataset")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use from the dataset. You might run into memory issues if you use too many samples.")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size to use for the tokenizer")
    parser.add_argument("--model_type", type=str, default="bpe", choices=["bpe", "unigram", "word", "char"], help="The type of the tokenizer model to use")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Name of the tokenizer to be uploaded to the hub")
    parser.add_argument("--unk_token", type=str, default="<|unk|>", help="The unknown token to use for the tokenizer")
    parser.add_argument("--bos_token", type=str, default="<|im_start|>", help="The beginning of sentence token to use for the tokenizer")
    parser.add_argument("--eos_token", type=str, default="<|im_end|>", help="The end of sentence token to use for the tokenizer")
    parser.add_argument("--pad_token", type=str, default="<|pad|>", help="The padding token to use for the tokenizer")
    parser.add_argument("--add_bos_token", action='store_true', help="Whether to add a BOS token during tokenization")
    parser.add_argument("--add_eos_token", action='store_true', help="Whether to add an EOS token during tokenization")
    parser.add_argument("--clean_up_tokenization_spaces", action='store_true', help="Whether to clean up tokenization spaces")
    parser.add_argument("--add_prefix_space", action='store_true', help="Whether to add a prefix space to the input")
    parser.add_argument("--padding_side", type=str, default="right", choices=["left", "right"], help="The side to use for padding")
    parser.add_argument("--truncation_side", type=str, default="right", choices=["left", "right"], help="The side to use for truncation")
    
    args = parser.parse_args()

    print("Training a tokenizer! 🚀")
    main(args)
    print("Tokenizer trained successfully! 🎉")
    