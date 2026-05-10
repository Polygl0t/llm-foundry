"""
Dataset Tokenization Script

Tokenizes datasets for both pretraining (causal LM) and supervised fine-tuning (SFT).
Supports standard text tokenization and chat-template formatting.

Output columns (all configurable via flags):
  input_ids         Token IDs (always included).
  seq_lengths       Length of each sequence              (--return_seq_lengths).
  attention_mask    All-ones mask over real tokens        (--return_attention_mask).
  labels            Token IDs for loss computation;
                    non-assistant positions masked to -100
                    when assistant masks are available    (--return_labels).
  assistant_masks   Binary per-token mask (1 = assistant) (--return_assistant_masks);
                    requires --apply_chat_template.

Usage:
Pretraining tokenization:
    python tokenize.py \
        --input_path  data/pretrain_raw \
        --output_dir  data/pretrain_tokenized \
        --tokenizer_name Qwen/Qwen3-0.6B \
        --add_bos_token --add_eos_token \
        --return_seq_lengths \
        --max_length 8192

SFT tokenization:
    python tokenize.py \
        --input_path  data/sft_raw \
        --output_dir  data/sft_tokenized \
        --tokenizer_name Qwen/Qwen3-0.6B \
        --text_column messages \
        --apply_chat_template \
        --return_seq_lengths \
        --return_labels \
        --return_assistant_masks
"""
import argparse
from transformers import AutoTokenizer
from utils import DatasetLoader, get_logger, save_dataset, save_metadata

logger = get_logger("Tokenize")


def load_tokenizer(args):
    """Load a HuggingFace tokenizer with optional chat template support."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        cache_dir=args.cache_dir,
        token=args.token,
        use_fast=True,
    )

    if args.apply_chat_template:
        if args.chat_template_path:
            with open(args.chat_template_path) as f:
                tokenizer.chat_template = f.read()
        elif tokenizer.chat_template is None:
            raise ValueError(
                "The tokenizer has no chat template. "
                "Provide --chat_template_path or use a tokenizer that already defines one."
            )

    if args.return_assistant_masks and not args.apply_chat_template:
        raise ValueError("--return_assistant_masks requires --apply_chat_template.")

    if args.apply_chat_template and (args.add_bos_token or args.add_eos_token):
        raise ValueError(
            "--add_bos_token / --add_eos_token must not be combined with "
            "--apply_chat_template. The chat template is responsible for all "
            "special tokens; adding extra BOS/EOS would corrupt the formatting "
            "and could mask assistant-ending EOS tokens out of the loss."
        )

    return tokenizer


def create_tokenize_function(tokenizer, args):
    """Return a batched tokenization function configured from *args*.

    `seq_lengths` is always produced so that downstream filtering and token
    counting work regardless of whether the user requests it in the output.
    It is removed from the final dataset if `--return_seq_lengths` is not set.
    """
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    want_assistant_masks = args.return_assistant_masks and args.apply_chat_template

    def tokenize(examples):
        if args.apply_chat_template:
            ids = tokenizer.apply_chat_template(
                examples[args.text_column],
                return_assistant_tokens_mask=want_assistant_masks,
                return_dict=True,
                add_generation_prompt=False,
            )
            input_ids = ids["input_ids"]
            assistant_masks = ids.get("assistant_masks") if want_assistant_masks else None

            # The chat template is solely responsible for BOS/EOS placement.
            # We never inject extra special tokens here — doing so would risk
            # masking the assistant's terminal EOS out of the loss and could
            # break the template's expected formatting.

            if assistant_masks is not None:
                for seq, mask in zip(input_ids, assistant_masks):
                    if len(seq) != len(mask):
                        raise ValueError("assistant_masks must have the same length as input_ids.")

        else:
            # Standard tokenization (no chat template).
            raw = tokenizer(
                examples[args.text_column],
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )
            input_ids = raw["input_ids"]
            for i, seq in enumerate(input_ids):
                prefix = [bos_id] if (args.add_bos_token and bos_id is not None) else []
                suffix = [eos_id] if (args.add_eos_token and eos_id is not None) else []
                input_ids[i] = prefix + seq + suffix

        result = {
            "input_ids": input_ids,
            # Always produced internally; dropped later if not requested.
            "seq_lengths": [len(seq) for seq in input_ids],
        }

        if args.return_attention_mask:
            result["attention_mask"] = [[1] * len(seq) for seq in input_ids]

        if want_assistant_masks:
            result["assistant_masks"] = assistant_masks

        if args.return_labels:
            if want_assistant_masks:
                result["labels"] = [
                    [t if m == 1 else -100 for t, m in zip(seq, mask)]
                    for seq, mask in zip(input_ids, assistant_masks)
                ]
            else:
                # No assistant mask available — treat every token as a label.
                result["labels"] = [seq[:] for seq in input_ids]

        return result

    return tokenize


def main(args):
    
    # Load dataset
    loader = DatasetLoader(
        path=args.input_path,
        cache_dir=args.cache_dir,
        seed=args.seed,
        split=args.split,
        subset=args.subset,
    )
    dataset = loader.load()
    logger.info(f"Loaded dataset: {len(dataset):,} examples.\n{dataset}")

    # Validate text column
    if args.text_column not in dataset.column_names:
        raise ValueError(
            f"Column '{args.text_column}' not found in dataset. "
            f"Available columns: {dataset.column_names}"
        )

    # Load tokenizer
    tokenizer = load_tokenizer(args)

    # Tokenize
    tokenize_fn = create_tokenize_function(tokenizer, args)
    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
        num_proc=args.num_proc,
        load_from_cache_file=True,
    )

    # Filter by max_length if specified
    if args.max_length is not None:
        before = len(dataset)
        dataset = dataset.filter(
            lambda x: x["seq_lengths"] <= args.max_length,
            num_proc=args.num_proc,
            desc=f"Filtering sequences longer than {args.max_length}",
        )
        dropped = before - len(dataset)
        if dropped:
            logger.info(f"Dropped {dropped:,} sequences exceeding max_length={args.max_length}.")

    sample_count = len(dataset)
    token_count = int(sum(dataset["seq_lengths"]))
    logger.info(f"Samples: {sample_count:,} | Tokens: {token_count:,}")

    if sample_count == 0:
        logger.warning("No samples remaining after filtering. Nothing saved.")
        return

    # Truncate to max_tokens if specified
    if args.max_tokens is not None and token_count > args.max_tokens:
        cumulative, max_rows = 0, 0
        for length in dataset["seq_lengths"]:
            if cumulative + length > args.max_tokens:
                break
            cumulative += length
            max_rows += 1
        logger.info(
            f"Truncating to {max_rows:,} samples (~{cumulative:,} tokens) "
            f"to stay within max_tokens={args.max_tokens:,}."
        )
        dataset = dataset.select(range(max_rows))
        sample_count = max_rows
        token_count = cumulative

    # Drop seq_lengths from output if not requested by the user
    if not args.return_seq_lengths:
        dataset = dataset.remove_columns("seq_lengths")

    # Save
    n_chunks = save_dataset(
        dataset, args.output_dir, args.output_type, args.tokens_per_chunk, token_count
    )

    save_metadata(
        args.output_dir,
        samples=sample_count,
        tokens=token_count,
        tokens_per_chunk=token_count // max(n_chunks, 1),
        chunks=n_chunks,
        tokenizer=args.tokenizer_name,
        apply_chat_template=args.apply_chat_template,
        add_bos_token=args.add_bos_token,
        add_eos_token=args.add_eos_token,
        max_length=args.max_length,
    )
    logger.info("Tokenization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O Stuff
    io = parser.add_argument_group("Input / Output")
    io.add_argument(
        "--input_path", required=True,
        help="Dataset source: local file, local directory, or HuggingFace Hub id.",
    )
    io.add_argument(
        "--output_dir", required=True,
        help="Directory to write the tokenized dataset into.",
    )
    io.add_argument(
        "--output_type", choices=["parquet", "jsonl"], default="parquet",
        help="Output file format.",
    )
    io.add_argument(
        "--split", default="train",
        help="Dataset split to use when loading from HuggingFace Hub.",
    )
    io.add_argument(
        "--subset", default=None,
        help="Dataset subset / config name for HuggingFace Hub datasets.",
    )

    # Tokenizer
    tok = parser.add_argument_group("Tokenizer")
    tok.add_argument(
        "--tokenizer_name", required=True,
        help="Name or local path of the tokenizer.",
    )
    tok.add_argument(
        "--token", default=None,
        help="HuggingFace API token (for gated models).",
    )
    tok.add_argument(
        "--chat_template_path", default=None,
        help="Path to a Jinja2 chat-template file. Overrides the tokenizer's built-in template.",
    )

    # Text input
    text = parser.add_argument_group("Text Input")
    text.add_argument(
        "--text_column", default="text",
        help="Dataset column that contains the text or messages to tokenize.",
    )
    text.add_argument(
        "--apply_chat_template", action="store_true",
        help=(
            "Apply the tokenizer's chat template. "
            "Use when the text column holds a list of message dicts (SFT)."
        ),
    )

    # Special tokens
    sp = parser.add_argument_group("Special Tokens")
    sp.add_argument(
        "--add_bos_token", action="store_true",
        help="Prepend the BOS token to each sequence. Incompatible with --apply_chat_template.",
    )
    sp.add_argument(
        "--add_eos_token", action="store_true",
        help="Append the EOS token to each sequence. Incompatible with --apply_chat_template.",
    )

    # Output fields
    out = parser.add_argument_group("Output Fields")
    out.add_argument(
        "--return_seq_lengths", action="store_true",
        help="Include the 'seq_lengths' column in the saved dataset.",
    )
    out.add_argument(
        "--return_attention_mask", action="store_true",
        help="Include an 'attention_mask' column (all 1s over real tokens).",
    )
    out.add_argument(
        "--return_labels", action="store_true",
        help=(
            "Include a 'labels' column for loss computation. "
            "Non-assistant tokens are masked to -100 when --return_assistant_masks is also set."
        ),
    )
    out.add_argument(
        "--return_assistant_masks", action="store_true",
        help=(
            "Include an 'assistant_masks' column (1 for assistant tokens, 0 otherwise). "
            "Requires --apply_chat_template."
        ),
    )

    # Filtering / limits
    filt = parser.add_argument_group("Filtering / Limits")
    filt.add_argument(
        "--max_length", type=int, default=None,
        help="Discard sequences longer than this many tokens.",
    )
    filt.add_argument(
        "--max_tokens", type=int, default=None,
        help="Truncate the output dataset to at most this many tokens in total.",
    )

    # Performance / saving
    perf = parser.add_argument_group("Performance / Saving")
    perf.add_argument(
        "--num_proc", type=int, default=8,
        help="Number of parallel worker processes.",
    )
    perf.add_argument(
        "--tokens_per_chunk", type=int, default=300_000_000,
        help="Maximum number of tokens per output file.",
    )
    perf.add_argument(
        "--cache_dir", default=None,
        help="Cache directory for the tokenizer and HuggingFace datasets.",
    )
    perf.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for dataset shuffling (shuffling disabled when not set).",
    )

    args = parser.parse_args()
    logger.info("Starting tokenization...")
    main(args)
