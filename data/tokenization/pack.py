"""
Dataset Packing Script

Packs a pre-tokenized dataset into fixed-length chunks using one of two strategies:

  concatenate
      Concatenates all token sequences end-to-end and splits the result into
      non-overlapping blocks of exactly `block_size` tokens.  No padding is added;
      any trailing tokens that do not fill a complete block are discarded.

  bfd  (Best-Fit Decreasing)
      Sorts sequences by length (longest first) and greedily assigns each one to
      the existing chunk that will leave the least remaining space while still
      fitting the sequence.  A new chunk is opened only when no existing chunk has
      enough room.  Chunks that are not completely full at the end are padded to
      `block_size` with per-column pad values.

Both strategies detect and pack the following columns when present in the dataset:
  input_ids, labels, attention_mask, assistant_masks

The output always contains a `seq_lengths` column whose value equals `block_size`
for every row (since every packed chunk is exactly `block_size` tokens long after
padding/discarding).

Padding values used by the BFD strategy:
  input_ids      -> --pad_token_id  (required for bfd)
  labels         -> -100
  attention_mask -> 0
  assistant_masks-> 0

Usage:
(Concatenation)
    python pack.py \\
        --input_path  data/data_tokenized \\
        --output_dir  data/data_packed \\
        --strategy    concatenate \\
        --block_size  4096

(Best-Fit Decreasing)
    python pack.py \\
        --input_path   data/data_tokenized \\
        --output_dir   data/data_packed \\
        --strategy     bfd \\
        --block_size   4096 \\
        --pad_token_id 0
"""
import argparse
from utils import DatasetLoader, get_logger, save_dataset, save_metadata

logger = get_logger("Pack")

# Ordered list of columns that carry per-token data and should be packed.
# Columns absent from the dataset are silently skipped.
_PACKABLE_COLUMNS = ["input_ids", "labels", "attention_mask", "assistant_masks"]

# Default pad value for each packable column (used by BFD strategy).
_DEFAULT_PAD = {
    "input_ids": None,   # overridden by --pad_token_id
    "labels": -100,
    "attention_mask": 0,
    "assistant_masks": 0,
}

# Strategy: concatenate
def create_concatenate_function(block_size, columns):
    """Return a batched map function that concatenates tokens and splits into blocks.

    Any tokens at the end that do not fill a complete block are discarded.
    No padding is applied.
    """
    def pack(examples):
        # Flatten every packable column across the batch into one long sequence.
        concat = {col: [tok for seq in examples[col] for tok in seq] for col in columns}
        total = len(concat[columns[0]])

        n_blocks = total // block_size
        usable = n_blocks * block_size

        result = {
            col: [concat[col][i: i + block_size] for i in range(0, usable, block_size)]
            for col in columns
        }
        result["seq_lengths"] = [block_size] * n_blocks
        return result

    return pack


# Strategy: BFD (Best-Fit Decreasing)
def create_bfd_function(block_size, columns, pad_values):
    """Return a batched map function that packs using Best-Fit Decreasing.

    Sequences longer than `block_size` are silently discarded.
    Partially filled chunks are padded to `block_size` using `pad_values`.
    """
    def pack(examples):
        # Determine per-sequence lengths.
        if "seq_lengths" in examples:
            lengths = examples["seq_lengths"]
        else:
            lengths = [len(seq) for seq in examples["input_ids"]]

        # Collect valid sequences (non-empty, fit within block_size).
        sequences = []
        for i, length in enumerate(lengths):
            if 0 < length <= block_size:
                entry = {"len": length}
                for col in columns:
                    if len(examples[col][i]) < length:
                        raise ValueError(
                            f"Column '{col}' is shorter than seq_lengths for row {i}."
                        )
                    entry[col] = list(examples[col][i][:length])
                sequences.append(entry)

        # Sort longest-first (Best-Fit Decreasing).
        sequences.sort(key=lambda s: s["len"], reverse=True)

        out = {col: [] for col in columns}
        out["seq_lengths"] = []
        partial_chunks = []  # each is a dict with "len" + one list per column

        for seq in sequences:
            L = seq["len"]

            # Find the partial chunk with the least leftover space that still fits L.
            best_idx, best_leftover = None, block_size + 1
            for idx, ch in enumerate(partial_chunks):
                space = block_size - ch["len"]
                if L <= space:
                    leftover = space - L
                    if leftover < best_leftover:
                        best_leftover = leftover
                        best_idx = idx

            if best_idx is None:
                if L == block_size:
                    # Exact fit — emit immediately without buffering.
                    for col in columns:
                        out[col].append(seq[col])
                    out["seq_lengths"].append(block_size)
                else:
                    # Open a new partial chunk.
                    new_chunk = {"len": L}
                    for col in columns:
                        new_chunk[col] = seq[col][:]
                    partial_chunks.append(new_chunk)
            else:
                ch = partial_chunks[best_idx]
                for col in columns:
                    ch[col].extend(seq[col])
                ch["len"] += L

                if ch["len"] == block_size:
                    # Chunk is exactly full — emit and remove from partial list.
                    for col in columns:
                        out[col].append(ch[col])
                    out["seq_lengths"].append(block_size)
                    partial_chunks.pop(best_idx)

        # Pad and emit any remaining partial chunks.
        for ch in partial_chunks:
            pad_len = block_size - ch["len"]
            for col in columns:
                ch[col].extend([pad_values[col]] * pad_len)
                out[col].append(ch[col])
            out["seq_lengths"].append(block_size)

        return out

    return pack


def main(args):
    # Load dataset
    loader = DatasetLoader(
        path=args.input_path,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    dataset = loader.load()
    logger.info(f"Loaded dataset: {len(dataset):,} examples.\n{dataset}")

    # Identify columns to pack (preserve declaration order).
    columns = [col for col in _PACKABLE_COLUMNS if col in dataset.column_names]
    if "input_ids" not in columns:
        raise ValueError("The dataset must contain an 'input_ids' column.")
    logger.info(f"Columns to pack: {columns}")

    # Build the packing function.
    if args.strategy == "bfd":
        if args.pad_token_id is None:
            raise ValueError("--pad_token_id is required when using the 'bfd' strategy.")

        pad_values = dict(_DEFAULT_PAD)
        pad_values["input_ids"] = args.pad_token_id

        pack_fn = create_bfd_function(args.block_size, columns, pad_values)
        desc = f"Packing with BFD (block_size={args.block_size:,})"

    else:  # concatenate
        pack_fn = create_concatenate_function(args.block_size, columns)
        desc = f"Packing with concatenation (block_size={args.block_size:,})"

    # Apply packing.
    # `remove_columns=dataset.column_names` drops all original columns; the
    # pack function's return value provides the new columns.
    dataset = dataset.map(
        pack_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc=desc,
        num_proc=args.num_proc,
        load_from_cache_file=True,
    )

    sample_count = len(dataset)
    token_count = sample_count * args.block_size
    logger.info(f"Samples after packing: {sample_count:,} | Tokens: {token_count:,}")

    if sample_count == 0:
        logger.warning("No samples after packing. Nothing saved.")
        return

    # Truncate to max_tokens if specified.
    if args.max_tokens is not None and token_count > args.max_tokens:
        max_rows = args.max_tokens // args.block_size
        actual_tokens = max_rows * args.block_size
        logger.info(
            f"Truncating to {max_rows:,} samples (~{actual_tokens:,} tokens) "
            f"to stay within max_tokens={args.max_tokens:,}."
        )
        dataset = dataset.select(range(max_rows))
        sample_count = max_rows
        token_count = actual_tokens

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
        strategy=args.strategy,
        block_size=args.block_size,
        packed_columns=",".join(columns),
    )
    logger.info("Packing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O stuff
    io = parser.add_argument_group("Input / Output")
    io.add_argument(
        "--input_path", required=True,
        help="Tokenized dataset source: local file, local directory, or HuggingFace Hub id.",
    )
    io.add_argument(
        "--output_dir", required=True,
        help="Directory to write the packed dataset into.",
    )
    io.add_argument(
        "--output_type", choices=["parquet", "jsonl"], default="parquet",
        help="Output file format.",
    )

    # Packing
    pack = parser.add_argument_group("Packing")
    pack.add_argument(
        "--strategy", choices=["concatenate", "bfd"], required=True,
        help=(
            "'concatenate': concatenate all tokens and split into blocks (pretraining). "
            "'bfd': Best-Fit Decreasing bin-packing with padding (SFT)."
        ),
    )
    pack.add_argument(
        "--block_size", type=int, required=True,
        help="Target sequence length for every packed chunk.",
    )
    pack.add_argument(
        "--pad_token_id", type=int, default=None,
        help="Token ID used to pad partial chunks. Required for the 'bfd' strategy.",
    )

    # Limits
    lim = parser.add_argument_group("Limits")
    lim.add_argument(
        "--max_tokens", type=int, default=None,
        help="Truncate the packed output to at most this many tokens in total.",
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
        help="Cache directory for HuggingFace datasets.",
    )
    perf.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for dataset shuffling before packing (disabled when not set).",
    )

    args = parser.parse_args()
    logger.info("Starting dataset packing...")
    main(args)
