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

When `--return_seq_lengths` is set, the output also includes a `seq_lengths`
column whose value equals `block_size` for every row (since every packed chunk
is exactly `block_size` tokens long after padding/discarding).

Padding values used by the BFD strategy:
  input_ids      -> --pad_token_id  (required for bfd)
  labels         -> -100
  attention_mask -> 0
  assistant_masks-> 0

Degenerate-sample guardrail (ON by default)
-------------------------------------------
The guardrail therefore drops, before/while packing:

  1. any *source* sequence that is degenerate, i.e. that
       - has no tokens at all, or
       - consists only of filler tokens (`--filler_token_ids`; a run of `<0x00>` or
         of pad tokens), or
       - has more than `--max_filler_fraction` of its tokens equal to a filler id, or
       - carries no trainable label (its `labels` are present and all -100);
  2. any emitted *block* that is still degenerate (a long filler run inside an
     otherwise valid document can line up with a block boundary, so filtering the
     sources alone is not enough to guarantee the invariant).

Both filters are on by default; disable with `--no-filter_filler`.

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

import numpy as np
from utils import DatasetLoader, get_logger, save_dataset, save_metadata

logger = get_logger("Pack")

# Ordered list of columns that carry per-token data and should be packed.
# Columns absent from the dataset are silently skipped.
_PACKABLE_COLUMNS = ["input_ids", "labels", "attention_mask", "assistant_masks"]

# Default pad value for each packable column (used by BFD strategy).
_DEFAULT_PAD = {
    "input_ids": None,  # overridden by --pad_token_id
    "labels": -100,
    "attention_mask": 0,
    "assistant_masks": 0,
}

# Token ids that carry no learning signal, so a sample made only of them is
# worthless (and produces a NaN loss for its batch). E.g., 3 is `<0x00>` -- the NUL byte
# in the Tucano2 tokenizer. `--pad_token_id` is added to this set at run
# time, since a run of pad tokens is equally uninformative.
DEFAULT_FILLER_TOKEN_IDS = (3,)

# HF causal-LM label value that means "not trained on".
_IGNORE_LABEL_ID = -100

# Guardrail counters. `dataset.map(num_proc=N)` runs `pack()` in worker processes,
# so these are only aggregate when num_proc == 1 (see `main`).
_FILTER_STATS = {"sequences": 0, "blocks": 0}


def make_degenerate_predicate(filler_token_ids, max_filler_fraction=1.0):
    """
    Build the predicate that decides whether a sample is degenerate (and dropped).

    A sample is degenerate when it is empty, when all (or, for a fraction below 1.0,
    at least that fraction) of its tokens are filler ids, or when it carries no
    trainable label (its `labels` are all `-100`).

    `max_filler_fraction` of 1.0 means "only samples made *entirely* of filler
    tokens" -- the case that actually breaks the loss.

    Returns `None` when `filler_token_ids` is empty, which disables the guardrail.
    """
    ids = tuple(sorted({int(i) for i in (filler_token_ids or ())}))
    if not ids:
        return None

    filler = np.asarray(ids, dtype=np.int64)
    single_filler = int(filler[0]) if filler.size == 1 else None
    threshold = float(max_filler_fraction)

    def is_degenerate(tokens, labels=None):
        if not tokens:
            return True

        # No trainable target at all. min/max avoids building an array for this.
        if labels and min(labels) == _IGNORE_LABEL_ID and max(labels) == _IGNORE_LABEL_ID:
            return True

        # Fast path for the default criterion (every token is the filler id).
        if threshold >= 1.0 and single_filler is not None:
            return min(tokens) == single_filler and max(tokens) == single_filler

        arr = np.asarray(tokens, dtype=np.int64)
        if single_filler is not None:
            n_filler = int(np.count_nonzero(arr == single_filler))
        else:
            n_filler = int(np.count_nonzero(np.isin(arr, filler)))
        return n_filler >= threshold * arr.size

    return is_degenerate


def _sequence_keep_mask(examples, columns, is_degenerate):
    """
    Row-level keep mask (True = keep) for the *input* examples, or None when the
    guardrail is disabled. Also updates the sequence drop counter.
    """
    if is_degenerate is None:
        return None

    labels = examples["labels"] if "labels" in columns else None
    keep = []
    for index, tokens in enumerate(examples["input_ids"]):
        row_labels = labels[index] if labels is not None else None
        if is_degenerate(tokens, row_labels):
            _FILTER_STATS["sequences"] += 1
            keep.append(False)
        else:
            keep.append(True)
    return keep


def _filter_examples(examples, columns, keep):
    """Apply a row-level keep mask to every packed column."""
    if keep is None:
        return examples
    return {col: [seq for seq, k in zip(examples[col], keep, strict=False) if k] for col in columns}


def _drop_degenerate_blocks(out, columns, is_degenerate_block):
    """
    Drop emitted blocks that are still degenerate, in place.

    This is the safety net that guarantees the output never contains a
    signal-free block, even when a filler run inside a valid document happens to
    line up with a block boundary. `is_degenerate_block` must use the strict
    criterion (all tokens / all labels), so padded BFD chunks are kept.
    """
    if is_degenerate_block is None or not out["input_ids"]:
        return

    labels = out["labels"] if "labels" in columns else None
    keep = []
    for index, block in enumerate(out["input_ids"]):
        row_labels = labels[index] if labels is not None else None
        keep.append(not is_degenerate_block(block, row_labels))

    dropped = len(keep) - int(sum(keep))
    if dropped == 0:
        return

    _FILTER_STATS["blocks"] += dropped
    logger.warning(
        "Dropped %d packed block(s) that were entirely filler tokens (no trainable signal).",
        dropped,
    )
    for col in columns:
        out[col] = [seq for seq, k in zip(out[col], keep, strict=False) if k]
    out["seq_lengths"] = [n for n, k in zip(out["seq_lengths"], keep, strict=False) if k]


# Strategy: concatenate
def create_concatenate_function(
    block_size,
    columns,
    filler_token_ids=DEFAULT_FILLER_TOKEN_IDS,
    max_filler_fraction=1.0,
):
    """Return a batched map function that concatenates tokens and splits into blocks.

    Any tokens at the end that do not fill a complete block are discarded.
    No padding is applied.

    Degenerate source sequences are dropped before concatenation (so a corrupted
    document cannot pollute the blocks around it), and degenerate blocks are dropped
    afterwards as a safety net. Pass `filler_token_ids=()` to disable both.
    """
    is_degenerate = make_degenerate_predicate(filler_token_ids, max_filler_fraction)
    is_degenerate_block = make_degenerate_predicate(filler_token_ids, 1.0)

    def pack(examples):
        # Guardrail 1: drop degenerate source sequences.
        keep = _sequence_keep_mask(examples, columns, is_degenerate)
        if keep is not None:
            examples = _filter_examples(examples, columns, keep)

        # Flatten every packable column across the batch into one long sequence.
        concat = {col: [tok for seq in examples[col] for tok in seq] for col in columns}
        total = len(concat[columns[0]])

        n_blocks = total // block_size
        usable = n_blocks * block_size

        result = {
            col: [concat[col][i : i + block_size] for i in range(0, usable, block_size)]
            for col in columns
        }
        result["seq_lengths"] = [block_size] * n_blocks

        # Guardrail 2: a filler run can still line up with a block boundary.
        _drop_degenerate_blocks(result, columns, is_degenerate_block)
        return result

    return pack


# Strategy: BFD (Best-Fit Decreasing)
def create_bfd_function(
    block_size,
    columns,
    pad_values,
    filler_token_ids=DEFAULT_FILLER_TOKEN_IDS,
    max_filler_fraction=1.0,
):
    """Return a batched map function that packs using Best-Fit Decreasing.

    Sequences longer than `block_size` are silently discarded.
    Partially filled chunks are padded to `block_size` using `pad_values`.

    Degenerate sequences (all filler tokens / no trainable label) are never
    scheduled into a chunk, and degenerate chunks are dropped as a safety net.
    Pass `filler_token_ids=()` to disable both.
    """
    is_degenerate = make_degenerate_predicate(filler_token_ids, max_filler_fraction)
    is_degenerate_block = make_degenerate_predicate(filler_token_ids, 1.0)

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
                        raise ValueError(f"Column '{col}' is shorter than seq_lengths for row {i}.")
                    entry[col] = list(examples[col][i][:length])

                # Guardrail 1: never schedule a signal-free sequence into a chunk.
                if is_degenerate is not None:
                    row_labels = entry["labels"] if "labels" in columns else None
                    if is_degenerate(entry["input_ids"], row_labels):
                        _FILTER_STATS["sequences"] += 1
                        continue

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
                if space >= L:
                    leftover = space - L
                    if leftover < best_leftover:
                        best_leftover = leftover
                        best_idx = idx

            if best_idx is None:
                if block_size == L:
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

        # Guardrail 2: drop any chunk that is still signal-free.
        _drop_degenerate_blocks(out, columns, is_degenerate_block)
        return out

    return pack


def main(args):
    # Counters are module level so the pack functions can update them; reset them
    # here because `main` may be called more than once in the same process.
    _FILTER_STATS["sequences"] = 0
    _FILTER_STATS["blocks"] = 0

    # Load dataset
    loader = DatasetLoader(
        path=args.input_path,
        cache_dir=args.cache_dir,
        seed=args.seed,
        num_proc=args.num_proc,
    )
    dataset = loader.load()
    logger.info(f"Loaded dataset: {len(dataset):,} examples.\n{dataset}")

    # Identify columns to pack (preserve declaration order).
    columns = [col for col in _PACKABLE_COLUMNS if col in dataset.column_names]
    if "input_ids" not in columns:
        raise ValueError("The dataset must contain an 'input_ids' column.")
    logger.info(f"Columns to pack: {columns}")

    # Resolve the degenerate-sample guardrail (see the module docstring).
    # `getattr` keeps programmatic callers that build an argparse.Namespace by hand
    # working; the guardrail is ON unless it is explicitly turned off.
    filter_filler = getattr(args, "filter_filler", True)
    max_filler_fraction = getattr(args, "max_filler_fraction", 1.0)
    filler_token_ids = set(getattr(args, "filler_token_ids", None) or DEFAULT_FILLER_TOKEN_IDS)
    if getattr(args, "pad_token_id", None) is not None:
        # A run of pad tokens carries no signal either.
        filler_token_ids.add(int(args.pad_token_id))
    if not filter_filler:
        filler_token_ids = set()

    if not 0.0 < max_filler_fraction <= 1.0:
        raise ValueError("--max_filler_fraction must be in (0, 1].")

    if filler_token_ids:
        logger.info(
            "Degenerate-sample guardrail: ON | filler token ids: %s | max filler fraction: %.2f",
            sorted(filler_token_ids),
            max_filler_fraction,
        )
    else:
        logger.warning(
            "Degenerate-sample guardrail: OFF. Signal-free samples (e.g. runs of NUL "
            "bytes) will be packed and will make the loss 'nan' for their batches."
        )

    # Build the packing function.
    if args.strategy == "bfd":
        if args.pad_token_id is None:
            raise ValueError("--pad_token_id is required when using the 'bfd' strategy.")

        pad_values = dict(_DEFAULT_PAD)
        pad_values["input_ids"] = args.pad_token_id

        pack_fn = create_bfd_function(
            args.block_size,
            columns,
            pad_values,
            filler_token_ids=filler_token_ids,
            max_filler_fraction=max_filler_fraction,
        )
        desc = f"Packing with BFD (block_size={args.block_size:,})"

    else:  # concatenate
        pack_fn = create_concatenate_function(
            args.block_size,
            columns,
            filler_token_ids=filler_token_ids,
            max_filler_fraction=max_filler_fraction,
        )
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

    # Report what the guardrail removed. With num_proc > 1 the counting happens in
    # the worker processes, so only the per-batch warnings (emitted for dropped
    # blocks) are reliable there.
    if filler_token_ids:
        if args.num_proc > 1:
            logger.info(
                "Degenerate-sample guardrail is active; drop counts are per worker and "
                "are not aggregated because num_proc=%d. Verify the output with "
                "tools/clean_packed_samples.py --verify.",
                args.num_proc,
            )
        else:
            logger.info(
                "Degenerate-sample guardrail dropped %d source sequence(s) and %d packed block(s).",
                _FILTER_STATS["sequences"],
                _FILTER_STATS["blocks"],
            )

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

    # Drop seq_lengths from output if not requested by the user.
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
        strategy=args.strategy,
        block_size=args.block_size,
        packed_columns=",".join(columns),
        filler_guardrail="on" if filler_token_ids else "off",
        filler_token_ids=",".join(str(i) for i in sorted(filler_token_ids)) or "none",
        max_filler_fraction=max_filler_fraction,
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
        "--input_path",
        required=True,
        help="Tokenized dataset source: local file, local directory, or HuggingFace Hub id.",
    )
    io.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write the packed dataset into.",
    )
    io.add_argument(
        "--output_type",
        choices=["parquet", "jsonl"],
        default="parquet",
        help="Output file format.",
    )

    # Packing
    pack = parser.add_argument_group("Packing")
    pack.add_argument(
        "--strategy",
        choices=["concatenate", "bfd"],
        required=True,
        help=(
            "'concatenate': concatenate all tokens and split into blocks (pretraining). "
            "'bfd': Best-Fit Decreasing bin-packing with padding (SFT)."
        ),
    )
    pack.add_argument(
        "--block_size",
        type=int,
        required=True,
        help="Target sequence length for every packed chunk.",
    )
    pack.add_argument(
        "--pad_token_id",
        type=int,
        default=None,
        help="Token ID used to pad partial chunks. Required for the 'bfd' strategy.",
    )
    pack.add_argument(
        "--return_seq_lengths",
        action="store_true",
        help="Include the 'seq_lengths' column in the saved dataset.",
    )

    # Degenerate-sample guardrail
    guard = parser.add_argument_group(
        "Degenerate-sample guardrail",
        "Drops samples that carry no learning signal: sequences/blocks made only of "
        "filler tokens (e.g. a run of NUL bytes) or with no trainable label.",
    )
    guard.add_argument(
        "--filter_filler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop degenerate source sequences and packed blocks. Use "
            "--no-filter_filler to reproduce the previous behaviour."
        ),
    )
    guard.add_argument(
        "--filler_token_ids",
        nargs="+",
        type=int,
        default=list(DEFAULT_FILLER_TOKEN_IDS),
        help=(
            "Token ids treated as filler (no learning signal); the default is the NUL "
            "byte token '<0x00>'. --pad_token_id is added to this set automatically."
        ),
    )
    guard.add_argument(
        "--max_filler_fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of filler tokens at or above which a *source* sequence is "
            "dropped. 1.0 keeps only the strict 'entirely filler' case; lower values "
            "clean more aggressively. Never applied to emitted blocks."
        ),
    )

    # Limits
    lim = parser.add_argument_group("Limits")
    lim.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Truncate the packed output to at most this many tokens in total.",
    )

    # Performance / saving
    perf = parser.add_argument_group("Performance / Saving")
    perf.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Number of parallel worker processes.",
    )
    perf.add_argument(
        "--tokens_per_chunk",
        type=int,
        default=300_000_000,
        help="Maximum number of tokens per output file.",
    )
    perf.add_argument(
        "--cache_dir",
        default=None,
        help="Cache directory for HuggingFace datasets.",
    )
    perf.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for dataset shuffling before packing (disabled when not set).",
    )

    args = parser.parse_args()
    logger.info("Starting dataset packing...")
    main(args)
