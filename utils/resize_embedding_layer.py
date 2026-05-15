"""
Check for size mismatches between a tokenizer and a model's embedding layer,
and optionally resize the embeddings to match the tokenizer vocabulary.

Useful when adapting models to new tokenizers that may have different vocab sizes,
e.g., after running tokensurgeon.

How to Use:
    python resize_embedding_layer.py <model_path> [options]

Examples:
    # Check for mismatch only
    python resize_embedding_layer.py /path/to/model

    # Check and resize, saving to a custom output directory
    python resize_embedding_layer.py /path/to/model --resize -o /path/to/output

    # Resize with vocab padded to the nearest multiple of 64 (for hardware efficiency)
    python resize_embedding_layer.py /path/to/model --resize --pad-to-multiple-of 64
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_path: str, dtype: str = "bfloat16") -> Tuple[Any, Any]:
    torch_dtype = getattr(torch, dtype)

    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Loading model from {model_path} (dtype={dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="cpu",
    )

    return tokenizer, model


def get_sizes(tokenizer: Any, model: Any) -> Tuple[int, int]:
    try:
        tokenizer_size = len(tokenizer)
    except Exception:
        tokenizer_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer.get_vocab())

    emb = model.get_input_embeddings()
    if emb is None:
        raise ValueError("Model has no input embeddings (get_input_embeddings() returned None).")

    embed_size = emb.weight.shape[0]
    return tokenizer_size, embed_size


def report_mismatch(
    tokenizer: Any,
    tokenizer_size: int,
    embed_size: int,
    save_missing: bool,
    out_dir: Path,
) -> None:
    if tokenizer_size > embed_size:
        missing_ids = list(range(embed_size, tokenizer_size))
        missing_tokens = tokenizer.convert_ids_to_tokens(missing_ids)

        print(f"\nTokenizer has {tokenizer_size} ids but embedding has {embed_size} rows.")
        print(f"Tokens without embeddings: {len(missing_tokens)}")
        print("\nShowing first 20 examples:")
        for i, t in enumerate(missing_tokens[:20]):
            print(f"  id={missing_ids[i]:6d}  token={t!r}")

        if save_missing:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "tokens_without_embeddings.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"ids": missing_ids, "tokens": missing_tokens, "count": len(missing_tokens)},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"\nWrote full list to {out_path}")

    else:
        extra_embed_ids = list(range(tokenizer_size, embed_size))

        print(f"\nEmbedding has {embed_size} rows but tokenizer has {tokenizer_size} ids.")
        print(f"Embedding rows without tokenizer tokens: {len(extra_embed_ids)}")
        print("\nShowing first 20 examples:")
        for eid in extra_embed_ids[:20]:
            print(f"  embed_id={eid}")

        if save_missing:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "embeddings_without_tokens.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"embed_ids": extra_embed_ids, "count": len(extra_embed_ids)},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"\nWrote full list to {out_path}")


def resize_and_save(
    model: Any,
    tokenizer: Any,
    tokenizer_size: int,
    embed_size: int,
    out_dir: Path,
    pad_to_multiple_of: int | None = None,
) -> None:
    print(f"\nResizing model embeddings from {embed_size} -> {tokenizer_size} ...")

    # resize_token_embeddings also updates the tied LM head, if present.
    model.resize_token_embeddings(tokenizer_size, pad_to_multiple_of=pad_to_multiple_of)

    new_embed_size = model.get_input_embeddings().weight.shape[0]
    if pad_to_multiple_of:
        print(f"Successfully resized to {new_embed_size} rows (padded to multiple of {pad_to_multiple_of})")
    else:
        print(f"Successfully resized to {new_embed_size} rows")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving resized model and tokenizer to {out_dir} ...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Done.")


def main(args: argparse.Namespace) -> int:

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.dtype)
    tokenizer_size, embed_size = get_sizes(tokenizer, model)

    print(f"\ntokenizer_size = {tokenizer_size}")
    print(f"embedding_rows = {embed_size}")

    if tokenizer_size == embed_size:
        print("\n✓ Sizes match: every tokenizer id has a corresponding embedding row.")
        sys.exit(0)

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.model_path) / "model_with_resized_embedding"

    report_mismatch(tokenizer, tokenizer_size, embed_size, args.save_missing, out_dir)

    if args.resize:
        resize_and_save(model, tokenizer, tokenizer_size, embed_size, out_dir, args.pad_to_multiple_of)
    else:
        print("\nSkipping resize (use --resize to resize and save the model)")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model directory containing model and tokenizer files",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory for the resized model (default: MODEL_PATH/model_with_resized_embedding)",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help="Resize the model embeddings and save (default: only check, don't resize)",
    )
    parser.add_argument(
        "--pad-to-multiple-of",
        type=int,
        default=None,
        metavar="N",
        help="Pad the resized vocabulary to the nearest multiple of N (e.g. 64) for hardware efficiency",
    )
    parser.add_argument(
        "--save-missing",
        action="store_true",
        help="Save the list of mismatched tokens/embeddings to a JSON file in the output directory",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type to load the model in (default: bfloat16)",
    )

    args = parser.parse_args()
    main(args)
