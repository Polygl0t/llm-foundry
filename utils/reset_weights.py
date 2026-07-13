"""
Reset non-attention weights in Llama and Qwen3.5 causal language models.

By default this keeps attention blocks untouched while re-initializing
embeddings, layer norms, MLPs, and any other non-attention modules via the
model's own `_init_weights` implementation. With `--embeddings_only`, only
the token embeddings (and the lm_head when untied) are re-initialized and
everything else is preserved.

Usage:
        python reset_weights.py --model Qwen/Qwen3.5-0.6B
        python reset_weights.py --model meta-llama/Llama-2-7b-hf --output_dir ./reset-model
        python reset_weights.py --model ./local-checkpoint --dry_run
        python reset_weights.py --model ./ckpt --seed 1337 --resize_vocab 49152 --embeddings_only --output_dir ./reset-model
"""

import argparse
from collections.abc import Iterable
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

SUPPORTED_MODEL_TYPES = {"llama", "qwen2", "qwen3", "qwen3_moe", "qwen3_5_text", "qwen3_5_moe_text"}
# RMSNorm classes whose `_init_weights` is a no-op and whose weight should be
# reset to 1.0 (forward computes `x * weight`, weight stored as ones).
# Qwen3_5RMSNorm / Qwen3_5MoeRMSNorm are intentionally excluded: their forward
# computes `x * (1.0 + weight)` with weight stored as zeros, and their model's
# `_init_weights` already handles them via `init.zeros_(module.weight)`.
RMSNORM_CLASS_NAMES = {"LlamaRMSNorm", "Qwen2RMSNorm", "Qwen3RMSNorm", "Qwen3MoeRMSNorm"}
ATTENTION_CLASS_NAMES = {
    "llama": {
        "LlamaAttention",
    },
    "qwen2": {"Qwen2Attention"},
    "qwen3": {"Qwen3Attention"},
    "qwen3_moe": {"Qwen3MoeAttention"},
    "qwen3_5_text": {
        "Qwen3_5Attention",
        "Qwen3_5GatedDeltaNet",
    },
    "qwen3_5_moe_text": {
        "Qwen3_5MoeAttention",
        "Qwen3_5MoeGatedDeltaNet",
    },
}


def resolve_dtype(dtype_name: str) -> torch.dtype | str:
    """Map CLI dtype choices to torch dtypes."""
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


def detect_model_type(model: AutoModelForCausalLM) -> str:
    """Detect whether the loaded model is a supported Llama or Qwen3.5 model."""
    model_type = getattr(model.config, "model_type", None)
    if model_type in SUPPORTED_MODEL_TYPES:
        return model_type

    architecture_names = set(getattr(model.config, "architectures", []) or [])
    if any("Llama" in name for name in architecture_names):
        return "llama"
    if any("Qwen3_5Moe" in name for name in architecture_names):
        return "qwen3_5_moe_text"
    if any("Qwen3_5" in name for name in architecture_names):
        return "qwen3_5_text"

    raise ValueError(
        f"Unsupported model_type={model_type!r}. Expected one of {sorted(SUPPORTED_MODEL_TYPES)}."
    )


def collect_attention_prefixes(model: AutoModelForCausalLM, model_type: str) -> set[str]:
    """Collect module-name prefixes for attention blocks that must be preserved."""
    attention_prefixes: set[str] = set()
    attention_class_names = ATTENTION_CLASS_NAMES[model_type]

    for name, module in model.named_modules():
        if type(module).__name__ in attention_class_names:
            attention_prefixes.add(name)

    if not attention_prefixes:
        raise ValueError(f"Could not find any attention blocks for model_type={model_type!r}.")

    return attention_prefixes


def is_inside_attention_block(module_name: str, attention_prefixes: Iterable[str]) -> bool:
    """Return True when a module is an attention block or nested inside one."""
    return any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in attention_prefixes
    )


def reset_non_attention_weights(
    model: AutoModelForCausalLM,
    *,
    dry_run: bool = False,
) -> tuple[list[str], list[str]]:
    """Reset all non-attention modules using the model's own weight initializer."""
    model_type = detect_model_type(model)
    attention_prefixes = collect_attention_prefixes(model, model_type)

    reset_modules: list[str] = []
    kept_modules: list[str] = []
    tie_word_embeddings = getattr(model.config, "tie_word_embeddings", False)

    for name, module in model.named_modules():
        if is_inside_attention_block(name, attention_prefixes):
            kept_modules.append(name)
            continue

        reset_modules.append(name)
        if not dry_run:
            # When embeddings are tied, lm_head shares weights with embed_tokens.
            # embed_tokens is visited first, so skip lm_head to avoid re-randomizing.
            if tie_word_embeddings and name == "lm_head":
                print(f"[Info]    Skipping tied module: {name}")
                continue
            model._init_weights(module)
            # Llama/Qwen2/Qwen3(_Moe) _init_weights skips RMSNorm; reset explicitly.
            # Qwen3_5(_Moe)RMSNorm is handled by their _init_weights (zeros_), and
            # its forward uses `(1 + weight)`, so we must NOT override here.
            if type(module).__name__ in RMSNORM_CLASS_NAMES:
                module.weight.data.fill_(1.0)
                if getattr(module, "bias", None) is not None:
                    module.bias.data.zero_()

    return reset_modules, kept_modules


def reset_embedding_weights(
    model: AutoModelForCausalLM,
    *,
    dry_run: bool = False,
) -> tuple[list[str], list[str]]:
    """Reset ONLY the token-embedding layers, keeping every other module.

    Re-initializes the input embeddings and, when `tie_word_embeddings` is
    False, the output embeddings (lm_head). With tied embeddings the lm_head
    shares its weight tensor with the input embedding, so resetting the input
    embedding covers both.
    """
    tie_word_embeddings = getattr(model.config, "tie_word_embeddings", False)
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    reset_modules: list[str] = []
    kept_modules: list[str] = []
    for name, module in model.named_modules():
        is_input = module is input_embeddings
        is_output = output_embeddings is not None and module is output_embeddings
        if not (is_input or is_output):
            kept_modules.append(name)
            continue
        if is_output and not is_input and tie_word_embeddings:
            # lm_head shares its weight with embed_tokens; resetting the
            # input embedding already re-initializes it.
            print(f"[Info]    Skipping tied module: {name}")
            kept_modules.append(name)
            continue
        reset_modules.append(name)
        if not dry_run:
            model._init_weights(module)

    return reset_modules, kept_modules


def main(args) -> None:
    """Load a model, reset non-attention weights, and optionally save the result."""
    torch_dtype = resolve_dtype(args.dtype)

    print("=" * 80)
    print("RESET NON-ATTENTION WEIGHTS")
    print("=" * 80)
    print(f"\n[1] Loading model from: {args.model}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(args.device)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"    Seeded re-initialization RNG with: {args.seed}")

    if args.resize_vocab is not None:
        old_vocab = model.get_input_embeddings().weight.shape[0]
        if not args.dry_run:
            model.resize_token_embeddings(args.resize_vocab)
            print(f"    Resized token embeddings: {old_vocab:,} -> {args.resize_vocab:,}")
        else:
            print(f"    Would resize token embeddings: {old_vocab:,} -> {args.resize_vocab:,}")

    if args.embeddings_only:
        print(f"\n[2] {'Inspecting' if args.dry_run else 'Resetting'} embedding modules only")
        reset_modules, kept_modules = reset_embedding_weights(model, dry_run=args.dry_run)
    else:
        model_type = detect_model_type(model)
        print(f"    Detected model type: {model_type}")
        print(f"\n[2] {'Inspecting' if args.dry_run else 'Resetting'} non-attention modules")
        reset_modules, kept_modules = reset_non_attention_weights(model, dry_run=args.dry_run)
    print(f"    Reset modules: {len(reset_modules):,}")
    print(f"    Kept modules:  {len(kept_modules):,}")

    preview_count = min(24, len(reset_modules))
    if preview_count:
        print("\n    First reset modules:")
        for module_name in reset_modules[:preview_count]:
            label = module_name or "<root>"
            print(f"      - {label}")

    if args.output_dir and not args.dry_run:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[3] Saving model to: {output_dir}")
        model.save_pretrained(output_dir)
        print("    Save completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model id or local path to load with AutoModelForCausalLM.from_pretrained().",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional directory to save the reset model. If omitted, the model is not saved.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to load the model on, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Torch dtype used while loading the model.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the model.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be reset without modifying the model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the random re-initialization (torch.manual_seed). "
        "Set this to make the reset reproducible and to share the exact "
        "initialization across experimental conditions.",
    )
    parser.add_argument(
        "--embeddings_only",
        action="store_true",
        help="Reset only the token embeddings (and lm_head when untied), "
        "keeping attention, MLPs, and norms intact.",
    )
    parser.add_argument(
        "--resize_vocab",
        type=int,
        default=None,
        help="Resize the model's token embeddings (and tied lm_head) to this "
        "vocabulary size before resetting, e.g. the downstream tokenizer's "
        "vocab size.",
    )
    args = parser.parse_args()

    main(args)
