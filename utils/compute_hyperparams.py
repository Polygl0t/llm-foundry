"""
Compute hyperparameters (batch size & learning rate) from scaling laws.
- Bi et al., 2024, https://arxiv.org/abs/2401.02954

Formulas from DeepSeek LLM scaling heuristics:
    C = (72 * n_layer * d_model^2 + 12 * n_layer * d_model * l_seq) * D
    lr_max = 0.3118 * C^(-0.125)
    batch_size_tokens = 0.2920 * C^(0.3271)
    batch_size_tokens -> rounded to nearest power of 2

Usage:
    python compute_hyperparams.py --n-layer 28 --d-model 1536 --l-seq 4096 --params 670e6 --tokens 408e9
    python compute_hyperparams.py --n-layer 36 --d-model 2560 --l-seq 4096 --params 4e9 --tokens 1e12
"""

import argparse
import math
from typing import Optional


def nearest_power_of_two(x: float) -> int:
    """Round a number to the nearest power of two."""
    if x <= 0:
        raise ValueError("Input must be positive")
    lower = 2 ** math.floor(math.log2(x))
    upper = 2 ** math.ceil(math.log2(x))
    # Pick the closer one; ties go to the upper power
    if x - lower < upper - x:
        return lower
    else:
        return upper


def estimate_params(n_layer: int, d_model: int) -> float:
    """
    Estimate non-embedding parameters for a decoder-only transformer.

    Uses the standard approximation from scaling law papers:
        N ≈ 12 * n_layer * d_model^2

    Breakdown per layer:
      - Attention QKV + O projections: 4 * d_model^2
      - FFN (4x expansion, up + down): 8 * d_model^2
      - Layer norms: negligible (2 * d_model)
    """
    return 12 * n_layer * d_model ** 2


def compute_hyperparams(n_layer: int, d_model: int, l_seq: int, tokens: float,
                       params: Optional[float] = None):
    """Compute compute budget C, then derive learning rate and batch size."""
    # Compute budget (FLOPs)
    C = (72 * n_layer * d_model ** 2 + 12 * n_layer * d_model * l_seq) * tokens

    # DeepSeek scaling heuristics
    lr = 0.3118 * (C ** -0.125)
    batch_size = 0.2920 * (C ** 0.3271)
    batch_size_pow2 = nearest_power_of_two(batch_size)

    # Parameter & data ratio (Chinchilla optimal: 20 tokens per param)
    # See https://arxiv.org/abs/2203.15556 for details on the Chinchilla scaling laws.
    if params is not None:
        n_params = params
        params_is_estimate = False
    else:
        n_params = estimate_params(n_layer, d_model)
        params_is_estimate = True
    tokens_per_param = tokens / n_params

    return {
        "C": C,
        "lr": lr,
        "batch_size": batch_size,
        "batch_size_pow2": batch_size_pow2,
        "n_params": n_params,
        "tokens_per_param": tokens_per_param,
        "params_is_estimate": params_is_estimate,
    }


def fmt(x: float) -> str:
    """Format a large number in a human-readable way."""
    if abs(x) >= 1e15:
        return f"{x:.4e}"
    elif abs(x) >= 1e12:
        return f"{x / 1e12:.4f} T"
    elif abs(x) >= 1e9:
        return f"{x / 1e9:.4f} B"
    elif abs(x) >= 1e6:
        return f"{x / 1e6:.4f} M"
    elif abs(x) >= 1e3:
        return f"{x / 1e3:.4f} K"
    else:
        return f"{x:.6f}"


def main(args):

    result = compute_hyperparams(
        n_layer=args.n_layer,
        d_model=args.d_model,
        l_seq=args.l_seq,
        tokens=args.tokens,
        params=args.params,
    )

    params_label = "Params (given)" if not result["params_is_estimate"] else "Params (estimated)"
    params_note = "" if not result["params_is_estimate"] else " (non-embedding)"

    print("=" * 56)
    print("  Hyperparameter Scaling (DeepSeek Heuristics)")
    print("=" * 56)
    print(f"  Architecture:")
    print(f"    n_layer  = {args.n_layer}")
    print(f"    d_model  = {args.d_model}")
    print(f"    l_seq    = {args.l_seq}")
    print(f"    tokens   = {fmt(args.tokens)} ({args.tokens:.4e})")
    print(f"  =========================================")
    print(f"  Compute budget C  = {fmt(result['C'])} ({result['C']:.4e} FLOPs)")
    print(f"  =========================================")
    print(f"  {params_label:16s} = {fmt(result['n_params'])}{params_note}")
    print(f"  Tokens / param    = {result['tokens_per_param']:.2f}  "
          f"(Chinchilla optimal: 20:1)")
    if result['tokens_per_param'] < 20:
        print(f"                       ⚠️ Below optimal — consider more tokens "
              f"or a smaller model")
    print(f"  =========================================")
    print(f"  Max Learning Rate = {result['lr']:.6e}")
    if not args.no_pow2:
        print(f"  Batch Size        = {result['batch_size']:,.0f} tokens")
        print(f"    -> rounded to   = {result['batch_size_pow2']:>13,} tokens")
        print(f"      (2^{int(math.log2(result['batch_size_pow2']))})")
        print(f"    -> per step (GAS micro-batches):")
        print(f"      GAS x micro_batch_size x l_seq = {result['batch_size_pow2']:,}")
        # Suggest some breakdowns
        for gas in [1, 2, 4, 8, 16, 32, 64, 128]:
            per_step = result["batch_size_pow2"] // gas
            if per_step >= args.l_seq and per_step % args.l_seq == 0:
                micro_bs = per_step // args.l_seq
                print(f"      GAS={gas:>3d}  ->  micro_batch_size={micro_bs:>4d}  (per device)")
        # Training steps
        tokens_per_step = result["batch_size_pow2"]
        steps = args.tokens / tokens_per_step
        print(f"  =========================================")
        print(f"  Training Run:")
        print(f"    Tokens / step     = {tokens_per_step:>13,}")
        print(f"    Total steps       = {steps:>13,.0f}")
        print(f"      ({steps:,.0f} steps × {tokens_per_step:,} tokens/step = "
              f"{fmt(tokens_per_step * steps)})")
    else:
        print(f"  Batch Size        = {result['batch_size']:,.0f} tokens")
    print("=" * 56)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-layer", type=int, required=True,
        help="Number of transformer layers (e.g. 28)"
    )
    parser.add_argument(
        "--d-model", type=int, required=True,
        help="Model hidden dimension (e.g. 1536)"
    )
    parser.add_argument(
        "--l-seq", type=int, default=4096,
        help="Sequence length / context window (default: 4096)"
    )
    parser.add_argument(
        "--tokens", type=float, required=True,
        help="Total training tokens. Can use scientific notation, e.g. 408e9"
    )
    parser.add_argument(
        "--params", type=float, default=None,
        help="Actual parameter count (e.g. 7e9 for a 7B model). If omitted, "
             "estimated from n_layer and d_model."
    )
    parser.add_argument(
        "--no-pow2", action="store_true",
        help="Skip rounding batch size to nearest power of two"
    )
    args = parser.parse_args()
    main(args)
