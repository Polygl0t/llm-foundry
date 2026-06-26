"""
GPU Communication Benchmark / LLM Gradient All-Reduce Simulation.
Benchmarks all_reduce (SUM) on gradient tensors for a Llama-style LLM.

Also benchmarks activation/logits-scale tensors (hidden states, logits)
whose size depends on --batch-size and --seq-length, so you can measure
communication cost at realistic token counts.

Usage (SLURM):
    srun --cpu-bind=none python3 distributed_test.py

Usage (torchrun):
    torchrun --nproc-per-node=<N> distributed_test.py
"""

import argparse
import logging
import os
import sys
import time

import torch
import torch.distributed as dist

MODEL_CONFIG = {
    "hidden_size": 512,
    "intermediate_size": 1536,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "vocab_size": 49152,
    "tie_word_embeddings": True,
}


def setup_distributed(logger):
    """Initialize the distributed process group and return env info.

    Discovers world size/rank from SLURM or torchrun env vars and exports
    `LOCAL_RANK` / `RANK` / `WORLD_SIZE` for downstream compatibility.

    Returns a dict with keys:
        world_size, rank, local_rank, master_process, device, ddp
    """
    if "SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ:
        world_size = int(os.environ["SLURM_NTASKS"])
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ.get("SLURM_LOCALID", rank % max(torch.cuda.device_count(), 1)))
    elif "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        world_size, rank, local_rank = 1, 0, 0

    # Export torchrun-style env vars unconditionally.
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if world_size > 1 and torch.cuda.is_available():
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            device_id=torch.device("cuda", local_rank),
        )
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master_process = rank == 0
        if master_process:
            logger.info(
                f"Running via '{dist.get_backend().upper()}' backend. "
                f"Rank {rank} / world size {world_size}."
            )
    else:
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda"):
            torch.cuda.set_device(device)
        master_process = True

    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "master_process": master_process,
        "device": device,
        "ddp": world_size > 1 and torch.cuda.is_available(),
    }


def cleanup_distributed(env):
    """Destroy the process group if running in distributed mode."""
    if env["ddp"]:
        dist.destroy_process_group()


def compute_gradient_sizes(config):
    """Return dict of {param_name: num_elements} for all gradient tensors."""
    hs = config["hidden_size"]
    is_ = config["intermediate_size"]
    n_layers = config["num_hidden_layers"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = hs // n_heads
    vocab = config["vocab_size"]

    sizes = {}

    # Embedding (tied with lm_head)
    sizes["embed_tokens"] = vocab * hs

    for i in range(n_layers):
        p = f"layers.{i}"
        sizes[f"{p}.attn.q_proj"] = hs * n_heads * head_dim  # = hs*hs
        sizes[f"{p}.attn.k_proj"] = hs * n_kv_heads * head_dim
        sizes[f"{p}.attn.v_proj"] = hs * n_kv_heads * head_dim
        sizes[f"{p}.attn.o_proj"] = n_heads * head_dim * hs  # = hs*hs
        sizes[f"{p}.mlp.gate_proj"] = hs * is_
        sizes[f"{p}.mlp.up_proj"] = hs * is_
        sizes[f"{p}.mlp.down_proj"] = is_ * hs
        sizes[f"{p}.input_layernorm"] = hs
        sizes[f"{p}.post_attention_layernorm"] = hs

    sizes["norm"] = hs
    return sizes


def benchmark_all_reduce(tensor, n_warmup, n_iter, dtype=torch.bfloat16):
    """Run all_reduce SUM repeatedly and return (mean_time_ms, algo_bw_gbps, size_mb)."""
    elements = tensor.numel()
    size_bytes = elements * tensor.element_size()
    size_mb = size_bytes / (1024 * 1024)

    # Warmup
    for _ in range(n_warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Timed iterations
    start = time.perf_counter()
    for _ in range(n_iter):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    mean_ms = elapsed_ms / n_iter
    # Algorithmic bandwidth: each rank sends its data and receives the result
    # For ring all-reduce the per-rank volume is 2 * (N-1)/N * size, but NCCL
    # reports "bus bandwidth" = size / time.  We report algo bandwidth here as
    # 2 * size_bytes / time (GB/s) which reflects the total data moved.
    algo_bw_gbps = (2 * size_bytes) / (mean_ms / 1000) / (1024**3)
    return mean_ms, algo_bw_gbps, size_mb


def main(args):
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    logger = logging.getLogger("GPU-Bench")
    logging.basicConfig(
        format="%(name)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    env = setup_distributed(logger)
    num_nodes = int(os.environ.get("SLURM_NNODES", 1))

    if env["world_size"] == 1:
        raise RuntimeError("World size is 1 — this benchmark needs ≥2 GPUs.")

    master = env["master_process"]
    device = env["device"]

    if master:
        logger.info("=" * 72)
        logger.info("GPU COMMUNICATION BENCHMARK  —  Gradient All-Reduce (DDP)")
        logger.info("=" * 72)
        logger.info(f"  World size      : {env['world_size']} GPUs / {num_nodes} node(s)")
        logger.info(f"  Backend         : {dist.get_backend().upper()}")
        logger.info(f"  Dtype           : {args.dtype}")
        logger.info(f"  Warmup / Iter   : {args.n_warmup} / {args.n_iter}")
        logger.info(f"  Sim. batch      : {args.batch_size} x {args.seq_length} tokens")
        logger.info(
            f"  Model           : {MODEL_CONFIG['num_hidden_layers']}L, "
            f"{MODEL_CONFIG['hidden_size']}D, "
            f"{MODEL_CONFIG['intermediate_size']}FF, "
            f"vocab={MODEL_CONFIG['vocab_size']}"
        )
        nccl_p2p_disabled = os.environ.get("NCCL_P2P_DISABLE", "0") == "1"
        fi = os.environ.get("FI_PROVIDER", "not set")
        logger.info(f"  NCCL P2P        : {'DISABLED' if nccl_p2p_disabled else 'ENABLED'}")
        logger.info(f"  FI_PROVIDER     : {fi}")
        logger.info("-" * 72)
        logger.info(f"{'Tensor':<48s} {'Size (MB)':>10s} {'Time (ms)':>10s} {'BW (GB/s)':>10s}")
        logger.info("-" * 72)

    dist.barrier()

    grad_sizes = compute_gradient_sizes(MODEL_CONFIG)
    all_results = []

    if master:
        logger.info(f"\n{'─' * 72}")
        logger.info("Single-parameter gradient all-reduce")
        logger.info(f"{'─' * 72}")

    for name, numel in grad_sizes.items():
        t = torch.randn(numel, dtype=dtype, device=device)
        dist.barrier()
        ms, bw, mb = benchmark_all_reduce(t, args.n_warmup, args.n_iter, dtype)
        if master:
            logger.info(f"  {name:<46s} {mb:>10.2f} {ms:>10.3f} {bw:>10.2f}")
        all_results.append((name, mb, ms, bw))
        dist.barrier()

    if master:
        logger.info(f"\n{'─' * 72}")
        logger.info("Component-grouped all-reduce (one layer)")
        logger.info(f"{'─' * 72}")

    # Attention for one layer
    attn_el = (
        grad_sizes["layers.0.attn.q_proj"]
        + grad_sizes["layers.0.attn.k_proj"]
        + grad_sizes["layers.0.attn.v_proj"]
        + grad_sizes["layers.0.attn.o_proj"]
    )
    t = torch.randn(attn_el, dtype=dtype, device=device)
    dist.barrier()
    ms, bw, mb = benchmark_all_reduce(t, args.n_warmup, args.n_iter, dtype)
    if master:
        logger.info(f"  {'attention (Q+K+V+O, 1 layer)':<46s} {mb:>10.2f} {ms:>10.3f} {bw:>10.2f}")
    all_results.append(("attention_1layer", mb, ms, bw))
    dist.barrier()

    # MLP for one layer
    mlp_el = (
        grad_sizes["layers.0.mlp.gate_proj"]
        + grad_sizes["layers.0.mlp.up_proj"]
        + grad_sizes["layers.0.mlp.down_proj"]
    )
    t = torch.randn(mlp_el, dtype=dtype, device=device)
    dist.barrier()
    ms, bw, mb = benchmark_all_reduce(t, args.n_warmup, args.n_iter, dtype)
    if master:
        logger.info(f"  {'MLP (gate+up+down, 1 layer)':<46s} {mb:>10.2f} {ms:>10.3f} {bw:>10.2f}")
    all_results.append(("mlp_1layer", mb, ms, bw))
    dist.barrier()

    # Full layer (attn + MLP + norms)
    layer0_el = sum(v for k, v in grad_sizes.items() if k.startswith("layers.0"))
    t = torch.randn(layer0_el, dtype=dtype, device=device)
    dist.barrier()
    ms, bw, mb = benchmark_all_reduce(t, args.n_warmup, args.n_iter, dtype)
    if master:
        logger.info(f"  {'full layer (attn+MLP+norms)':<46s} {mb:>10.2f} {ms:>10.3f} {bw:>10.2f}")
    all_results.append(("full_layer", mb, ms, bw))
    dist.barrier()

    if master:
        logger.info(f"\n{'─' * 72}")
        logger.info("Full-model gradient all-reduce")
        logger.info(f"{'─' * 72}")

    total_params = sum(grad_sizes.values())
    t = torch.randn(total_params, dtype=dtype, device=device)
    dist.barrier()
    ms, bw, mb = benchmark_all_reduce(t, args.n_warmup, args.n_iter, dtype)
    if master:
        logger.info(f"  {'full model (all parameters)':<46s} {mb:>10.2f} {ms:>10.3f} {bw:>10.2f}")
    all_results.append(("full_model", mb, ms, bw))
    dist.barrier()

    if master:
        logger.info(f"\n{'─' * 72}")
        logger.info("Gradient-accumulation-scale all-reduce")
        logger.info(f"{'─' * 72}")

    for copies in [2, 4, 8]:
        t = torch.randn(total_params * copies, dtype=dtype, device=device)
        dist.barrier()
        ms, bw, mb = benchmark_all_reduce(t, args.n_warmup, args.n_iter, dtype)
        label = f"model x {copies}"
        if master:
            logger.info(f"  {label:<46s} {mb:>10.2f} {ms:>10.3f} {bw:>10.2f}")
        all_results.append((label, mb, ms, bw))
        dist.barrier()

    # Activations / logits-scale all-reduce
    # Unlike gradient tensors (fixed by model shape), activations and
    # logits scale with batch_size * seq_length.  This section shows how
    # much communication you'd pay when exchanging hidden states
    # (e.g. sequence parallelism) or materialising logits.
    num_tokens = args.batch_size * args.seq_length
    hs = MODEL_CONFIG["hidden_size"]
    vocab = MODEL_CONFIG["vocab_size"]

    if master:
        logger.info(f"\n{'─' * 72}")
        logger.info(
            f"Activation / hidden-state all-reduce "
            f"(batch={args.batch_size} x seq={args.seq_length} "
            f"= {num_tokens} tokens)"
        )
        logger.info(f"{'─' * 72}")

    # 1) Hidden states: [B*T, hidden_size] — activations between layers
    hidden_el = num_tokens * hs
    t = torch.randn(hidden_el, dtype=dtype, device=device)
    dist.barrier()
    ms, bw, mb = benchmark_all_reduce(t, args.n_warmup, args.n_iter, dtype)
    if master:
        label = f"hidden states (B*T={num_tokens} x H={hs})"
        logger.info(f"  {label:<46s} {mb:>10.2f} {ms:>10.3f} {bw:>10.2f}")
    all_results.append(("hidden_states", mb, ms, bw))
    dist.barrier()

    # 2) Logits: [B*T, vocab_size] — final projection output.
    #    This is often the largest transient tensor in a Transformer.
    logits_el = num_tokens * vocab
    elem_bytes = 4 if dtype == torch.float32 else 2  # bf16 / fp16 = 2 B, fp32 = 4 B
    logits_gb = logits_el * elem_bytes / (1024**3)

    gpu_mem_bytes = torch.cuda.get_device_properties(device).total_memory
    if logits_el * elem_bytes < gpu_mem_bytes * 0.85:
        t = torch.randn(logits_el, dtype=dtype, device=device)
        dist.barrier()
        ms, bw, mb = benchmark_all_reduce(t, args.n_warmup, args.n_iter, dtype)
        if master:
            label = f"logits (B*T={num_tokens} x V={vocab})"
            logger.info(f"  {label:<46s} {mb:>10.2f} {ms:>10.3f} {bw:>10.2f}")
        all_results.append(("logits_full", mb, ms, bw))
        dist.barrier()
    else:
        if master:
            logger.info(f"  {'logits (B*T x V)':<46s} {'—':>10s} {'—':>10s} {'—':>10s}")
            logger.info(
                f"    → {logits_el:,} elements ≈ {logits_gb:.1f} GB "
                f"({args.dtype}) — exceeds GPU memory, skipped"
            )
        all_results.append(("logits_full", logits_gb * 1024, float("nan"), float("nan")))

    if master:
        logger.info(f"\n{'=' * 72}")
        logger.info("SUMMARY")
        logger.info(f"{'=' * 72}")
        logger.info(f"{'Tensor':<48s} {'Size (MB)':>10s} {'Time (ms)':>10s} {'BW (GB/s)':>10s}")
        logger.info("-" * 72)
        for name, mb, ms, bw in all_results:
            logger.info(f"  {name:<46s} {mb:>10.2f} {ms:>10.3f} {bw:>10.2f}")
        logger.info("-" * 72)
        total_mb = total_params * 2 / (1024 * 1024)
        logger.info(f"  Total params : {total_params:>12,}  ({total_mb:.2f} MB @ bf16)")
        logger.info("  Bus bandwidth: size / time  (reported BW = 2xsize / time)")
        logger.info("=" * 72)

    cleanup_distributed(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--n-warmup", type=int, default=5, help="Warmup iterations (default: 5)")
    parser.add_argument("--n-iter", type=int, default=20, help="Timed iterations (default: 20)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Micro batch size — controls activation/logits "
        "tensor sizes in the hidden-states benchmark "
        "(default: 128)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=4096,
        help="Sequence length — controls activation/logits "
        "tensor sizes in the hidden-states benchmark "
        "(default: 4096)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Data type for gradients (default: bfloat16)",
    )
    args = parser.parse_args()

    main(args)
