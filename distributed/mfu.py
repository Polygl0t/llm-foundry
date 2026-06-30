"""
MFU calculation helpers for the distributed trainers.

This module supports the model families this codebase currently targets:

  * Standard GPT-style dense transformers (e.g. `LlamaForCausalLM`,
    `Qwen3ForCausalLM`, dense `Qwen3_5ForCausalLM`).
  * Mixture-of-Experts transformers (e.g. `Qwen3MoeForCausalLM`,
    MoE-configured `Qwen3_5MoeForCausalLM`). For MoE, the trainer passes the
    *active* parameter count (params actually used per token) as
    `num_parameters`; the dense-transformer FLOPs formula then yields the
    correct per-token compute.
  * Qwen3.5 hybrid models that mix full attention with gated-delta-net
    linear attention layers (`layer_types` contains `"linear_attention"`).
    A structural per-layer formula is used for these so the linear-attention
    layers are accounted for properly.

Provides:
    - `MFUContext` dataclass capturing the model + hardware shape.
    - `TrainingPerformanceMetrics` dataclass with the per-step results.
    - `create_mfu_context()` to build a context from training args.
    - `calculate_training_metrics()` to compute MFU + throughput per step.
"""

from dataclasses import dataclass

# Peak FLOPs (BF16) for supported hardware (Bender|Marvin|Jupiter|OP).
PEAK_FLOPS_BY_HARDWARE = {
    "a100": 312e12,  # --> https://www.nvidia.com/en-us/data-center/a100/
    "a40": 150e12,  # --> https://www.nvidia.com/en-us/data-center/a40/
    "gh200": 990e12,  # --> https://www.nvidia.com/en-eu/data-center/grace-hopper-superchip/
    "h200": 989.5e12,    # H200 SXM — 1,979 TFLOPS (sparsity) / 2 = 989.5 dense (https://resources.nvidia.com/en-us-gpu-resources/hpc-datasheet-sc23)
    "h100": 756.5e12,    # H100 PCIe — 1,513 TFLOPS (sparsity) / 2 = 756.5 dense  (https://www.megware.com/fileadmin/user_upload/LandingPage%20NVIDIA/nvidia-h100-datasheet.pdf)
    # Extend with more hardware as needed.
}


@dataclass(frozen=True)
class MFUContext:
    """Architecture + hardware information used by the MFU calculation."""

    peak_flops: float
    num_parameters: int  # For MoE models this is the *active* param count.
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    sequence_length: int

    # Optional structural fields. Required only when the model has any
    # `linear_attention` layers (e.g., Qwen3.5 hybrid).
    hidden_size: int = 0
    vocab_size: int = 0
    intermediate_size: int = 0
    num_key_value_heads: int = 0
    layer_types: tuple[str, ...] = ()

    # Linear attention (Gated DeltaNet) parameters.
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_conv_kernel_dim: int = 4
    linear_chunk_size: int = 256

    # MoE parameters (used by the structural / hybrid path).
    # The dense-transformer path instead consumes `num_parameters` set to
    # the active parameter count.
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_intermediate_size: int = 0


@dataclass(frozen=True)
class TrainingPerformanceMetrics:
    tokens_processed: int
    global_tokens_per_sec: float
    tokens_per_sec_per_gpu: float
    mfu: float


def _dense_transformer_flops(ctx, micro_batch_size, gradient_accumulation_steps, dt):
    """
    FLOPs for a standard dense transformer.

    Uses the well-known approximation::

        flops/token = 6N + 12 * L * H * d_head * S

    For MoE models the trainer is expected to set `num_parameters` to the
    *active* parameter count (only the experts selected per token), so this
    same formula yields a correct active-FLOPs estimate.

    Source: https://www.adamcasson.com/posts/transformer-flops
    """
    flops_per_token = (
        6 * ctx.num_parameters
        + 12 * ctx.num_hidden_layers * ctx.num_attention_heads * ctx.head_dim * ctx.sequence_length
    )
    flops_per_iter = (
        flops_per_token * ctx.sequence_length * micro_batch_size * gradient_accumulation_steps
    )
    return flops_per_iter / dt


def _mlp_macs_per_token(ctx):
    """
    Per-token MACs for the MLP block of a single layer.

    Returns the dense SwiGLU cost `3 * d * intermediate_size` for non-MoE
    layers, and a MoE-active cost (routed experts + optional shared expert)
    when MoE fields are populated.
    """
    d = ctx.hidden_size
    if ctx.num_experts_per_tok > 0 and ctx.moe_intermediate_size > 0:
        routed = 3 * d * ctx.moe_intermediate_size * ctx.num_experts_per_tok
        shared = 3 * d * ctx.shared_intermediate_size
        return routed + shared
    return 3 * d * ctx.intermediate_size


def _full_attention_macs_per_token(ctx):
    """
    Per-token MACs for a full causal-attention layer with grouped-query
    attention (GQA) support.

    We use the non-causal attention cost (`2 * d * s`) so that the
    structural path matches the standard `6N + 12·L·H·d_head·S`
    approximation used by `_dense_transformer_flops`.
    """
    d = ctx.hidden_size
    s = ctx.sequence_length
    h = ctx.num_attention_heads
    kv_h = ctx.num_key_value_heads if ctx.num_key_value_heads > 0 else h
    q_dim = h * ctx.head_dim
    kv_dim = kv_h * ctx.head_dim
    projections = 2 * d * q_dim + 2 * d * kv_dim
    attention = 2 * d * s
    return projections + attention + _mlp_macs_per_token(ctx)


def _linear_attention_macs_per_token(ctx):
    """
    Per-token MACs for a Gated-DeltaNet linear-attention layer
    (training, chunkwise parallel).

    Counts every projection, the depthwise causal conv1d, the pre-chunk
    element-wise transforms, the intra- and inter-chunk kernel operations,
    the output gated-RMSNorm, and the final out-projection.

    After `query` / `key` are repeated to match `num_value_heads`
    (see `transformers.Qwen3_5GatedDeltaNet`), all chunk-level
    work uses the *value*-head count, not the key-head count.

    Reference: https://arxiv.org/abs/2604.03444 (Eqs. 8 and 10)
    """
    d = ctx.hidden_size
    nk = ctx.linear_num_key_heads
    nv = ctx.linear_num_value_heads
    dk = ctx.linear_key_head_dim
    dv = ctx.linear_value_head_dim
    k = nk * dk  # total key dimension (before repeat)
    v = nv * dv  # total value dimension
    L = ctx.linear_chunk_size
    K = ctx.linear_conv_kernel_dim

    #  1. Input projections
    # in_proj_qkv : d → (k + k + v)      = 2k + v
    # in_proj_z   : d → v
    # in_proj_b   : d → nv
    # in_proj_a   : d → nv
    projections = d * (2 * k + 2 * v + 2 * nv)

    #  2. Causal depthwise conv1d -
    conv = K * (2 * k + v)

    #  3. Chunked gated-delta-rule kernel
    # Intra-chunk: per-chunk kernel mixing, triangular solve, value aggregation.
    #   - k_beta @ keyᵀ  (L×L matmul)         →  nv·L·dk  / token
    #   - triangular recurrence                →  nv·2L²/3  / token
    #   - attn @ v_beta                        →  nv·L·dv  / token
    #   - attn @ (k_beta ⊙ g)                  →  nv·L·dk  / token
    intra = nv * (2 * L * dk + L * dv + (2 * L * L) // 3)

    # Inter-chunk: state-passing between consecutive chunks.
    #   - q_i @ k_iᵀ (causal average)          →  nv·L·dk/2 / token
    #   - k_cumdecay @ last_state              →  nv·dk·dv  / token
    #   - (q_i ⊙ g_exp) @ last_state           →  nv·dk·dv  / token
    #   - attn @ v_new (causal average)        →  nv·L·dv/2 / token
    #   - (k_i ⊙ decay)ᵀ @ v_new (state upd)   →  nv·dk·dv  / token
    inter = nv * ((L * dk) // 2 + (L * dv) // 2 + 3 * dk * dv)

    #  4. Pre- / post-chunk element-wise ops
    # Q/K L2-norm, beta / g discretisation, output gated-RMSNorm.
    elem_ops = nv * (6 * dk + 5 * dv)

    #  5. Output projection
    out_proj = d * v

    return projections + conv + intra + inter + elem_ops + out_proj + _mlp_macs_per_token(ctx)


_LAYER_MAC_FNS = {
    "full_attention": _full_attention_macs_per_token,
    "attention": _full_attention_macs_per_token,  # alias
    "linear_attention": _linear_attention_macs_per_token,
}


def _hybrid_attention_flops(ctx, micro_batch_size, gradient_accumulation_steps, dt):
    """
    Structural FLOPs for Qwen3.5-style hybrid models that mix full attention
    with linear-attention layers. Per-layer MLP cost is MoE-aware when the
    MoE fields are populated.
    """
    lm_head_macs = ctx.hidden_size * ctx.vocab_size
    total_layer_macs = sum(_LAYER_MAC_FNS[lt](ctx) for lt in ctx.layer_types)
    flops_per_token = 2 * (lm_head_macs + total_layer_macs)  # FLOPs = 2 * MACs
    flops_per_fwdbwd = 3 * flops_per_token * ctx.sequence_length  # fwd + bwd ~= 3 * fwd
    flops_per_iter = flops_per_fwdbwd * micro_batch_size * gradient_accumulation_steps
    return flops_per_iter / dt


def _has_linear_attention(layer_types):
    return any(lt == "linear_attention" for lt in layer_types)


def create_mfu_context(args, hardware, num_parameters):
    """Build an :class:`MFUContext` from runtime training args and hardware."""
    peak_flops = PEAK_FLOPS_BY_HARDWARE.get(hardware.lower())
    if peak_flops is None:
        supported = ", ".join(sorted(PEAK_FLOPS_BY_HARDWARE))
        raise ValueError(f"Hardware '{hardware}' not supported for MFU. Supported: {supported}.")

    return MFUContext(
        peak_flops=peak_flops,
        num_parameters=num_parameters,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        head_dim=args.head_dim,
        sequence_length=args.max_position_embeddings,
        hidden_size=getattr(args, "hidden_size", 0) or 0,
        vocab_size=getattr(args, "vocab_size", 0) or 0,
        intermediate_size=getattr(args, "intermediate_size", 0) or 0,
        num_key_value_heads=getattr(args, "num_key_value_heads", 0) or 0,
        layer_types=tuple(getattr(args, "layer_types", ()) or ()),
        linear_num_key_heads=getattr(args, "linear_num_key_heads", 0) or 0,
        linear_num_value_heads=getattr(args, "linear_num_value_heads", 0) or 0,
        linear_key_head_dim=getattr(args, "linear_key_head_dim", 0) or 0,
        linear_value_head_dim=getattr(args, "linear_value_head_dim", 0) or 0,
        linear_conv_kernel_dim=getattr(args, "linear_conv_kernel_dim", 4) or 4,
        num_experts_per_tok=getattr(args, "num_experts_per_tok", 0) or 0,
        moe_intermediate_size=getattr(args, "moe_intermediate_size", 0) or 0,
        shared_intermediate_size=getattr(args, "shared_intermediate_size", 0) or 0,
    )


def calculate_training_metrics(
    mfu_context, micro_batch_size, gradient_accumulation_steps, world_size, dt
):
    """Compute throughput + MFU for the current step."""
    if dt <= 0:
        raise ValueError("Step duration must be positive for MFU calculation.")

    tokens_processed = (
        micro_batch_size * gradient_accumulation_steps * mfu_context.sequence_length * world_size
    )
    global_tokens_per_sec = tokens_processed / dt
    tokens_per_sec_per_gpu = global_tokens_per_sec / world_size

    if _has_linear_attention(mfu_context.layer_types):
        flops_achieved = _hybrid_attention_flops(
            mfu_context,
            micro_batch_size,
            gradient_accumulation_steps,
            dt,
        )
    else:
        flops_achieved = _dense_transformer_flops(
            mfu_context,
            micro_batch_size,
            gradient_accumulation_steps,
            dt,
        )
    mfu = (flops_achieved / mfu_context.peak_flops) * 100

    return TrainingPerformanceMetrics(
        tokens_processed=tokens_processed,
        global_tokens_per_sec=global_tokens_per_sec,
        tokens_per_sec_per_gpu=tokens_per_sec_per_gpu,
        mfu=mfu,
    )
