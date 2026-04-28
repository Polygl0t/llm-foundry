"""
MFU calculation helpers for the distributed trainers.

The strategy registry makes it straightforward to add alternative formulas for
other architectures (MoE, Mamba, Hybrid) without changing the training loop.

Provides:
    - `MFUContext` dataclass that encapsulates all necessary information about the model and hardware for MFU calculation.
    - `TrainingPerformanceMetrics` dataclass that encapsulates the results of the MFU calculation and related performance metrics.
    - `create_mfu_context()` function to create an `MFUContext` from training arguments and hardware information.
    - `calculate_training_metrics()` function that computes the MFU and related performance metrics given the context and training step information.
"""
from dataclasses import dataclass, field
from typing import Tuple


PEAK_FLOPS_BY_HARDWARE = {
    "a100": 300e12,
    "a40": 150e12,
}


@dataclass(frozen=True)
class MFUContext:
    mfu_type: str
    peak_flops: float
    num_parameters: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    sequence_length: int
    
    # Extended fields for hybrid / mamba / linear-attention models
    hidden_size: int = 0
    vocab_size: int = 0
    intermediate_size: int = 0
    layer_types: Tuple[str, ...] = ()
    # Number of K/V heads for grouped-query attention. When 0 or equal to
    # `num_attention_heads`, attention behaves as multi-head attention (MHA).
    num_key_value_heads: int = 0
    
    # Mamba2 parameters
    mamba_d_state: int = 0
    mamba_chunk_size: int = 256
    mamba_d_conv: int = 4
    mamba_n_heads: int = 0
    mamba_d_head: int = 0
    mamba_n_groups: int = 1
    
    # Linear attention (GDN / DeltaNet) parameters
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_conv_kernel_dim: int = 4
    linear_chunk_size: int = 256

    # MoE parameters (used by hybrid / mamba paths when num_experts_per_tok > 0).
    # The dense_transformer / moe paths instead derive active FLOPs from
    # `num_parameters`, which the trainer is expected to set to the active count.
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0      # per-routed-expert MLP intermediate size
    shared_intermediate_size: int = 0   # always-active shared expert intermediate size


@dataclass(frozen=True)
class TrainingPerformanceMetrics:
    tokens_processed: int
    global_tokens_per_sec: float
    tokens_per_sec_per_gpu: float
    mfu: float


def _dense_transformer_mfu(context, micro_batch_size, gradient_accumulation_steps, dt):
    """
    Calculate MFU for a standard dense transformer architecture.
        - source: https://www.adamcasson.com/posts/transformer-flops
    """
    flops_per_token = (
        6 * context.num_parameters
        + 12 * context.num_hidden_layers * context.num_attention_heads * context.head_dim * context.sequence_length
    )
    flops_per_fwdbwd = flops_per_token * context.sequence_length
    flops_per_iter = flops_per_fwdbwd * (micro_batch_size * gradient_accumulation_steps)
    flops_achieved = flops_per_iter / dt
    return (flops_achieved / context.peak_flops) * 100

def _active_mlp_macs(ctx):
    """
    Per-token MACs for the active part of the MLP block in a single layer.

    Handles three configurations:
      * Dense SwiGLU MLP (default): 3 * d * intermediate_size.
      * Top-k routed MoE (e.g. Qwen-MoE): only the `num_experts_per_tok`
        routed experts contribute, each with `moe_intermediate_size`.
      * Routed MoE + shared expert(s) (e.g. GraniteMoeHybrid, DeepSeek-V2/V3):
        adds a constant `shared_intermediate_size` term on top of the routed
        contribution.
    """
    d = ctx.hidden_size
    if ctx.num_experts_per_tok > 0 and ctx.moe_intermediate_size > 0:
        # SwiGLU per expert: gate + up + down projections.
        routed = 3 * d * ctx.moe_intermediate_size * ctx.num_experts_per_tok
        # Shared expert(s) are dense and always active.
        shared = 3 * d * ctx.shared_intermediate_size
        return routed + shared
    return 3 * d * ctx.intermediate_size


def _mamba_layer_macs(ctx):
    """
    MACs per token for a single Mamba2 layer (training / chunkwise parallel).

    Combines the sequence-independent terms from Eq. 9 (in-proj, conv1d,
    out-proj, MLP) with the chunk-parallel overheads from Eq. 11
    (intra-chunk SSD mixing and inter-chunk state passing).
        - source: https://arxiv.org/abs/2604.03444 (Eqs. 9 and 11)
    """
    d = ctx.hidden_size
    e = ctx.mamba_n_heads * ctx.mamba_d_head          # expanded dim
    g = ctx.mamba_n_groups
    n = ctx.mamba_d_state
    h = ctx.mamba_n_heads

    # In-projection: d -> (2e + 2gn + h) for x, z, B, C, dt   (Eq. 9: d.p)
    in_proj = d * (2 * e + 2 * g * n + h)
    # Depthwise conv1d: kernel_size x channels                 (Eq. 9: 4c with c=e)
    conv = ctx.mamba_d_conv * e
    # Out-projection: e -> d                                    (Eq. 9: e.d)
    out_proj = e * d
    # Intra-chunk SSD mixing                                   (Eq. 11)
    intra_chunk = 2 * ctx.mamba_chunk_size * e
    # Inter-chunk state passing                                (Eq. 11)
    inter_chunk = 2 * e * n
    # MLP (SwiGLU). MoE-aware: only active experts contribute.   (Eq. 9: 3d.d_MLP)
    mlp = _active_mlp_macs(ctx)

    return in_proj + conv + out_proj + intra_chunk + inter_chunk + mlp


def _attention_layer_macs(ctx):
    """
    MACs per token for a full (causal) attention layer.

    Supports grouped-query attention (GQA): when `num_key_value_heads` is set
    and smaller than `num_attention_heads`, the K and V projection costs are
    scaled down by `num_key_value_heads / num_attention_heads`.
    """
    d = ctx.hidden_size
    s = ctx.sequence_length
    h = ctx.num_attention_heads
    kv_h = ctx.num_key_value_heads if ctx.num_key_value_heads > 0 else h
    q_dim = h * ctx.head_dim       # query projection output dim
    kv_dim = kv_h * ctx.head_dim   # key / value projection output dim (shared across grouped heads)
    # Q + O projections are full d->d; K and V are d->kv_dim each.
    projections = 2 * d * q_dim + 2 * d * kv_dim
    attention   = d * s            # QK^T + attn.V  (causal avg, dominated by query dim)
    mlp         = _active_mlp_macs(ctx)
    return projections + attention + mlp


def _linear_attention_layer_macs(ctx):
    """
    MACs per token for a GDN / DeltaNet linear-attention layer (training).

    Combines the sequence-independent terms from Eq. 8 with the
    chunk-parallel overheads from Eq. 10.
        - source: https://arxiv.org/abs/2604.03444 (Eqs. 8 and 10)
    """
    d = ctx.hidden_size
    k = ctx.linear_num_key_heads * ctx.linear_key_head_dim     # total key dim
    v = ctx.linear_num_value_heads * ctx.linear_value_head_dim # total value dim
    h = ctx.linear_num_key_heads                               # head count
    L = ctx.linear_chunk_size

    # Linear projections: Q, K (each d->k), V (d->v), 2 gating (d->h)   (Eq. 8)
    projections = d * (2 * k + v + 2 * h)
    # Depthwise convs: kernel x (2k+v) channels                       (Eq. 8)
    conv = ctx.linear_conv_kernel_dim * (2 * k + v)
    # Gate projection (d->v) + output projection (v->d)                 (Eq. 8)
    gate_out = 2 * d * v
    # Intra-chunk overhead (Kernel, W/U, Attn)                         (Eq. 10)
    intra_chunk = L * (3 * k + 2 * v)
    # Inter-chunk state passing: 3 mat-muls of size k_h x v_h per head (Eq. 10)
    inter_chunk = 3 * k * v // h if h > 0 else 0
    # MLP, MoE-aware.                                                 (Eq. 8)
    mlp = _active_mlp_macs(ctx)

    return projections + conv + gate_out + intra_chunk + inter_chunk + mlp


_LAYER_MAC_FNS = {
    "mamba":            _mamba_layer_macs,
    "attention":        _attention_layer_macs,
    "full_attention":   _attention_layer_macs,
    "linear_attention": _linear_attention_layer_macs,
}


def _mamba_mfu(context, micro_batch_size, gradient_accumulation_steps, dt):
    """
    Calculate MFU for Mamba / hybrid architectures.

    Supports pure Mamba2 models as well as hybrid models that mix
    Mamba2, full attention, and linear attention (GDN / DeltaNet) layers.

    When ``context.layer_types`` is provided, per-layer costs are computed
    according to each layer's type.  Otherwise all layers are assumed
    to be Mamba2.

    References:
        - Mamba2 training FLOPs:  Eq. 9 + 11 in mamba_flops.md
        - GDN training FLOPs:     Eq. 8 + 10 in mamba_flops.md
        - Attention FLOPs:        Eq. 7  in mamba_flops.md
        - Overall formula:        Eq. 6  (F = 2 x total MACs per token)
        - source: https://arxiv.org/abs/2604.03444
    """
    d = context.hidden_size
    V = context.vocab_size

    # LM head MACs (embedding lookups excluded, only output projection).
    lm_head_macs = d * V

    # Per-layer MACs, dispatched by layer type.
    if context.layer_types:
        total_layer_macs = sum(
            _LAYER_MAC_FNS[lt](context) for lt in context.layer_types
        )
    else:
        # No explicit layer_types -> assume every layer is Mamba2.
        total_layer_macs = context.num_hidden_layers * _mamba_layer_macs(context)

    # FLOPs = 2 x MACs  (Eq. 6)
    flops_per_token = 2 * (lm_head_macs + total_layer_macs)
    # Forward + backward ≈ 3 x forward
    flops_per_fwdbwd = 3 * flops_per_token * context.sequence_length
    flops_per_iter = flops_per_fwdbwd * (micro_batch_size * gradient_accumulation_steps)
    flops_achieved = flops_per_iter / dt
    return (flops_achieved / context.peak_flops) * 100


MFU_REGISTRY = {
    "dense_transformer": _dense_transformer_mfu,
    "mamba": _mamba_mfu,
    "hybrid": _mamba_mfu,
    # Pure MoE transformers reuse the dense-transformer formula; the trainer
    # passes the *active* parameter count (params actually used per token)
    # as `num_parameters`. See _compute_active_trainable_params in model_setup.py.
    "moe": _dense_transformer_mfu,
    # MoE-hybrid (e.g. GraniteMoeHybrid: mamba + attention layers, with a
    # routed-MoE MLP on every layer, optionally plus a shared expert).
    # Reuses the structural mamba/hybrid formula; the per-layer MLP term is
    # MoE-aware via `num_experts_per_tok`, `moe_intermediate_size`, and
    # `shared_intermediate_size` on the MFUContext.
    "moe_hybrid": _mamba_mfu,
}


def create_mfu_context(args, hardware, num_parameters):
    peak_flops = PEAK_FLOPS_BY_HARDWARE.get(hardware.lower())
    if peak_flops is None:
        raise ValueError("Hardware not supported for MFU calculation.")

    layer_types = tuple(getattr(args, 'layer_types', ()) or ())

    return MFUContext(
        mfu_type=args.mfu_type,
        peak_flops=peak_flops,
        num_parameters=num_parameters,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        head_dim=args.head_dim,
        sequence_length=args.max_position_embeddings,
        # Extended fields
        hidden_size=getattr(args, 'hidden_size', 0),
        vocab_size=getattr(args, 'vocab_size', 0),
        intermediate_size=getattr(args, 'intermediate_size', 0),
        layer_types=layer_types,
        num_key_value_heads=getattr(args, 'num_key_value_heads', 0) or 0,
        # Mamba2
        mamba_d_state=getattr(args, 'mamba_d_state', 0),
        mamba_chunk_size=getattr(args, 'mamba_chunk_size', 256),
        mamba_d_conv=getattr(args, 'mamba_d_conv', 4),
        mamba_n_heads=getattr(args, 'mamba_n_heads', 0),
        mamba_d_head=getattr(args, 'mamba_d_head', 0),
        mamba_n_groups=getattr(args, 'mamba_n_groups', 1),
        # Linear attention
        linear_num_key_heads=getattr(args, 'linear_num_key_heads', 0),
        linear_num_value_heads=getattr(args, 'linear_num_value_heads', 0),
        linear_key_head_dim=getattr(args, 'linear_key_head_dim', 0),
        linear_value_head_dim=getattr(args, 'linear_value_head_dim', 0),
        linear_conv_kernel_dim=getattr(args, 'linear_conv_kernel_dim', 4),
        # MoE
        num_experts_per_tok=getattr(args, 'num_experts_per_tok', 0) or 0,
        moe_intermediate_size=getattr(args, 'moe_intermediate_size', 0) or 0,
        shared_intermediate_size=getattr(args, 'shared_intermediate_size', 0) or 0,
    )


def calculate_training_metrics(mfu_context, micro_batch_size, gradient_accumulation_steps, world_size, dt):
    if dt <= 0:
        raise ValueError("Step duration must be positive for MFU calculation.")

    strategy = MFU_REGISTRY.get(mfu_context.mfu_type)
    if strategy is None:
        supported_types = ", ".join(sorted(MFU_REGISTRY))
        raise ValueError(
            f"Invalid MFU type: '{mfu_context.mfu_type}'. Supported types are: {supported_types}."
        )

    tokens_processed = micro_batch_size * gradient_accumulation_steps * mfu_context.sequence_length * world_size
    global_tokens_per_sec = tokens_processed / dt
    tokens_per_sec_per_gpu = global_tokens_per_sec / world_size
    mfu = strategy(mfu_context, micro_batch_size, gradient_accumulation_steps, dt)

    return TrainingPerformanceMetrics(
        tokens_processed=tokens_processed,
        global_tokens_per_sec=global_tokens_per_sec,
        tokens_per_sec_per_gpu=tokens_per_sec_per_gpu,
        mfu=mfu,
    )