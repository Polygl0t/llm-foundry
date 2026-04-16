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
    # MLP (SwiGLU: gate + up + down projections)               (Eq. 9: 3d.d_MLP)
    mlp = 3 * d * ctx.intermediate_size

    return in_proj + conv + out_proj + intra_chunk + inter_chunk + mlp


def _attention_layer_macs(ctx):
    """
    MACs per token for a full (causal) attention layer.
    """
    d = ctx.hidden_size
    s = ctx.sequence_length
    projections = 4 * d * d                     # Q, K, V, O projections
    attention   = d * s                         # QK^T + attn.V  (causal avg)
    mlp         = 3 * d * ctx.intermediate_size # SwiGLU MLP
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
    # MLP                                                              (Eq. 8)
    mlp = 3 * d * ctx.intermediate_size

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