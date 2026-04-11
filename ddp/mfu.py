"""
MFU calculation helpers for the DDP trainer.

The strategy registry makes it straightforward to add alternative formulas for
other architectures (MoE, Mamba, Hybrid) without changing the training loop.

Provides:
    - `MFUContext` dataclass that encapsulates all necessary information about the model and hardware for MFU calculation.
    - `TrainingPerformanceMetrics` dataclass that encapsulates the results of the MFU calculation and related performance metrics.
    - `create_mfu_context()` function to create an `MFUContext` from training arguments and hardware information.
    - `calculate_training_metrics()` function that computes the MFU and related performance metrics given the context and training step information.
"""
from dataclasses import dataclass


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


@dataclass(frozen=True)
class TrainingPerformanceMetrics:
    tokens_processed: int
    global_tokens_per_sec: float
    tokens_per_sec_per_gpu: float
    mfu: float


def _dense_transformer_mfu(context, micro_batch_size, gradient_accumulation_steps, dt):
    flops_per_token = (
        6 * context.num_parameters
        + 12 * context.num_hidden_layers * context.num_attention_heads * context.head_dim * context.sequence_length
    )
    flops_per_fwdbwd = flops_per_token * context.sequence_length
    flops_per_iter = flops_per_fwdbwd * (micro_batch_size * gradient_accumulation_steps)
    flops_achieved = flops_per_iter / dt
    return (flops_achieved / context.peak_flops) * 100


MFU_REGISTRY = {
    "dense_transformer": _dense_transformer_mfu,
}


def create_mfu_context(args, hardware, num_parameters):
    peak_flops = PEAK_FLOPS_BY_HARDWARE.get(hardware.lower())
    if peak_flops is None:
        raise ValueError("Hardware not supported for MFU calculation.")

    return MFUContext(
        mfu_type=args.mfu_type,
        peak_flops=peak_flops,
        num_parameters=num_parameters,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        head_dim=args.head_dim,
        sequence_length=args.max_position_embeddings,
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