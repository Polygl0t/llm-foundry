"""
Tokenizer and model initialization utilities for the distributed trainers.

This module owns the pre-distributed model setup path so the trainer can consume a
single explicit result object.

Provides:
    - `ModelInitializationResult` dataclass that encapsulates the tokenizer, model, and related state.
    - `prepare_training_components()` function that initializes the tokenizer and model.
    - `apply_fsdp_wrapping()` function that applies FSDP2 sharding to the model.
    - `get_full_model_state_dict()` utility to gather the full model state dict for checkpointing.
    - `get_full_optimizer_state_dict()` utility to gather the full optimizer state dict for checkpointing.
"""
from dataclasses import dataclass
from typing import Any, Optional
import importlib
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelInitializationResult:
    """Explicit state returned by the tokenizer/model setup pipeline."""

    args: Any
    tokenizer: Any
    model: torch.nn.Module
    precision: torch.dtype
    checkpoint_path: Optional[str]
    trainable_params: int
    active_trainable_params: int


def _log_message(master_process, logger, file_logger, message):
    """Helper function to log messages to both the console and the file logger."""
    if not master_process:
        return

    if logger is not None:
        logger.info(message)
    if file_logger is not None:
        file_logger.log_metadata(message)


def _resolve_checkpoint_path(resume_from_checkpoint):
    """
    Determine the correct checkpoint path to resume from, if any.
    We always want to resume from the latest checkpoint in the specified directory, 
    but we also want to allow users to specify a specific checkpoint path if they choose to.
    """
    if not resume_from_checkpoint:
        return None

    checkpoint_path = resume_from_checkpoint
    try:
        checkpoint_dirs = os.listdir(checkpoint_path)
        checkpoint_dirs = [
            directory for directory in checkpoint_dirs if directory.startswith("step_")
        ]
        checkpoint_path = os.path.join(
            checkpoint_path,
            sorted(
                checkpoint_dirs,
                key=lambda directory: int(directory.split("_")[-1].split(".")[0]),
            )[-1],
        )
    except Exception:
        pass

    return checkpoint_path


def _create_tokenizer(args, master_process, logger=None, file_logger=None):
    """
    Create and return a tokenizer based on the provided arguments.
    """
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": True,
        "token": args.hub_token,
    }

    if args.tokenizer_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            **tokenizer_kwargs,
        )
    elif args.base_model is not None:
        _log_message(
            master_process,
            logger,
            file_logger,
            f"No tokenizer name specified, using {args.base_model} to load the tokenizer.",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            **tokenizer_kwargs,
        )
    elif not args.continual_pretraining:
        _log_message(
            master_process,
            logger,
            file_logger,
            "No tokenizer specified. Continuing without a tokenizer because training is configured from scratch.",
        )
        tokenizer = None
    else:
        raise ValueError(
            "Either `tokenizer_name_or_path` or `base_model` must be set to load a tokenizer for continual pretraining."
        )

    if tokenizer is not None and args.chat_template_path is not None:
        with open(args.chat_template_path, "r") as handle:
            tokenizer.chat_template = handle.read()
        _log_message(
            master_process,
            logger,
            file_logger,
            f"Loaded chat template from {args.chat_template_path}. Chat template added to the tokenizer.",
        )
    elif tokenizer is None and args.chat_template_path is not None:
        _log_message(
            master_process,
            logger,
            file_logger,
            f"WARNING: chat_template_path={args.chat_template_path} was provided but no tokenizer was loaded. Skipping chat template setup.",
        )

    return tokenizer


def _build_model_from_config(args, tokenizer, precision, master_process, distributed_config=None, use_kernels=False):
    """
    Build and return a model with random weights from a Hugging Face config file.

    The config file (pointed to by `args.path_to_model_config`) defines all
    architecture parameters. Only runtime kwargs (token IDs, dtype) are injected here.

    To gain access to `from_pretrained`-only features (`use_kernels`,
    `distributed_config`/expert parallelism, `tp_plan`, `kernel_config`, ...), the
    randomly initialized model is materialized once on the master rank, written to a
    bootstrap checkpoint directory (prefixed with `.` so the resume logic in
    `_resolve_checkpoint_path`, which filters by `startswith("step_")`, ignores it),
    then reloaded on every rank via `from_pretrained`.
    """
    if args.path_to_model_config is None:
        raise ValueError(
            "`path_to_model_config` must be set when training from scratch. "
            "Point it to a Hugging Face-compatible config file (e.g., config.json) or directory."
        )

    runtime_kwargs = {
        "token": args.hub_token,
        "dtype": precision,
    }
    if tokenizer is not None:
        runtime_kwargs.update(
            {
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "unk_token_id": tokenizer.unk_token_id,
            }
        )

    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=args.path_to_model_config,
        cache_dir=args.cache_dir,
        **runtime_kwargs,
    )

    # When a tokenizer is available, keep the config large enough to host it.
    if tokenizer is not None:
        config.vocab_size = max(config.vocab_size, len(tokenizer))

    # Bootstrap checkpoint path. Leading `.` keeps it invisible to the resume logic
    # in `_resolve_checkpoint_path` (which filters dirs by `startswith("step_")`).
    bootstrap_dir = os.path.join(args.checkpoint_dir, args.stage_name, ".step_00000")

    # Step 1 (master only): build the random model with `from_config` and persist it.
    if master_process:
        os.makedirs(bootstrap_dir, exist_ok=True)
        random_model = AutoModelForCausalLM.from_config(
            config,
            attn_implementation=args.attn_implementation,
        )
        random_model.save_pretrained(bootstrap_dir)
        # Free memory before all ranks reload via `from_pretrained`.
        del random_model

    # Step 2: synchronize so every rank sees the bootstrap checkpoint on disk.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Step 3 (all ranks): reload through `from_pretrained` so kernels and
    # distributed_config are applied via the supported code path.
    return AutoModelForCausalLM.from_pretrained(
        bootstrap_dir,
        dtype=precision,
        attn_implementation=args.attn_implementation,
        cache_dir=args.cache_dir,
        **({"distributed_config": distributed_config} if distributed_config is not None else {}),
        **({"use_kernels": True} if use_kernels else {}),
    )


def _load_model(args, tokenizer, precision, master_process, logger=None, file_logger=None, distributed_config=None, use_kernels=False):
    """
    Load a model from a checkpoint or initialize a new model based on the provided arguments.
    """
    checkpoint_path = _resolve_checkpoint_path(args.resume_from_checkpoint)

    if checkpoint_path is not None:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            dtype=precision,
            attn_implementation=args.attn_implementation,
            cache_dir=args.cache_dir,
            **({"distributed_config": distributed_config} if distributed_config is not None else {}),
            **({"use_kernels": True} if use_kernels else {}),
        )
        _log_message(
            master_process,
            logger,
            file_logger,
            f"Resumed model from checkpoint: {checkpoint_path}",
        )
        return model, checkpoint_path

    if not args.continual_pretraining:
        _log_message(master_process, logger, file_logger, "Initializing model from `AutoConfig`.")
        return _build_model_from_config(
            args, tokenizer, precision, master_process,
            distributed_config=distributed_config, use_kernels=use_kernels,
        ), None

    _log_message(
        master_process,
        logger,
        file_logger,
        f"Initializing model from base model: {args.base_model} for continual pretraining/fine-tuning.",
    )

    config = None
    needs_context_extension = (
        args.new_max_position_embeddings is not None
        or args.rope_scale_factor is not None
        or args.new_rope_theta is not None
    )

    if needs_context_extension:
        config = AutoConfig.from_pretrained(args.base_model, cache_dir=args.cache_dir)
        original_max_pos = config.max_position_embeddings

        # Apply max_position_embeddings override (explicit value takes priority over scale factor)
        if args.new_max_position_embeddings is not None:
            config.max_position_embeddings = args.new_max_position_embeddings
        elif args.rope_scale_factor is not None:
            config.max_position_embeddings = int(config.max_position_embeddings * args.rope_scale_factor)

        # Apply rope_theta override
        if args.new_rope_theta is not None:
            config.rope_theta = args.new_rope_theta
        elif config.max_position_embeddings != original_max_pos:
            # Warn if scaling positions without scaling theta
            if master_process and logger is not None:
                logger.info(
                    "WARNING: max_position_embeddings was scaled but rope_theta was not overridden. "
                    "Consider setting `new_rope_theta` to a larger value for context extension."
                )

        _log_message(
            master_process,
            logger,
            file_logger,
            f"Context extension: max_position_embeddings={config.max_position_embeddings}, rope_theta={config.rope_theta}.",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=precision,
        attn_implementation=args.attn_implementation,
        cache_dir=args.cache_dir,
        config=config,
        **({"distributed_config": distributed_config} if distributed_config is not None else {}),
        **({"use_kernels": True} if use_kernels else {}),
    )
    return model, None


def _resize_embeddings_for_tokenizer(model, tokenizer, master_process, logger=None, file_logger=None):
    """Ensure pretrained models can represent every tokenizer token."""
    if tokenizer is None:
        return model

    tokenizer_vocab_size = len(tokenizer)
    current_vocab_size = model.get_input_embeddings().num_embeddings
    target_vocab_size = max(current_vocab_size, tokenizer_vocab_size)

    if target_vocab_size == current_vocab_size:
        return model

    model.resize_token_embeddings(target_vocab_size)
    _log_message(
        master_process,
        logger,
        file_logger,
        f"Resized token embeddings from {current_vocab_size:,} to {target_vocab_size:,} to cover the tokenizer vocabulary.",
    )
    return model


def _apply_liger_kernels(model, args):
    """
    Apply Liger kernels to the model for optimized performance.

    Liger's RoPE replacement is only valid for HF rotary embedding modules
    with the standard interface (Llama / Qwen3 / Qwen2.5 ...). Qwen3.5 uses
    a customized rotary embedding (partial rotation, per-layer shapes) that
    is not compatible with Liger's RoPE kernel, so we disable it there.
    """
    liger_transformers = importlib.import_module("liger_kernel.transformers")
    apply_liger_kernel = getattr(liger_transformers, "_apply_liger_kernel_to_instance")
    model_type = str(getattr(model.config, "model_type", "") or "")
    rope_compatible = not model_type.startswith("qwen3_5")
    liger_kwargs = {
        "rope": rope_compatible,
        "cross_entropy": False,
        "fused_linear_cross_entropy": True,
        "rms_norm": True,
        "swiglu": True,
    }
    apply_liger_kernel(model=model, **liger_kwargs)


def _try_create_distributed_config(enable_expert_parallelism, master_process, logger=None, file_logger=None):
    """
    Attempt to create a DistributedConfig with expert parallelism enabled.

    Returns the config object if successful, or None if the import fails
    (e.g. older transformers version) or the feature is not requested.
    """
    if not enable_expert_parallelism:
        return None

    try:
        from transformers.distributed.configuration_utils import DistributedConfig
        _log_message(
            master_process, logger, file_logger,
            "Expert parallelism enabled via DistributedConfig.",
        )
        return DistributedConfig(enable_expert_parallel=True)
    except (ImportError, ModuleNotFoundError):
        _log_message(
            master_process, logger, file_logger,
            "WARNING: enable_expert_parallelism is True but DistributedConfig could not be imported. "
            "Expert parallelism requires transformers >= 5.x. Continuing without it.",
        )
        return None


def _check_kernels_available(use_kernels, master_process, logger=None, file_logger=None):
    """
    Check whether the `kernels` library and transformers' `use_kernels`
    support are available.

    Returns True if `use_kernels` was requested **and** both the `kernels`
    package and the transformers kwarg are importable, False otherwise.
    """
    if not use_kernels:
        return False

    try:
        import kernels as _kernels  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        _log_message(
            master_process, logger, file_logger,
            "WARNING: use_kernels is True but the `kernels` package is not installed. "
            "Install it with `pip install -U kernels` (>= 0.11.0). Continuing without kernels.",
        )
        return False

    # Verify that the installed transformers version actually supports use_kernels.
    # Probe for `transformers.KernelConfig` to make sure the kwarg is recognized, 
    # since older versions of transformers may ignore the `use_kernels` 
    # argument without error.
    try:
        from transformers import KernelConfig  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        _log_message(
            master_process, logger, file_logger,
            "WARNING: use_kernels is True but the installed transformers version does not support "
            "the `use_kernels` kwarg. Upgrade transformers to a compatible version. Continuing without kernels.",
        )
        return False

    _log_message(
        master_process, logger, file_logger,
        "Optimized HF Hub kernels enabled (use_kernels=True).",
    )
    # To use specific kernel mappings, create a KernelConfig:
    #   from transformers import KernelConfig
    #   kernel_config = KernelConfig(
    #       kernel_mapping={
    #           "RMSNorm": "kernels-community/liger_kernels:LigerRMSNorm",
    #       }
    #   )
    # and pass `kernel_config=kernel_config` alongside `use_kernels=True`.
    return True


def _compute_active_trainable_params(config, trainable_params):
    """
    Compute the number of active trainable parameters.

    For dense models, active_trainable_params == trainable_params.
    For MoE models, only the routed experts selected per token
    (num_experts_per_tok) are counted, since the remaining experts
    are inactive during each forward pass.

    - Note: Handles some naming conventions for MoE-related config fields, 
            but you might need to adjust this function if your model uses 
            different field names or MoE architecture.
    """
    # Detect MoE: try both naming conventions for the total expert count.
    num_experts = getattr(config, 'num_experts', None) or getattr(config, 'num_local_experts', None)
    if num_experts is None or num_experts <= 1:
        return trainable_params

    num_experts_per_tok = getattr(config, 'num_experts_per_tok', None)
    if num_experts_per_tok is None or num_experts_per_tok >= num_experts:
        return trainable_params

    hidden_size = config.hidden_size

    # Per-expert MLP intermediate size.
    expert_intermediate_size = getattr(config, 'moe_intermediate_size', None) or config.intermediate_size

    # SwiGLU MLP per expert: gate_proj + up_proj + down_proj
    params_per_expert = 3 * hidden_size * expert_intermediate_size

    # Number of MoE layers. Qwen-style configs use `decoder_sparse_step` (every
    # k-th layer is MoE) and `mlp_only_layers` (explicit indices that use a
    # dense MLP instead of MoE). Both are honored when present; otherwise the
    # default assumption is that every layer is a MoE layer.
    decoder_sparse_step = getattr(config, 'decoder_sparse_step', 1) or 1
    mlp_only_layers = set(getattr(config, 'mlp_only_layers', None) or [])
    num_moe_layers = sum(
        1 for layer_idx in range(config.num_hidden_layers)
        if layer_idx not in mlp_only_layers
        and (layer_idx + 1) % decoder_sparse_step == 0
    )

    inactive_params = num_moe_layers * (num_experts - num_experts_per_tok) * params_per_expert
    return trainable_params - inactive_params


def prepare_training_components(args, device, master_process, logger=None, file_logger=None):
    """Build tokenizer/model state needed by the trainer before DDP|FSDP wrapping."""
    torch.set_float32_matmul_precision(args.mat_mul_precision)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = args.bf16
    precision = torch.bfloat16 if args.bf16 else torch.float32

    tokenizer = _create_tokenizer(args, master_process, logger, file_logger)

    distributed_config = _try_create_distributed_config(
        args.enable_expert_parallelism, master_process, logger, file_logger,
    )

    use_kernels = _check_kernels_available(
        args.use_kernels, master_process, logger, file_logger,
    )

    model, checkpoint_path = _load_model(
        args, tokenizer, precision, master_process, logger, file_logger,
        distributed_config=distributed_config,
        use_kernels=use_kernels,
    )

    if args.continual_pretraining:
        model = _resize_embeddings_for_tokenizer(
            model, tokenizer, master_process, logger, file_logger,
        )

    # Backfill runtime architecture fields declared in TrainingArguments
    # (consumed by mfu.py, data_loading.py, utils.py, train_ddp.py)
    args.max_position_embeddings = model.config.max_position_embeddings
    args.vocab_size = model.config.vocab_size
    args.num_hidden_layers = model.config.num_hidden_layers
    args.num_attention_heads = model.config.num_attention_heads
    args.head_dim = getattr(
        model.config, 'head_dim',
        model.config.hidden_size // model.config.num_attention_heads,
    )
    # GQA: fall back to num_attention_heads (MHA) when not specified.
    args.num_key_value_heads = getattr(
        model.config, 'num_key_value_heads', model.config.num_attention_heads,
    )

    # Architecture fields consumed by the structural MFU path for hybrid models.
    # No-ops for the standard dense / MoE-active dense_transformer path.
    args.hidden_size = getattr(model.config, 'hidden_size', 0)
    args.intermediate_size = getattr(model.config, 'intermediate_size', 0)
    # `layer_types` lists each block's flavour. Supported values for the
    # families this codebase targets: "full_attention" / "attention" and
    # "linear_attention" (Qwen3.5 Gated-DeltaNet hybrid). Some configs encode
    # the schedule via `full_attention_interval` (every k-th layer is full
    # attention, the rest are linear-attention); synthesize `layer_types`
    # from it when present so the MFU path works without architecture-specific
    # branching.
    layer_types = tuple(getattr(model.config, 'layer_types', ()) or ())
    if not layer_types:
        full_attention_interval = getattr(model.config, 'full_attention_interval', None)
        if full_attention_interval and full_attention_interval > 0:
            layer_types = tuple(
                "full_attention" if (layer_idx + 1) % full_attention_interval == 0
                else "linear_attention"
                for layer_idx in range(model.config.num_hidden_layers)
            )
    args.layer_types = layer_types

    # Linear attention (GDN / DeltaNet) hyperparameters (e.g. Qwen3.5 hybrid).
    args.linear_num_key_heads = getattr(model.config, 'linear_num_key_heads', 0) or 0
    args.linear_num_value_heads = getattr(model.config, 'linear_num_value_heads', 0) or 0
    args.linear_key_head_dim = getattr(model.config, 'linear_key_head_dim', 0) or 0
    args.linear_value_head_dim = getattr(model.config, 'linear_value_head_dim', 0) or 0
    args.linear_conv_kernel_dim = getattr(model.config, 'linear_conv_kernel_dim', 4) or 4

    # MoE fields. Used by the hybrid structural FLOPs path for per-layer MLP
    # cost; for the dense path, MoE accounting goes through `num_parameters`
    # (active parameters, computed in `_compute_active_trainable_params`).
    _num_experts = (
        getattr(model.config, 'num_experts', None)
        or getattr(model.config, 'num_local_experts', None)
        or 0
    )
    if _num_experts and _num_experts > 1:
        args.num_experts_per_tok = getattr(model.config, 'num_experts_per_tok', 0) or 0
        args.moe_intermediate_size = (
            getattr(model.config, 'moe_intermediate_size', None)
            or getattr(model.config, 'intermediate_size', 0)
            or 0
        )
        # Qwen-MoE / Qwen3.5-MoE name the shared-expert size
        # `shared_expert_intermediate_size`; accept `shared_intermediate_size`
        # as an alias for backwards compatibility.
        args.shared_intermediate_size = (
            getattr(model.config, 'shared_expert_intermediate_size', None)
            or getattr(model.config, 'shared_intermediate_size', None)
            or 0
        )
    else:
        args.num_experts_per_tok = 0
        args.moe_intermediate_size = 0
        args.shared_intermediate_size = 0

    if tokenizer is not None:
        tokenizer.model_max_length = model.config.max_position_embeddings

    # Warn if training a hybrid (used linear-attention) model without the
    # fast-path kernels. E.g., the Qwen3.5 modeling code only takes the optimized
    # path when BOTH `flash-linear-attention` (chunk / fused gated-delta-rule)
    # AND `causal-conv1d` (the short-conv branch of GatedDeltaNet) are
    # importable; missing either falls back to a slow PyTorch reference path.
    if "linear_attention" in args.layer_types:
        missing = []
        try:
            import fla  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            missing.append("flash-linear-attention")
        try:
            import causal_conv1d  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            missing.append("causal-conv1d")
        if missing:
            _log_message(
                master_process, logger, file_logger,
                "WARNING: the model has linear-attention layers but the following fast-path "
                f"package(s) are not installed: {', '.join(missing)}. Training will fall back "
                "to a slow PyTorch reference path. Install with:\n"
                "    pip install flash-linear-attention causal-conv1d\n"
                "See https://github.com/fla-org/flash-linear-attention#installation and "
                "https://github.com/Dao-AILab/causal-conv1d for details.",
            )

    if args.use_liger_kernel:
        _apply_liger_kernels(model, args)
        _log_message(master_process, logger, file_logger, "Applied Liger kernels to the model.")

    model.config.name_or_path = args.hub_model_id

    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    active_trainable_params = _compute_active_trainable_params(model.config, trainable_params)
    _log_message(
        master_process,
        logger,
        file_logger,
        f"Number of trainable parameters: {trainable_params:,}",
    )
    if active_trainable_params != trainable_params:
        _log_message(
            master_process,
            logger,
            file_logger,
            f"Number of active trainable parameters (MoE): {active_trainable_params:,}",
        )

    # Disable KV cache during training — it's only needed for autoregressive generation.
    # With use_cache=True the model outputs past_key_values (the full KV tensors for the
    # sequence) on every forward pass, which wastes memory and can cause apparent VRAM
    # spikes (especially when switching between train/eval mode at checkpoint steps).
    model.config.use_cache = False

    if args.gradient_checkpointing:
        _log_message(master_process, logger, file_logger, "Gradient checkpointing enabled.")
        # IMPORTANT: For FSDP, always use `use_reentrant=False`. Reentrant checkpointing is incompatible
        # with FSDP because it doesn't properly handle the sharded parameter semantics.
        # Using reentrant=True with FSDP can cause:
        # - Incorrect gradient computation
        # - Memory leaks due to retained activation graphs
        # - Deadlocks during backward pass
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
            }
        )

    # Torch Compile
    # See https://docs.pytorch.org/docs/stable/generated/torch.compile.html
    # WARNING: Torch compile is not working good with liger kernel: https://github.com/linkedin/Liger-Kernel/issues/174
    if args.torch_compile and not args.use_liger_kernel:
        if master_process and logger is not None:
            logger.info("Compiling model with torch.compile.")
        model = torch.compile(model)

    model.to(device)

    return ModelInitializationResult(
        args=args,
        tokenizer=tokenizer,
        model=model,
        precision=precision,
        checkpoint_path=checkpoint_path,
        trainable_params=trainable_params,
        active_trainable_params=active_trainable_params,
    )

# FSDP wrapping and state-dict utilities
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import CPUOffloadPolicy
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh

def _iter_transformer_blocks(model):
    """
    Yield every transformer block that should be individually sharded by FSDP.

    Modern HF causal-LM models (dense, MoE, Qwen3.5 hybrid) all expose their
    per-layer blocks under `model.model.layers` as a `ModuleList`. Sharding
    every entry in that list is the standard FSDP2 idiom (cf. the official
    PyTorch FSDP2 tutorial and torchtitan), and is architecture-agnostic: it
    works for dense, MoE, and hybrid models without registering any
    decoder-layer class up front.

    This helper centralizes the assumption and provides a clearer error if a
    new architecture deviates from it.
    """
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        raise ValueError(
            f"Model of type '{getattr(model.config, 'model_type', type(model).__name__)}' "
            f"does not expose `model.model.layers`. FSDP wrapping in this codebase "
            f"assumes that convention. Update `_iter_transformer_blocks` if your model "
            f"places its decoder blocks elsewhere."
        )
    return layers


def _set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    """Set the modules to be prefetched for forward pass."""
    for i, layer in enumerate(model.model.layers):
        if i >= len(model.model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def _set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    """Set the modules to be prefetched for backward pass."""
    for i, layer in enumerate(model.model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)


def apply_fsdp_wrapping(model, args, device_type, world_size, rank, master_process, logger=None, file_logger=None):
    """
    Apply FSDP2 (fully_shard) wrapping to the model.

    This function shards each decoder layer individually, then shards the root
    model.  It supports mixed precision, CPU offload, HSDP (2-D device mesh),
    and explicit prefetching — all controlled by the fields on `args`.

    Returns:
        effective_world_size (int): The data-parallel world size after accounting
            for HSDP.  Callers should use this for gradient-accumulation and
            sampler calculations.
    """
    fsdp_kwargs = {}

    # Mixed precision
    if args.fsdp_mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        _log_message(
            master_process, logger, file_logger,
            "Enabled mixed precision policy for FSDP. Param type = torch.bfloat16, Reduce type = torch.float32",
        )

    # Device mesh and HSDP setup
    effective_world_size = world_size

    if args.dp_shard is None:
        mesh_config = init_device_mesh(
            device_type=device_type,
            mesh_shape=(world_size,),
        )
        _log_message(
            master_process, logger, file_logger,
            f"Initialized 1D device mesh with shape: ({world_size},) for Fully Sharded Data Parallel (FSDP).",
        )
    else:
        assert world_size % args.dp_shard == 0, (
            f"World size {world_size} needs to be divisible by `dp_shard` size "
            f"(dp_shard={args.dp_shard}, world_size={world_size})"
        )
        assert args.dp_shard > 1, f"dp_shard needs to be greater than 1 (dp_shard={args.dp_shard})."

        data_parallel_size = world_size // args.dp_shard
        mesh_config = init_device_mesh(
            device_type=device_type,
            mesh_shape=(data_parallel_size, args.dp_shard),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        effective_world_size = data_parallel_size
        _log_message(
            master_process, logger, file_logger,
            f"Initialized 2D device mesh with shape: (dp_replicate={data_parallel_size}, dp_shard={args.dp_shard}) for Hybrid Sharding Data Parallel (HSDP).",
        )

    fsdp_kwargs["mesh"] = mesh_config

    # Sharding strategy (ZeRO-3 vs ZeRO-2)
    fsdp_kwargs["reshard_after_forward"] = True if args.full_shard else False
    _log_message(
        master_process, logger, file_logger,
        f"FSDP / ZeRO Stage is set to {'ZeroStage3' if args.full_shard else 'ZeroStage2'}",
    )

    # CPU offload
    if args.cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=True)
        _log_message(master_process, logger, file_logger, "Enabled CPU offload policy for FSDP.")

    # Per-layer sharding (bottom-up, as required by FSDP2). We wrap every
    # block in `model.model.layers` regardless of its concrete class. This is
    # architecture-agnostic and supports dense (Llama, Qwen3, Qwen3.5), MoE
    # (Qwen3.5-MoE), and Qwen3.5 linear-attention hybrid models without
    # needing to register their decoder-layer class first.
    layer_classes = set()
    for layer in _iter_transformer_blocks(model):
        fully_shard(layer, **fsdp_kwargs)
        layer_classes.add(type(layer).__name__)
    _log_message(
        master_process, logger, file_logger,
        f"FSDP per-layer sharding applied to block classes: {sorted(layer_classes)}.",
    )

    # Shard the root model (covers embeddings, output projection, etc.).
    fully_shard(model, **fsdp_kwargs)

    # Explicit prefetching
    if args.explicit_prefetching:
        _set_modules_to_forward_prefetch(model, num_to_forward_prefetch=2)
        _set_modules_to_backward_prefetch(model, num_to_backward_prefetch=2)

    return effective_world_size


def get_full_model_state_dict(model):
    """
    Retrieve the full (un-sharded) model state dict from an FSDP-wrapped model.
    Must be called on **all** ranks; rank-0 receives the complete dict.

    References:
        - https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#state-dict-with-dcp-apis
        - https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
    """
    return get_model_state_dict(
        model=model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        ),
    )


def get_full_optimizer_state_dict(model, optimizer):
    """
    Retrieve the full (un-sharded) optimizer state dict from an FSDP-wrapped model.
    Must be called on **all** ranks; rank-0 receives the complete dict.

    References:
        - https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#state-dict-with-dcp-apis
        - https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
    """
    return get_optimizer_state_dict(
        model=model,
        optimizers=optimizer,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        ),
    )