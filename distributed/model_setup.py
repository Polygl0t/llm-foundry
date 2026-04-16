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
    else:
        raise ValueError(
            "Either `tokenizer_name_or_path` or `base_model` must be set to load a tokenizer."
        )

    if args.chat_template_path is not None:
        with open(args.chat_template_path, "r") as handle:
            tokenizer.chat_template = handle.read()
        _log_message(
            master_process,
            logger,
            file_logger,
            f"Loaded chat template from {args.chat_template_path}. Chat template added to the tokenizer.",
        )

    return tokenizer


def _build_model_from_config(args, tokenizer, precision, distributed_config=None, use_kernels=False):
    """
    Build and return a model with random weights from a Hugging Face config file.

    The config file (pointed to by `args.path_to_model_config`) defines all
    architecture parameters. Only runtime kwargs (token IDs, dtype) are injected here.
    """
    if args.path_to_model_config is None:
        raise ValueError(
            "`path_to_model_config` must be set when training from scratch. "
            "Point it to a Hugging Face-compatible config file (e.g., config.json) or directory."
        )

    runtime_kwargs = {
        "token": args.hub_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "torch_dtype": precision,
    }

    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=args.path_to_model_config,
        cache_dir=args.cache_dir,
        **runtime_kwargs,
    )

    # Ensure vocab_size is at least as large as the tokenizer
    config.vocab_size = max(config.vocab_size, len(tokenizer))

    return AutoModelForCausalLM.from_config(
        config,
        attn_implementation=args.attn_implementation,
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
            torch_dtype=precision,
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
        return _build_model_from_config(args, tokenizer, precision, distributed_config=distributed_config, use_kernels=use_kernels), None

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
        torch_dtype=precision,
        attn_implementation=args.attn_implementation,
        cache_dir=args.cache_dir,
        config=config,
        **({"distributed_config": distributed_config} if distributed_config is not None else {}),
        **({"use_kernels": True} if use_kernels else {}),
    )
    return model, None


def _apply_liger_kernels(model, args):
    """
    Apply Liger kernels to the model for optimized performance.
    """
    liger_transformers = importlib.import_module("liger_kernel.transformers")
    apply_liger_kernel = getattr(liger_transformers, "_apply_liger_kernel_to_instance")
    liger_kwargs = {
        "rope": True,
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
    Check whether the ``kernels`` library and transformers' ``use_kernels``
    support are available.

    Returns True if ``use_kernels`` was requested **and** both the ``kernels``
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

    # Verify that the installed transformers version actually accepts use_kernels.
    import inspect
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    if "use_kernels" not in sig.parameters:
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

    # Number of MoE layers (Qwen uses decoder_sparse_step; default = all layers).
    decoder_sparse_step = getattr(config, 'decoder_sparse_step', 1) or 1
    num_moe_layers = config.num_hidden_layers // decoder_sparse_step

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

    tokenizer.model_max_length = model.config.max_position_embeddings

    # Warn if training a hybrid/mamba model without the optimized CUDA kernels.
    if args.mfu_type in ("mamba", "hybrid"):
        _mamba_kernels_available = True
        try:
            from mamba_ssm import Mamba
        except (ImportError, ModuleNotFoundError):
            _mamba_kernels_available = False
        try:
            from causal_conv1d import causal_conv1d_fn
        except (ImportError, ModuleNotFoundError):
            _mamba_kernels_available = False

        if not _mamba_kernels_available:
            _log_message(
                master_process, logger, file_logger,
                "WARNING: mfu_type is set to '{}' but the mamba-ssm and/or causal-conv1d "
                "packages are not installed. Training will be significantly slower without "
                "the optimized CUDA kernels. Install them with:\n"
                "    pip install mamba-ssm[causal-conv1d] --no-build-isolation\n"
                "See https://github.com/state-spaces/mamba/#installation for details.".format(args.mfu_type),
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
        model.config.use_cache = False

    # [Torch Compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
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

# Decoder layer classes used to identify which sub-modules should be
# individually sharded by FSDP.  Add more mappings as needed.
_DECODER_LAYER_MAP = {}

def _get_decoder_layer_class(model_type):
    """
    Lazily import and return the decoder layer class for a given model type.
    Caches the result in _DECODER_LAYER_MAP.
    """
    if model_type in _DECODER_LAYER_MAP:
        return _DECODER_LAYER_MAP[model_type]

    _import_map = {
        "smollm3": ("transformers.models.smollm3.modeling_smollm3", "SmolLM3DecoderLayer"),
        "llama": ("transformers.models.llama.modeling_llama", "LlamaDecoderLayer"),
        "gemma3_text": ("transformers.models.gemma3.modeling_gemma3", "Gemma3DecoderLayer"),
        "qwen3": ("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer"),
        "qwen2": ("transformers.models.qwen2.modeling_qwen2", "Qwen2DecoderLayer"),
    }

    if model_type not in _import_map:
        raise ValueError(
            f"Unsupported model_type '{model_type}' for FSDP decoder-layer sharding. "
            f"Supported types: {sorted(_import_map)}. "
            f"Add your model's decoder layer class to the mapping in model_setup.py."
        )

    module_path, class_name = _import_map[model_type]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    _DECODER_LAYER_MAP[model_type] = cls
    return cls


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
    and explicit prefetching — all controlled by the fields on ``args``.

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

    # Per-layer sharding (bottom-up, as required by FSDP2)
    decoder_cls = _get_decoder_layer_class(model.config.model_type)
    for layer in model.model.layers:
        if isinstance(layer, decoder_cls):
            fully_shard(layer, **fsdp_kwargs)

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