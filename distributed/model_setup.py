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

import importlib
import os
from dataclasses import dataclass
from typing import Any

import torch
from gdn_patch import patch_qwen3_5_gdn_initialization
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Attention (and attention-like) module class names per supported model family.
# Anything listed here is treated as an "attention block" whose parameters
# remain trainable during context-extension fine-tuning, while everything else
# (embeddings, norms, MLPs / MoE experts, lm_head) is frozen.
#
# Qwen3.5 hybrids list both the regular attention AND the linear-attention
# variant (GatedDeltaNet), because both implement positional / sequence mixing
# and should be trained when extending the context window.
ATTENTION_CLASS_NAMES = {
    "llama": {"LlamaAttention"},
    "qwen2": {"Qwen2Attention"},
    "qwen3": {"Qwen3Attention"},
    "qwen3_moe": {"Qwen3MoeAttention"},
    "qwen3_5_text": {"Qwen3_5Attention", "Qwen3_5GatedDeltaNet"},
    "qwen3_5_moe_text": {"Qwen3_5MoeAttention", "Qwen3_5MoeGatedDeltaNet"},
}


@dataclass
class ModelInitializationResult:
    """Explicit state returned by the tokenizer/model setup pipeline."""

    args: Any
    tokenizer: Any
    model: torch.nn.Module
    precision: torch.dtype
    checkpoint_path: str | None
    trainable_params: int
    active_trainable_params: int
    non_attention_frozen: bool = False
    fp8_enabled: bool = False
    linear_attention_fast_path: bool = False


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
        with open(args.chat_template_path) as handle:
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


def _build_model_from_config(
    args, tokenizer, precision, master_process, distributed_config=None, use_kernels=False
):
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
        # Upstream Qwen3.5 GatedDeltaNet initializes `A_log` in the model dtype,
        # so a bf16 `uniform_(0, 16)` can round to 0 and make `log(0) = -inf` on
        # some heads. Re-initialize the decay params (finite) before persisting.
        patch_qwen3_5_gdn_initialization(random_model, force_reinit=True)
        random_model.save_pretrained(bootstrap_dir, max_shard_size="5GB")
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


# Per-decoder-layer RMSNorms (siblings of the attention block) that we keep
# trainable during context-extension fine-tuning. The norm immediately
# preceding attention shapes the activation distribution that attention
# sees; freezing it can become a bottleneck at large RoPE scale factors
# (cf. LongRoPE, LLaMA-Pro). `post_attention_layernorm` is included for
# symmetry — the parameter count is negligible (one weight vector per norm
# per layer) and it keeps the residual-stream conditioning consistent.
TRAINABLE_PER_LAYER_NORM_CHILDREN = ("input_layernorm", "post_attention_layernorm")


def _freeze_non_attention_blocks(model, master_process, logger=None, file_logger=None):
    """
    Freeze every parameter that does not live inside an attention block,
    except for the per-decoder-layer RMSNorms adjacent to attention
    (`input_layernorm`, `post_attention_layernorm`), which are kept
    trainable.

    Used for context-extension fine-tuning: we only want to spend VRAM /
    optimizer state on the modules that actually need to adapt to the new
    sequence length (the attention blocks and, for Qwen3.5 hybrids, the
    linear-attention `GatedDeltaNet` blocks, plus the per-layer norms
    feeding them). Embeddings, the final `model.norm`, MLPs / MoE experts,
    and `lm_head` are all set to `requires_grad=False`.

    Returns the number of trainable / frozen parameters after the operation
    for logging convenience.
    """
    model_type = str(getattr(model.config, "model_type", "") or "")
    attention_class_names = ATTENTION_CLASS_NAMES.get(model_type)
    if attention_class_names is None:
        raise ValueError(
            f"Cannot freeze non-attention blocks: unsupported model_type={model_type!r}. "
            f"Expected one of {sorted(ATTENTION_CLASS_NAMES)}. Add the model's attention "
            f"class name(s) to `ATTENTION_CLASS_NAMES` to support context-extension freezing."
        )

    # Collect module-name prefixes of attention blocks; anything that matches
    # one of these prefixes (or is nested under one) stays trainable.
    attention_prefixes = set()
    for module_name, module in model.named_modules():
        if type(module).__name__ in attention_class_names:
            attention_prefixes.add(module_name)

    if not attention_prefixes:
        raise ValueError(
            f"Could not find any attention blocks for model_type={model_type!r}; "
            f"refusing to freeze the entire model."
        )

    # Each attention block's immediate parent is its decoder layer. The
    # per-layer norms (`input_layernorm`, `post_attention_layernorm`) are
    # direct children of that decoder layer and should remain trainable.
    # Top-level attention prefixes (no parent) — unusual, but possible —
    # contribute nothing here and are simply skipped.
    decoder_layer_prefixes = set()
    for prefix in attention_prefixes:
        parent_path, _, _ = prefix.rpartition(".")
        if parent_path:
            decoder_layer_prefixes.add(parent_path)

    trainable_norm_prefixes = tuple(
        f"{layer_prefix}.{norm_child}"
        for layer_prefix in decoder_layer_prefixes
        for norm_child in TRAINABLE_PER_LAYER_NORM_CHILDREN
    )

    def _is_inside_attention(param_name: str) -> bool:
        return any(
            param_name.startswith(prefix + ".") or param_name == prefix
            for prefix in attention_prefixes
        )

    def _is_per_layer_attention_norm(param_name: str) -> bool:
        return any(
            param_name.startswith(prefix + ".") or param_name == prefix
            for prefix in trainable_norm_prefixes
        )

    trainable_after = 0
    frozen_after = 0
    for param_name, parameter in model.named_parameters():
        if _is_inside_attention(param_name) or _is_per_layer_attention_norm(param_name):
            # Leave attention params (and per-layer attention-adjacent norms)
            # trainable. Do not flip to True if the user already disabled
            # them elsewhere.
            if parameter.requires_grad:
                trainable_after += parameter.numel()
            else:
                frozen_after += parameter.numel()
        else:
            parameter.requires_grad = False
            frozen_after += parameter.numel()

    _log_message(
        master_process,
        logger,
        file_logger,
        f"Context extension: froze non-attention parameters "
        f"(kept per-layer {'/'.join(TRAINABLE_PER_LAYER_NORM_CHILDREN)} trainable). "
        f"Trainable: {trainable_after:,} | Frozen: {frozen_after:,}.",
    )
    return trainable_after, frozen_after


def _load_model(
    args,
    tokenizer,
    precision,
    master_process,
    logger=None,
    file_logger=None,
    distributed_config=None,
    use_kernels=False,
):
    """
    Load a model from a checkpoint or initialize a new model based on the provided arguments.

    Returns a 3-tuple `(model, checkpoint_path, non_attention_frozen)`. The
    `non_attention_frozen` flag is True when `_freeze_non_attention_blocks`
    was applied (only happens during continual pretraining with context
    extension); downstream code uses it to skip the MoE active-params
    adjustment, since all remaining trainable params are dense attention.
    """
    checkpoint_path = _resolve_checkpoint_path(args.resume_from_checkpoint)

    if not args.continual_pretraining:
        if checkpoint_path is not None:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                dtype=precision,
                attn_implementation=args.attn_implementation,
                cache_dir=args.cache_dir,
                **(
                    {"distributed_config": distributed_config}
                    if distributed_config is not None
                    else {}
                ),
                **({"use_kernels": True} if use_kernels else {}),
            )
            _log_message(
                master_process,
                logger,
                file_logger,
                f"Resumed model from checkpoint: {checkpoint_path}",
            )
            return model, checkpoint_path, False

        _log_message(master_process, logger, file_logger, "Initializing model from `AutoConfig`.")
        return (
            _build_model_from_config(
                args,
                tokenizer,
                precision,
                master_process,
                distributed_config=distributed_config,
                use_kernels=use_kernels,
            ),
            None,
            False,
        )

    # Continual pretraining / fine-tuning. The config may come from a resumed
    # checkpoint (staged context extension, e.g. 4k -> 32k -> 64k) or from the
    # base model (first stage). Context-extension overrides are applied to the
    # config before the weights are loaded, in both cases.
    config_source = checkpoint_path if checkpoint_path is not None else args.base_model
    _log_message(
        master_process,
        logger,
        file_logger,
        f"Resumed model from checkpoint for context extension: {checkpoint_path}"
        if checkpoint_path is not None
        else f"Initializing model from base model: {args.base_model} for continual pretraining/fine-tuning.",
    )

    needs_context_extension = args.new_max_position_embeddings is not None

    config = None
    if needs_context_extension:
        config = AutoConfig.from_pretrained(config_source, cache_dir=args.cache_dir)
        original_max_pos = config.max_position_embeddings
        target_max_pos = args.new_max_position_embeddings

        config.max_position_embeddings = target_max_pos

        if target_max_pos > original_max_pos:
            # Position interpolation ("linear" RoPE): positions are effectively
            # divided by `factor`, so the pretrained frequencies are reused over
            # the longer window. The factor is cumulative so staged extensions
            # (4k -> 32k -> 64k) keep stacking correctly when resuming from a
            # checkpoint that already carries a factor.
            ratio = target_max_pos / original_max_pos
            # Only auto-apply linear scaling when the source uses the default RoPE
            # or a previous stage's linear scaling. A resumed staged checkpoint
            # already carries rope_type="linear" + a factor; multiply into it so
            # the cumulative factor keeps stacking. Other types (yarn, llama3,
            # longrope, dynamic, ...) encode a different scaling scheme that we
            # must not clobber.
            current_rope_type = config.rope_parameters.get("rope_type", "default")
            if current_rope_type in {"default", "linear"}:
                config.rope_parameters["rope_type"] = "linear"
                config.rope_parameters["factor"] = config.rope_parameters.get("factor", 1.0) * ratio
            else:
                _log_message(
                    master_process,
                    logger,
                    file_logger,
                    f"WARNING: the source model already uses "
                    f"rope_type={current_rope_type!r}; skipping automatic "
                    "linear scaling. Configure RoPE manually in the source model config.",
                )

        _log_message(
            master_process,
            logger,
            file_logger,
            f"Context extension: max_position_embeddings={config.max_position_embeddings}, "
            f"rope_type={config.rope_parameters.get('rope_type', 'default')}, "
            f"factor={config.rope_parameters.get('factor', 1.0):g}, "
            f"rope_theta={config.rope_parameters['rope_theta']}.",
        )

    model = AutoModelForCausalLM.from_pretrained(
        config_source,
        dtype=precision,
        attn_implementation=args.attn_implementation,
        cache_dir=args.cache_dir,
        config=config,
        **({"distributed_config": distributed_config} if distributed_config is not None else {}),
        **({"use_kernels": True} if use_kernels else {}),
    )

    # Context-extension fine-tuning: freeze everything outside the attention
    # blocks so VRAM / optimizer state goes entirely to the modules that need
    # to adapt to the new sequence length.
    non_attention_frozen = False
    if needs_context_extension:
        _freeze_non_attention_blocks(model, master_process, logger, file_logger)
        non_attention_frozen = True

    return model, checkpoint_path, non_attention_frozen


def _resize_embeddings_for_tokenizer(
    model, tokenizer, master_process, logger=None, file_logger=None
):
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
    apply_liger_kernel = liger_transformers._apply_liger_kernel_to_instance
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


# Modules that are never converted to fp8. `lm_head` is either fused into
# Liger's `fused_linear_cross_entropy` (converting it would break the fusion)
# or is the numerically most sensitive projection in the network, so both
# torchtitan and the torchao examples keep it in high precision.
FP8_EXCLUDED_MODULE_SUFFIXES = ("lm_head",)

# fp8 gemms require both matmul dimensions to be divisible by 16.
FP8_DIM_ALIGNMENT = 16

# Minimum CUDA compute capability with hardware fp8 support
# (8.9 = Ada, 9.0 = Hopper / Grace Hopper, 10.0+ = Blackwell).
FP8_MIN_COMPUTE_CAPABILITY = (8, 9)


def _fp8_module_filter_fn(module, fqn):
    """
    Decide whether a module is eligible for conversion to `Float8Linear`.

    Skips the excluded modules (see `FP8_EXCLUDED_MODULE_SUFFIXES`) and any
    linear whose `in_features` / `out_features` are not divisible by
    `FP8_DIM_ALIGNMENT`, since the fp8 gemms require that alignment.
    """
    if any(fqn == suffix or fqn.endswith(f".{suffix}") for suffix in FP8_EXCLUDED_MODULE_SUFFIXES):
        return False

    return not (
        isinstance(module, torch.nn.Linear)
        and (
            module.in_features % FP8_DIM_ALIGNMENT != 0
            or module.out_features % FP8_DIM_ALIGNMENT != 0
        )
    )


def _apply_fp8_training(model, args, master_process, logger=None, file_logger=None):
    """
    Convert eligible `torch.nn.Linear` modules to `Float8Linear` for fp8
    mixed precision training via `torchao.float8`.

    Only the matmuls in the forward/backward of a linear are computed in fp8;
    parameters, gradients, optimizer states, and every non-linear op stay in
    bf16/fp32, so the rest of the training logic is unchanged.

    This is a best-effort optimization: if `torchao` is not installed, the GPU
    does not support fp8, or the requested recipe is unknown, a warning is
    logged and training proceeds with the default precision configuration
    (tf32 / bf16).

    IMPORTANT: this must be called AFTER every other model-level optimization
    (Liger kernels, gradient checkpointing) so that fp8 wraps the final module
    tree, and BEFORE DDP / FSDP wrapping and `torch.compile`.

    Reference: https://docs.pytorch.org/ao/stable/workflows/training.html

    Returns True when the conversion was applied, False otherwise.
    """
    if not args.fp8:
        return False

    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except (ImportError, ModuleNotFoundError):
        _log_message(
            master_process,
            logger,
            file_logger,
            "WARNING: fp8 is True but `torchao` is not installed. "
            "Install it with `pip install torchao`. "
            "Continuing with the default precision configuration (tf32 / bf16).",
        )
        return False

    if not torch.cuda.is_available():
        _log_message(
            master_process,
            logger,
            file_logger,
            "WARNING: fp8 is True but no CUDA device is available. "
            "Continuing with the default precision configuration (tf32 / bf16).",
        )
        return False

    compute_capability = torch.cuda.get_device_capability()
    if compute_capability < FP8_MIN_COMPUTE_CAPABILITY:
        _log_message(
            master_process,
            logger,
            file_logger,
            f"WARNING: fp8 is True but the GPU compute capability is "
            f"{compute_capability[0]}.{compute_capability[1]}, below the "
            f"{FP8_MIN_COMPUTE_CAPABILITY[0]}.{FP8_MIN_COMPUTE_CAPABILITY[1]} required for "
            "hardware fp8 (Ada / Hopper / Grace Hopper / Blackwell). "
            "Continuing with the default precision configuration (tf32 / bf16).",
        )
        return False

    try:
        config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
    except Exception as error:
        _log_message(
            master_process,
            logger,
            file_logger,
            f"WARNING: fp8 is True but the recipe {args.fp8_recipe!r} could not be resolved "
            f"({error}). Valid recipes are `tensorwise`, `rowwise`, and `rowwise_with_gw_hp`. "
            "Continuing with the default precision configuration (tf32 / bf16).",
        )
        return False

    if not args.bf16:
        _log_message(
            master_process,
            logger,
            file_logger,
            "WARNING: fp8 is enabled but `bf16` is False. fp8 training is designed to run on "
            "top of a bf16 model; running it on fp32 parameters wastes memory and bandwidth.",
        )

    convert_to_float8_training(model, config=config, module_filter_fn=_fp8_module_filter_fn)

    converted = sum(1 for module in model.modules() if type(module).__name__ == "Float8Linear")
    if converted == 0:
        _log_message(
            master_process,
            logger,
            file_logger,
            "WARNING: fp8 is True but no `torch.nn.Linear` module was eligible for conversion "
            f"(all were excluded or had dimensions not divisible by {FP8_DIM_ALIGNMENT}). "
            "Continuing with the default precision configuration (tf32 / bf16).",
        )
        return False

    _log_message(
        master_process,
        logger,
        file_logger,
        f"Enabled fp8 mixed precision training via torchao (recipe={args.fp8_recipe}, "
        f"converted {converted} linear module(s) to Float8Linear).",
    )

    if not args.torch_compile:
        _log_message(
            master_process,
            logger,
            file_logger,
            "NOTE: fp8 is enabled without `torch_compile`. torchao recommends `torch.compile` "
            "for competitive fp8 performance; without it the casting/scaling overhead may "
            "offset the fp8 gemm speedup.",
        )

    return True


def _try_create_distributed_config(
    enable_expert_parallelism, master_process, logger=None, file_logger=None
):
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
            master_process,
            logger,
            file_logger,
            "Expert parallelism enabled via DistributedConfig.",
        )
        return DistributedConfig(enable_expert_parallel=True)
    except (ImportError, ModuleNotFoundError):
        _log_message(
            master_process,
            logger,
            file_logger,
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
            master_process,
            logger,
            file_logger,
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
            master_process,
            logger,
            file_logger,
            "WARNING: use_kernels is True but the installed transformers version does not support "
            "the `use_kernels` kwarg. Upgrade transformers to a compatible version. Continuing without kernels.",
        )
        return False

    _log_message(
        master_process,
        logger,
        file_logger,
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


def _compute_active_trainable_params(config, trainable_params, non_attention_frozen=False):
    """
    Compute the number of active trainable parameters.

    For dense models, active_trainable_params == trainable_params.
    For MoE models, only the routed experts selected per token
    (num_experts_per_tok) are counted, since the remaining experts
    are inactive during each forward pass.

    When `non_attention_frozen` is True (context-extension fine-tuning),
    the MoE experts have already been frozen and are therefore excluded
    from `trainable_params` entirely. Subtracting inactive-expert params
    again would over-count (and could even go negative), so we short-circuit
    and return `trainable_params` as-is — all remaining trainable params
    (dense attention blocks plus per-layer attention-adjacent norms) are
    fully active per forward pass.

    - Note: Handles some naming conventions for MoE-related config fields,
            but you might need to adjust this function if your model uses
            different field names or MoE architecture.
    """
    if non_attention_frozen:
        return trainable_params

    # Detect MoE: try both naming conventions for the total expert count.
    num_experts = getattr(config, "num_experts", None) or getattr(config, "num_local_experts", None)
    if num_experts is None or num_experts <= 1:
        return trainable_params

    num_experts_per_tok = getattr(config, "num_experts_per_tok", None)
    if num_experts_per_tok is None or num_experts_per_tok >= num_experts:
        return trainable_params

    hidden_size = config.hidden_size

    # Per-expert MLP intermediate size.
    expert_intermediate_size = (
        getattr(config, "moe_intermediate_size", None) or config.intermediate_size
    )

    # SwiGLU MLP per expert: gate_proj + up_proj + down_proj
    params_per_expert = 3 * hidden_size * expert_intermediate_size

    # Number of MoE layers. Qwen-style configs use `decoder_sparse_step` (every
    # k-th layer is MoE) and `mlp_only_layers` (explicit indices that use a
    # dense MLP instead of MoE). Both are honored when present; otherwise the
    # default assumption is that every layer is a MoE layer.
    decoder_sparse_step = getattr(config, "decoder_sparse_step", 1) or 1
    mlp_only_layers = set(getattr(config, "mlp_only_layers", None) or [])
    num_moe_layers = sum(
        1
        for layer_idx in range(config.num_hidden_layers)
        if layer_idx not in mlp_only_layers and (layer_idx + 1) % decoder_sparse_step == 0
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
        args.enable_expert_parallelism,
        master_process,
        logger,
        file_logger,
    )

    use_kernels = _check_kernels_available(
        args.use_kernels,
        master_process,
        logger,
        file_logger,
    )

    model, checkpoint_path, non_attention_frozen = _load_model(
        args,
        tokenizer,
        precision,
        master_process,
        logger,
        file_logger,
        distributed_config=distributed_config,
        use_kernels=use_kernels,
    )

    if args.continual_pretraining:
        model = _resize_embeddings_for_tokenizer(
            model,
            tokenizer,
            master_process,
            logger,
            file_logger,
        )

    # Backfill runtime architecture fields declared in TrainingArguments
    # (consumed by mfu.py, data_loading.py, utils.py, train_ddp.py)
    args.max_position_embeddings = model.config.max_position_embeddings
    args.vocab_size = model.config.vocab_size
    args.num_hidden_layers = model.config.num_hidden_layers
    args.num_attention_heads = model.config.num_attention_heads
    args.head_dim = getattr(
        model.config,
        "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )
    # GQA: fall back to num_attention_heads (MHA) when not specified.
    args.num_key_value_heads = getattr(
        model.config,
        "num_key_value_heads",
        model.config.num_attention_heads,
    )

    # Architecture fields consumed by the structural MFU path for hybrid models.
    # No-ops for the standard dense / MoE-active dense_transformer path.
    args.hidden_size = getattr(model.config, "hidden_size", 0)
    args.intermediate_size = getattr(model.config, "intermediate_size", 0)
    # `layer_types` lists each block's flavour. Supported values for the
    # families this codebase targets: "full_attention" / "attention" and
    # "linear_attention" (Qwen3.5 Gated-DeltaNet hybrid). Some configs encode
    # the schedule via `full_attention_interval` (every k-th layer is full
    # attention, the rest are linear-attention); synthesize `layer_types`
    # from it when present so the MFU path works without architecture-specific
    # branching.
    layer_types = tuple(getattr(model.config, "layer_types", ()) or ())
    if not layer_types:
        full_attention_interval = getattr(model.config, "full_attention_interval", None)
        if full_attention_interval and full_attention_interval > 0:
            layer_types = tuple(
                "full_attention"
                if (layer_idx + 1) % full_attention_interval == 0
                else "linear_attention"
                for layer_idx in range(model.config.num_hidden_layers)
            )
    args.layer_types = layer_types

    # Linear attention (GDN / DeltaNet) hyperparameters (e.g. Qwen3.5 hybrid).
    args.linear_num_key_heads = getattr(model.config, "linear_num_key_heads", 0) or 0
    args.linear_num_value_heads = getattr(model.config, "linear_num_value_heads", 0) or 0
    args.linear_key_head_dim = getattr(model.config, "linear_key_head_dim", 0) or 0
    args.linear_value_head_dim = getattr(model.config, "linear_value_head_dim", 0) or 0
    args.linear_conv_kernel_dim = getattr(model.config, "linear_conv_kernel_dim", 4) or 4

    # MoE fields. Used by the hybrid structural FLOPs path for per-layer MLP
    # cost; for the dense path, MoE accounting goes through `num_parameters`
    # (active parameters, computed in `_compute_active_trainable_params`).
    _num_experts = (
        getattr(model.config, "num_experts", None)
        or getattr(model.config, "num_local_experts", None)
        or 0
    )
    if _num_experts and _num_experts > 1:
        args.num_experts_per_tok = getattr(model.config, "num_experts_per_tok", 0) or 0
        args.moe_intermediate_size = (
            getattr(model.config, "moe_intermediate_size", None)
            or getattr(model.config, "intermediate_size", 0)
            or 0
        )
        # Qwen-MoE / Qwen3.5-MoE name the shared-expert size
        # `shared_expert_intermediate_size`; accept `shared_intermediate_size`
        # as an alias for backwards compatibility.
        args.shared_intermediate_size = (
            getattr(model.config, "shared_expert_intermediate_size", None)
            or getattr(model.config, "shared_intermediate_size", None)
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
    linear_attention_fast_path = False
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
                master_process,
                logger,
                file_logger,
                "WARNING: the model has linear-attention layers but the following fast-path "
                f"package(s) are not installed: {', '.join(missing)}. Training will fall back "
                "to a slow PyTorch reference path. Install with:\n"
                "    pip install flash-linear-attention causal-conv1d\n"
                "See https://github.com/fla-org/flash-linear-attention#installation and "
                "https://github.com/Dao-AILab/causal-conv1d for details.",
            )
        else:
            linear_attention_fast_path = True
            _log_message(
                master_process,
                logger,
                file_logger,
                "Linear-attention fast path is enabled: flash-linear-attention and causal-conv1d are both installed.",
            )

        # Defensive repair (no-op on healthy models): guarantee no GatedDeltaNet
        # layer carries a non-finite `A_log`, including checkpoints produced by a
        # pre-fix run. Deterministic, so it stays consistent across ranks.
        patch_qwen3_5_gdn_initialization(
            model, logger=logger if master_process else None, force_reinit=False
        )

    if args.use_liger_kernel:
        _apply_liger_kernels(model, args)
        _log_message(master_process, logger, file_logger, "Applied Liger kernels to the model.")

    model.config.name_or_path = args.hub_model_id
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    active_trainable_params = _compute_active_trainable_params(
        model.config,
        trainable_params,
        non_attention_frozen=non_attention_frozen,
    )
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

    # fp8 mixed precision (torchao). Applied AFTER every other model-level
    # optimization (Liger kernels, gradient checkpointing) so the fp8 swap sees
    # the final module tree, and BEFORE `torch.compile` / DDP / FSDP wrapping.
    # The model is moved to the device first so `Float8Linear` is built directly
    # on the accelerator. No-op (with a warning) when `torchao` is unavailable or
    # the hardware does not support fp8.
    model.to(device)

    fp8_enabled = _apply_fp8_training(model, args, master_process, logger, file_logger)

    return ModelInitializationResult(
        args=args,
        tokenizer=tokenizer,
        model=model,
        precision=precision,
        checkpoint_path=checkpoint_path,
        trainable_params=trainable_params,
        active_trainable_params=active_trainable_params,
        non_attention_frozen=non_attention_frozen,
        fp8_enabled=fp8_enabled,
        linear_attention_fast_path=linear_attention_fast_path,
    )


# FSDP wrapping and state-dict utilities
from torch.distributed.checkpoint.state_dict import (  # noqa: E402
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh  # noqa: E402
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard  # noqa: E402
from torch.distributed.tensor import Replicate, Shard  # noqa: E402
from torch.distributed.tensor.parallel import (  # noqa: E402
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

SUPPORTED_SEQUENCE_PARALLEL_MODEL_TYPES = {
    "llama",
    "qwen3",
    "qwen3_moe",
    "qwen3_5_text",
    "qwen3_5_moe_text",
}


class _OptionalPrepareModuleInput(PrepareModuleInput):
    def _prepare_input_arg(self, input_value, mesh, input_layout, desired_layout):
        if input_value is None:
            return None
        return super()._prepare_input_arg(input_value, mesh, input_layout, desired_layout)


class _OptionalPrepareModuleInputOutput(PrepareModuleInputOutput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        prepare_input = self.prepare_module_input
        self.prepare_module_input = _OptionalPrepareModuleInput(
            input_layouts=prepare_input.input_layouts,
            desired_input_layouts=prepare_input.desired_input_layouts,
            input_kwarg_layouts=prepare_input.input_kwarg_layouts,
            desired_input_kwarg_layouts=prepare_input.desired_input_kwarg_layouts,
            use_local_output=prepare_input.use_local_output,
        )


def _is_float8_linear(module):
    """True when `module` is a torchao `Float8Linear` (fp8 training swap applied)."""
    return type(module).__name__ == "Float8Linear"


def _colwise_style(module, fp8_enabled, **kwargs):
    """
    Return a ColwiseParallel style for `module`, upgraded to torchao's
    fp8-aware `Float8ColwiseParallel` when the module is a `Float8Linear`.

    Plain `ColwiseParallel` on a `Float8Linear` casts to fp8 *inside* the
    linear's compiled forward, which trips `as_strided not supported with
    DTensor` under torch.compile + DTensor. The fp8-aware style instead casts
    at the TP boundary (producing a DTensor(Float8TrainingTensor)) so the
    linear sees an already-fp8 input and skips the offending path.
    """
    if fp8_enabled and _is_float8_linear(module):
        from torchao.float8.float8_tensor_parallel import Float8ColwiseParallel

        return Float8ColwiseParallel(**kwargs)
    return ColwiseParallel(**kwargs)


def _rowwise_style(module, fp8_enabled, **kwargs):
    """RowwiseParallel style, fp8-aware for `Float8Linear` (see `_colwise_style`)."""
    if fp8_enabled and _is_float8_linear(module):
        from torchao.float8.float8_tensor_parallel import Float8RowwiseParallel

        return Float8RowwiseParallel(**kwargs)
    return RowwiseParallel(**kwargs)


def _prepare_module_input_style(fp8_enabled, contains_float8, **kwargs):
    """
    Return a `PrepareModuleInput` style, upgraded to torchao's
    `PrepareFloat8ModuleInput` when the downstream consumers are `Float8Linear`.

    Casting the shared module input to fp8 once (before the Shard -> Replicate
    all-gather) lets multiple fp8 consumers (e.g. `gate_proj` and `up_proj`)
    reuse a single fp8 all-gather instead of each re-casting.
    """
    if fp8_enabled and contains_float8:
        from torchao.float8.float8_tensor_parallel import PrepareFloat8ModuleInput

        return PrepareFloat8ModuleInput(**kwargs)
    return PrepareModuleInput(**kwargs)


def build_sequence_parallel_plan(model, fp8_enabled=False):
    """Build a TP/SP plan from the modules present in a supported causal LM."""
    model_type = getattr(model.config, "model_type", None)
    if model_type not in SUPPORTED_SEQUENCE_PARALLEL_MODEL_TYPES:
        raise ValueError(
            f"Sequence parallelism does not support model type {model_type!r}. "
            f"Supported model types: {sorted(SUPPORTED_SEQUENCE_PARALLEL_MODEL_TYPES)}."
        )

    if model_type == "llama":
        rotary_input_layouts = (Shard(1),)
        rotary_desired_layouts = (Replicate(),)
    else:
        rotary_input_layouts = (Shard(1), None)
        rotary_desired_layouts = (Replicate(), None)

    plan = {
        "model.embed_tokens": RowwiseParallel(
            input_layouts=Replicate(), output_layouts=Shard(1), use_local_output=False
        ),
        "model.norm": SequenceParallel(),
        "model.rotary_emb": PrepareModuleInputOutput(
            input_layouts=rotary_input_layouts,
            desired_input_layouts=rotary_desired_layouts,
            use_local_input=True,
            output_layouts=(Replicate(), Replicate()),
            desired_output_layouts=(Replicate(), Replicate()),
            use_local_output=True,
        ),
        "lm_head": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
            use_local_output=True,
        ),
    }

    for layer_id, layer in enumerate(_iter_transformer_blocks(model)):
        prefix = f"model.layers.{layer_id}"
        if model_type in {"qwen3_5_text", "qwen3_5_moe_text"}:
            layer_input_layouts = (Shard(1), None)
            layer_desired_layouts = (Shard(1), None)
        else:
            layer_input_layouts = (Shard(1),)
            layer_desired_layouts = (Shard(1),)
        plan[prefix] = PrepareModuleInput(
            input_layouts=layer_input_layouts,
            desired_input_layouts=layer_desired_layouts,
            # Keep the block boundary as a DTensor. With per-block torch.compile,
            # emitting a `.to_local()` here makes the compiled block return a
            # plain-tensor output whose backward tangent AOTAutograd guesses as a
            # DTensor/AsyncCollectiveTensor, crashing in backward with
            # "Expected a AsyncCollectiveTensor tangent but got a plain Tensor"
            # (pytorch#172556). SequenceParallel on `input_layernorm` consumes
            # the DTensor directly.
            use_local_output=False,
        )
        plan[f"{prefix}.input_layernorm"] = SequenceParallel()
        plan[f"{prefix}.post_attention_layernorm"] = SequenceParallel()

        if hasattr(layer, "self_attn"):
            attention_prefix = f"{prefix}.self_attn"
            plan[attention_prefix] = _OptionalPrepareModuleInput(
                input_kwarg_layouts={
                    "hidden_states": Shard(1),
                    "attention_mask": Replicate(),
                },
                desired_input_kwarg_layouts={
                    "hidden_states": Replicate(),
                    "attention_mask": Replicate(),
                },
            )
            for projection in ("q_proj", "k_proj", "v_proj"):
                plan[f"{attention_prefix}.{projection}"] = _colwise_style(
                    getattr(layer.self_attn, projection), fp8_enabled
                )
            # Emit a DTensor (not a local tensor) so the decoder layer's
            # `residual + attn_out` stays DTensor+DTensor. The residual is a
            # DTensor(Shard(1)) coming from the block-input prepare, and mixing
            # it with a plain local tensor raises "aten.add.Tensor got mixed
            # torch.Tensor and DTensor" under torch.compile.
            plan[f"{attention_prefix}.o_proj"] = _rowwise_style(
                layer.self_attn.o_proj,
                fp8_enabled,
                output_layouts=Shard(1),
                use_local_output=False,
            )

        if hasattr(layer, "linear_attn"):
            linear_attention_prefix = f"{prefix}.linear_attn"
            plan[linear_attention_prefix] = _OptionalPrepareModuleInputOutput(
                input_kwarg_layouts={
                    "hidden_states": Shard(1),
                    "attention_mask": Replicate(),
                },
                desired_input_kwarg_layouts={
                    "hidden_states": Replicate(),
                    "attention_mask": Replicate(),
                },
                use_local_input=True,
                output_layouts=Replicate(),
                desired_output_layouts=Shard(1),
            )
            for projection in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                plan[f"{linear_attention_prefix}.{projection}"] = _colwise_style(
                    getattr(layer.linear_attn, projection),
                    fp8_enabled,
                    output_layouts=Replicate(),
                )

        mlp = layer.mlp
        if all(hasattr(mlp, projection) for projection in ("gate_proj", "up_proj", "down_proj")):
            # `post_attention_layernorm` (SequenceParallel) emits Shard(1); the
            # colwise projections need a replicated input, so redistribute
            # Shard(1) -> Replicate once at the MLP boundary (fp8-aware so the
            # cast/all-gather is shared by `gate_proj` and `up_proj`). Without
            # this prepare, ColwiseParallel would silently receive a Shard(1)
            # input (its declared/desired layouts are both Replicate, so it
            # never redistributes) and produce incorrect results.
            mlp_has_float8 = _is_float8_linear(mlp.gate_proj)
            plan[f"{prefix}.mlp"] = _prepare_module_input_style(
                fp8_enabled,
                mlp_has_float8,
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            )
            plan[f"{prefix}.mlp.gate_proj"] = _colwise_style(mlp.gate_proj, fp8_enabled)
            plan[f"{prefix}.mlp.up_proj"] = _colwise_style(mlp.up_proj, fp8_enabled)
            # DTensor output so `residual + mlp_out` stays DTensor+DTensor.
            plan[f"{prefix}.mlp.down_proj"] = _rowwise_style(
                mlp.down_proj,
                fp8_enabled,
                output_layouts=Shard(1),
                use_local_output=False,
            )
        else:
            plan[f"{prefix}.mlp"] = PrepareModuleInputOutput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
                use_local_input=True,
                output_layouts=Replicate(),
                desired_output_layouts=Shard(1),
            )

    module_names = dict(model.named_modules())
    missing = sorted(set(plan) - set(module_names))
    if missing:
        raise ValueError(f"Sequence-parallel plan references missing modules: {missing}")
    return plan


def _validate_fp8_sequence_parallel_plan(model, plan, sp_size):
    """Ensure every TP-sharded Float8Linear keeps FP8-aligned local dimensions."""
    modules = dict(model.named_modules())
    invalid_shards = []
    for module_name, style in plan.items():
        module = modules[module_name]
        if type(module).__name__ != "Float8Linear":
            continue

        if isinstance(style, ColwiseParallel):
            dimension_name = "out_features"
        elif isinstance(style, RowwiseParallel):
            dimension_name = "in_features"
        else:
            continue

        dimension = getattr(module, dimension_name)
        if dimension % sp_size != 0 or dimension // sp_size % FP8_DIM_ALIGNMENT != 0:
            invalid_shards.append(
                f"{module_name}.{dimension_name}={dimension} -> {dimension / sp_size:g} per SP rank"
            )

    if invalid_shards:
        raise ValueError(
            "FP8 sequence parallelism requires each sharded Float8Linear GEMM dimension "
            f"to remain divisible by {FP8_DIM_ALIGNMENT}. Invalid shards: "
            + ", ".join(invalid_shards)
        )


def _iter_transformer_blocks(model):
    """
    Yield every transformer block that should be individually sharded by FSDP.

    Modern HF causal-LM models (dense, MoE, Qwen3.5 hybrid) all expose their
    per-layer blocks under `model.model.layers` as a `ModuleList`. Sharding
    every entry in that list is the standard FSDP2 idiom, and is architecture-agnostic:
    it works for dense, MoE, and hybrid models without registering any
    decoder-layer class up front.

    This helper provides a clearer error if a new architecture deviates from it.
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


def apply_fsdp_wrapping(
    model,
    args,
    fp8_enabled,
    device_type,
    world_size,
    rank,
    local_world_size,
    master_process,
    logger=None,
    file_logger=None,
):
    """
    Apply FSDP2 (fully_shard) wrapping to the model.

    This function shards each decoder layer individually, then shards the root
    model. It supports mixed precision, CPU offload, sequence parallelism,
    and explicit prefetching, all controlled by the fields on `args`. When
    `args.torch_compile` is set, each decoder layer is also `torch.compile`
    individually, right before it is sharded (the torchtitan pattern);
    the root module is left uncompiled.

    Returns:
        tuple[int, int]: The data-parallel world size and this process's rank in
            that dimension, for gradient-accumulation and sampler calculations.
    """
    fsdp_kwargs = {}

    # Mixed precision
    if args.fsdp_mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        _log_message(
            master_process,
            logger,
            file_logger,
            "Enabled mixed precision policy for FSDP. Param type = torch.bfloat16, Reduce type = torch.float32",
        )

    # Device mesh and sequence-parallel setup
    data_parallel_size = world_size
    data_parallel_rank = rank

    if not args.sequence_parallel:
        mesh_config = init_device_mesh(
            device_type=device_type,
            mesh_shape=(world_size,),
        )
        _log_message(
            master_process,
            logger,
            file_logger,
            f"Initialized 1D device mesh with shape: ({world_size},) for Fully Sharded Data Parallel (FSDP).",
        )
        fsdp_mesh = mesh_config
    else:
        sp_size = local_world_size if args.sp_shard is None else args.sp_shard
        assert sp_size > 0, f"sp_shard must be positive (sp_shard={sp_size})."
        assert local_world_size % sp_size == 0, (
            f"Local world size {local_world_size} must be divisible by the sequence-parallel "
            f"size (sp_shard={sp_size})."
        )
        assert world_size % sp_size == 0, (
            f"World size {world_size} must be divisible by the sequence-parallel size "
            f"(sp_shard={sp_size})."
        )
        data_parallel_size = world_size // sp_size
        mesh_config = init_device_mesh(
            device_type=device_type,
            mesh_shape=(data_parallel_size, sp_size),
            mesh_dim_names=("dp", "sp"),
        )
        fsdp_mesh = mesh_config["dp"]
        tp_mesh = mesh_config["sp"]
        data_parallel_rank = fsdp_mesh.get_local_rank()
        _log_message(
            master_process,
            logger,
            file_logger,
            f"Initialized 2D device mesh with shape: (dp={data_parallel_size}, sp={sp_size}) "
            "for FSDP2 + sequence parallelism.",
        )
        tp_plan = build_sequence_parallel_plan(model, fp8_enabled=fp8_enabled)
        if fp8_enabled:
            _validate_fp8_sequence_parallel_plan(model, tp_plan, sp_size)
        parallelize_module(model, tp_mesh, tp_plan)
        _log_message(master_process, logger, file_logger, "Applied tensor/sequence parallelism.")

    fsdp_kwargs["mesh"] = fsdp_mesh

    # Sharding strategy (ZeRO-3 vs ZeRO-2)
    fsdp_kwargs["reshard_after_forward"] = bool(args.full_shard)
    _log_message(
        master_process,
        logger,
        file_logger,
        f"FSDP / ZeRO Stage is set to {'ZeroStage3' if args.full_shard else 'ZeroStage2'}",
    )

    # CPU offload
    if args.cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=True)
        _log_message(master_process, logger, file_logger, "Enabled CPU offload policy for FSDP.")

    # Per-block torch.compile. Following the torchtitan pattern, each
    # transformer block is compiled individually and BEFORE it is sharded, so
    # the compiled-region boundaries line up with FSDP's all-gather/reshard
    # boundaries instead of Dynamo having to graph-break around FSDP hooks
    # inside one giant whole-model graph. All blocks share a single compiled
    # graph (same class, same shape), which is what collapses compile time.
    compile_blocks = args.torch_compile
    if compile_blocks:
        if args.use_liger_kernel:
            _log_message(
                master_process,
                logger,
                file_logger,
                "WARNING: torch_compile + Liger kernel is enabled together. Some versions of "
                "PyTorch/Liger kernel don't play well combined (e.g. "
                "https://github.com/linkedin/Liger-Kernel/issues/174) and this combination may fail.",
            )
        _log_message(
            master_process,
            logger,
            file_logger,
            "Compiling each transformer block individually with torch.compile before FSDP sharding "
            "(root module stays uncompiled).",
        )

    # Per-layer sharding (bottom-up, as required by FSDP2). We wrap every
    # block in `model.model.layers` regardless of its concrete class. This is
    # architecture-agnostic and supports dense (Llama, Qwen3, Qwen3.5), MoE
    # (Qwen3.5-MoE), and Qwen3.5 linear-attention hybrid models without
    # needing to register their decoder-layer class first.
    layers = _iter_transformer_blocks(model)
    layer_classes = set()
    for layer_id, layer in layers.named_children():
        layer_classes.add(type(layer).__name__)
        if compile_blocks:
            layer = torch.compile(layer)
            # Swap the compiled block back into the ModuleList in place so
            # `model.model.layers[i]` (and `_set_modules_to_forward_prefetch` /
            # `_set_modules_to_backward_prefetch` below) see the compiled
            # `OptimizedModule`, which `fully_shard` then wraps directly.
            layers.register_module(layer_id, layer)
        fully_shard(layer, **fsdp_kwargs)
    _log_message(
        master_process,
        logger,
        file_logger,
        f"FSDP per-layer sharding applied to block classes: {sorted(layer_classes)}.",
    )

    # Shard the root model (embeddings, lm_head, etc.).
    fully_shard(model, **fsdp_kwargs)

    # Explicit prefetching
    if args.explicit_prefetching:
        _set_modules_to_forward_prefetch(model, num_to_forward_prefetch=2)
        _set_modules_to_backward_prefetch(model, num_to_backward_prefetch=2)

    return data_parallel_size, data_parallel_rank


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
