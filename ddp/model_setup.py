"""
Tokenizer and model initialization utilities for the DDP trainer.

This module owns the pre-DDP model setup path so the trainer can consume a
single explicit result object.

Provides:
    - `ModelInitializationResult` dataclass that encapsulates the tokenizer, model, and related state.
    - `prepare_training_components()` function that initializes the tokenizer and model.
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


def _build_model_from_config(args, tokenizer, precision):
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
    )


def _load_model(args, tokenizer, precision, master_process, logger=None, file_logger=None):
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
        return _build_model_from_config(args, tokenizer, precision), None

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


def prepare_training_components(args, device, master_process, logger=None, file_logger=None):
    """Build tokenizer/model state needed by the trainer before DDP wrapping."""
    torch.set_float32_matmul_precision(args.mat_mul_precision)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = args.bf16
    precision = torch.bfloat16 if args.bf16 else torch.float32

    tokenizer = _create_tokenizer(args, master_process, logger, file_logger)

    model, checkpoint_path = _load_model(args, tokenizer, precision, master_process, logger, file_logger)

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

    if args.use_liger_kernel:
        _apply_liger_kernels(model, args)
        _log_message(master_process, logger, file_logger, "Applied Liger kernels to the model.")

    model.config.name_or_path = args.hub_model_id

    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    _log_message(
        master_process,
        logger,
        file_logger,
        f"Number of trainable parameters: {trainable_params:,}",
    )

    if args.gradient_checkpointing:
        _log_message(master_process, logger, file_logger, "Gradient checkpointing enabled.")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False if torch.cuda.device_count() > 1 else True,
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
    )