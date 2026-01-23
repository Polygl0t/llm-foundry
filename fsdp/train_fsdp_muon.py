"""
Fully Sharded Data Parallel (FSDP) Training Script with Muon Optimizer

This script implements distributed training for large language models using PyTorch FSDP
with a hybrid Muon+AdamW optimization strategy for improved training efficiency.

Training pipeline:
1. SLURM setup: Initialize DDP process group
2. Model loading: From scratch, checkpoint, or reference model
3. Data loading: Create distributed samplers and dataloaders
4. Optimizer setup: MuonWithAuxAdam with 4 parameter groups
5. LR scheduler: Cosine or WSD decay (separate for Adam/Muon)
6. Training loop: Forward, backward, gradient accumulation, optimizer step
7. Validation: Periodic evaluation on val_dataset
8. Checkpointing: Save model, optimizer (both Adam and Muon states)
9. Monitoring: Log to W&B with separate Adam/Muon learning rates

Requirements:
    - SLURM environment (SLURM_NTASKS, SLURM_PROCID)
    - PyTorch with NCCL backend
    - transformers, liger-kernel, codecarbon, wandb
    - muon optimizer (https://github.com/KellerJordan/Muon)
"""
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)

from torch.distributed.fsdp import FSDPModule, CPUOffloadPolicy
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch

from transformers import default_data_collator, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.smollm3.modeling_smollm3 import SmolLM3DecoderLayer
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from liger_kernel.transformers import _apply_liger_kernel_to_instance
from muon import MuonWithAuxAdam, get_muon_momentum
from specifications import TrainingArguments
from functools import partial

from codecarbon import EmissionsTracker
import numpy as np
import datasets
import argparse
import logging
import wandb
import glob
import yaml
import time
import math
import sys
import os

def setup_triton_cache():
    """
    Setup Triton cache directory with proper permissions and cleanup.

    -   This helps to avoid conflicts where different processes 
        might try to access cache files that have been modified
        or deleted.
    """

    # Use SLURM_JOB_ID to create a unique cache directory for each job.
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
    cache_dir = os.environ.get('TRITON_CACHE_DIR', f'./.cache/triton_cache/{slurm_job_id}')

    # Create rank-specific cache directory to avoid conflicts.
    rank = dist.get_rank() if dist.is_initialized() else 0
    rank_cache_dir = f"{cache_dir}/rank_{rank}"
    
    os.makedirs(rank_cache_dir, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = rank_cache_dir
    
    # Cleanup old cache files older than 1 hour.
    try:
        for root, _, files in os.walk(rank_cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if os.path.getmtime(file_path) < time.time() - 3600:
                        os.remove(file_path)
                except (OSError, IOError):
                    pass
    except Exception:
        pass

def get_full_model_state_dict(model: FSDPModule):
    """
    Utility function for checkpointing with Fully Sharded Data Parallel (FSDP) and 
    `torch.distributed.checkpoint`:

    - `get_full_model_state_dict`: Retrieves the full model state dict by all-gathering 
    weights from all ranks and offloading them to CPU.

    References:
    - FSDP State Dict Tutorial: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#state-dict-with-dcp-apis  
    - Distributed Checkpoint API: https://pytorch.org/docs/stable/distributed.checkpoint.html
    - Model State Dict API: https://docs.pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_model_state_dict
    - State Dict Options API: https://docs.pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.StateDictOptions
    """
    return get_model_state_dict(
        model=model,
        options=StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
            ),
        )    

def get_full_optimizer_state_dict( 
        model: FSDPModule,
        opt: torch.optim.Optimizer,
    ):
    """
    Utility functions for checkpointing with Fully Sharded Data Parallel (FSDP) and 
    `torch.distributed.checkpoint`:

    - `get_full_optimizer_state_dict`: Collects the full optimizer state dict across 
    all ranks.

    References:
    - FSDP State Dict Tutorial: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#state-dict-with-dcp-apis  
    - Distributed Checkpoint API: https://pytorch.org/docs/stable/distributed.checkpoint.html
    - Optimizer State Dict API: https://docs.pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_optimizer_state_dict
    - State Dict Options API: https://docs.pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.StateDictOptions
    """
    return get_optimizer_state_dict(
                model=model,
                optimizers=opt,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )

def set_optimizer(
        model: FSDPModule,
        optimizers: torch.optim.Optimizer,
        optim_state_dict,
):
    """
    Utility function for checkpointing with Fully Sharded Data Parallel (FSDP) and 
    `torch.distributed.checkpoint`:

    - `set_optimizer_state_dict`: Loads a full optimizer state dict into the distributed optimizer.

    References:
    - FSDP State Dict Tutorial: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#state-dict-with-dcp-apis
    - Distributed Checkpoint API: https://pytorch.org/docs/stable/distributed.checkpoint.html
    - Optimizer State Dict API: https://docs.pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_optimizer_state_dict
    - State Dict Options API: https://docs.pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.StateDictOptions
    """
    return set_optimizer_state_dict(
            model=model,
            optimizers=optimizers,
            optim_state_dict=optim_state_dict,
            options=StateDictOptions(
                        full_state_dict=True,
                        broadcast_from_rank0=True,
            ),
        )

def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    """
    Set the modules to be prefetched for forward pass.
    """
    for i, layer in enumerate(model.model.layers):
        if i >= len(model.model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)

def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    """
    Set the modules to be prefetched for backward pass.
    """
    for i, layer in enumerate(model.model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)

def cleanup_log_file(log_file):
    """
    Clean up the log file by removing incomplete entries after the last "Validation" line.
    This ensures the log remains consistent when resuming training.
    """
    if not os.path.exists(log_file):
        return
    
    try:
        # Read all lines from the log file
        with open(log_file, "r") as f:
            lines = f.readlines()
        
        # Find the last occurrence of a line starting with "Validation"
        last_validation_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("Validation"):
                last_validation_idx = i
                break
        
        # If we found a validation line, keep everything up to and including it
        if last_validation_idx != -1:
            with open(log_file, "w") as f:
                f.writelines(lines[:last_validation_idx + 1])
    except Exception as e:
        # If cleanup fails, just continue - we don't want to crash the training
        print(f"Warning: Failed to cleanup log file: {e}")

def checkpoint_already_validated(checkpoint_dir, stage_name, step, log_file):
    """
    Check if a checkpoint has already been validated by verifying:
    1. The checkpoint directory exists
    2. The log file contains a validation entry for this step
    
    Returns:
        bool: True if checkpoint exists and has been validated, False otherwise
    """
    # Check if checkpoint directory exists
    checkpoint_name = f"step_{step:05d}"
    output_dir = os.path.join(checkpoint_dir, stage_name, checkpoint_name)
    
    if not os.path.exists(output_dir):
        return False
    
    # Check if log file exists and contains validation for this step
    if not os.path.exists(log_file):
        return False
    
    try:
        with open(log_file, "r") as f:
            for line in f:
                # Look for validation log entry for this specific step
                if line.startswith("Validation") and f"step: {step:5d}" in line:
                    return True
    except Exception:
        # If we can't read the log, assume not validated to be safe
        return False
    
    return False

def main(specs, slurm_job_id, hardware):

    # Load the training arguments from the specifications.yaml file
    with open(specs, "r") as stream:
        kwargs = yaml.safe_load(stream)
    # Check the `specifications.py` script to see all available arguments.
    args = TrainingArguments(**kwargs)

    # [Logging facility for Python](https://docs.python.org/3/library/logging.html#)
    logger = logging.getLogger(f"FSDP-Trainer-{slurm_job_id}-{args.stage_name}")

    logging.basicConfig(
        format="%(name)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    if "SLURM_NTASKS" and "SLURM_PROCID" in os.environ:

        # SLURM_NTASKS is the total number of processes (aka, world size).
        world_size = int(os.environ["SLURM_NTASKS"])

        if world_size > 1:

            # SLURM_PROCID is the rank of the current process in SLURM.
            rank = int(os.environ['SLURM_PROCID'])

            # [PyTorch Distributed Documentation](https://docs.pytorch.org/docs/stable/distributed.html)
            dist.init_process_group(
                backend="nccl",
                world_size=world_size, 
                rank=rank,
                device_id=rank % torch.cuda.device_count()
            )

            # Set the device to the current rank.
            device = f"cuda:{rank % torch.cuda.device_count()}"
            torch.cuda.set_device(device)
            # The first process is the master process.
            master_process = rank == 0
            fsdp = True
            if master_process:
                logger.info(f"Running FSDP via '{dist.get_backend()}' backend. Logging process: {rank}. World size: {world_size}.")

        else:
            # If the world size is 1, then we are not using distributed training.
            rank = 0
            device = "cuda:0"
            torch.cuda.set_device(device)
            master_process = True
            fsdp = False
            logger.info("Running single process training.")

    else:
        raise ValueError("SLURM_NTASKS or SLURM_PROCID environment variable is not set. This script is intended to be run with SLURM.")

    # Setup Triton cache before any GPU operations.
    setup_triton_cache()

    # If we are `resume_from_checkpoint`, we use the slurm job id from the checkpoint path.
    if args.resume_from_checkpoint:
        slurm_job_id = args.resume_from_checkpoint.split("slurm_job_")[-1].split("/")[0]
    
    # Update the checkpoint directory to include the SLURM job id.
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, f"slurm_job_{slurm_job_id}")
    
    if master_process: 
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        # Create a log file to store the training logs.
        log_file = os.path.join(args.checkpoint_dir, f"logs-{slurm_job_id}.txt")
        
        # Clean up the log file if resuming from checkpoint
        if args.resume_from_checkpoint:
            cleanup_log_file(log_file)
        
        # Ensure log file exists (create if it doesn't)
        with open(log_file, "a") as f: 
            pass

    # [Common PyTorch Functions](https://docs.pytorch.org/docs/stable/torch.html)
    # Set the random state seed for reproducibility.
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set the precision for matrix multiplication, the TF32 mode, the CuDNN TF32 mode, 
    # and the model precision to bfloat16 if `bf16` is set to True. 
    # - Note: If you wish to train a model in fp16, you will need to worry about the [loss/gradient scaling](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-gradscaler).
    torch.set_float32_matmul_precision(args.mat_mul_precision)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = args.bf16
    precision = torch.bfloat16 if args.bf16 else torch.float32

    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": True,
        "token": args.hub_token,
    }

    # Load the tokenizer from HuggingFace or a local path.
    if args.tokenizer_name_or_path is not None:

        # [AutoTokenizer](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer)
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path, 
            **tokenizer_kwargs
        )
    else:
        if master_process:
            logger.info(f"No tokenizer name specified, using the {args.reference_model} to load the tokenizer.")
            with open(log_file, "a") as f:
                f.write(f"No tokenizer name specified, using the {args.reference_model} to load the tokenizer.\n")

        tokenizer = AutoTokenizer.from_pretrained(
            args.reference_model, 
            **tokenizer_kwargs
        )
    
    # Add the `chat_template` to the tokenizer if a path (`args.chat_template_path`) is specified.
    if args.chat_template_path is not None:
        with open(args.chat_template_path, "r") as f:
            tokenizer.chat_template = f.read()
        
        if master_process:
            logger.info(f"Loaded chat template from {args.chat_template_path}. Chat template added to the tokenizer.")
            with open(log_file, "a") as f:
                f.write(f"Loaded chat template from {args.chat_template_path}. Chat template added to the tokenizer.\n")

    # We override the model's vocab size with the tokenizer's vocab size only if the tokenizer's 
    # vocab size is larger than the model's vocab size. It is okay for the model's vocab size 
    # to be larger than the tokenizer's, which is basically a way of padding the vocab size 
    # and leaving some embeddings unused. This is useful when you want to use a tokenizer
    # with a vocab size that is not a nice round number (e.g., 50257).
    args.vocab_size = len(tokenizer) if len(tokenizer) > args.vocab_size else args.vocab_size
    
    # Potentially load the model from a previous checkpoint.
    if args.resume_from_checkpoint:

        # We expect the `resume_from_checkpoint` argument to be the path to a directory of checkpoints,
        # the logic below will find the latest checkpoint in that directory.
        checkpoint_path = args.resume_from_checkpoint

        try:
            # We try to find the latest checkpoint in the directory.
            checkpoint_dirs = os.listdir(checkpoint_path)
            checkpoint_dirs = [dir for dir in checkpoint_dirs if dir.startswith(f"step_")] # Checkpoints are intended to be named like "step_1000"
            checkpoint_path = os.path.join(checkpoint_path, sorted(checkpoint_dirs, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1])
        except:
            # If the checkpoint directory does not contain any checkpoints, we will assume
            # that `resume_from_checkpoint` is already set to the latest checkpoint.
            pass
            
        # [AutoModelForCausalLM](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForCausalLM)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, 
            torch_dtype=precision, 
            attn_implementation=args.attn_implementation, 
            cache_dir=args.cache_dir
        )

        if master_process:
            logger.info(f"Resumed model from checkpoint: {checkpoint_path}")
            with open(log_file, "a") as f:
                f.write(f"\nResumed model from checkpoint: {checkpoint_path}\n")
    
    else:

        # If we are not resuming from a checkpoint, and we are not doing continual pretraining,
        # we initialize the model from `AutoConfig`.
        if not args.continual_pretraining:
            if master_process:
                logger.info(f"Initializing model from `AutoConfig`.")
                with open(log_file, "a") as f:
                    f.write(f"Initializing model from `AutoConfig`.\n")

            # Define the model architecture.
            config_kwargs = {
                "cache_dir": args.cache_dir,
                "token": args.hub_token,
                "output_hidden_states": args.output_hidden_states,
                "hidden_size": args.hidden_size,
                "intermediate_size": args.hidden_size * 4 if args.intermediate_size is None else args.intermediate_size,
                "max_position_embeddings": args.max_position_embeddings,
                "num_attention_heads": args.num_attention_heads,
                "num_key_value_heads": args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads,
                "head_dim": args.hidden_size // args.num_attention_heads if args.head_dim is None else args.head_dim,
                "num_hidden_layers": args.num_hidden_layers,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "unk_token_id": tokenizer.unk_token_id,
                "torch_dtype": precision,
                "vocab_size": args.vocab_size,
                "rope_theta": args.rope_theta,
                "use_cache": args.use_cache,
                "hidden_act": args.hidden_act,
                "rms_norm_eps": args.rms_norm_eps,
                "tie_word_embeddings": args.tie_word_embeddings,
                "layer_types": ["full_attention"] * args.num_hidden_layers, # If you want to alternate full_attention and sparse_attention ("sliding_attention"), you can do it here.
                "no_rope_layer_interval": args.no_rope_layer_interval, # [NoPE: The Counting Power of Transformers with No Positional Encodings](https://arxiv.org/pdf/2505.11199)
                "no_rope_layers": [0 if (i + 1) % args.no_rope_layer_interval == 0 else 1 for i in range(args.num_hidden_layers)] \
                    if args.no_rope_layer_interval is not None else [1] * args.num_hidden_layers,
                "pretraining_tp": 1, # Just to make sure we are no inheriting from a tensor-parallel pretraining config
                "transformers.js_config": None, # Just to make sure we are not inheriting from a transformers.js config
            }

            # Create an instance of the model using AutoConfig and the set configuration.
            # [AutoConfig](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoConfig)
            # [AutoModelForCausalLM](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForCausalLM)
            model = AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(
                    args.reference_model if args.reference_model is not None else "HuggingFaceTB/SmolLM2-360M", # Llama!
                    **config_kwargs,
                ), 
                attn_implementation=args.attn_implementation,
            )
        
        # If we are doing continual pretraining, we load the model from the reference model.
        else:
            if master_process:
                logger.info(f"Initializing model from reference model: {args.reference_model} for continual pretraining/fine-tuning.")
                with open(log_file, "a") as f:
                    f.write(f"Initializing model from reference model: {args.reference_model} for continual pretraining/fine-tuning.\n")

            # Check if we are performing RoPE extension/scaling.
            # What is RoPE scaling?
            # RoPE (Rotary Position Embedding) scaling is a technique used to adjust the position embeddings
            # of a model to better fit the input sequence length and improve performance on longer sequences.
            # By increasing the rope_theta parameter and adjusting the max_position_embeddings, we can
            # effectively scale the maximum sequence length our model can handle by training on longer sequences.
            if args.rope_scale_factor is not None:
                
                # Get the model configuration from the reference model.
                config = AutoConfig.from_pretrained(args.reference_model, cache_dir=args.cache_dir)

                # Increase the max position embeddings based on the RoPE scale factor.
                config.max_position_embeddings = int(args.max_position_embeddings * args.rope_scale_factor)
                args.max_position_embeddings = config.max_position_embeddings

                # Check if we scale the base frequency (rope_theta).
                if model.config.rope_theta == args.rope_theta:
                    if master_process:
                        logger.info(f"WARNING: RoPE theta should scale when performing RoPE scaling. Check your configurations and perhaps adjust the rope_theta to a larger value.")
                
                # Set the new RoPE theta (hopefully scaled).
                config.rope_theta = args.rope_theta

                if master_process:
                    logger.info(f"Performing RoPE scaling to {model.config.max_position_embeddings} max position embeddings and {model.config.rope_theta} rope theta.")
                    logger.info(f"WARNING: `args.max_position_embeddings` has been reset to {args.max_position_embeddings}.")
                    with open(log_file, "a") as f:
                        f.write(f"Performing RoPE scaling to {model.config.max_position_embeddings} max position embeddings and {model.config.rope_theta} rope theta.\n")
                        

            # [AutoModelForCausalLM](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForCausalLM)
            model = AutoModelForCausalLM.from_pretrained(
                args.reference_model, 
                torch_dtype=precision, 
                attn_implementation=args.attn_implementation, 
                cache_dir=args.cache_dir,
                config=config if args.rope_scale_factor is not None else None,
            )
    
    # Set the tokenizer `model_max_length` to the models max position embeddings.
    tokenizer.model_max_length = model.config.max_position_embeddings

    # [Liger Kernel: Efficient Triton Kernels for LLM Training](https://github.com/linkedin/Liger-Kernel)
    if args.use_liger_kernel:
        # Apply the Liger kernels to the model.
        liger_kwargs = {
            "rope": True,
            "cross_entropy": False,
            "fused_linear_cross_entropy": True,
            "rms_norm": True,
            "swiglu": True,
        }
        # Learn more on how to CORRECTLY apply the Liger kernels to the model here:
        # https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py
        _apply_liger_kernel_to_instance(model=model, **liger_kwargs)

        if master_process:
            logger.info(f"Applied Liger kernels to the model.")
            with open(log_file, "a") as f:
                f.write(f"Applied Liger kernels to the model.\n")

    # Change the model's `name_or_path`. If set to
    # None this will not show in the config (which is totally fine).
    model.config.name_or_path = args.hub_model_id
  
    # Print the number of trainable parameters in the model.
    if master_process:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters: {params:,}")
        with open(log_file, "a") as f:
            f.write(f"Number of trainable parameters: {params:,}\n")

    # Set the gradient checkpointing for the model
    # What is Gradient Checkpointing? -> https://arxiv.org/abs/1604.06174
    if args.gradient_checkpointing:

        if master_process:
            logger.info(f"Gradient checkpointing enabled.")
            with open(log_file, "a") as f:
                f.write(f"Gradient checkpointing enabled.\n")

        # Enable gradient checkpointing (https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForCausalLM)
        # IMPORTANT: For FSDP, always use `use_reentrant=False`. Reentrant checkpointing is incompatible
        # with FSDP because it doesn't properly handle the sharded parameter semantics.
        # Using reentrant=True with FSDP can cause:
        # - Incorrect gradient computation
        # - Memory leaks due to retained activation graphs  
        # - Deadlocks during backward pass
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    # [Torch Compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
    # WARNING: Torch compile is not working good with liger kernel: https://github.com/linkedin/Liger-Kernel/issues/174
    if args.torch_compile and not args.use_liger_kernel:
        if master_process:
            logger.info(f"Compiling model with torch.compile.")
        model = torch.compile(model)

    model.to(device)

    if fsdp:
        # Wrap the model with FullyShardedDataParallel (FSDP).
        # FSDP partitions model parameters, gradients, and optimizer states across
        # all processes, rather than replicating the full model on each GPU (as DDP does).
        # This enables training very large models that would not fit into a single GPU's
        # memory by "sharding" parameters and only gathering them when needed.
        #
        #        +---------+    +---------+    +---------+
        #        | Rank 0  |    | Rank 1  |    | Rank 2  |
        #        |  GPU 0  |    |  GPU 1  |    |  GPU 2  |
        #        | Shard A |    | Shard B |    | Shard C |
        #        +---------+    +---------+    +---------+
        #            \             |             /
        #             \            |            /
        #              \           |           /
        #               +---------------------+
        #               |   All-Gather /      |
        #               |   Reduce-Scatter    |
        #               +---------------------+
        #
        # Training flow:
        #    - Each GPU stores only a shard of the parameters and optimizer states.
        #    - During forward/backward, parameter shards are "all-gathered" so each
        #      rank has what it needs for computation.
        #    - Gradients are reduced and then "reduce-scattered" back to shards.
        #    - Optimizer updates happen on the sharded states locally.
        #
        # Reference: https://pytorch.org/docs/stable/fsdp.html
        # FSDP2 documentation -> https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-to-use-fsdp2
        # FSDP2 Example Implementation -> https://github.com/pytorch/examples/tree/main/distributed/FSDP2

        fsdp_kwargs = {}
        if args.fsdp_mixed_precision:
            # Enable module-level mixed precision policy in FSDP:
            # - param_dtype=torch.bfloat16: Casts unsharded parameters to bfloat16 for forward/backward computation.
            # - reduce_dtype=torch.float32: Upcasts gradients to float32 during reduce-scatter to preserve numerical accuracy.
            #
            # Refer to the FSDP2 documentation for more information:
            # -> https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#enabling-mixed-precision
            # -> https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.MixedPrecisionPolicy
            # Here, we are transferring half-precision gradients, but accumulating/reducing them in full-precision, since it's 
            # well known that accumulation operations are often not super stable in half-precision.
            # ([source](https://github.com/deepspeedai/DeepSpeed/issues/2352))
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )
            if master_process:
                logger.info(f"Enabled mixed precision policy for FSDP. Param type = torch.bfloat16, Reduce type = torch.float32")

        # What is a DeviceMesh?
        # This is a useful abstraction for managing device placement and sharding
        # when working with multi-D parallelism.
        # -> https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html
        # What is a DeviceMesh?
        # This is a useful abstraction for managing device placement and sharding
        # when working with multi-D parallelism.
        # -> https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html
        if args.dp_shard is None:
            # We assume the strategy desired if to fully shard the model
            # across all available devices.
            mesh_config = init_device_mesh(
                device_type=device_type,
                mesh_shape=(world_size,)
            )

            if master_process:
                logger.info(f"Initialized 1D device mesh with shape: ({world_size},) for Fully Sharded Data Parallel (FSDP).")
                logger.info("If you want to customize the device mesh topology, you can do so by modifying the `dp_shard` argument.")

        else:
            assert world_size % args.dp_shard == 0, f"World size {world_size} needs to be divisible by `dp_shard` size (dp_shard={args.dp_shard}, world_size={world_size})"
            assert args.dp_shard > 1, f"dp_shard needs to be greater than 1 (dp_shard={args.dp_shard})."
            # If args.dp_shard is not None, we assume the strategy desired is to perform HSDP.
            # Hybrid Sharding Data Parallel (HSDP) is 2D strategy to perform FSDP within a
            # host and DDP across hosts.
            #
            # Data Parallel ("dp") across hosts
            # FSDP ("fsdp") within each host
            # Why? Communication costs are lower within a host compared to across hosts.
            # So we leave the costly communication across hosts to DDP, which requires
            # simple all-reduce operations to synchronize gradients. And we use the faster
            # in-host communication to perform FSDP, which has much more costly communication
            # operations (All-Gather, Reduce-Scatter).
            #
            # For example, if we have 2 hosts (nodes) with 4 GPUs each,
            # =========================================================
            # ------------       ------------
            # | Host 1   |       | Host 2   |
            # | 4 GPUs   |       | 4 GPUs   |
            # |          |       |          |
            # | (FSDP)   |       | (FSDP)   |
            # |[0,1,..,3]|       |[4,5..,7] |
            # |          |       |          |
            # |          |       |          |
            # ------------       ------------
            # DP:
            # [0,..., 3], [4, ..., 7]
            # I.e., Data parallel in two nodes and FSDP within each node.
            # Documentation: https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html#how-to-use-devicemesh-with-hsdp
            # ==========================================================
            
            # Create a sharding plan based on the given `world_size`.
            data_parallel_size = world_size // args.dp_shard

            # `dp_replicate` is the submesh that connects intra-host devices.
            # `dp_shard` is the submesh that connects inter-host devices.
            mesh_config = init_device_mesh(
                device_type=device_type,
                mesh_shape=(data_parallel_size, args.dp_shard),
                mesh_dim_names=("dp_replicate", "dp_shard")
            )
            
            # For simplicity sake, we set the `world_size` to be `data_parallel_size`.
            # Like this, we don't have to recalculate it later, and the rest of the code
            # can keep using `world_size` without changes.
            world_size = data_parallel_size

            if master_process:
                logger.info(f"Initialized 2D device mesh with shape: (dp_replicate={data_parallel_size}, dp_shard={args.dp_shard}) for Hybrid Sharding Data Parallel (HSDP).")
                logger.info(f"Current world size: {world_size}")

        fsdp_kwargs["mesh"] = mesh_config

        # ZeRO / FSDP Sharding Stages
        # ZeRO and FSDP progressively shard model states (optimizer, gradients, parameters)
        # across processes to reduce memory usage. Each stage adds more sharding:
        #   Stage 1 – Optimizer State Sharding
        #       - Only optimizer states (e.g., Adam’s m, v) are sharded.
        #       - Parameters and gradients remain fully replicated.
        #       - Memory saving: modest.
        #       - Not recommended but if you wish to use, look for [ZeroRedundancyOptimizer](https://docs.pytorch.org/docs/main/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer)
        #
        #       Params   : [replicated]
        #       Gradients: [replicated]
        #       Optimizer: [sharded]
        #
        #   Stage 2 – + Gradient Sharding
        #       - Optimizer states and gradients are sharded.
        #       - Parameters still fully replicated.
        #       - Memory saving: significant.
        #
        #       Params   : [replicated]
        #       Gradients: [sharded]
        #       Optimizer: [sharded]
        #
        #   Stage 3 – + Parameter Sharding (Full Shard = FSDP)
        #       - Optimizer states, gradients, and parameters are all sharded.
        #       - Parameters are all-gathered just-in-time for compute, then freed.
        #       - Memory saving: maximal.
        #
        #       Params   : [sharded]
        #       Gradients: [sharded]
        #       Optimizer: [sharded]
        #
        # - Note: FSDP2 only supports stages 3 and 2.
        #
        # In FSDP2, we control the level of sharding by setting the `reshard_after_forward` flag.
        # If True, then this reshards parameters after forward and re-all-gathers in backward.
        # This is equivalent to ZeRO stage 3.
        # If False, then this does not reshards parameters.
        # If False, then this keeps the unsharded parameters in memory after forward and avoids the all-gather in backward.
        # This is equivalent to ZeRO stage 2.
        # If None, it is set to True for non-root modules and False for root modules.
        # To learn more about all the possible ways in wich you can perform FSDP,
        # e.g., full shard, shard only optimizer and gradients (i.e., ZeRO stage 2),
        # check this migration tutorial: https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
        fsdp_kwargs["reshard_after_forward"] = True if args.full_shard else False
        if master_process:
            logger.info(f"FSDP / ZeRO Stage is set to {"ZeroStage3" if args.full_shard else "ZeroStage2"}")

        # This offload policy offloads parameters, gradients, and optimizer states to CPU.
        # If you don't have enough CPU memory, set `pin_memory` to False.
        # [CPUOffloadPolicy](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.CPUOffloadPolicy)
        if args.cpu_offload:
            fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=True)
            if master_process:
                logger.info("Enabled CPU offload policy for FSDP." if args.cpu_offload else "Disabled CPU offload policy for FSDP.")

        # According to FSDP2 documentation:
        # User should apply fully_shard in a bottom-up manner. For example, in a Transformer model, fully_shard should be applied to each layer before applying it to the root model. 
        # When applied to the root model, fully_shard excludes model.parameters() from each layer and groups the remaining parameters (e.g., embeddings, output projection) into a single all-gather group
        # fully_shard :
           # - Apply FSDP to module
           # - shards parameters across ranks, and convert model.parameters() from plain torch.Tensor to DTensor to represent sharded parameters.

        # Refer to the FSDP2 documentation for more information: 
        # -> https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#model-initialization
        # -> https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html#torch-distributed-fsdp-fully-shard
        # Here are some classics, but feel free to add more options if the need arises.
        decoder_mapping_dict = {
            "smollm3": SmolLM3DecoderLayer,
            "llama": LlamaDecoderLayer,
            "gemma3_text": Gemma3DecoderLayer,
            "qwen3": Qwen3DecoderLayer,
            "qwen2": Qwen2DecoderLayer
        }

        for layer in model.model.layers:
            # This will shard every layer.
            # fully_shard(layer, **fsdp_kwargs)
            # This will shard only blocks of the decoder-transformer architecture.
            # But you can also, for example, shard the nn.Embedding layer, or the lm_head.
            if isinstance(layer, decoder_mapping_dict[model.config.model_type]):
                fully_shard(layer, **fsdp_kwargs)
        
        # Finally, we shard the full model.
        # [fully_shard](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
        fully_shard(model, **fsdp_kwargs)

        # If explicit prefetching is enabled, we set the modules to be prefetched.
        # This can promote a slight increase in performance/throughput.
        if args.explicit_prefetching:
            set_modules_to_forward_prefetch(model, num_to_forward_prefetch=2)
            set_modules_to_backward_prefetch(model, num_to_backward_prefetch=2)
    
    # What is a sanity check?
    # A sanity check is a quick test to ensure that the training script
    # is working as expected.
    if args.sanity_check:

        # Create a list of tokens to be used for the sanity check.
        input_ids = [torch.arange(args.max_position_embeddings, device="cpu")]  * args.sanity_check_num_samples
        
        dataset = datasets.Dataset.from_dict(
            {
                "input_ids": input_ids,
            }
        ).with_format("torch")

        # Use 10% of the dataset for validation.
        train_dataset, val_dataset = dataset, dataset.select(range(int(args.sanity_check_num_samples * 0.1)))

    else:

        # Load our training dataset.
        # We expect the dataset to be in a specific format: jsonl or parquet.
        assert args.dataset_type in ["jsonl", "parquet"], f"Dataset type must be either 'jsonl' or 'parquet', got {args.dataset_type}."

        train_dataset_files = []
        train_dirs = args.train_dataset_dir
        if isinstance(train_dirs, str):
            train_dirs = [train_dirs]

        # Below, we loop over all training directories and collect the dataset files that
        # have the correct file extension.
        for train_dir in train_dirs:
            if os.path.isfile(train_dir) and train_dir.endswith(f".{args.dataset_type}"):
                train_dataset_files.append(train_dir)
            elif os.path.isdir(train_dir):
                train_dataset_files += glob.glob(f"{train_dir}/*.{args.dataset_type}")

        # If shuffling is enabled, we shuffle the dataset files.
        if args.shuffle_dataset:
            if master_process:
                logger.info(f"Shuffling enabled. Shuffling {len(train_dataset_files)} dataset files.")
                with open(log_file, "a") as f:
                    f.write(f"Shuffling enabled. Shuffling {len(train_dataset_files)} dataset files.\n")
            np.random.seed(args.seed)
            np.random.shuffle(train_dataset_files)

        # Validation dataset is expected to always be in a single directory.
        # Validation files should be of the same type as the training files.
        val_dataset_files = glob.glob(f"{args.val_dataset_dir}/*.{args.dataset_type}")
        
        # Change jsonl to json if the dataset type is jsonl (dont know how to make this cleaner)
        args.dataset_type = "json" if args.dataset_type == "jsonl" else args.dataset_type

        # Load the datasets from disk
        # [datasets.load_dataset](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset)
        train_dataset = datasets.load_dataset(
            args.dataset_type,
            data_files=train_dataset_files,
            split='train',
            num_proc=len(train_dataset_files),
            cache_dir=args.cache_dir,
        )

        val_dataset = datasets.load_dataset(
            args.dataset_type,
            data_files=val_dataset_files,
            split='train',
            num_proc=len(val_dataset_files),
            cache_dir=args.cache_dir,
        )

        # Shuffle the indicies of the training dataset.
        if args.shuffle_dataset:
            train_dataset = train_dataset.shuffle(seed=args.seed)
            if master_process:
                logger.info(f"Shuffling enabled. Shuffling indices.")

        # Format the input_ids lists as torch.tensor.
        train_dataset = train_dataset.with_format("torch")
        val_dataset = val_dataset.with_format("torch")
        

    # Initialize the DistributedSampler.
    # This sampler is designed to partition the dataset among multiple processes.
    # [DistributedSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)
    # DistributedSampler is used for data parallelism, so its rank should match the data parallel replica's index.
    # If we are performing HSDP, then the rank of the sampler should be adjusted accordingly.
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank // args.dp_shard if args.dp_shard else rank,
        shuffle=args.shuffle_dataset,
        drop_last=False,
        )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank // args.dp_shard if args.dp_shard else rank,
        shuffle=False,
        drop_last=False,
        )
    
    # Create the dataloaders.
    # [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(args.seed)

    # Create a collate function that will generate batches with labels.
    # This function allows us to selectively mask certain tokens (e.g., eos or pad)
    if args.mask_eos_token:
        assert tokenizer.eos_token_id is not None, "The tokenizer does not have an eos token id."
    if args.mask_pad_token:
        assert tokenizer.pad_token_id is not None, "The tokenizer does not have a pad token id."
        assert tokenizer.pad_token_id != tokenizer.eos_token_id, "The pad token and eos token are the same. This can lead to issues when masking."

    if master_process:
        logger.info(f"Collate function masking settings: mask_eos_token={args.mask_eos_token}, mask_pad_token={args.mask_pad_token}.")
        with open(log_file, "a") as f:
            f.write(f"Collate function masking settings: mask_eos_token={args.mask_eos_token}, mask_pad_token={args.mask_pad_token}.\n")
    
    def collate_with_masking(
            examples, 
            eos_id=tokenizer.eos_token_id, 
            pad_id=tokenizer.pad_token_id
        ):
        """Collate function that masks specified tokens (e.g., eos or pad)."""

        # [default_data_collator](https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.default_data_collator)
        batch = default_data_collator(examples)

        # If we already have labels in the batch, we do not need to create them again.
        # IMPORTANT: This assumes that the labels are already masked correctly.
        if "labels" in batch:
            return batch
        
        # If not, we create them here.
        input_ids = batch.get("input_ids")
        labels = input_ids.clone()

        # Mask the loss of the eos token (see the [Apertus Report](https://arxiv.org/abs/2509.14233))
        if args.mask_eos_token:
            labels[labels == eos_id] = -100
        # Mask the loss of the pad token (useful when using padding in the dataset)
        if args.mask_pad_token:
            labels[labels == pad_id] = -100

        # Attach the masked copy to the batch
        batch["labels"] = labels

        return batch
    
    collate_fn = partial(
        collate_with_masking, 
        eos_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=collate_fn,
        batch_size=args.micro_batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers_for_dataloader,
        generator=dataloader_generator,
        # The number of samples loaded in advance by the dataloader.
        # If you see that your optimization steps are lagging in a
        # periodic manner, try to increase the `prefetch_factor`.
        prefetch_factor=args.prefetch_factor
    )

    validation_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        collate_fn=collate_fn,
        batch_size=args.eval_micro_batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers_for_dataloader,
        prefetch_factor=args.prefetch_factor, 
        # We do not need to shuffle the validation set, so we dont pass in the generator
    )

    # Calculate the gradient accumulation steps.
    # Instead of breaking your head calculating this manually, we set the script to this on the fly.
    # The number of gradient accumulation steps is estimated based on how many tokens you want (`total_batch_size`) 
    # to be processed in a single step. If you stipulate a `total_batch_size` that does not divide evenly by
    # `micro_batch_size` * `max_position_embeddings` * `world_size`, we will throw an error.
    assert args.total_batch_size % (args.micro_batch_size * args.max_position_embeddings * world_size) == 0, "Make sure your `total_batch_size` is divisible by `micro_batch_size` * `max_position_embeddings` * `world_size`"
    gradient_accumulation_steps = args.total_batch_size // (args.micro_batch_size * args.max_position_embeddings * world_size)

    # Now, we need to calculate the math around the number of steps per epoch and the total number of steps.
    # Initially, we will set the number of steps per epoch to the number of batches in the training dataloader.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Then, `max_steps` is just `num_update_steps_per_epoch` * your number of epochs
    max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)

    # If the `args.max_steps` is set, we will override the `max_steps` with the value from the arguments.
    if args.max_steps is not None:
        max_steps = args.max_steps
        args.num_train_epochs = max_steps // num_update_steps_per_epoch if max_steps > num_update_steps_per_epoch else 1
        if master_process:
            logger.info(f"Overriding the number of steps to {max_steps} as per the `max_steps` argument (check the YAML file if you are not sure).")
            with open(log_file, "a") as f:
                f.write(f"Overriding the number of steps to {max_steps} as per the `max_steps` argument (check the YAML file if you are not sure).\n")

    # Cosine decay learning rate scheduler
    if args.lr_decay_type.lower() == "cosine":

        lr_decay_iters = max_steps * args.lr_decay_iters_coef

        def lr_scheduler(it):

            # 1) Linear warmup for `warmup_steps` steps
            if it < args.warmup_steps:
                # `it + 1` to avoid giving 0 learning rate at the first step
                adam_lr = args.max_learning_rate * (it + 1) / args.warmup_steps
                muon_lr = args.muon_learning_rate * (it + 1) / args.warmup_steps
                return (adam_lr, muon_lr, "warmup")
            # 2) If it > `lr_decay_iters``, return min learning rate
            if it > lr_decay_iters:
                return (args.min_learning_rate, args.min_learning_rate, "stable")
            # 3) In between, use cosine decay down to `min_learning_rate`
            decay_ratio = (it - args.warmup_steps) / (lr_decay_iters - args.warmup_steps)
            assert 0 <= decay_ratio <= 1
            
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            adam_lr = args.min_learning_rate + coeff * (args.max_learning_rate - args.min_learning_rate)
            muon_lr = args.min_learning_rate + coeff * (args.muon_learning_rate - args.min_learning_rate)
            return (adam_lr, muon_lr, "cosine_decay")
    
    # Warmup, Stable, Decay (WSD) learning rate scheduler -> https://arxiv.org/abs/2405.18392
    elif args.lr_decay_type.lower() == "wsd":

        lr_decay_iters = max_steps * args.lr_decay_iters_coef
        stable_iters = max_steps - lr_decay_iters
        
        def lr_scheduler(it):
            # 1) Linear warmup for `warmup_steps` steps
            if it < args.warmup_steps:
                # `it + 1` to avoid giving 0 learning rate at the first step
                adam_lr = args.max_learning_rate * (it + 1) / args.warmup_steps
                muon_lr = args.muon_learning_rate * (it + 1) / args.warmup_steps
                return (adam_lr, muon_lr, "warmup")
            # 2) If it > `stable_iters`, linear decay to `min_learning_rate`
            if it > stable_iters and lr_decay_iters > 0:
                decay_ratio = (it - stable_iters) / (max_steps - stable_iters)
                assert 0 <= decay_ratio <= 1
                # If `use_sqrt` is set, we use 1 - sqrt decay instead of linear decay.
                if args.use_sqrt:
                    decay_ratio = np.sqrt(decay_ratio)
                coeff = 1.0 - decay_ratio
                adam_lr = args.min_learning_rate + coeff * (args.max_learning_rate - args.min_learning_rate)
                muon_lr = args.min_learning_rate + coeff * (args.muon_learning_rate - args.min_learning_rate)
                return (adam_lr, muon_lr, "linear_decay" if not args.use_sqrt else "1-sqrt")
            # 3) In between, use constant learning rate
            return (args.max_learning_rate, args.muon_learning_rate, "stable")
                
    else:
        raise ValueError(f"Invalid learning rate decay type: '{args.lr_decay_type}'. Supported types are: `cosine` and `wsd`.")

    if master_process:
        logger.info(f"Using learning rate decay type: {args.lr_decay_type}")
        with open(log_file, "a") as f:
            f.write(f"Using learning rate decay type: {args.lr_decay_type}\n")

    # For the optimizer, we will split weights in two groups: 
    # - one with weight decay; 
    # - and the other without weight decay (biases, layer norms, and embeddings).
    #
    # Following OLMo 2, we remove weight decay from embedding layers to improve training stability. 
    # Source: https://arxiv.org/abs/2501.00656
    #
    # Also, we create two parameter groups: 
    # - one for AdamW; 
    # - and one for Muon.
    #
    # Muon will optimize the large matrix multiplications (ndim >= 2 that are not embeddings).
    # AdamW will optimize the rest of the parameters (ndim < 2 and embeddings).
    # Source: https://kellerjordan.github.io/posts/muon/
    no_decay = ["bias", "layer_norm.weight", "embed_tokens.weight"]
    hidden_matrix_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed_tokens.weight" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed_tokens.weight" in n]
    scalar_params_with_decay = [p for n, p in model.named_parameters() if p.ndim < 2 and not any(nd in n for nd in no_decay)]
    scalar_params_no_decay = [p for n, p in model.named_parameters() if p.ndim < 2 and any(nd in n for nd in no_decay)]

    optimizer_grouped_parameters = [
        {
            "params": embed_params,
            "weight_decay": 0.0,
            "lr": args.max_learning_rate,
            "betas": (args.beta1, args.beta2),
            "eps": args.eps,
            "use_muon": False,
        },
        {
            "params": scalar_params_no_decay,
            "weight_decay": 0.0,
            "lr": args.max_learning_rate,
            "betas": (args.beta1, args.beta2),
            "eps": args.eps,
            "use_muon": False,
        },
        {
            "params": scalar_params_with_decay,
            "weight_decay": args.weight_decay,
            "lr": args.max_learning_rate,
            "betas": (args.beta1, args.beta2),
            "eps": args.eps,
            "use_muon": False,
        },
        {
            "params": hidden_matrix_params,
            "weight_decay": args.weight_decay,
            "lr": args.muon_learning_rate,
            "momentum": args.beta2,
            "use_muon": True,
        }

    ]

    # Initialize the AdamW + Muon optimizer.
    # [MuonWithAuxAdam](https://github.com/KellerJordan/Muon/blob/master/muon.py#L138)
    optimizer = MuonWithAuxAdam(optimizer_grouped_parameters)

    # Compile optimizer step.
    if args.torch_compile:

        if master_process:
            logger.info("Compiling optimizer step with torch.compile.")
        
        @torch.compile(fullgraph=False)
        def optimizer_step(adam_lr, muon_lr, step):
            """
            This just helps to bundle the optimizer step with the learning rate.
            """
            for param_group in optimizer.param_groups:
                if param_group["use_muon"]:
                    param_group['lr'] = muon_lr
                    param_group['momentum'] = get_muon_momentum(step)
                else:
                    param_group['lr'] = adam_lr
            optimizer.step()
    
    else:
        def optimizer_step(adam_lr, muon_lr, step):
            for param_group in optimizer.param_groups:
                if param_group["use_muon"]:
                    param_group['lr'] = muon_lr
                    param_group['momentum'] = get_muon_momentum(step)
                else:
                    param_group['lr'] = adam_lr
            optimizer.step()

    # Resume the optimizer from the checkpoint.
    if args.resume_from_checkpoint:

        checkpoint = os.path.join(checkpoint_path, 'checkpoint.pt')
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'), weights_only=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if master_process:
            logger.info(f"Resumed optimizer from checkpoint: {checkpoint_path}")
            with open(log_file, "a") as f:
                f.write(f"Resumed optimizer from checkpoint: {checkpoint_path}\n")

    if master_process:

        logger.info("="*50)
        logger.info("***** Running training *****")
        logger.info(f"  Training stage | {args.stage_name}")
        logger.info(f"  SLURM Job ID | {slurm_job_id}")
        logger.info(f"  Hardware | {hardware.upper()}")
        logger.info(f"  World size (total GPUs) | {world_size}")
        logger.info(f"  Precision | {'bfloat16' if args.bf16 else 'float32'}")
        logger.info(f"  Resuming from checkpoint | {args.resume_from_checkpoint is not None}")
        if args.resume_from_checkpoint:
            logger.info(f"    Checkpoint path | {args.resume_from_checkpoint}")
            logger.info(f"    Starting from step | {checkpoint.get('resume_step', None) if not args.begin_new_stage else None}")
        logger.info("="*50)
        logger.info("Dataset Configuration:")
        logger.info(f"  Num train examples | {len(train_dataset):,}")
        logger.info(f"  Num validation examples | {len(val_dataset):,}")
        logger.info(f"  Length of train dataloader | {len(train_dataloader):,}")
        logger.info(f"  Max position embeddings (seq length) | {args.max_position_embeddings:,}")
        logger.info(f"  Shuffle dataset | {args.shuffle_dataset}")
        logger.info(f"  Mask EOS token | {args.mask_eos_token}")
        logger.info(f"  Mask PAD token | {args.mask_pad_token}")
        logger.info("="*50)
        logger.info("Batch Configuration:")
        logger.info(f"  Num Epochs | {args.num_train_epochs}")
        logger.info(f"  Micro batch size per device | {args.micro_batch_size}")
        logger.info(f"  Gradient accumulation steps | {gradient_accumulation_steps}")
        logger.info(f"  Total batch size (samples) | {args.micro_batch_size * gradient_accumulation_steps * world_size}")
        logger.info(f"  Total batch size (tokens) | {args.total_batch_size:,}")
        logger.info(f"  Total optimization steps | {max_steps:,}")
        logger.info(f"  Steps per epoch | {num_update_steps_per_epoch:,}")
        logger.info(f"  Checkpointing every | {args.checkpointing_steps} steps")
        logger.info("="*50)
        logger.info("Model Architecture:")
        logger.info(f"  Model type | {args.reference_model if args.reference_model else 'Custom'}")
        logger.info(f"  Attention implementation | {args.attn_implementation}")
        logger.info(f"  Gradient checkpointing | {args.gradient_checkpointing}")
        logger.info(f"  Liger kernel | {args.use_liger_kernel}")
        logger.info(f"  Torch compile | {args.torch_compile}")
        logger.info(f"  Trainable parameters | {params:,}")
        logger.info("="*50)
        logger.info("FSDP Configuration:")
        logger.info(f"  Full shard (ZeRO-3) | {args.full_shard}")
        logger.info(f"  Mixed precision | {args.fsdp_mixed_precision}")
        logger.info(f"  CPU offload | {args.cpu_offload}")
        logger.info(f"  DP shard (HSDP) | {args.dp_shard if args.dp_shard else 'None'}")
        logger.info(f"  Explicit prefetching | {args.explicit_prefetching}")
        logger.info("="*50)
        logger.info("Optimizer Configuration (Adam + Muon):")
        logger.info(f"  Max learning rate (Adam) | {args.max_learning_rate}")
        logger.info(f"  Min learning rate | {args.min_learning_rate}")
        logger.info(f"  Muon learning rate | {args.muon_learning_rate}")
        logger.info(f"  LR scheduler type | {args.lr_decay_type.upper()}")
        logger.info(f"  LR decay iterations coef | {args.lr_decay_iters_coef}")
        logger.info(f"  Warmup steps | {args.warmup_steps}")
        logger.info(f"  Weight decay | {args.weight_decay}")
        logger.info(f"  Beta1 | {args.beta1}")
        logger.info(f"  Beta2 | {args.beta2}")
        logger.info(f"  Epsilon | {args.eps}")
        logger.info(f"  Max grad norm | {args.max_grad_norm}")
        logger.info("="*50)

        if args.resume_from_checkpoint is None:
            # No need to log this again if we are resuming from a checkpoint.
            with open(log_file, "a") as f:
                f.write("="*50 + "\n")
                f.write("***** Running training *****\n")
                f.write(f"  Training stage | {args.stage_name}\n")
                f.write(f"  SLURM Job ID | {slurm_job_id}\n")
                f.write(f"  Hardware | {hardware.upper()}\n")
                f.write(f"  World size (total GPUs) | {world_size}\n")
                f.write(f"  Precision | {'bfloat16' if args.bf16 else 'float32'}\n")
                f.write("="*50 + "\n")
                f.write("Dataset Configuration:\n")
                f.write(f"  Num train examples | {len(train_dataset):,}\n")
                f.write(f"  Num validation examples | {len(val_dataset):,}\n")
                f.write(f"  Length of train dataloader | {len(train_dataloader):,}\n")
                f.write(f"  Max position embeddings (seq length) | {args.max_position_embeddings:,}\n")
                f.write(f"  Shuffle dataset | {args.shuffle_dataset}\n")
                f.write(f"  Mask EOS token | {args.mask_eos_token}\n")
                f.write(f"  Mask PAD token | {args.mask_pad_token}\n")
                f.write("="*50 + "\n")
                f.write("Batch Configuration:\n")
                f.write(f"  Num Epochs | {args.num_train_epochs}\n")
                f.write(f"  Micro batch size per device | {args.micro_batch_size}\n")
                f.write(f"  Gradient accumulation steps | {gradient_accumulation_steps}\n")
                f.write(f"  Total batch size (samples) | {args.micro_batch_size * gradient_accumulation_steps * world_size}\n")
                f.write(f"  Total batch size (tokens) | {args.total_batch_size:,}\n")
                f.write(f"  Total optimization steps | {max_steps:,}\n")
                f.write(f"  Steps per epoch | {num_update_steps_per_epoch:,}\n")
                f.write(f"  Checkpointing every | {args.checkpointing_steps} steps\n")
                f.write("="*50 + "\n")
                f.write("Model Architecture:\n")
                f.write(f"  Model type | {args.reference_model if args.reference_model else 'Custom'}\n")
                f.write(f"  Attention implementation | {args.attn_implementation}\n")
                f.write(f"  Gradient checkpointing | {args.gradient_checkpointing}\n")
                f.write(f"  Liger kernel | {args.use_liger_kernel}\n")
                f.write(f"  Torch compile | {args.torch_compile}\n")
                f.write(f"  Trainable parameters | {params:,}\n")
                f.write("="*50 + "\n")
                f.write("FSDP Configuration:\n")
                f.write(f"  Full shard (ZeRO-3) | {args.full_shard}\n")
                f.write(f"  Mixed precision | {args.fsdp_mixed_precision}\n")
                f.write(f"  CPU offload | {args.cpu_offload}\n")
                f.write(f"  DP shard (HSDP) | {args.dp_shard if args.dp_shard else 'None'}\n")
                f.write(f"  Explicit prefetching | {args.explicit_prefetching}\n")
                f.write("="*50 + "\n")
                f.write("Optimizer Configuration (Adam + Muon):\n")
                f.write(f"  Max learning rate (Adam) | {args.max_learning_rate}\n")
                f.write(f"  Min learning rate | {args.min_learning_rate}\n")
                f.write(f"  Muon learning rate | {args.muon_learning_rate}\n")
                f.write(f"  LR scheduler type | {args.lr_decay_type.upper()}\n")
                f.write(f"  LR decay iterations coef | {args.lr_decay_iters_coef}\n")
                f.write(f"  Warmup steps | {args.warmup_steps}\n")
                f.write(f"  Weight decay | {args.weight_decay}\n")
                f.write(f"  Beta1 | {args.beta1}\n")
                f.write(f"  Beta2 | {args.beta2}\n")
                f.write(f"  Epsilon | {args.eps}\n")
                f.write(f"  Max grad norm | {args.max_grad_norm}\n")
                f.write("="*50 + "\n")

    # If we are resuming from a checkpoint (inside the same stage), we need to get the current iteration count, 
    # which is the number of batches processed by the dataloader so far, the current epoch, and the completed steps 
    # from the checkpoint.
    if args.resume_from_checkpoint and not args.begin_new_stage:

        resume_step = int(checkpoint['resume_step'])
        iter_count = int(checkpoint['iteration'])
        epoch = int(checkpoint['epoch'])
        if epoch > 1:
            # Shuffle the sampler every epoch.
            train_sampler.set_epoch(epoch)

    else:
        # For the beginning of training, and every subsequent stage, we reset all counters.
        # WARNING: Don't forget to set `begin_new_stage` to True every time you start a new stage!!!
        resume_step = 0
        iter_count = 0
        epoch = 1

        if args.resume_from_checkpoint and args.begin_new_stage:
            if master_process:
                logger.info(f"Starting new training stage | {args.stage_name}")
                with open(log_file, "a") as f:
                    f.write(f"Starting new training stage | {args.stage_name}\n")
    
    if not args.begin_new_stage:
        if master_process:
            logger.info(f"WARNING: `begin_new_stage` is set to False. If this is a multistage training, make sure you set it to True for the new stages.")

    # Initialize  W&B and CodeCarbon.
    # If you dont want to use W&B, you maybe should consider checking Trackio.
    # -> https://github.com/gradio-app/trackio
    if master_process:

        if args.wandb_token is not None: 

            # Login to wandb.
            # [wandb.login](https://docs.wandb.ai/ref/python/sdk/functions/login)
            wandb.login(key=args.wandb_token)

            # Initialize wandb.
            # [wandb.init](https://docs.wandb.ai/ref/python/sdk/functions/init/)
            wandb.init(
                project=args.wandb_project if args.wandb_project is not None else "default", 
                notes=args.wandb_desc if args.wandb_desc is not None else "N/A",
                name=f"""{args.wandb_id}-{args.stage_name}-{time.strftime("%d-%m-%Y")}-bs-{args.total_batch_size}-epochs-{args.num_train_epochs}-steps-{max_steps}-lr-{args.max_learning_rate}-sch-{args.lr_decay_type}""",
                config=kwargs,
                resume="allow", # Allows resuming runs that stopped before completion.
                id=f"{args.wandb_id}-{slurm_job_id}" if args.wandb_id is not None else f"{slurm_job_id}",
            )
        
        # We would also like to track the energy consumption of the training process. 
        # For this, we are going to use the `codecarbon` library.
        # To do this, we need to initialize the `EmissionsTracker` and then track 
        # the energy consumption on [only the main process](https://github.com/mlco2/codecarbon/issues/544).
        # [EmissionsTracker](https://mlco2.github.io/codecarbon/usage.html#explicit-object)
        tracker = EmissionsTracker(
            project_name=args.wandb_project if args.wandb_project is not None else "default",
            log_level="critical", # Set to "critical" to silence codecarbon.
            output_dir=args.checkpoint_dir,
            output_file=f"emissions_{slurm_job_id}.csv",
            tracking_mode='machine', # We are tracking the energy consumption of all processes (all GPUS in a given machine/node).
        )

        logger.info(f'Geo Location: ISO: {tracker._geo.country_iso_code} | Country: {tracker._geo.country_name} | Region : {tracker._geo.region}')
        tracker.start()

        # MFU is a performance efficiency metric that measures how well a model training run utilizes the available peak compute of the hardware.
        # P = is the peak FLOPS of the hardware (the promise)
        # C = is the total number of parameters in the model (the capacity)
        # L = is the number of layers in the model
        # H = is the number of attention heads in the model
        # Q = is the hidden size per attention head
        # T = is the max position embeddings
        # What is the formula?
        # MFU = (Achieved FLOPS) / (Peak FLOPS)
        # Achieved FLOPS = (FLOPS per iteration) * (1 / dt)
        # dt = (time taken for the step in seconds)
        # FLOPS per iteration = (FLOPS per forward and backward pass) * (batch size)
        # FLOPS per forward and backward pass = (FLOPS per token) * T
        # FLOPS per token = 6 * C + 12 * L * H * Q * T
        P = 300e12 if hardware == "a100" else 150e12 if hardware == "a40" else None
        C = params
        L = args.num_hidden_layers
        H = args.num_attention_heads
        Q = args.hidden_size // args.num_attention_heads if args.head_dim is None else args.head_dim
        T = args.max_position_embeddings
        if P is None:
            raise ValueError("Hardware not supported for MFU calculation.")

    # Set the model to training mode.
    model.train()

    # Create an iterator from the train dataloader.
    iter_train_dataloader = iter(train_dataloader)

    if master_process:
        logger.info(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}\n")

    # Flag to indicate if the learning rate stage has changed (starts as False)
    # We use this to save a checkpoint if the learning rate stage changes.
    lr_stage_change = False
    # Get the current learning rate stage
    current_lr_stage = lr_scheduler(resume_step)[-1]

    # Start the training loop!!! 🚀🚀🚀
    for completed_steps in range(1, max_steps + 1):

        # Skip the steps that have already been completed when resuming from a checkpoint.
        if resume_step >= completed_steps:
            for micro_step in range(gradient_accumulation_steps):
                try:
                    next(iter_train_dataloader)
                except StopIteration:
                    # If we reach the end of the dataloader, we need to reset the iterator.
                    epoch += 1
                    train_sampler.set_epoch(epoch)
                    iter_train_dataloader = iter(train_dataloader)
                    next(iter_train_dataloader)
                iter_count += 1
            continue
            
        # Reset the sampler if we have exhausted the dataloader for this epoch.
        # iter_count tracks the number of BATCHES consumed, not optimizer steps.
        if iter_count >= len(train_dataloader):
            if epoch < math.ceil(args.num_train_epochs):
                epoch += 1
                iter_count = 0
                train_sampler.set_epoch(epoch)
                # IMPORTANT: Must recreate iterator after changing sampler epoch
                iter_train_dataloader = iter(train_dataloader)
                if master_process:
                    logger.info(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")
                    with open(log_file, "a") as f:
                        f.write(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}\n")
         
        # Evaluate the model when:
        # - We have completed `args.checkpointing_steps` steps (excluding step 0).
        # - The learning rate stage has changed.
        # - We are at the last step.
        if (
            completed_steps % args.checkpointing_steps == 0
            or lr_stage_change
            or completed_steps == max_steps
        ):
            # Check if this checkpoint has already been validated (to avoid re-validation on resume)
            already_validated = checkpoint_already_validated(
                args.checkpoint_dir, 
                args.stage_name, 
                completed_steps, 
                log_file if master_process else os.path.join(args.checkpoint_dir, f"logs-{slurm_job_id}.txt")
            )
            
            # Skip validation if checkpoint already exists and has been validated
            if already_validated:
                if master_process:
                    logger.info(f"Skipping validation for step {completed_steps} - checkpoint already validated.")
                pass
            else:
                if master_process:
                    logger.info("***** Running validation *****")

                model.eval()

                with torch.no_grad():

                    val_loss_accum = 0.0
                    num_batches = 0

                    # Time the validation loop.
                    val_t0 = time.time()

                    for _, batch in enumerate(validation_dataloader):

                        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                        
                        with torch.autocast(device_type=device_type, dtype=precision):
                            loss = model(
                                input_ids=batch["input_ids"],
                                labels=batch["labels"],
                            ).loss

                        val_loss_accum += loss.detach()
                        num_batches += 1

                    val_t1 = time.time()
                    val_time = val_t1 - val_t0
                    
                    # Average the loss over the number of batches on this process
                    if num_batches > 0:
                        val_loss_accum = val_loss_accum / num_batches
                
                if fsdp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.SUM)
                    val_loss_accum = val_loss_accum / world_size

                model.train()

                # Retrieves the full model state dict and full optimizer state dict before saving the checkpoint.
                # Need to be called on all processes.
                model_state_dict = get_full_model_state_dict(model)
                opt_state_dict = get_full_optimizer_state_dict(model, optimizer)

                if master_process:
                    logger.info(f"Validation | step: {completed_steps:5d} | loss: {float(val_loss_accum.item()):.4f} | kWh: {float(tracker._total_energy.kWh):.2f} | val_time: {val_time:.2f}s")
                    
                    with open(log_file, "a") as f:
                        f.write(f"Validation | step: {completed_steps:5d} | loss: {float(val_loss_accum.item()):.4f} |  kWh: {float(tracker._total_energy.kWh):.2f} | val_time: {val_time:.2f}s\n")
                    
                    if args.wandb_token is not None:
                        wandb.log({"val_loss": val_loss_accum.item()})
                        
                    # Create the checkpoint directory.
                    checkpoint_name = f"step_{completed_steps:05d}"
                    output_dir = os.path.join(args.checkpoint_dir, args.stage_name, checkpoint_name)
                    os.makedirs(output_dir, exist_ok=True)

                    # Save the model and tokenizer.           
                    model.save_pretrained(output_dir,state_dict=model_state_dict)
                    tokenizer.save_pretrained(output_dir)

                    # Save the optimizer state and other metadata.
                    torch.save(
                        {
                        'resume_step' : completed_steps,
                        'iteration': iter_count,
                        'epoch': epoch,
                        'config': model.config,
                        'optimizer': opt_state_dict,
                        }, 
                        f"{output_dir}/checkpoint.pt",
                    )
                
                    # Push it to the hub.
                    if args.push_to_hub and args.hub_token is not None and args.hub_model_id is not None:
                        hub_model_id = f"{args.hub_model_id}-{args.stage_name}-{completed_steps}"
                        model.push_to_hub(hub_model_id, token=args.hub_token, private=True)
                        tokenizer.push_to_hub(hub_model_id, token=args.hub_token)

                    # Flush the codecarbon tracker at the end of the validation step.
                    tracker.flush()
                
                # Set barrier to ensure that all processes have finished the validation step before continuing.
                if fsdp:
                    dist.barrier()
        
        # We are timing the training loop to measure the MFU.
        t0 = time.time()

        # Initiate a counter for the accumulated loss.
        accumulated_loss = 0.0
        
        # Perform one optimization step.
        optimizer.zero_grad(set_to_none=True)
        
        # Perform the gradient accumulation inner loop.
        for micro_step in range(gradient_accumulation_steps):

            # Get the next batch.
            try:
                batch = next(iter_train_dataloader)
            except StopIteration:
                # Epoch boundary crossed during gradient accumulation
                # Must update sampler epoch before recreating iterator
                epoch += 1
                train_sampler.set_epoch(epoch)
                iter_train_dataloader = iter(train_dataloader)
                batch = next(iter_train_dataloader)
                if master_process:
                    logger.info(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")
                    with open(log_file, "a") as f:
                        f.write(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}\n")
                # Reset iter_count since we're starting a new epoch
                iter_count = 0

            # Move the batch to the device.
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # For FSDP2 gradient accumulation:
            # - Use `set_requires_gradient_sync(False)` to skip gradient reduce-scatter on intermediate steps
            # - On the final micro-step, enable sync so gradients are properly reduced across ranks
            # 
            # Why this matters:
            # - Without this, every backward() triggers an expensive reduce-scatter operation
            # - With gradient accumulation, we only need to sync after all micro-batches are processed
            # - This can provide up to gradient_accumulation_steps speedup in communication
            #
            # Note: FSDP2 does NOT have set_requires_all_reduce() or set_reshard_after_backward() as runtime methods.
            # The reshard behavior is controlled at initialization time via the `reshard_after_forward` kwarg.
            is_last_micro_step = (micro_step == gradient_accumulation_steps - 1)
            
            if fsdp:
                # Only sync gradients on the last micro-step
                model.set_requires_gradient_sync(is_last_micro_step)

            # Autocast is a PyTorch context manager that enables mixed precision training.
            # [torch.autocast](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-torch-autocast)
            with torch.autocast(device_type=device_type, dtype=precision):

                # Transformers from HF perform the loss calculation internally,
                # as well as the shifting of the labels in the case of a causal language model.
                # - Note: the loss is already an average loss over the micro-batch.
                loss = model(
                            input_ids=batch["input_ids"],
                            labels=batch["labels"],
                        ).loss

            # Accumulate the raw loss for logging (detached to avoid graph retention).
            accumulated_loss += loss.detach()
            
            # Scale the loss for gradient accumulation before backward pass.
            # This ensures gradients are properly averaged across all micro-batches.
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()
            
            # Track batch consumption for epoch boundary detection
            iter_count += 1

        # Average the accumulated loss over gradient accumulation steps.
        accumulated_loss = accumulated_loss / gradient_accumulation_steps

        if fsdp:
            dist.all_reduce(accumulated_loss, op=dist.ReduceOp.SUM)
            accumulated_loss = accumulated_loss / world_size

        # Clip gradients up to `args.max_grad_norm`.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Determine the learning rate for the current step.
        adam_lr, muon_lr, lr_stage = lr_scheduler(completed_steps)

        # And check if the learning rate stage has changed.
        if lr_stage != current_lr_stage:
            lr_stage_change = True
            current_lr_stage = lr_stage
            if master_process:
                logger.info(f"Learning rate stage changed to: {current_lr_stage} at step {completed_steps} | {args.stage_name}.")
                with open(log_file, "a") as f:
                    f.write(f"Learning rate stage changed to: {current_lr_stage} at step {completed_steps} | {args.stage_name}.\n")
        else:
            lr_stage_change = False

        optimizer_step(adam_lr, muon_lr, completed_steps)
        torch.cuda.synchronize() if device_type == "cuda" else None
        t1 = time.time()

        if master_process:

            # Calculate the MFU and other performance metrics.
            dt = t1 - t0
            tokens_processed = (args.micro_batch_size * gradient_accumulation_steps) * args.max_position_embeddings * world_size
            tokens_per_sec = tokens_processed / dt
            flops_per_token  = 6*C + 12*L*H*Q*T
            flops_per_fwdbwd = flops_per_token * T
            flops_per_iter = flops_per_fwdbwd * (args.micro_batch_size * gradient_accumulation_steps)
            flops_achieved = flops_per_iter * (1 / dt)
            mfu = (flops_achieved / P) * 100

            # Get the current VRAM usage.
            if device.startswith("cuda"):
                used_vram = torch.cuda.max_memory_allocated(device) // (1024 ** 3)
            else:
                used_vram = 0

            logger.info(f"Training | step: {completed_steps:5d} | loss: {accumulated_loss.item():.6f} | adam-lr: {adam_lr:.4e}  | muon-lr: {muon_lr:.4e} | lr stage: '{current_lr_stage}' | norm: {float(norm):.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | VRAM: {used_vram:.2f} | MFU: {mfu:.2f}%")

            with open(log_file, "a") as f:
                f.write(f"Training | step: {completed_steps:5d} | loss: {accumulated_loss.item():.6f} | adam-lr: {adam_lr:.4e}  | muon-lr: {muon_lr:.4e} | lr stage: '{current_lr_stage}' | norm: {float(norm):.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | VRAM: {used_vram:.2f} | MFU: {mfu:.2f}%\n")
            if args.wandb_token is not None:

                wandb.log({
                            "loss": accumulated_loss.item(),
                            "lr": adam_lr,
                            "muon_lr": muon_lr,
                            "grad_norm": norm,
                            "dt": dt,
                            "tokens_per_sec": tokens_per_sec,
                            "mfu": mfu,
                            })
    
    # Terminate the W&B tracker and the CodeCarbon tracker at the end of the training loop.
    if master_process:
        tracker.stop()
        if args.wandb_token is not None:
            wandb.finish()

    # Cleanup.
    if fsdp:
        dist.destroy_process_group()
    # Done!

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully Sharded Data Parallel Training")
    parser.add_argument(
        "--specs",
        type=str,
        required=True,
        help="The path to the specifications file.",
    )
    parser.add_argument(
        "--slurm-job-id",
        type=str,
        required=True,
        help="The SLURM job id.",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        required=True,
        help="The hardware used for training.",
    )
    args = parser.parse_args()

    main(args.specs, args.slurm_job_id, args.hardware)
