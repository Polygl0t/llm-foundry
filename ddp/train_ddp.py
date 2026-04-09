"""
Distributed Data Parallel (DDP) Training for Large Language Models

Production-ready training script for transformer-based causal language models using
PyTorch DDP with either standard AdamW or a hybrid Muon + Adam optimizer.
Designed for multi-GPU, multi-node SLURM clusters.

How to Use:

1. Configure `specifications.yaml` with your training settings, including dataset paths, 
    model architecture, and optimization parameters.

2. Launch the training script with SLURM, specifying the number of nodes and GPUs.
    See the `train_ddp.sh` script for an example SLURM job submission.
"""
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch
import contextlib
import os
import time

from transformers import (
    default_data_collator, 
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForCausalLM
)
from liger_kernel.transformers import _apply_liger_kernel_to_instance
from specifications import TrainingArguments
from functools import partial

from optimizers import (
    normalize_optimizer_type, 
    create_lr_scheduler, 
    create_optimizer, 
    get_optimizer_summary_lines
)
from utils import (
    StructuredTrainingLogger,
    setup_triton_cache, 
    cleanup_log_file, 
    checkpoint_already_validated
)

from codecarbon import EmissionsTracker
import numpy as np
import datasets
import argparse
import logging
import wandb
import glob
import yaml
import math
import sys

def main(specs, slurm_job_id, hardware, optimizer_type_override=None):

    # Load the training arguments from the specifications.yaml file
    with open(specs, "r") as stream:
        kwargs = yaml.safe_load(stream)
    if optimizer_type_override is not None:
        kwargs["optimizer_type"] = optimizer_type_override
    
    # Create the `args` object from the loaded specifications.
    # Check the `specifications.py` script to see all available arguments.
    args = TrainingArguments(**kwargs)
    args.optimizer_type = normalize_optimizer_type(args.optimizer_type)
    kwargs["optimizer_type"] = args.optimizer_type

    # [Logging facility for Python](https://docs.python.org/3/library/logging.html#)
    logger = logging.getLogger(f"DDP-Trainer-{slurm_job_id}-{args.stage_name}")

    logging.basicConfig(
        format="%(name)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    if "SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ:

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
            ddp = True
            if master_process:
                logger.info(f"Running DDP via '{dist.get_backend()}' backend. Logging process: {rank}. World size: {world_size}.")
        
        else:
            # If the world size is 1, then we are not using distributed training.
            rank = 0
            device = "cuda:0"
            torch.cuda.set_device(device)
            master_process = True
            ddp = False
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
    
    log_file = None
    file_logger = None

    if master_process: 
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        # Create a log file to store the training logs.
        log_file = os.path.join(args.checkpoint_dir, f"{slurm_job_id}.log")
        
        # Clean up the log file if resuming from checkpoint
        if args.resume_from_checkpoint:
            cleanup_log_file(log_file)
        
        file_logger = StructuredTrainingLogger(log_file)

    def append_metadata(message):
        if file_logger is not None:
            file_logger.log_metadata(message)

    def append_stats(payload):
        if file_logger is not None:
            file_logger.log_stats(payload)

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
            append_metadata(f"No tokenizer name specified, using the {args.reference_model} to load the tokenizer.")

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
            append_metadata(f"Loaded chat template from {args.chat_template_path}. Chat template added to the tokenizer.")

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
            append_metadata(f"Resumed model from checkpoint: {checkpoint_path}")
    
    else:

        # If we are not resuming from a checkpoint, and we are not doing continual pretraining,
        # we initialize the model from `AutoConfig`.
        if not args.continual_pretraining:
            if master_process:
                logger.info(f"Initializing model from `AutoConfig`.")
                append_metadata("Initializing model from `AutoConfig`.")

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
                append_metadata(f"Initializing model from reference model: {args.reference_model} for continual pretraining/fine-tuning.")

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
                if config.rope_theta == args.rope_theta:
                    if master_process:
                        logger.info(f"WARNING: RoPE theta should scale when performing RoPE scaling. Check your configurations and perhaps adjust the rope_theta to a larger value.")
                
                # Set the new RoPE theta (hopefully scaled).
                config.rope_theta = args.rope_theta

                if master_process:
                    logger.info(f"Performing RoPE scaling to {config.max_position_embeddings} max position embeddings and {config.rope_theta} rope theta.")
                    logger.info(f"WARNING: `args.max_position_embeddings` has been reset to {args.max_position_embeddings}.")
                    append_metadata(f"Performing RoPE scaling to {config.max_position_embeddings} max position embeddings and {config.rope_theta} rope theta.")
                        

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
            append_metadata("Applied Liger kernels to the model.")

    # Change the model's `name_or_path`. If set to
    # None this will not show in the config (which is totally fine).
    model.config.name_or_path = args.hub_model_id
  
    # Print the number of trainable parameters in the model.
    if master_process:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters: {params:,}")
        append_metadata(f"Number of trainable parameters: {params:,}")

    # Set the gradient checkpointing for the model
    # What is Gradient Checkpointing? -> https://arxiv.org/abs/1604.06174
    if args.gradient_checkpointing:

        if master_process:
            logger.info(f"Gradient checkpointing enabled.")
            append_metadata("Gradient checkpointing enabled.")

        # Enable gradient checkpointing (https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForCausalLM)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False if torch.cuda.device_count() > 1 else True})
        model.config.use_cache = False

    # [Torch Compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
    # WARNING: Torch compile is not working good with liger kernel: https://github.com/linkedin/Liger-Kernel/issues/174
    if args.torch_compile and not args.use_liger_kernel:
        if master_process:
            logger.info(f"Compiling model with torch.compile.")
        model = torch.compile(model)

    model.to(device)

    if ddp:
        # Wrap the model with DistributedDataParallel (DDP).
        # DDP enables multi-process, multi-GPU training by replicating the model
        # across processes, synchronizing gradients between them, and ensuring
        # efficient scaling across devices. Each process is responsible for one
        # GPU and communicates with others to keep model replicas in sync.
        #
        #        +---------+    +---------+    +---------+
        #        | Rank 0  |    | Rank 1  |    | Rank 2  |
        #        |  GPU 0  |    |  GPU 1  |    |  GPU 2  |
        #        | Model   |    | Model   |    | Model   |
        #        +---------+    +---------+    +---------+
        #            \             |             /
        #             \            |            /
        #              \           |           /
        #               +---------------------+
        #               | Gradient All-Reduce |
        #               +---------------------+
        #
        # Key arguments here:
        #    - device_ids: Pin each process to a specific GPU based on its rank.
        #    - static_graph: 
        #        Enables optimizations for models with a fixed forward/backward
        #        graph (no dynamic control flow). Should be False when using
        #        gradient accumulation or dynamic graphs, since it may otherwise
        #        skip needed synchronizations.
        #    - gradient_as_bucket_view:
        #        Lets gradients be viewed directly from the communication buckets 
        #        to save memory. More efficient, but may complicate custom 
        #        gradient handling.
        # Note:
        #     Gradient accumulation can break if `static_graph=True`
        #     (see https://github.com/pytorch/pytorch/issues/143580).
        #Reference: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        model = DDP(
            model, 
            device_ids=[rank % torch.cuda.device_count()],
            static_graph=args.static_graph,
            gradient_as_bucket_view=True,
        ) 

    # Unwrap version of the model if it is wrapped in DDP.
    # Some methods we need to call on the unwrapped model
    # and this just make things simpler.
    raw_model = model.module if ddp else model 

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
                append_metadata(f"Shuffling enabled. Shuffling {len(train_dataset_files)} dataset files.")
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
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=args.shuffle_dataset,
        drop_last=False,
        )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
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
        append_metadata(f"Collate function masking settings: mask_eos_token={args.mask_eos_token}, mask_pad_token={args.mask_pad_token}.")
    
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
            append_metadata(f"Overriding the number of steps to {max_steps} as per the `max_steps` argument (check the YAML file if you are not sure).")

    lr_scheduler = create_lr_scheduler(args, max_steps, args.optimizer_type)

    if master_process:
        logger.info(f"Using learning rate decay type: {args.lr_decay_type}")
        append_metadata(f"Using learning rate decay type: {args.lr_decay_type}")

    optimizer, optimizer_step, optimizer_label = create_optimizer(
        model,
        args,
        device_type,
        args.optimizer_type,
        master_process,
        logger,
    )

    # Resume the optimizer from the checkpoint.
    if args.resume_from_checkpoint:

        checkpoint = os.path.join(checkpoint_path, 'checkpoint.pt')
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'), weights_only=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if master_process:
            logger.info(f"Resumed optimizer from checkpoint: {checkpoint_path}")
            append_metadata(f"Resumed optimizer from checkpoint: {checkpoint_path}")

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
        optimizer_summary_lines = get_optimizer_summary_lines(args, args.optimizer_type)
        logger.info(f"Optimizer Configuration ({optimizer_label}):")
        for line in optimizer_summary_lines:
            logger.info(line)
        logger.info("="*50)

        if args.resume_from_checkpoint is None:
            append_metadata("="*50)
            append_metadata("***** Running training *****")
            append_metadata(f"  Training stage | {args.stage_name}")
            append_metadata(f"  SLURM Job ID | {slurm_job_id}")
            append_metadata(f"  Hardware | {hardware.upper()}")
            append_metadata(f"  World size (total GPUs) | {world_size}")
            append_metadata(f"  Precision | {'bfloat16' if args.bf16 else 'float32'}")
            append_metadata("="*50)
            append_metadata("Dataset Configuration:")
            append_metadata(f"  Num train examples | {len(train_dataset):,}")
            append_metadata(f"  Num validation examples | {len(val_dataset):,}")
            append_metadata(f"  Length of train dataloader | {len(train_dataloader):,}")
            append_metadata(f"  Max position embeddings (seq length) | {args.max_position_embeddings:,}")
            append_metadata(f"  Shuffle dataset | {args.shuffle_dataset}")
            append_metadata(f"  Mask EOS token | {args.mask_eos_token}")
            append_metadata(f"  Mask PAD token | {args.mask_pad_token}")
            append_metadata("="*50)
            append_metadata("Batch Configuration:")
            append_metadata(f"  Num Epochs | {args.num_train_epochs}")
            append_metadata(f"  Micro batch size per device | {args.micro_batch_size}")
            append_metadata(f"  Gradient accumulation steps | {gradient_accumulation_steps}")
            append_metadata(f"  Total batch size (samples) | {args.micro_batch_size * gradient_accumulation_steps * world_size}")
            append_metadata(f"  Total batch size (tokens) | {args.total_batch_size:,}")
            append_metadata(f"  Total optimization steps | {max_steps:,}")
            append_metadata(f"  Steps per epoch | {num_update_steps_per_epoch:,}")
            append_metadata(f"  Checkpointing every | {args.checkpointing_steps} steps")
            append_metadata("="*50)
            append_metadata("Model Architecture:")
            append_metadata(f"  Model type | {args.reference_model if args.reference_model else 'Custom'}")
            append_metadata(f"  Attention implementation | {args.attn_implementation}")
            append_metadata(f"  Gradient checkpointing | {args.gradient_checkpointing}")
            append_metadata(f"  Liger kernel | {args.use_liger_kernel}")
            append_metadata(f"  Torch compile | {args.torch_compile}")
            append_metadata(f"  Trainable parameters | {params:,}")
            append_metadata("="*50)
            append_metadata(f"Optimizer Configuration ({optimizer_label}):")
            for line in optimizer_summary_lines:
                append_metadata(line)
            append_metadata("="*50)

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
                append_metadata(f"Starting new training stage | {args.stage_name}")
    
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

    # Prepare a null context manager to use in combination with `model.no_sync()`
    # This is used to avoid synchronizing gradients during gradient accumulation steps.
    # [nullcontext](https://docs.python.org/3/library/contextlib.html#contextlib.nullcontext)
    null_context = contextlib.nullcontext()

    # Create an iterator from the train dataloader.
    iter_train_dataloader = iter(train_dataloader)

    if master_process:
        logger.info(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")
        append_metadata(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")

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
                    append_metadata(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")
         
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
                log_file if master_process else os.path.join(args.checkpoint_dir, f"{slurm_job_id}.log")
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

                if ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.SUM)
                    val_loss_accum = val_loss_accum / world_size

                model.train()

                if master_process:
                    logger.info(f"Validation | step: {completed_steps:5d} | loss: {val_loss_accum.item():.4f} | kWh: {tracker._total_energy.kWh:.2f} | val_time: {val_time:.2f}s")
                    append_stats(
                        {
                            "status": "validation",
                            "step": completed_steps,
                            "loss": round(val_loss_accum.item(), 6),
                            "kwh": round(tracker._total_energy.kWh, 6),
                            "val_time_s": round(val_time, 6),
                            "stage_name": args.stage_name,
                        }
                    )

                    if args.wandb_token is not None:
                        wandb.log({"val_loss": val_loss_accum.item()})

                    # Create the checkpoint directory.
                    checkpoint_name = f"step_{completed_steps:05d}"
                    output_dir = os.path.join(args.checkpoint_dir, args.stage_name, checkpoint_name)
                    os.makedirs(output_dir, exist_ok=True)

                    # Save the model and tokenizer.
                    raw_model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    # Save the optimizer state and other metadata.
                    torch.save(
                        {
                        'resume_step' : completed_steps,
                        'iteration': iter_count,
                        'epoch': epoch,
                        'config': raw_model.config,
                        'optimizer': optimizer.state_dict(),
                        }, 
                        f"{output_dir}/checkpoint.pt",
                    )
                
                    # Push it to the hub.
                    if args.push_to_hub and args.hub_token is not None and args.hub_model_id is not None:
                        hub_model_id = f"{args.hub_model_id}-{args.stage_name}-{completed_steps}"
                        raw_model.push_to_hub(hub_model_id, token=args.hub_token, private=True)
                        tokenizer.push_to_hub(hub_model_id, token=args.hub_token)

                    # Flush the codecarbon tracker at the end of the validation step.
                    tracker.flush()
                
                # Set barrier to ensure that all processes have finished the validation step before continuing.
                if ddp:
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
                    append_metadata(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")
                # Reset iter_count since we're starting a new epoch
                iter_count = 0

            # Move the batch to the device.
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # Use no_sync context manager for all steps except the last one,
            # which is the only one that requires gradient synchronization.
            sync_context = model.no_sync() if ddp and micro_step < gradient_accumulation_steps - 1 else null_context

            with sync_context:
                # Autocast is a PyTorch context manager that enables mixed precision training.
                # [torch.autocast](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-torch-autocast)
                with torch.autocast(device_type=device_type, dtype=precision):

                    # Transformers from HF perform the loss calculation internelly,
                    # as well as the shifting of the labels in the case of a causal language model.
                    # - Note: the loss is already an average loss over the micro-batch.
                    loss = model(
                                input_ids=batch["input_ids"],
                                labels=batch["labels"],
                            ).loss

                # Accumulate the raw loss.
                accumulated_loss += loss.detach()
                
                # Scale the loss for gradient accumulation before backward pass.
                scaled_loss = loss / gradient_accumulation_steps
                scaled_loss.backward()
                # Track batch consumption for epoch boundary detection
                iter_count += 1

        # Average the accumulated loss over gradient accumulation steps.
        accumulated_loss = accumulated_loss / gradient_accumulation_steps

        if ddp:
            dist.all_reduce(accumulated_loss, op=dist.ReduceOp.SUM)
            accumulated_loss = accumulated_loss / world_size

        # Clip gradients up to `args.max_grad_norm`.
        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)

        # Determine the learning rate for the current step.
        adam_lr, muon_lr, lr_stage = lr_scheduler(completed_steps)

        # And check if the learning rate stage has changed.
        if lr_stage != current_lr_stage:
            lr_stage_change = True
            current_lr_stage = lr_stage
            if master_process:
                logger.info(f"Learning rate stage changed to: {current_lr_stage} at step {completed_steps} | {args.stage_name}.")
                append_metadata(f"Learning rate stage changed to: {current_lr_stage} at step {completed_steps} | {args.stage_name}.")
        else:
            lr_stage_change = False

        optimizer_step(adam_lr, muon_lr, completed_steps)
        torch.cuda.synchronize() if device_type == "cuda" else None
        t1 = time.time()

        if master_process:

            # Calculate the MFU and other performance metrics.
            dt = t1 - t0
            tokens_processed = (args.micro_batch_size * gradient_accumulation_steps) * args.max_position_embeddings * world_size
            global_tokens_per_sec = tokens_processed / dt
            tokens_per_sec_per_gpu = global_tokens_per_sec / world_size
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

            lr_log = f"adam-lr: {adam_lr:.4e}"
            if muon_lr is not None:
                lr_log += f" | muon-lr: {muon_lr:.4e}"

            logger.info(f"Training | step: {completed_steps:5d} | loss: {accumulated_loss.item():.6f} | {lr_log} | lr stage: '{current_lr_stage}' | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | global tok/sec: {global_tokens_per_sec:.2f} | tok/sec/gpu: {tokens_per_sec_per_gpu:.2f} | VRAM: {used_vram:.2f} | MFU: {mfu:.2f}%")
            training_stats = {
                "status": "training",
                "step": completed_steps,
                "loss": round(accumulated_loss.item(), 6),
                "adam_lr": adam_lr,
                "lr_stage": current_lr_stage,
                "grad_norm": round(float(norm), 6),
                "dt_ms": round(dt * 1000, 6),
                "global_tokens_per_sec": round(global_tokens_per_sec, 6),
                "tokens_per_sec_per_gpu": round(tokens_per_sec_per_gpu, 6),
                "vram_gb": round(float(used_vram), 6),
                "mfu": round(mfu, 6),
                "stage_name": args.stage_name,
            }
            if muon_lr is not None:
                training_stats["muon_lr"] = muon_lr
            append_stats(training_stats)

            if args.wandb_token is not None:

                metrics = {
                    "loss": accumulated_loss.item(),
                    "step": completed_steps,
                    "adam_lr": adam_lr,
                    "grad_norm": norm,
                    "dt_ms": dt * 1000,
                    "global_tokens_per_sec": global_tokens_per_sec,
                    "tokens_per_sec_per_gpu": tokens_per_sec_per_gpu,
                    "mfu": mfu,
                }
                if muon_lr is not None:
                    metrics["muon_lr"] = muon_lr
                wandb.log(metrics)
    
    # Terminate the W&B tracker and the CodeCarbon tracker at the end of the training loop.
    if master_process:
        tracker.stop()
        if args.wandb_token is not None:
            wandb.finish()

    # Cleanup.
    if ddp:
        dist.destroy_process_group()
    # Done!

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Data Parallel Training")
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
    parser.add_argument(
        "--optimizer-type",
        type=str,
        default=None,
        help="Optional override for the optimizer type (`adamw` or `muon_adam`).",
    )
    args = parser.parse_args()

    main(args.specs, args.slurm_job_id, args.hardware, optimizer_type_override=args.optimizer_type)
