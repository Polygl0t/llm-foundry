"""
Distributed Data Parallel (DDP) Training for Large Language Models

Production-ready training script for transformer-based causal language models using
PyTorch DDP with either standard AdamW or a hybrid Muon + Adam optimizer.
Designed for multi-GPU, multi-node SLURM clusters.

Modules:

- `data_loading`: Dataset loading and DataLoader creation.
- `mfu`: MFU calculation utilities for performance monitoring.
- `model_setup`: Pre-DDP model and tokenizer initialization.
- `optimizers`: Optimizer and learning rate scheduler creation.
- `specifications`: Dataclass definitions for training arguments.
- `trainer`: Encapsulates the training and validation loop in a `DDPTrainer` class.
- `utils`: Logging, checkpointing, and miscellaneous utilities.

How to Use:

1. Configure `specifications.yaml` with your training settings, including dataset paths, 
    model architecture, and optimization parameters.

2. Launch the training script with SLURM, specifying the number of nodes and GPUs.
    See the `train_ddp.sh` script for an example SLURM job submission.
"""
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

import argparse
import os

from model_setup import prepare_training_components
from specifications import TrainingArguments
from data_loading import prepare_dataloaders
from trainer import DDPTrainer
from mfu import create_mfu_context

from optimizers import (
    create_lr_scheduler, 
    create_optimizer, 
    get_optimizer_summary_lines
)

from utils import (
    StructuredTrainingLogger,
    DistributedEnvironment,
    compute_training_schedule,
    load_checkpoint_state,
    initialize_wandb,
    create_emissions_tracker,
    setup_triton_cache, 
    cleanup_log_file
)

def main(specs, slurm_job_id, hardware):

    # Load the training arguments from the specifications.yaml file.
    args = TrainingArguments.from_yaml(specs)

    # Initiate a logger for the training process.
    logger = StructuredTrainingLogger.create_python_logger(f"DDP-Trainer-{slurm_job_id}-{args.stage_name}")
    
    # Initialize the distributed environment.
    # See the `DistributedEnvironment` class in `utils.py` for details on what this does.
    # It returns:
    # - rank: the global rank of the current process (int).
    # - device: the torch.device object for the current process.
    # - device_type: the type of the device (e.g., "cuda" or "mps").
    # - world_size: the total number of processes across all nodes (int).
    # - master_process: a boolean indicating if this is the master process (rank 0).
    # - ddp: a boolean indicating if Distributed Data Parallel (DDP) is being used.
    env = DistributedEnvironment(logger)
    rank = env.rank
    device = env.device
    device_type = env.device_type
    world_size = env.world_size
    master_process = env.master_process
    ddp = env.ddp

    # Setup Triton cache before any GPU operations.
    setup_triton_cache()

    # If we are `resume_from_checkpoint`, we use the SLURM job ID from the checkpoint path.
    # The SLURM job ID is used to create a unique checkpoint directory for this training run, 
    # which allows us to avoid conflicts between different runs.
    if args.resume_from_checkpoint:
        rel = os.path.relpath(args.resume_from_checkpoint, args.checkpoint_dir)
        slurm_job_id = rel.split(os.sep)[0]
    
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, f"{slurm_job_id}")
    log_file = os.path.join(args.checkpoint_dir, f"{slurm_job_id}.log")
    file_logger = None
    
    if master_process:

        # Create the checkpoint directory if it doesn't exist.
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Clean up the log file if resuming from checkpoint.
        if args.resume_from_checkpoint:
            cleanup_log_file(log_file)
        
        # Initialize a structured training logger to log metadata and stats to a file.
        file_logger = StructuredTrainingLogger(log_file)

    # Set the random state seed for reproducibility.
    env.seed_everything(args.seed)

    # See the `model_setup.py` script for details on what this function does.
    model_state = prepare_training_components(
        args=args,
        device=device,
        master_process=master_process,
        logger=logger,
        file_logger=file_logger if master_process else None,
    )

    # It returns:
    # - the updated `args` object.
    # - the tokenizer (a HuggingFace tokenizer object).
    # - the model (a PyTorch nn.Module object, still unwrapped in DDP).
    # - the precision (torch.bfloat16 or torch.float32).
    # - the checkpoint path (if resuming from checkpoint, otherwise None).
    # - the number of trainable parameters in the model (int).
    # - the number of active trainable parameters in the model (int, counts only experts in MoE models).
    args = model_state.args
    tokenizer = model_state.tokenizer
    model = model_state.model
    precision = model_state.precision
    checkpoint_path = model_state.checkpoint_path
    trainable_params = model_state.trainable_params
    active_trainable_params = model_state.active_trainable_params

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
    raw_model = model.module if ddp else model 

    # See the `data_loading.py` module for details on dataset loading and dataloader creation.
    data = prepare_dataloaders(
        args=args,
        tokenizer=tokenizer,
        world_size=world_size,
        rank=rank,
        logger=logger if master_process else None,
        file_logger=file_logger if master_process else None,
    )

    train_dataloader = data.train_dataloader
    validation_dataloader = data.val_dataloader
    train_sampler = data.train_sampler

    # Calculate gradient accumulation steps, steps per epoch, and total training steps.
    gradient_accumulation_steps, num_update_steps_per_epoch, max_steps = compute_training_schedule(
        args, len(train_dataloader), world_size
    )
    if args.max_steps is not None and master_process:
        logger.info(f"Overriding the number of steps to {max_steps} as per the `max_steps` argument (check the YAML file if you are not sure).")
        file_logger.log_metadata(f"Overriding the number of steps to {max_steps} as per the `max_steps` argument (check the YAML file if you are not sure).")

    # Create the learning rate scheduler.
    # See the `optimizers.py` script for details on what this function does.
    lr_scheduler = create_lr_scheduler(args, max_steps)

    if master_process:
        logger.info(f"Using learning rate decay type: {args.lr_decay_type}")
        file_logger.log_metadata(f"Using learning rate decay type: {args.lr_decay_type}")

    # See the `optimizers.py` script for details on what this function does.
    # It returns: 
    # - the optimizer.
    # - the optimizer step (a function that we will call to update the optimizer).
    # - a label describing the optimizer (for logging purposes).
    optimizer, optimizer_step, optimizer_label = create_optimizer(
        model=model,
        args=args,
        device_type=device_type,
        master_process=master_process,
        logger=logger,
    )

    # If we are resuming from checkpoint, we load the optimizer state from the checkpoint.
    # It returns:
    # - resume_step: the step to resume from (int).
    # - iter_count: the number of batches consumed in the current epoch (int).
    # - epoch: the epoch to resume from (int).
    resume_step, iter_count, epoch = load_checkpoint_state(
        args=args,
        checkpoint_path=checkpoint_path,
        optimizer=optimizer,
        device=device,
        master_process=master_process,
        logger=logger,
        file_logger=file_logger if master_process else None,
    )

    if master_process:

        logger.info("="*50)
        logger.info(f"  Training stage | {args.stage_name}")
        logger.info(f"  Run ID | {slurm_job_id}")
        logger.info(f"  Hardware | {hardware.upper()}")
        logger.info(f"  World size (total GPUs) | {world_size}")
        logger.info(f"  Precision | {'bfloat16' if args.bf16 else 'float32'}")
        logger.info(f"  Resuming from checkpoint | {args.resume_from_checkpoint is not None}")
        if args.resume_from_checkpoint:
            logger.info(f"    Checkpoint path | {args.resume_from_checkpoint}")
            logger.info(f"    Starting from step | {resume_step if not args.begin_new_stage else None}")
        logger.info("="*50)
        logger.info("Dataset Configuration:")
        logger.info(f"  Num train examples | {data.num_train_samples:,}")
        logger.info(f"  Num validation examples | {data.num_val_samples:,}")
        logger.info(f"  Length of train dataloader | {len(train_dataloader):,}")
        logger.info(f"  Max position embeddings (seq length) | {args.max_position_embeddings:,}")
        logger.info(f"  Shuffle dataset | {args.shuffle_dataset}")
        logger.info(f"  Additional mask token IDs | {args.additional_mask_token_ids}")
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
        logger.info(f"  Model config | {args.path_to_model_config or args.base_model or 'From checkpoint'}")
        logger.info(f"  Attention implementation | {args.attn_implementation}")
        logger.info(f"  Gradient checkpointing | {args.gradient_checkpointing}")
        logger.info(f"  Liger kernel | {args.use_liger_kernel}")
        logger.info(f"  Torch compile | {args.torch_compile}")
        logger.info(f"  MFU type | {args.mfu_type}")
        logger.info(f"  Trainable parameters | {trainable_params:,}")
        if trainable_params != active_trainable_params:
            logger.info(f"  Active trainable parameters (counting only experts in MoE models) | {active_trainable_params:,}")
        logger.info("="*50)
        optimizer_summary_lines = get_optimizer_summary_lines(args)
        logger.info(f"Optimizer Configuration ({optimizer_label}):")
        for line in optimizer_summary_lines:
            logger.info(line)
        logger.info("="*50)

        if args.resume_from_checkpoint is None:
            file_logger.log_metadata("="*50)
            file_logger.log_metadata(f"  Training stage | {args.stage_name}")
            file_logger.log_metadata(f"  Run ID | {slurm_job_id}")
            file_logger.log_metadata(f"  Hardware | {hardware.upper()}")
            file_logger.log_metadata(f"  World size (total GPUs) | {world_size}")
            file_logger.log_metadata(f"  Precision | {'bfloat16' if args.bf16 else 'float32'}")
            file_logger.log_metadata("="*50)
            file_logger.log_metadata("Dataset Configuration:")
            file_logger.log_metadata(f"  Num train examples | {data.num_train_samples:,}")
            file_logger.log_metadata(f"  Num validation examples | {data.num_val_samples:,}")
            file_logger.log_metadata(f"  Length of train dataloader | {len(train_dataloader):,}")
            file_logger.log_metadata(f"  Max position embeddings (seq length) | {args.max_position_embeddings:,}")
            file_logger.log_metadata(f"  Shuffle dataset | {args.shuffle_dataset}")
            file_logger.log_metadata(f"  Additional mask token IDs | {args.additional_mask_token_ids}")
            file_logger.log_metadata("="*50)
            file_logger.log_metadata("Batch Configuration:")
            file_logger.log_metadata(f"  Num Epochs | {args.num_train_epochs}")
            file_logger.log_metadata(f"  Micro batch size per device | {args.micro_batch_size}")
            file_logger.log_metadata(f"  Gradient accumulation steps | {gradient_accumulation_steps}")
            file_logger.log_metadata(f"  Total batch size (samples) | {args.micro_batch_size * gradient_accumulation_steps * world_size}")
            file_logger.log_metadata(f"  Total batch size (tokens) | {args.total_batch_size:,}")
            file_logger.log_metadata(f"  Total optimization steps | {max_steps:,}")
            file_logger.log_metadata(f"  Steps per epoch | {num_update_steps_per_epoch:,}")
            file_logger.log_metadata(f"  Checkpointing every | {args.checkpointing_steps} steps")
            file_logger.log_metadata("="*50)
            file_logger.log_metadata("Model Architecture:")
            file_logger.log_metadata(f"  Model config | {args.path_to_model_config or args.base_model or 'From checkpoint'}")
            file_logger.log_metadata(f"  Attention implementation | {args.attn_implementation}")
            file_logger.log_metadata(f"  Gradient checkpointing | {args.gradient_checkpointing}")
            file_logger.log_metadata(f"  Liger kernel | {args.use_liger_kernel}")
            file_logger.log_metadata(f"  Torch compile | {args.torch_compile}")
            file_logger.log_metadata(f"  MFU type | {args.mfu_type}")
            file_logger.log_metadata(f"  Trainable parameters | {trainable_params:,}")
            if trainable_params != active_trainable_params:
                file_logger.log_metadata(f"  Active trainable parameters (counting only experts in MoE models) | {active_trainable_params:,}")
            file_logger.log_metadata("="*50)
            file_logger.log_metadata(f"Optimizer Configuration ({optimizer_label}):")
            for line in optimizer_summary_lines:
                file_logger.log_metadata(line)
            file_logger.log_metadata("="*50)

    tracker = None
    mfu_context = None

    if master_process:

        # Initialize W&B (if configured) and CodeCarbon.
        if args.wandb_token is not None:
            initialize_wandb(args, slurm_job_id, max_steps)

        # Create and start the CodeCarbon emissions tracker.
        tracker = create_emissions_tracker(args, logger)

        # Get the correct MFU calculation settings.
        # See the `mfu.py` script for details on what this function does.
        # Use active_trainable_params so MoE models count only the experts
        # that participate in each forward pass.
        mfu_context = create_mfu_context(args=args, hardware=hardware, num_parameters=active_trainable_params)

    # Create the DDPTrainer and run the training loop.
    trainer = DDPTrainer(
        args=args,
        model=model,
        raw_model=raw_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        optimizer_step=optimizer_step,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        train_sampler=train_sampler,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        resume_step=resume_step,
        iter_count=iter_count,
        epoch=epoch,
        device=device,
        device_type=device_type,
        ddp=ddp,
        world_size=world_size,
        master_process=master_process,
        precision=precision,
        logger=logger,
        file_logger=file_logger,
        log_file=log_file,
        slurm_job_id=slurm_job_id,
        tracker=tracker,
        mfu_context=mfu_context,
    )
    trainer.train()

    # Cleanup.
    env.cleanup()

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
    args = parser.parse_args()

    main(args.specs, args.slurm_job_id, args.hardware)
