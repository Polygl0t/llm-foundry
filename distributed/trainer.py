"""
Trainer class for the distributed trainers.

Encapsulates the main training and validation loop, checkpointing,
and per-step logging.

Provides:
    - `DDPTrainer` class with a `train()` method that runs the DDP training loop.
    - `FSDPTrainer` class with a `train()` method that runs the FSDP training loop.
"""
import torch
import torch.distributed as dist

import contextlib
import time
import math
import os

try:
    import wandb
except ImportError:
    wandb = None

from model_setup import get_full_model_state_dict, get_full_optimizer_state_dict
from utils import checkpoint_already_validated
from mfu import calculate_training_metrics


# Shared helper functions used by both DDPTrainer and FSDPTrainer
def _run_validation_forward(model, validation_dataloader, device, device_type, precision):
    """
    Run the validation forward pass. Returns (val_loss_accum, val_time).
    
    NOTE: We intentionally do NOT call model.eval() here. Liger kernel's
    fused_linear_cross_entropy only activates when self.training=True. In eval
    mode it falls back to materializing the full [batch, seq, vocab] logits
    tensor, which causes OOM with large batch sizes. Modern causal LLMs have no
    dropout or batchnorm, so train/eval mode produces identical outputs.
    """
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

    return val_loss_accum, val_time


def _log_validation(*, logger, file_logger, completed_steps, val_loss_accum, val_time, tracker, stage_name, wandb_token):
    """Log validation results to console, file logger, and optionally W&B."""
    logger.info(f"Validation | step: {completed_steps:5d} | loss: {val_loss_accum.item():.4f} | kWh: {tracker._total_energy.kWh:.2f} | val_time: {val_time:.2f}s")
    file_logger.log_stats(
        {
            "status": "validation",
            "step": completed_steps,
            "loss": round(val_loss_accum.item(), 6),
            "kwh": round(tracker._total_energy.kWh, 6),
            "val_time_s": round(val_time, 6),
            "stage_name": stage_name,
        }
    )

    if wandb_token is not None:
        wandb.log({"val_loss": val_loss_accum.item()})


def _save_checkpoint(*, output_dir, model_to_save, tokenizer, completed_steps, iter_count, epoch, config, optimizer_state, model_save_kwargs=None):
    """Save model, tokenizer, and training state to a checkpoint directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Save the model and tokenizer.
    if model_save_kwargs:
        model_to_save.save_pretrained(output_dir, **model_save_kwargs)
    else:
        model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save the optimizer state and other metadata.
    torch.save(
        {
        'resume_step' : completed_steps,
        'iteration': iter_count,
        'epoch': epoch,
        'config': config,
        'optimizer': optimizer_state,
        }, 
        f"{output_dir}/checkpoint.pt",
    )


def _push_to_hub(*, model_to_push, tokenizer, hub_model_id, hub_token, stage_name, completed_steps):
    """Push model and tokenizer to the Hugging Face Hub."""
    hub_id = f"{hub_model_id}-{stage_name}-{completed_steps}"
    model_to_push.push_to_hub(hub_id, token=hub_token, private=True)
    tokenizer.push_to_hub(hub_id, token=hub_token)


def _log_training_step(
    *, logger, file_logger, mfu_context,
    completed_steps, accumulated_loss, adam_lr, muon_lr, current_lr_stage, norm, dt,
    micro_batch_size, gradient_accumulation_steps, world_size, device, stage_name, wandb_token,
):
    """Calculate MFU and log training step metrics to console, file logger, and W&B."""
    # Calculate the MFU and other performance metrics.
    # See the `mfu.py` script for details on what this function does.
    performance_metrics = calculate_training_metrics(
        mfu_context=mfu_context,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        world_size=world_size,
        dt=dt,
    )
    
    # It returns:
    # - The global tokens per second (tokens/s processed across all GPUs).
    # - The tokens per second per GPU.
    # - The MFU (the GPU utilization metric). 
    global_tokens_per_sec = performance_metrics.global_tokens_per_sec
    tokens_per_sec_per_gpu = performance_metrics.tokens_per_sec_per_gpu
    mfu = performance_metrics.mfu

    # Get the peak VRAM usage during this step (reset before each step's t0).
    if device.startswith("cuda"):
        used_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
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
        "stage_name": stage_name,
    }
    if muon_lr is not None:
        training_stats["muon_lr"] = muon_lr
    file_logger.log_stats(training_stats)

    if wandb_token is not None:

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


def _finalize_training(*, tracker, wandb_token):
    """Stop the CodeCarbon tracker and finish the W&B run."""
    tracker.stop()
    if wandb_token is not None:
        wandb.finish()


class DDPTrainer:
    """Runs the training and validation loop."""

    def __init__(
        self, *,
        args,
        model,
        raw_model,
        tokenizer,
        optimizer,
        optimizer_step,
        lr_scheduler,
        train_dataloader,
        validation_dataloader,
        train_sampler,
        gradient_accumulation_steps,
        max_steps,
        resume_step,
        iter_count,
        epoch,
        device,
        device_type,
        ddp,
        world_size,
        master_process,
        precision,
        logger,
        file_logger,
        log_file,
        slurm_job_id,
        tracker,
        mfu_context,
    ):
        self.args = args
        self.model = model
        self.raw_model = raw_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.optimizer_step = optimizer_step
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.train_sampler = train_sampler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_steps = max_steps
        self.resume_step = resume_step
        self.iter_count = iter_count
        self.epoch = epoch
        self.device = device
        self.device_type = device_type
        self.ddp = ddp
        self.world_size = world_size
        self.master_process = master_process
        self.precision = precision
        self.logger = logger
        self.file_logger = file_logger
        self.log_file = log_file
        self.slurm_job_id = slurm_job_id
        self.tracker = tracker
        self.mfu_context = mfu_context

    def train(self):
        """Run the training loop."""

        # Local aliases for readability.
        args = self.args
        model = self.model
        raw_model = self.raw_model
        tokenizer = self.tokenizer
        optimizer = self.optimizer
        optimizer_step = self.optimizer_step
        lr_scheduler = self.lr_scheduler
        train_dataloader = self.train_dataloader
        validation_dataloader = self.validation_dataloader
        train_sampler = self.train_sampler
        gradient_accumulation_steps = self.gradient_accumulation_steps
        max_steps = self.max_steps
        resume_step = self.resume_step
        iter_count = self.iter_count
        epoch = self.epoch
        device = self.device
        device_type = self.device_type
        ddp = self.ddp
        world_size = self.world_size
        master_process = self.master_process
        precision = self.precision
        logger = self.logger
        file_logger = self.file_logger
        log_file = self.log_file
        slurm_job_id = self.slurm_job_id
        tracker = self.tracker
        mfu_context = self.mfu_context

        if args.resume_from_checkpoint and not args.begin_new_stage and epoch > 1:
            # Shuffle the sampler every epoch.
            train_sampler.set_epoch(epoch)
        
        if not args.begin_new_stage:
            if master_process:
                logger.info(f"WARNING: `begin_new_stage` is set to False. If this is a multistage training, make sure you set it to True for the new stages.")

        # Set the model to training mode.
        model.train()

        # Prepare a null context manager to use in combination with `model.no_sync()`
        # This is used to avoid synchronizing gradients during gradient accumulation steps.
        # See https://docs.python.org/3/library/contextlib.html#contextlib.nullcontext
        null_context = contextlib.nullcontext()

        # Create an iterator from the train dataloader.
        iter_train_dataloader = iter(train_dataloader)

        if master_process:
            logger.info(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")
            file_logger.log_metadata(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")

        # Flag to indicate if the learning rate stage has changed (starts as False)
        # We use this to save a checkpoint if the learning rate stage changes.
        lr_stage_change = False
        # Get the current learning rate stage
        current_lr_stage = lr_scheduler(resume_step)[-1]

        # Start the training loop.
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
                        file_logger.log_metadata(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")
             
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
                        logger.info("Running validation ...")

                    val_loss_accum, val_time = _run_validation_forward(
                        model, validation_dataloader, device, device_type, precision
                    )

                    if ddp:
                        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.SUM)
                        val_loss_accum = val_loss_accum / world_size

                    if master_process:
                        _log_validation(
                            logger=logger, file_logger=file_logger,
                            completed_steps=completed_steps, val_loss_accum=val_loss_accum,
                            val_time=val_time, tracker=tracker,
                            stage_name=args.stage_name, wandb_token=args.wandb_token,
                        )

                        # Create the checkpoint directory.
                        checkpoint_name = f"step_{completed_steps:05d}"
                        output_dir = os.path.join(args.checkpoint_dir, args.stage_name, checkpoint_name)
                        _save_checkpoint(
                            output_dir=output_dir, model_to_save=raw_model,
                            tokenizer=tokenizer, completed_steps=completed_steps,
                            iter_count=iter_count, epoch=epoch,
                            config=raw_model.config,
                            optimizer_state=optimizer.state_dict(),
                        )
                    
                        # Push it to the hub.
                        if args.push_to_hub and args.hub_token is not None and args.hub_model_id is not None:
                            _push_to_hub(
                                model_to_push=raw_model, tokenizer=tokenizer,
                                hub_model_id=args.hub_model_id, hub_token=args.hub_token,
                                stage_name=args.stage_name, completed_steps=completed_steps,
                            )

                        # Flush the codecarbon tracker at the end of the validation step.
                        tracker.flush()
                    
                    # Set barrier to ensure that all processes have finished the validation step before continuing.
                    if ddp:
                        dist.barrier()
            
            # We are timing the training loop to measure the MFU.
            # Reset peak memory stats so max_memory_allocated captures only this step.
            if device_type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
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
                        file_logger.log_metadata(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")
                    # Reset iter_count since we're starting a new epoch
                    iter_count = 0

                # Move the batch to the device.
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                
                # Use no_sync context manager for all steps except the last one,
                # which is the only one that requires gradient synchronization.
                sync_context = model.no_sync() if ddp and micro_step < gradient_accumulation_steps - 1 else null_context

                with sync_context:
                    # Autocast is a PyTorch context manager that enables mixed precision training.
                    # See https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-torch-autocast
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
                    file_logger.log_metadata(f"Learning rate stage changed to: {current_lr_stage} at step {completed_steps} | {args.stage_name}.")
            else:
                lr_stage_change = False

            optimizer_step(adam_lr, muon_lr, completed_steps)
            torch.cuda.synchronize() if device_type == "cuda" else None
            t1 = time.time()

            if master_process:
                _log_training_step(
                    logger=logger, file_logger=file_logger, mfu_context=mfu_context,
                    completed_steps=completed_steps, accumulated_loss=accumulated_loss,
                    adam_lr=adam_lr, muon_lr=muon_lr, current_lr_stage=current_lr_stage,
                    norm=norm, dt=t1 - t0,
                    micro_batch_size=args.micro_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    world_size=world_size, device=device,
                    stage_name=args.stage_name, wandb_token=args.wandb_token,
                )
        
        # Terminate the W&B tracker and the CodeCarbon tracker at the end of the training loop.
        if master_process:
            _finalize_training(tracker=tracker, wandb_token=args.wandb_token)

class FSDPTrainer:
    """Runs the training and validation loop."""

    def __init__(
        self, *,
        args,
        model,
        tokenizer,
        optimizer,
        optimizer_step,
        lr_scheduler,
        train_dataloader,
        validation_dataloader,
        train_sampler,
        gradient_accumulation_steps,
        max_steps,
        resume_step,
        iter_count,
        epoch,
        device,
        device_type,
        fsdp,
        world_size,
        master_process,
        precision,
        logger,
        file_logger,
        log_file,
        slurm_job_id,
        tracker,
        mfu_context,
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.optimizer_step = optimizer_step
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.train_sampler = train_sampler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_steps = max_steps
        self.resume_step = resume_step
        self.iter_count = iter_count
        self.epoch = epoch
        self.device = device
        self.device_type = device_type
        self.fsdp = fsdp
        self.world_size = world_size
        self.master_process = master_process
        self.precision = precision
        self.logger = logger
        self.file_logger = file_logger
        self.log_file = log_file
        self.slurm_job_id = slurm_job_id
        self.tracker = tracker
        self.mfu_context = mfu_context

    def train(self):
        """Run the training loop."""

        # Local aliases for readability.
        args = self.args
        model = self.model
        tokenizer = self.tokenizer
        optimizer = self.optimizer
        optimizer_step = self.optimizer_step
        lr_scheduler = self.lr_scheduler
        train_dataloader = self.train_dataloader
        validation_dataloader = self.validation_dataloader
        train_sampler = self.train_sampler
        gradient_accumulation_steps = self.gradient_accumulation_steps
        max_steps = self.max_steps
        resume_step = self.resume_step
        iter_count = self.iter_count
        epoch = self.epoch
        device = self.device
        device_type = self.device_type
        fsdp = self.fsdp
        world_size = self.world_size
        master_process = self.master_process
        precision = self.precision
        logger = self.logger
        file_logger = self.file_logger
        log_file = self.log_file
        slurm_job_id = self.slurm_job_id
        tracker = self.tracker
        mfu_context = self.mfu_context

        if args.resume_from_checkpoint and not args.begin_new_stage and epoch > 1:
            # Shuffle the sampler every epoch.
            train_sampler.set_epoch(epoch)
        
        if not args.begin_new_stage:
            if master_process:
                logger.info(f"WARNING: `begin_new_stage` is set to False. If this is a multistage training, make sure you set it to True for the new stages.")

        # Set the model to training mode.
        model.train()

        # Create an iterator from the train dataloader.
        iter_train_dataloader = iter(train_dataloader)

        if master_process:
            logger.info(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")
            file_logger.log_metadata(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")

        # Flag to indicate if the learning rate stage has changed (starts as False)
        # We use this to save a checkpoint if the learning rate stage changes.
        lr_stage_change = False
        # Get the current learning rate stage
        current_lr_stage = lr_scheduler(resume_step)[-1]

        # Start the training loop.
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
                        file_logger.log_metadata(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")
             
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
                        logger.info("Running validation ...")

                    val_loss_accum, val_time = _run_validation_forward(
                        model, validation_dataloader, device, device_type, precision
                    )

                    if fsdp:
                        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.SUM)
                        val_loss_accum = val_loss_accum / world_size

                    # Retrieve full model and optimizer state dicts from all FSDP ranks.
                    # These calls must happen on ALL processes (they trigger all-gather internally).
                    model_state_dict = get_full_model_state_dict(model) if fsdp else None
                    opt_state_dict = get_full_optimizer_state_dict(model, optimizer) if fsdp else None

                    if master_process:
                        _log_validation(
                            logger=logger, file_logger=file_logger,
                            completed_steps=completed_steps, val_loss_accum=val_loss_accum,
                            val_time=val_time, tracker=tracker,
                            stage_name=args.stage_name, wandb_token=args.wandb_token,
                        )

                        # Create the checkpoint directory.
                        checkpoint_name = f"step_{completed_steps:05d}"
                        output_dir = os.path.join(args.checkpoint_dir, args.stage_name, checkpoint_name)
                        _save_checkpoint(
                            output_dir=output_dir, model_to_save=model,
                            tokenizer=tokenizer, completed_steps=completed_steps,
                            iter_count=iter_count, epoch=epoch,
                            config=model.config,
                            optimizer_state=opt_state_dict if fsdp else optimizer.state_dict(),
                            model_save_kwargs={"state_dict": model_state_dict} if fsdp else None,
                        )
                    
                        # Push it to the hub.
                        if args.push_to_hub and args.hub_token is not None and args.hub_model_id is not None:
                            _push_to_hub(
                                model_to_push=model, tokenizer=tokenizer,
                                hub_model_id=args.hub_model_id, hub_token=args.hub_token,
                                stage_name=args.stage_name, completed_steps=completed_steps,
                            )

                        # Flush the codecarbon tracker at the end of the validation step.
                        tracker.flush()
                    
                    # Set barrier to ensure that all processes have finished the validation step before continuing.
                    if fsdp:
                        dist.barrier()
            
            # We are timing the training loop to measure the MFU.
            # Reset peak memory stats so max_memory_allocated captures only this step.
            if device_type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
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
                        file_logger.log_metadata(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")
                    # Reset iter_count since we're starting a new epoch
                    iter_count = 0

                # Move the batch to the device.
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                
                # For FSDP2 gradient accumulation:
                # - Use `set_requires_gradient_sync(False)` to skip gradient reduce-scatter on intermediate steps
                # - On the final micro-step, enable sync so gradients are properly reduced across ranks
                #
                # Note: FSDP2 does NOT support the `model.no_sync()` context manager used by DDP.
                # Instead, gradient synchronization is controlled via `set_requires_gradient_sync()`.
                is_last_micro_step = (micro_step == gradient_accumulation_steps - 1)
                if fsdp:
                    model.set_requires_gradient_sync(is_last_micro_step)

                # Autocast is a PyTorch context manager that enables mixed precision training.
                # See https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-torch-autocast
                with torch.autocast(device_type=device_type, dtype=precision):

                    # Transformers from HF perform the loss calculation internally,
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
                    file_logger.log_metadata(f"Learning rate stage changed to: {current_lr_stage} at step {completed_steps} | {args.stage_name}.")
            else:
                lr_stage_change = False

            optimizer_step(adam_lr, muon_lr, completed_steps)
            torch.cuda.synchronize() if device_type == "cuda" else None
            t1 = time.time()

            if master_process:
                _log_training_step(
                    logger=logger, file_logger=file_logger, mfu_context=mfu_context,
                    completed_steps=completed_steps, accumulated_loss=accumulated_loss,
                    adam_lr=adam_lr, muon_lr=muon_lr, current_lr_stage=current_lr_stage,
                    norm=norm, dt=t1 - t0,
                    micro_batch_size=args.micro_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    world_size=world_size, device=device,
                    stage_name=args.stage_name, wandb_token=args.wandb_token,
                )
        
        # Terminate the W&B tracker and the CodeCarbon tracker at the end of the training loop.
        if master_process:
            _finalize_training(tracker=tracker, wandb_token=args.wandb_token)
