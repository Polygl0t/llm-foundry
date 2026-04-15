"""
Utility helpers for the DDP trainer.

Provides:
    - compute_training_schedule:    gradient accumulation steps and step counts
    - setup_triton_cache:           per-rank Triton cache with cleanup
    - StructuredTrainingLogger:     structured metadata/stats file writer
    - DistributedEnvironment:       DDP environment manager (SLURM, torchrun, or local)
    - load_checkpoint_state:        load optimizer and training state from checkpoint
    - initialize_wandb:             login and init W&B run
    - create_emissions_tracker:     create and start a CodeCarbon EmissionsTracker
    - cleanup_log_file:             truncate log after last validation entry
    - checkpoint_already_validated: check if a step was already validated
"""
import json
import logging
import math
import os
import sys
import time
import torch
import torch.distributed as dist
import numpy as np


class StructuredTrainingLogger:
    """
    Write training logs as metadata lines and JSON stats entries.
    
    This class has mainly two methods:
    - log_metadata: for writing human-readable metadata lines (e.g. hyperparameters)
    - log_stats: for writing structured JSON lines (e.g. training/validation metrics)
    """

    def __init__(self, log_file):
        self.log_file = log_file
        self.current_section = None
        with open(self.log_file, "a"):
            pass

    @classmethod
    def create_python_logger(cls, name):
        """Create a Python logger using the trainer's default console configuration."""
        # [Logging facility for Python](https://docs.python.org/3/library/logging.html#)
        logger = logging.getLogger(name)

        logging.basicConfig(
            format="%(name)s - %(message)s",
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        return logger

    def _switch_section(self, section):
        if self.current_section == section:
            return
        with open(self.log_file, "a") as file_handle:
            file_handle.write("---\n")
            file_handle.write(f"[{section}]\n")
        self.current_section = section

    def log(self, message, log_type):
        if log_type not in {"metadata", "stats"}:
            raise ValueError(f"Unsupported log type: {log_type}")

        self._switch_section(log_type)

        if log_type == "stats":
            payload = message if isinstance(message, dict) else {"message": str(message)}
            line = json.dumps(payload, sort_keys=True)
        else:
            if isinstance(message, dict):
                line = " | ".join(f"{key}: {value}" for key, value in message.items())
            else:
                line = str(message)

        with open(self.log_file, "a") as file_handle:
            file_handle.write(f"{line}\n")

    def log_metadata(self, message):
        self.log(message, "metadata")

    def log_stats(self, message):
        self.log(message, "stats")


class DistributedEnvironment:
    """Manages the distributed training environment setup and cleanup.

    Discovery order for world size and rank:
        1. SLURM variables  (SLURM_NTASKS / SLURM_PROCID)
        2. PyTorch launcher variables (WORLD_SIZE / RANK / LOCAL_RANK)
        3. Local fallback: use available GPUs or CPU
    """

    def __init__(self, logger):

        if "SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ:
            # SLURM cluster
            self.world_size = int(os.environ["SLURM_NTASKS"])
            self.rank = int(os.environ["SLURM_PROCID"])
            self.local_rank = int(os.environ.get("SLURM_LOCALID", self.rank % max(torch.cuda.device_count(), 1)))

        elif "WORLD_SIZE" in os.environ and "RANK" in os.environ:
            # torchrun / torch.distributed.launch
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        else:
            # Local single-process fallback (single GPU or CPU)
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0

        if self.world_size > 1:
            # Multi-process distributed training.
            if torch.cuda.is_available():
                # [PyTorch Distributed Documentation](https://docs.pytorch.org/docs/stable/distributed.html)
                dist.init_process_group(
                    backend="nccl",
                    world_size=self.world_size,
                    rank=self.rank,
                    device_id=self.local_rank,
                )
                self.device = f"cuda:{self.local_rank}"
                torch.cuda.set_device(self.device)
            else:
                dist.init_process_group(
                    backend="gloo",
                    world_size=self.world_size,
                    rank=self.rank,
                )
                self.device = "cpu"

            self.master_process = self.rank == 0
            self.ddp = True
            if self.master_process:
                logger.info(f"Running DDP via '{dist.get_backend()}' backend. Logging process: {self.rank}. World size: {self.world_size}.")

        else:
            # Single-process training (1 GPU or CPU).
            self.rank = 0
            if torch.cuda.is_available():
                self.device = "cuda:0"
                torch.cuda.set_device(self.device)
            else:
                self.device = "cpu"
            self.master_process = True
            self.ddp = False
            logger.info(f"Running single process training on {self.device}.")

        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"

    @staticmethod
    def seed_everything(seed):
        """Set the random state seed for reproducibility."""
        # [Common PyTorch Functions](https://docs.pytorch.org/docs/stable/torch.html)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def cleanup(self):
        """Clean up the distributed environment."""
        if self.ddp:
            dist.destroy_process_group()


def load_checkpoint_state(args, checkpoint_path, optimizer, master_process=False, logger=None, file_logger=None):
    """
    Load checkpoint state and restore the optimizer if resuming from a checkpoint.

    Returns a tuple of (resume_step, iter_count, epoch).
    """
    if args.resume_from_checkpoint:

        checkpoint = os.path.join(checkpoint_path, 'checkpoint.pt')
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'), weights_only=False)

        # The optimizer is updated in-place, so we don't need to return it.
        optimizer.load_state_dict(checkpoint['optimizer'])
        if master_process:
            logger.info(f"Resumed optimizer from checkpoint: {checkpoint_path}")
            file_logger.log_metadata(f"Resumed optimizer from checkpoint: {checkpoint_path}")

        if not args.begin_new_stage:
            resume_step = int(checkpoint['resume_step'])
            iter_count = int(checkpoint['iteration'])
            epoch = int(checkpoint['epoch'])
            return resume_step, iter_count, epoch

        else:
            if master_process:
                logger.info(f"Starting new training stage | {args.stage_name}")
                file_logger.log_metadata(f"Starting new training stage | {args.stage_name}")

    return 0, 0, 1


def _parse_legacy_validation_step(line):
    """
    Parse legacy validation step from a log line.
    This is for backward compatibility with older log formats that may not have structured JSON entries.
    """
    if not line.startswith("Validation") or "step:" not in line:
        return None
    try:
        step_fragment = line.split("step:", maxsplit=1)[1].split("|", maxsplit=1)[0]
        return int(step_fragment.strip())
    except (IndexError, ValueError):
        return None


def _iter_log_entries(log_file):
    """Helper generator to iterate through log entries, yielding (index, section, line) tuples."""
    current_section = None

    with open(log_file, "r") as file_handle:
        for index, raw_line in enumerate(file_handle):
            line = raw_line.rstrip("\n")

            if line == "---":
                continue

            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1]
                continue

            yield index, current_section, line


def compute_training_schedule(args, train_dataloader_length, world_size):
    """
    Compute gradient accumulation steps, steps per epoch, and total training steps.

    Returns a tuple of (gradient_accumulation_steps, num_update_steps_per_epoch, max_steps).
    May update ``args.num_train_epochs`` in-place when ``args.max_steps`` overrides the schedule.
    """
    tokens_per_step = args.micro_batch_size * args.max_position_embeddings * world_size
    assert args.total_batch_size % tokens_per_step == 0, (
        f"Make sure your `total_batch_size` ({args.total_batch_size}) is divisible by "
        f"`micro_batch_size` * `max_position_embeddings` * `world_size` ({tokens_per_step})"
    )
    gradient_accumulation_steps = args.total_batch_size // tokens_per_step

    num_update_steps_per_epoch = math.ceil(train_dataloader_length / gradient_accumulation_steps)
    max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)

    if args.max_steps is not None:
        max_steps = args.max_steps
        args.num_train_epochs = (
            max_steps // num_update_steps_per_epoch
            if max_steps > num_update_steps_per_epoch
            else 1
        )

    return gradient_accumulation_steps, num_update_steps_per_epoch, max_steps


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

def cleanup_log_file(log_file):
    """
    Clean up the log file by removing incomplete entries after the last validation entry.
    This ensures the log remains consistent when resuming training.
    """
    if not os.path.exists(log_file):
        return
    
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()

        last_validation_idx = -1

        for index, section, line in _iter_log_entries(log_file):
            if section == "stats":
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("status") == "validation":
                    last_validation_idx = index
                    continue

            legacy_step = _parse_legacy_validation_step(line)
            if legacy_step is not None:
                last_validation_idx = index

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
        for _, section, line in _iter_log_entries(log_file):
            if section == "stats":
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("status") == "validation" and payload.get("step") == step:
                    return True

            legacy_step = _parse_legacy_validation_step(line)
            if legacy_step == step:
                return True
    except Exception:
        # If we can't read the log, assume not validated to be safe
        return False
    
    return False


def initialize_wandb(args, slurm_job_id, max_steps):
    """
    Login to W&B and initialize a run.

    Only call this on the master process and when ``args.wandb_token`` is not None.

    References:
        - [wandb.login](https://docs.wandb.ai/ref/python/sdk/functions/login)
        - [wandb.init](https://docs.wandb.ai/ref/python/sdk/functions/init/)
    """
    import wandb
    import time as _time

    wandb.login(key=args.wandb_token)

    wandb.init(
        project=args.wandb_project if args.wandb_project is not None else "default",
        notes=args.wandb_desc if args.wandb_desc is not None else "N/A",
        name=f"""{args.wandb_id}-{args.stage_name}-{_time.strftime("%d-%m-%Y")}-bs-{args.total_batch_size}-epochs-{args.num_train_epochs}-steps-{max_steps}-lr-{args.max_learning_rate}-sch-{args.lr_decay_type}""",
        config=args.to_dict(),
        resume="allow",
        id=f"{args.wandb_id}-{slurm_job_id}" if args.wandb_id is not None else f"{slurm_job_id}",
    )


def create_emissions_tracker(args, logger):
    """
    Create and start a CodeCarbon EmissionsTracker.

    Only call this on the master process.

    References:
        - [EmissionsTracker](https://mlco2.github.io/codecarbon/usage.html#explicit-object)
        - [Tracking on the main process only](https://github.com/mlco2/codecarbon/issues/544)
    """
    from codecarbon import EmissionsTracker

    tracker = EmissionsTracker(
        project_name=args.wandb_project if args.wandb_project is not None else "default",
        log_level="critical",
        output_dir=args.checkpoint_dir,
        output_file="emissions.csv",
        tracking_mode="machine",
    )

    logger.info(
        f"Geo Location: ISO: {tracker._geo.country_iso_code} "
        f"| Country: {tracker._geo.country_name} "
        f"| Region : {tracker._geo.region}"
    )
    tracker.start()
    return tracker
