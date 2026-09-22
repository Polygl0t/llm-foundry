"""
Utility helpers for the distributed trainers.

Provides:
    - compute_training_schedule:    gradient accumulation steps and step counts
    - setup_triton_cache:           per-rank Triton cache with cleanup
    - StructuredTrainingLogger:     structured metadata/stats file writer
    - DistributedEnvironment:       Environment manager for distributed training (SLURM, torchrun, or local)
    - is_context_extension:         whether the run freezes down to attention-only training
    - load_checkpoint_state:        load optimizer and training state from checkpoint
                                    (context extension skips the optimizer state restore)
    - initialize_wandb:             login and init W&B run (one run per training stage, all
                                    stages of a multistage run sharing one `group` and one
                                    continuous step axis)
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

import numpy as np
import torch
import torch.distributed as dist


def _get_local_world_size():
    for variable in ("SLURM_NTASKS_PER_NODE", "SLURM_TASKS_PER_NODE"):
        if value := os.environ.get(variable):
            return int(value.split(",", 1)[0].split("(", 1)[0])
    if value := os.environ.get("LOCAL_WORLD_SIZE"):
        return int(value)
    return max(torch.cuda.device_count(), 1)


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
        # See https://docs.python.org/3/library/logging.html#
        logger = logging.getLogger(name)

        logging.basicConfig(
            format="%(name)s - %(message)s",
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        # basicConfig() only configures the root logger on the first call, so
        # set the level explicitly in case the root logger already has handlers.
        logger.setLevel(logging.INFO)

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

    def __init__(self, logger, mode="Distributed"):
        self.mode = mode
        self.local_world_size = _get_local_world_size()

        if "SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ:
            # SLURM cluster
            self.world_size = int(os.environ["SLURM_NTASKS"])
            self.rank = int(os.environ["SLURM_PROCID"])
            self.local_rank = int(
                os.environ.get("SLURM_LOCALID", self.rank % max(torch.cuda.device_count(), 1))
            )

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

        # Export torchrun-style env vars unconditionally. Some transformers code paths
        # (e.g. `initialize_tensor_parallelism`, triggered when `distributed_config` is
        # passed to `from_pretrained`) read `LOCAL_RANK` / `RANK` / `WORLD_SIZE` directly
        # from the environment and do not fall back to SLURM_* equivalents.
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)

        if self.world_size > 1:
            # Multi-process distributed training.
            if torch.cuda.is_available():
                # See https://docs.pytorch.org/docs/stable/distributed.html
                dist.init_process_group(
                    backend="nccl",
                    world_size=self.world_size,
                    rank=self.rank,
                    device_id=torch.device("cuda", self.local_rank),
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
                logger.info(
                    f"Running {self.mode} via '{dist.get_backend()}' backend. Logging process: {self.rank}. World size: {self.world_size}."
                )

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

        self.fsdp = self.ddp
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"

    @staticmethod
    def seed_everything(seed):
        """Set the random state seed for reproducibility."""
        # See https://docs.pytorch.org/docs/stable/torch.html
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


def is_context_extension(args):
    """
    Whether this run performs context extension.
    """
    return bool(getattr(args, "continual_pretraining", False)) and (
        getattr(args, "new_max_position_embeddings", None) is not None
    )


def _optimizer_state_is_loadable(optimizer, optimizer_state):
    """
    Whether a saved optimizer state dict describes the same parameter set as `optimizer`.

    Returns True when the state can be loaded (also when the pair cannot be inspected, e.g. a
    duck-typed optimizer without `param_groups` — `load_state_dict` itself then decides).
    """
    current_groups = getattr(optimizer, "param_groups", None)
    saved_groups = (
        optimizer_state.get("param_groups") if isinstance(optimizer_state, dict) else None
    )

    if current_groups is None or not isinstance(saved_groups, list):
        return True

    if len(current_groups) != len(saved_groups):
        return False

    return all(
        len(current_group.get("params", ())) == len(saved_group.get("params", ()))
        for current_group, saved_group in zip(current_groups, saved_groups, strict=False)
    )


def load_checkpoint_state(
    args,
    checkpoint_path,
    optimizer,
    device="cpu",
    master_process=False,
    logger=None,
    file_logger=None,
):
    """
    Load checkpoint state and restore the optimizer if resuming from a checkpoint.

    Context extension is exempt from the optimizer restore: the checkpoint was saved by a stage
    whose optimizer held the full (unfrozen) parameter set, while this stage's optimizer only
    holds the frozen-down attention subset, so `load_state_dict` would raise.

    Note that this exemption only kicks in when the saved state really is incompatible: a
    re-queued job of the *same* extension stage restores its optimizer state as usual, since the
    parameter set then matches.

    Returns a tuple of (resume_step, iter_count, epoch).
    """
    if args.resume_from_checkpoint:
        checkpoint = os.path.join(checkpoint_path, "checkpoint.pt")
        checkpoint = torch.load(checkpoint, map_location=torch.device(device), weights_only=False)

        if is_context_extension(args) and not _optimizer_state_is_loadable(
            optimizer, checkpoint["optimizer"]
        ):
            if master_process:
                message = (
                    f"Context extension detected: skipping the optimizer state restore from "
                    f"{checkpoint_path} because the checkpoint was saved with a different "
                    f"(non-frozen) parameter set. Continuing with a freshly initialized optimizer."
                )
                logger.info(message)
                file_logger.log_metadata(message)
        else:
            # The optimizer is updated in-place, so we don't need to return it.
            optimizer.load_state_dict(checkpoint["optimizer"])
            if master_process:
                logger.info(f"Resumed optimizer from checkpoint: {checkpoint_path}")
                file_logger.log_metadata(f"Resumed optimizer from checkpoint: {checkpoint_path}")

        if not args.begin_new_stage:
            resume_step = int(checkpoint["resume_step"])
            iter_count = int(checkpoint["iteration"])
            epoch = int(checkpoint["epoch"])
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

    with open(log_file) as file_handle:
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
    May update `args.num_train_epochs` in-place when `args.max_steps` overrides the schedule.
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
            math.ceil(max_steps / num_update_steps_per_epoch)
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
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    cache_dir = os.environ.get("TRITON_CACHE_DIR", f"./.cache/triton_cache/{slurm_job_id}")

    # Create rank-specific cache directory to avoid conflicts.
    rank = dist.get_rank() if dist.is_initialized() else 0
    rank_cache_dir = f"{cache_dir}/rank_{rank}"

    os.makedirs(rank_cache_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = rank_cache_dir

    # Cleanup old cache files older than 1 hour.
    try:
        for root, _, files in os.walk(rank_cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if os.path.getmtime(file_path) < time.time() - 3600:
                        os.remove(file_path)
                except OSError:
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
        with open(log_file) as f:
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
                f.writelines(lines[: last_validation_idx + 1])
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


def _resolve_multistage_group(args):
    """
    Return the W&B / Trackio group name that ties the stages of one training run together.
    """
    if args.wandb_id is not None:
        return str(args.wandb_id)
    if args.wandb_project is not None:
        return str(args.wandb_project)
    return "default"


def _inspect_multistage_state(args, group, project):
    """
    Inspect the Trackio project database for the state of this stage's multistage group.

    Returns a tuple of `(step_offset, previous_attempt_run_id)`:

    Both lookups are best effort: a missing or unreadable database (or an older trackio
    version) degrades to "no offset, no previous attempt" rather than failing the job.
    """
    step_offset = 0
    previous_attempt_run_id = None

    try:
        import pathlib

        from trackio import sqlite_storage
        from trackio import utils as trackio_utils

        # trackio resolves `TRACKIO_DIR` once, when it is first imported. The trainer sets the
        # environment variable before importing it, but re-align the cached value anyway so the
        # lookup is deterministic no matter what imported trackio first (an earlier stage in the
        # same process, a notebook, a test, ...).
        if os.environ.get("TRACKIO_DIR"):
            configured_dir = pathlib.Path(os.environ["TRACKIO_DIR"])
            if configured_dir != trackio_utils.TRACKIO_DIR:
                trackio_utils.TRACKIO_DIR = configured_dir
                sqlite_storage.TRACKIO_DIR = configured_dir

        for record in sqlite_storage.SQLiteStorage.get_run_records(project):
            config = (
                sqlite_storage.SQLiteStorage.get_run_config(
                    project, run=record["name"], run_id=record["id"]
                )
                or {}
            )

            record_group = config.get("_Group") or config.get("wandb_id")
            if record_group != group:
                continue

            if config.get("stage_name") == args.stage_name:
                # Our own stage: never counted towards the offset (see above). When the stage
                # is being restarted from scratch, its rows are a previous attempt.
                if args.begin_new_stage:
                    previous_attempt_run_id = record["id"]
                continue

            last_step = sqlite_storage.SQLiteStorage.get_max_step_for_run(
                project, run=record["name"], run_id=record["id"]
            )
            if last_step is not None:
                step_offset = max(step_offset, int(last_step))
    except Exception as error:
        print(
            "* Warning: trackio could not read the project database to continue the step "
            f"axis across stages: {error}"
        )

    return step_offset, previous_attempt_run_id


def initialize_wandb(args):
    """
    Login to W&B (or trackio, in offline mode) and initialize a run for the current stage.

    When `args.offline_mode` is True (for HPC clusters without internet access on
    compute nodes), trackio is used instead of W&B. trackio exposes a W&B-compatible
    API, so `trainer.py` keeps calling `wandb.log(...)` / `wandb.finish()` unchanged;
    here we simply patch the `wandb` name that `trainer.py` already imported to point
    at trackio instead. trackio's log directory comes from `args.trackio_dir` (falling
    back to `TRACKIO_DIR`, then `~/.trackio`).

    Multistage runs (`begin_new_stage: [...Warmup-Stable..., ...Cooldown...]`) get:

    - one run per stage, named `<group>-<stage_name>` -- stable across job requeues, so a
      resumed stage appends to its own run instead of creating a new one;
    - a shared `group` (= `wandb_id`) covering every stage of the run;
    - a step offset, so that all stages sit on one continuous x axis. `trainer.py` picks the
      offset up from `trainer._step_offset` and adds it to `completed_steps` before logging.

    Only call this on the master process and when `args.wandb_token` is not None.

    References:
        - See https://docs.wandb.ai/models/ref/python/functions/login
        - See https://docs.wandb.ai/models/ref/python/functions/init
        - See https://github.com/gradio-app/trackio
    """
    import time as _time

    # `trainer.py` owns the `wandb` name that the training loop logs through (it does
    # `import wandb` at module load time) and the step offset applied to those logs.
    import trainer

    project = args.wandb_project if args.wandb_project is not None else "default"
    group = _resolve_multistage_group(args)
    step_offset = 0
    previous_attempt_run_id = None

    if args.offline_mode:
        # Resolve TRACKIO_DIR with a three-tier precedence:
        #   1. args.trackio_dir  (explicit config)
        #   2. TRACKIO_DIR env var
        #   3. ~/.trackio  (default fallback)
        # It has to be set before the lookup below, since trackio resolves its database path
        # from the environment.
        if args.trackio_dir is not None:
            os.environ["TRACKIO_DIR"] = str(args.trackio_dir)
        elif "TRACKIO_DIR" not in os.environ:
            os.environ["TRACKIO_DIR"] = os.path.expanduser("~/.trackio")

        step_offset, previous_attempt_run_id = _inspect_multistage_state(args, group, project)

    # W&B run id / step axis: unlike trackio there is no local database to read the offset
    # from, so online W&B runs keep the per-stage step numbering (`trainer._step_offset` = 0).
    run_name = f"{group}-{args.stage_name}"
    resume = "allow"

    if previous_attempt_run_id is not None:
        # `begin_new_stage` restarts this stage's step counter, so its rows must not be mixed
        # with the earlier attempt's. Keep that attempt untouched and log this one separately.
        run_name = f"{run_name}-restart-{_time.strftime('%Y%m%d-%H%M%S')}"
        resume = "never"
        print(
            f"* An earlier attempt of stage '{args.stage_name}' already exists in group "
            f"'{group}'; logging this attempt as a separate run '{run_name}'."
        )

    # `trainer.py` turns these into the step number it reports to the tracker (see
    # `trainer._step_offset`). The `.log` file and the console keep the LOCAL per-stage steps.
    trainer._step_offset = step_offset

    if args.offline_mode:
        import trackio

        # `trainer.py` does `import wandb` at module load time. Since trackio's API
        # is W&B-compatible, we patch that already-imported reference in place so the
        # rest of the codebase does not need to change.
        trainer.wandb = trackio

        trackio.init(
            project=project,
            name=run_name,
            group=group,
            config=args.to_dict(),
            resume=resume,
            auto_log_gpu=args.trackio_auto_log_gpu,
        )
        return

    import wandb

    wandb.login(key=args.wandb_token)

    wandb.init(
        project=project,
        notes=args.wandb_desc if args.wandb_desc is not None else "N/A",
        name=run_name,
        group=group,
        config=args.to_dict(),
        resume=resume,
        id=f"{group}-{args.stage_name}",
    )


def create_emissions_tracker(args, logger):
    """
    Create and start a CodeCarbon EmissionsTracker (or OfflineEmissionsTracker).

    When `args.offline_mode` is True (for HPC clusters without internet access on
    compute nodes, which the default EmissionsTracker needs for IP-based geolocation),
    the OfflineEmissionsTracker is used instead, with the country/region supplied
    explicitly via `args.codecarbon_country_iso_code` / `args.codecarbon_region`.

    Only call this on the master process.

    References:
        - See https://docs.codecarbon.io/latest/reference/api/?h=Emission#emissionstracker-baseemissionstracker
        - See https://docs.codecarbon.io/latest/reference/api/?h=Emission#offlineemissionstracker-additional-parameters
    """
    common_kwargs = {
        "project_name": args.wandb_project if args.wandb_project is not None else "default",
        "log_level": "critical",
        "output_dir": args.checkpoint_dir,
        "output_file": "emissions.csv",
        "tracking_mode": "machine",
    }

    if args.offline_mode:
        from codecarbon import OfflineEmissionsTracker

        tracker = OfflineEmissionsTracker(
            country_iso_code=args.codecarbon_country_iso_code,
            region=args.codecarbon_region,
            **common_kwargs,
        )
        country_iso = args.codecarbon_country_iso_code or "unknown"
        region = args.codecarbon_region or "unknown"
    else:
        from codecarbon import EmissionsTracker

        # EmissionsTracker uses IP-based geolocation.
        # On air-gapped HPC nodes the lookup may fail, leaving _geo as None.
        tracker = EmissionsTracker(**common_kwargs)
        country = getattr(tracker, "_geo", None)
        country_iso = getattr(country, "country_iso_code", None) or "unknown"
        region = getattr(country, "region", None) or "unknown"

    logger.info(f"Geo Location: ISO: {country_iso} | Region: {region}")

    tracker.start()
    return tracker
