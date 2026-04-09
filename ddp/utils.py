"""
Utility helpers for the DDP trainer.

Provides:
    - setup_triton_cache:           per-rank Triton cache with cleanup
    - StructuredTrainingLogger:     structured metadata/stats file writer
    - cleanup_log_file:             truncate log after last validation entry
    - checkpoint_already_validated: check if a step was already validated
"""
import json
import os
import time
import torch.distributed as dist


class StructuredTrainingLogger:
    """Write training logs as metadata lines and JSON stats entries."""

    def __init__(self, log_file):
        self.log_file = log_file
        self.current_section = None
        with open(self.log_file, "a"):
            pass

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


def _parse_legacy_validation_step(line):
    if not line.startswith("Validation") or "step:" not in line:
        return None
    try:
        step_fragment = line.split("step:", maxsplit=1)[1].split("|", maxsplit=1)[0]
        return int(step_fragment.strip())
    except (IndexError, ValueError):
        return None


def _iter_log_entries(log_file):
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
