"""Utilities for managing per-rank Triton cache directories."""

import os
import time

TRITON_CACHE_CLEANUP_AGE = 3600


def setup_triton_cache() -> None:
    """Set up a per-rank Triton cache directory with stale-file cleanup.

    Rank resolution order:
    1. torch.distributed.get_rank() if distributed is initialized
    2. LOCAL_RANK
    3. CUDA_VISIBLE_DEVICES (commas replaced with hyphens)
    4. "0"
    """
    cache_dir = os.environ.get("TRITON_CACHE_DIR", "./.cache/triton_cache")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")

    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = str(torch.distributed.get_rank())
        elif "LOCAL_RANK" in os.environ:
            rank = os.environ["LOCAL_RANK"]
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            rank = os.environ["CUDA_VISIBLE_DEVICES"].replace(",", "-")
        else:
            rank = "0"
    except ImportError:
        rank = os.environ.get(
            "LOCAL_RANK",
            os.environ.get("CUDA_VISIBLE_DEVICES", "0").replace(",", "-"),
        )

    rank_cache_dir = f"{cache_dir}/{slurm_job_id}/rank_{rank}"
    os.makedirs(rank_cache_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = rank_cache_dir

    try:
        current_time = time.time()
        for root, _, files in os.walk(rank_cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if os.path.getmtime(file_path) < current_time - TRITON_CACHE_CLEANUP_AGE:
                        os.remove(file_path)
                except OSError:
                    # Ignore files that cannot be inspected or removed.
                    pass
    except Exception:
        # Cache cleanup should not prevent execution.
        pass
