"""Utilities for managing per-rank Triton cache directories."""

import os
import time

TRITON_CACHE_CLEANUP_AGE = 3600


def setup_triton_cache() -> None:
    """Set up a per-rank Triton cache directory and clean up stale files.

    Rank resolution order:
    1. torch.distributed.get_rank() if distributed is initialized
    2. LOCAL_RANK
    3. CUDA_VISIBLE_DEVICES (commas replaced with hyphens)
    4. "0"

    If TRITON_CACHE_DIR is set, it is used as the cache base verbatim.
    Otherwise, the default base includes SLURM_JOB_ID (or "local").
    The rank-specific directory is appended to either base.
    """
    if "TRITON_CACHE_DIR" in os.environ:
        cache_dir = os.environ["TRITON_CACHE_DIR"]
    else:
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
        cache_dir = os.path.join(".", ".cache", "triton_cache", slurm_job_id)

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

    rank_cache_dir = os.path.join(cache_dir, f"rank_{rank}")
    os.makedirs(rank_cache_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = rank_cache_dir

    try:
        current_time = time.time()
        cutoff_time = current_time - TRITON_CACHE_CLEANUP_AGE

        for root, _, files in os.walk(rank_cache_dir):
            for filename in files:
                file_path = os.path.join(root, filename)

                try:
                    if os.path.getmtime(file_path) < cutoff_time:
                        os.remove(file_path)
                except OSError:
                    # Ignore files that cannot be inspected or removed.
                    pass
    except Exception:
        # Cache cleanup should not prevent execution.
        pass
