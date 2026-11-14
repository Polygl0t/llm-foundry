"""Per-rank Triton cache directory setup with stale-file cleanup."""

import os
import time

# Cache files older than this (in seconds) are removed at startup.
TRITON_CACHE_CLEANUP_AGE = 3600


def setup_triton_cache() -> None:
    """Point `TRITON_CACHE_DIR` at a per-rank directory and remove stale cache files.

    Rank resolution order:
    1. torch.distributed.get_rank() if distributed is initialized
    2. LOCAL_RANK
    3. CUDA_VISIBLE_DEVICES (commas replaced with hyphens)
    4. "0"

    If `TRITON_CACHE_DIR` is set it is used as the base as-is (SLURM scripts already scope it
    per job); otherwise the base is `./.cache/triton_cache/<SLURM_JOB_ID or "local">`.
    """
    if "TRITON_CACHE_DIR" in os.environ:
        cache_dir = os.environ["TRITON_CACHE_DIR"]
    else:
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
        cache_dir = f"./.cache/triton_cache/{slurm_job_id}"

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

    rank_cache_dir = f"{cache_dir}/rank_{rank}"
    os.makedirs(rank_cache_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = rank_cache_dir

    # Cleanup must never block startup.
    try:
        cutoff = time.time() - TRITON_CACHE_CLEANUP_AGE
        for root, _, files in os.walk(rank_cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if os.path.getmtime(file_path) < cutoff:
                        os.remove(file_path)
                except OSError:
                    pass
    except Exception:
        pass
