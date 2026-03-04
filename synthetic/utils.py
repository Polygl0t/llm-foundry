"""Utility functions for the vLLM inference scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from datatrove.utils.logging import logger

if TYPE_CHECKING:
    from transformers import AutoConfig


MAX_GPUS_PER_NODE = 8

# Failure pattern definitions: (pattern_string, failure_reason)
_FAILURE_PATTERNS: list[tuple[str, str]] = [
    # OOM errors
    (r"torch\.OutOfMemoryError.*CUDA out of memory", "OOM"),
    (r"ValueError.*No available memory for the cache blocks", "OOM"),
    (r"OutOfMemoryError", "OOM"),
    (r"CUDA out of memory", "OOM"),
    (r"Failed to load model - not enough GPU memory", "OOM"),
    # Time limit exceeded
    (r"DUE TO TIME LIMIT", "timeout"),
    # Server startup failures
    (r"Failed to start VLLMServer server", "server_fail"),
    (r"Server encountered unrecoverable error", "server_fail"),
]
FAILURE_PATTERNS = [(re.compile(p, re.IGNORECASE), reason) for p, reason in _FAILURE_PATTERNS]


def detect_failure_reason(log_path: Path | None, max_bytes: int = 100_000) -> str | None:
    """Detect the failure reason from a log file by reading head and tail.

    Args:
        log_path: Path to the log file to scan.
        max_bytes: Maximum bytes to read from head/tail of the file.

    Returns:
        Failure reason string ("OOM", "timeout", "server_fail") or None if no failure detected.
    """
    if log_path is None or not log_path.exists():
        return None

    file_size = log_path.stat().st_size
    if file_size == 0:
        return None

    with open(log_path, errors="ignore") as f:
        # Read tail first (final status like timeout takes priority)
        if file_size > max_bytes:
            f.seek(file_size - max_bytes)
        tail = f.read(max_bytes)

        # Also read head for startup failures (OOM) if file is large
        f.seek(0)
        head = f.read(max_bytes) if file_size > max_bytes else ""

    # Check tail first, then head
    for content in (tail, head):
        for pattern, reason in FAILURE_PATTERNS:
            if pattern.search(content):
                return reason
    return None


# Valid quantization methods
QUANTIZATION_METHODS = ("bitsandbytes",)

# Valid KV cache dtype options
KV_CACHE_DTYPE_OPTIONS = ("auto", "fp8_e4m3", "fp8_e5m2")


def normalize_speculative(spec) -> str:
    """
    Accepts dict/str/bool and returns a canonical JSON string or empty string.

    For ngram method: prompt_lookup_max = num_speculative_tokens - 1 (if present).
    For suffix method: no additional parameters are added.
    Any provided prompt_lookup_max in the input is ignored and recomputed for ngram.
    """
    if not spec:
        return ""
    obj = None
    if isinstance(spec, dict):
        obj = dict(spec)
    elif isinstance(spec, str):
        try:
            parsed = json.loads(spec)
            if isinstance(parsed, dict):
                obj = parsed
        except Exception:
            obj = None
    else:
        obj = None

    if isinstance(obj, dict):
        method = str(obj.get("method", "")).lower()
        # Only add prompt_lookup_max for ngram method
        if method == "ngram" and "num_speculative_tokens" in obj:
            try:
                n = int(obj["num_speculative_tokens"])
                obj["prompt_lookup_max"] = max(n - 1, 0)
            except Exception:
                obj.pop("prompt_lookup_max", None)
        return json.dumps(obj, separators=(",", ":"))
    return str(spec)


def normalize_quantization(quant: str | None) -> str | None:
    """
    Normalize quantization configuration string.

    Returns:
        Normalized quantization string or None if disabled.

    Supported methods:
        - "bitsandbytes": 4-bit quantization using BitsAndBytes
    """
    if quant is None:
        return None
    if isinstance(quant, str):
        quant_lower = quant.strip().lower()
        if quant_lower in ("none", "null", ""):
            return None
        if quant_lower in QUANTIZATION_METHODS:
            return quant_lower
        raise ValueError(f"Unknown quantization method: {quant}. Supported: {QUANTIZATION_METHODS}")
    return None


def normalize_kvc_dtype(kv_dtype: str | None) -> str:
    """
    Normalize KV cache dtype configuration string.

    Returns:
        Normalized KV cache dtype string. Defaults to "auto".

    Supported options:
        - "auto": Uses the model's default "unquantized" data type
        - "fp8_e4m3": FP8 E4M3 format (CUDA 11.8+)
        - "fp8_e5m2": FP8 E5M2 format (CUDA 11.8+)
    """
    if kv_dtype is None:
        return "auto"
    if isinstance(kv_dtype, str):
        kv_lower = kv_dtype.strip().lower()
        if kv_lower in ("none", "null", ""):
            return "auto"
        if kv_lower in KV_CACHE_DTYPE_OPTIONS:
            return kv_lower
        raise ValueError(f"Unknown kvc_dtype: {kv_dtype}. Supported: {KV_CACHE_DTYPE_OPTIONS}")
    return "auto"


def validate_config(
    tp: int,
    pp: int,
    dp: int,
    config: AutoConfig,
    prompt_template: str | None = None,
) -> None:
    """Validate configuration parameters for single-node inference.

    Raises ValueError if any configuration is invalid.
    """
    if prompt_template and "[[DOCUMENT]]" not in prompt_template:
        raise ValueError("Prompt template must contain [[DOCUMENT]] variable")

    if tp < 1:
        raise ValueError(f"tp must be >= 1, got {tp}.")
    if pp < 1:
        raise ValueError(f"pp must be >= 1, got {pp}.")
    if dp < 1:
        raise ValueError(f"dp must be >= 1, got {dp}.")

    total_gpus = tp * pp * dp
    if total_gpus > MAX_GPUS_PER_NODE:
        raise ValueError(
            f"TPxPPxDP ({tp}x{pp}x{dp}={total_gpus}) exceeds max GPUs per node ({MAX_GPUS_PER_NODE})."
        )

    # Check if tp is valid for vLLM
    # Handle multi-modal configs (e.g., Gemma3) where num_attention_heads is in text_config
    num_heads = int(getattr(config, "num_attention_heads", None) or config.text_config.num_attention_heads)
    if num_heads % tp != 0:
        raise ValueError(
            f"num_attention_heads ({num_heads}) must be divisible by tensor parallel size (tp={tp})."
        )
