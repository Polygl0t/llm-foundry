"""Utility functions for the vLLM inference scripts."""

from __future__ import annotations

import os
import re
import sys
import glob
import time
import json
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from datatrove.utils.logging import logger

if TYPE_CHECKING:
    from transformers import AutoConfig
# Maximum GPUs per node for validation (adjust as needed for your cluster)
MAX_GPUS_PER_NODE = 8

VRAM_MB_TO_GB = 1024

# Time threshold for cleaning up old Triton cache files (in seconds)
TRITON_CACHE_CLEANUP_AGE = 3600

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

# Valid quantization methods
QUANTIZATION_METHODS = ("bitsandbytes",)

# Valid KV cache dtype options
KV_CACHE_DTYPE_OPTIONS = ("auto", "fp8_e4m3", "fp8_e5m2")

class DatasetLoader:
    """Loads datasets from a local file, local directory, or HuggingFace Hub.

    Source type is detected automatically:
    - Directory  -> all .jsonl or .parquet files inside are loaded.
    - Local file -> .jsonl or .parquet are supported.
    - Anything else is treated as a HuggingFace Hub dataset identifier.
    """

    _FILE_FORMATS = {".jsonl": "json", ".json": "json", ".parquet": "parquet"}

    def __init__(self, path: str, cache_dir: str | None = None, seed: int | None = None,
                 split: str = "train", subset: str | None = None) -> None:
        self.path = path
        self.cache_dir = cache_dir
        self.seed = seed
        self.split = split
        self.subset = subset

    def load(self):
        if os.path.isdir(self.path):
            dataset = self._from_directory()
        elif os.path.isfile(self.path):
            dataset = self._from_file()
        else:
            dataset = self._from_hf()
        return dataset.shuffle(seed=self.seed) if self.seed is not None else dataset

    def _from_file(self):
        ext = os.path.splitext(self.path)[1].lower()
        fmt = self._FILE_FORMATS.get(ext)
        if fmt is None:
            raise ValueError(f"Unsupported file format '{ext}'. Expected .jsonl or .parquet.")
        return datasets.load_dataset(fmt, data_files=self.path, split="train", cache_dir=self.cache_dir)

    def _from_directory(self):
        for ext, fmt in (("*.jsonl", "json"), ("*.parquet", "parquet")):
            files = sorted(glob.glob(os.path.join(self.path, ext)))
            if files:
                return datasets.load_dataset(
                    fmt, data_files=files, split="train",
                    num_proc=len(files), cache_dir=self.cache_dir,
                )
        raise ValueError(f"No .jsonl or .parquet files found in '{self.path}'.")

    def _from_hf(self):
        load_args = {"path": self.path, "split": self.split, "cache_dir": self.cache_dir}
        if self.subset is not None:
            load_args["name"] = self.subset
        return datasets.load_dataset(**load_args)


def setup_triton_cache() -> None:
    """
    Setup Triton cache directory with proper permissions and cleanup.

    -   This helps to avoid conflicts where different processes 
        might try to access cache files that have been modified
        or deleted.
    """
    cache_dir = os.environ.get('TRITON_CACHE_DIR', './.cache/triton_cache')
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
    cuda_visible_device = os.environ.get('CUDA_VISIBLE_DEVICES', '0').replace(',', '-')
    rank_cache_dir = f"{cache_dir}/{slurm_job_id}/rank_{cuda_visible_device}"

    logger.info(rank_cache_dir)
    os.makedirs(rank_cache_dir, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = rank_cache_dir

    # Clean up stale cache files
    try:
        current_time = time.time()
        for root, _, files in os.walk(rank_cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if os.path.getmtime(file_path) < current_time - TRITON_CACHE_CLEANUP_AGE:
                        os.remove(file_path)
                except (OSError, IOError):
                    pass  # Ignore errors when cleaning up
    except Exception:
        pass

def get_nvidia_smi_vram() -> list[float]:
    """Get the current VRAM usage of NVIDIA GPUs in GB."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        vram_list = result.decode("utf-8").strip().split("\n")
        return [float(v) / VRAM_MB_TO_GB for v in vram_list]
    except Exception:
        return [0.0]  # Return 0 instead of error string


def load_model_and_tokenizer(
        model_name_or_path: str,
        cache_dir: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
    ) -> tuple[AutoTokenizer, LLM]:
    """Load the model and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        cache_dir=cache_dir,
    )

    # [`vllm.LLM`](https://docs.vllm.ai/en/latest/api/vllm/#vllm.LLM)
    model = LLM(
        model=model_name_or_path,
        dtype=torch.float16 if "AWQ" in model_name_or_path else torch.bfloat16,
        download_dir=cache_dir,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    return tokenizer, model


def generate_rollouts(
        model: LLM,
        tokenizer: AutoTokenizer,
        input_string: str,
        system: str,
        sampling_params: SamplingParams,
        track_vram: bool = False,
        enable_thinking: bool = False,
    ) -> list[str]:
    """Generate text samples using the model."""

    # [`apply_chat_template`](https://huggingface.co/docs/transformers/main/chat_templating#using-applychattemplate)
    raw_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": input_string}
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )

    t0 = time.time()
    outputs = model.generate([raw_text], sampling_params, use_tqdm=False)
    elapsed_time = time.time() - t0

    nvidia_smi_vram = None
    if track_vram:
        nvidia_smi_vram = f'VRAM: {get_nvidia_smi_vram()[0]:.2f} GB'
    tokens_generated = len(tokenizer(outputs[0].outputs[0].text).input_ids)

    log_message = f"Time taken: {elapsed_time:.2f}s | Tokens: {tokens_generated}"
    if nvidia_smi_vram is not None:
        log_message += f" | {nvidia_smi_vram}"
    logger.info(log_message)

    return [seq.text for seq in outputs[0].outputs]


def get_logger(name: str) -> logging.Logger:
    """Create a simple logger."""
    # [Logging facility for Python](https://docs.python.org/3/library/logging.html#)
    logger = logging.getLogger(name)

    logging.basicConfig(
        format="%(name)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    return logger


def chunk_text(
        text: str, 
        tokenizer: AutoTokenizer, 
        max_chunk_size: int, 
        chunk_once: bool
    ) -> list[str]:
    """Chunk text into smaller pieces if it exceeds max_chunk_size"""
    tokenized_text = tokenizer(text).input_ids
    chunks = [
        tokenized_text[i:i + max_chunk_size] 
        for i in range(0, len(tokenized_text), max_chunk_size)
    ]
    decoded_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
    
    return [decoded_chunks[0]] if chunk_once else decoded_chunks


def get_starting_row(file_path: str, row_start: int | None) -> int:
    """
    Determine the starting row for processing.

    Here, we:
        1. Check if `row_start` is provided as an argument. If so, we use that.
        2. If not, we check the existing file to determine the next row to start from.
    This allows for resuming the generation process without redoing work.
    """
    if row_start is not None:
        return row_start

    if not os.path.exists(file_path):
        return 0

    max_row = -1
    with open(file_path, "r") as f:
        for line in f:
            try:
                max_row = max(max_row, int(json.loads(line)["row"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    return max_row + 1


def save_samples(
        output_file: str,
        row: int,
        seed_text: str,
        rollouts: list[str],
        metadata: dict,
        chunk: int | None = None,
    ) -> None:
    """
    Save generated samples to a file.
    Saved files look like this:

        ```json
        {"row": 0, "seed_text": "What is the capital of France?", "rollouts": ["The capital of France is Paris.", "Paris is the capital of France."], "chunk": 0, "metadata": {"difficulty": "easy"}}
        ```
    """

    record: dict = {
        "row": row, 
        "seed_text": seed_text, 
        "rollouts": rollouts, 
        "chunk": chunk, 
        "metadata": metadata
    }
    
    if chunk is not None:
        record["chunk"] = chunk
    if metadata:
        record["metadata"] = metadata
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_rollouts(
        sample: dict,
        counter: int,
        text_column: str,
        metadata_columns: list[str],
        model: LLM,
        tokenizer: AutoTokenizer,
        sampling_params: SamplingParams,
        file_path: str,
        system: str,
        prompt_prefix: str,
        prompt_suffix: str,
        max_chunk_size: int,
        chunk_once: bool,
        track_vram: bool = False,
        enable_thinking: bool = False,
        
    ) -> None:
    """
    Process a single dataset sample: 
    
        1. Chunk if needed;
        2. Generate rollouts;
        3. Save the results.
    """
    text_content = sample[text_column]
    metadata = {col: sample[col] for col in metadata_columns if col in sample}
    token_count = len(tokenizer(text_content).input_ids)

    if token_count > max_chunk_size:
        logger.info(f"Chunking row {counter} ({token_count} tokens) into {max_chunk_size}-token chunks...")
        text_samples = chunk_text(text_content, tokenizer, max_chunk_size, chunk_once)
    else:
        text_samples = [text_content]

    for i, text in enumerate(text_samples):
        full_prompt = f"{prompt_prefix}{text}{prompt_suffix}"
        chunk_label = i if len(text_samples) > 1 else None
        logger.info(f"Generating samples for row {counter}. Chunk {i + 1}/{len(text_samples)}...")

        rollouts = generate_rollouts(
            model=model,
            tokenizer=tokenizer,
            input_string=full_prompt,
            system=system,
            sampling_params=sampling_params,
            track_vram=track_vram,
            enable_thinking=enable_thinking
        )

        save_samples(
            file_path,
            row=counter,
            seed_text=text,
            rollouts=rollouts,
            metadata=metadata,
            chunk=chunk_label,
        )


def critique_response(
        model: LLM,
        tokenizer: AutoTokenizer,
        user_prompt: str,
        responses: list[str],
        system: str,
        sampling_params: SamplingParams,
        enable_thinking: bool = False,
    ) -> list[str]:
    """Critique each of the n responses in a single batched call. Returns one critique per response."""
    raw_prompts = []
    for response in responses:
        critique_prompt = f"""Review the following response and identify any ways it could be improved to better align with the constitutional principles.

Original user request: {user_prompt}

Response to critique:
{response}

Provide specific suggestions for improvement based on the constitution's guidelines for clarity, helpfulness, safety, and honesty. Be concise."""
        raw_prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": critique_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ))
    outputs = model.generate(raw_prompts, sampling_params, use_tqdm=False)
    return [o.outputs[0].text for o in outputs]


def revise_response(
        model: LLM,
        tokenizer: AutoTokenizer,
        user_prompt: str,
        original_responses: list[str],
        critiques: list[str],
        system: str,
        sampling_params: SamplingParams,
        enable_thinking: bool = False,
    ) -> list[str]:
    """Revise each of the n responses based on its critique in a single batched call."""
    raw_prompts = []
    for original_response, critique in zip(original_responses, critiques):
        revision_prompt = f"""Based on the following critique, provide an improved response to the original user request.

Original user request: {user_prompt}

Original response:
{original_response}

Critique:
{critique}

Provide the revised response that addresses the critique while following all constitutional principles:"""
        raw_prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": revision_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ))
    outputs = model.generate(raw_prompts, sampling_params, use_tqdm=False)
    return [o.outputs[0].text for o in outputs]


def constitutional_generation(
        model: LLM,
        tokenizer: AutoTokenizer,
        user_prompt: str,
        system: str,
        sampling_params: SamplingParams,
        enable_thinking: bool = False,
        enable_critique: bool = True,
        max_revisions: int = 1,
        track_vram: bool = False,
    ) -> dict:
    """Perform Constitutional AI generation with optional critique and revision.

    Each step generates num_return_sequences responses, running n parallel trajectories.
    critiques and revisions are lists of lists: shape [max_revisions][num_return_sequences].
    """
    # Step 1: Generate n initial responses
    logger.info("Generating initial responses...")
    initial_responses = generate_rollouts(
        model=model,
        tokenizer=tokenizer,
        input_string=user_prompt,
        system=system,
        sampling_params=sampling_params,
        enable_thinking=enable_thinking,
        track_vram=track_vram,
    )

    if not enable_critique:
        return {
            "initial_responses": initial_responses,
            "final_responses": initial_responses,
            "critiques": [],
            "revisions": [],
        }

    # Step 2: Critique and revise loop (n trajectories in parallel)
    current_responses = initial_responses
    critiques = []
    revisions = []

    for i in range(max_revisions):
        logger.info(f"Critique iteration {i + 1}/{max_revisions}...")
        iteration_critiques = critique_response(
            model=model,
            tokenizer=tokenizer,
            user_prompt=user_prompt,
            responses=current_responses,
            system=system,
            sampling_params=sampling_params,
            enable_thinking=enable_thinking,
        )
        critiques.append(iteration_critiques)

        logger.info(f"Revision iteration {i + 1}/{max_revisions}...")
        iteration_revisions = revise_response(
            model=model,
            tokenizer=tokenizer,
            user_prompt=user_prompt,
            original_responses=current_responses,
            critiques=iteration_critiques,
            system=system,
            sampling_params=sampling_params,
            enable_thinking=enable_thinking,
        )
        revisions.append(iteration_revisions)
        current_responses = iteration_revisions

    return {
        "initial_responses": initial_responses,
        "final_responses": current_responses,
        "critiques": critiques,
        "revisions": revisions,
    }


def save_cai_sample(
        output_file: str,
        row: int,
        instruction: str,
        cai_result: dict,
        metadata: dict,
    ) -> None:
    """Save Constitutional AI generation results to the output file."""
    record = {
        "row": row,
        "instruction": instruction,
        "initial_responses": cai_result["initial_responses"],
        "final_responses": cai_result["final_responses"],
        "critiques": cai_result["critiques"],
        "revisions": cai_result["revisions"],
        "metadata": metadata,
    }
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_cai_rollouts(
        sample: dict,
        counter: int,
        prompt_column: str,
        metadata_columns: list[str],
        model: LLM,
        tokenizer: AutoTokenizer,
        sampling_params: SamplingParams,
        file_path: str,
        system: str,
        prompt_prefix: str,
        prompt_suffix: str,
        max_chunk_size: int,
        enable_thinking: bool = False,
        enable_critique: bool = True,
        max_revisions: int = 1,
        track_vram: bool = False,
    ) -> None:
    """Process a single CAI dataset sample: run constitutional generation and save."""
    text_content = sample[prompt_column]
    metadata = {col: sample[col] for col in metadata_columns if col in sample}
    token_count = len(tokenizer(text_content).input_ids)

    if token_count > max_chunk_size:
        logger.info(f"Skipping row {counter} with {token_count} tokens (exceeds max chunk size of {max_chunk_size} tokens).")
        return

    full_prompt = f"{prompt_prefix}{text_content}{prompt_suffix}"
    logger.info(f"Generating CAI samples for row {counter}...")

    cai_result = constitutional_generation(
        model=model,
        tokenizer=tokenizer,
        user_prompt=full_prompt,
        system=system,
        sampling_params=sampling_params,
        enable_thinking=enable_thinking,
        enable_critique=enable_critique,
        max_revisions=max_revisions,
        track_vram=track_vram,
    )

    save_cai_sample(
        output_file=file_path,
        row=counter,
        instruction=text_content,
        cai_result=cai_result,
        metadata=metadata,
    )


def detect_failure_reason(log_path: Path | None, max_bytes: int = 100_000) -> str | None:
    """Detect the failure reason from a log file by reading head and tail."""
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
    """
    Validate configuration parameters for single-node inference.

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
