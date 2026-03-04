"""
Single-node script for generating synthetic data using vLLM inference and datatrove pipelines.

Input:  local JSONL or Parquet files.
Output: local JSONL files.

Supports resuming interrupted jobs via checkpoints — just re-run the same
command and already-completed chunks will be skipped.

Usage:

    # View all options
    python generate_datatrove.py --help

    # Basic: generate from local JSONL data
    python generate_datatrove.py \
        --input-path /data/prompts \
        --prompt-column text \
        --model-name-or-path Qwen/Qwen3-0.6B \
        --output-path /data/output

    # With a prompt template (must contain [[DOCUMENT]])
    python generate_datatrove.py \
        --input-path /data/documents \
        --prompt-column text \
        --prompt-template "Summarize the following document: [[DOCUMENT]]" \
        --model-name-or-path Qwen/Qwen3-0.6B \
        --output-path /data/summaries

    # Resume an interrupted job (re-run the exact same command)
    python generate_datatrove.py \
        --input-path /data/prompts \
        --prompt-column text \
        --model-name-or-path Qwen/Qwen3-0.6B \
        --output-path /data/output
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

from datatrove.data import Document
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceResult, InferenceRunner
from datatrove.pipeline.readers import JsonlReader, ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.utils.logging import logger

import torch
from transformers import AutoConfig, GenerationConfig

# Import normalization and validation utils (utils.py should be in the same directory as this script)
SCRIPT_DIR = str(Path(__file__).parent)
sys.path.insert(0, SCRIPT_DIR)
from utils import (
    normalize_kvc_dtype,
    normalize_quantization,
    normalize_speculative,
    validate_config,
)


def _detect_input_format(input_path: str) -> str:
    """Auto-detect input format by inspecting file extensions in the directory."""
    p = Path(input_path)
    if not p.is_dir():
        raise ValueError(f"Input path is not a directory: {input_path}")

    for f in p.rglob("*"):
        if not f.is_file():
            continue
        name = f.name.lower()
        if name.endswith(".parquet"):
            return "parquet"
        if name.endswith(".jsonl") or name.endswith(".jsonl.gz") or name.endswith(".jsonl.zst"):
            return "jsonl"

    raise ValueError(
        f"Could not detect input format in {input_path}. "
        "Expected .jsonl, .jsonl.gz, .jsonl.zst, or .parquet files."
    )


def _compute_reader_limit(max_examples: int, tasks: int) -> int:
    """Compute per-task reader limit so max_examples is respected globally.

    Each datatrove task applies ``limit`` independently. For multi-task runs
    this would multiply total output by the number of tasks, so we split the
    global budget evenly.
    """
    if max_examples <= 0:
        return max_examples
    if tasks < 1:
        raise ValueError("tasks must be >= 1 when max_examples is set.")
    reader_limit = (max_examples + tasks - 1) // tasks
    if tasks > 1:
        logger.info(
            f"Applying global max_examples={max_examples} across {tasks} tasks "
            f"({reader_limit} docs per task)"
        )
    return reader_limit


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic data using vLLM on a single node with local I/O.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input and prompt configuration
    parser.add_argument("--input-path", required=True, help="Directory containing JSONL or Parquet input files")
    parser.add_argument("--input-format", default="auto", help="Input format: 'jsonl', 'parquet', or 'auto'")
    parser.add_argument("--prompt-column", default="text", help="Column name containing the prompt text")
    parser.add_argument("--prompt-template", default=None, help="Template with [[DOCUMENT]] placeholder")
    parser.add_argument("--max-examples", type=int, default=-1, help="Max total examples to process (-1 = all)")

    # Output configuration
    parser.add_argument("--output-path", required=True, help="Local directory for output JSONL files")

    # Model and inference configuration
    parser.add_argument("--server-type", default="vllm", help="Inference server type")
    parser.add_argument("--model-name-or-path", required=True, help="Model name or local path")
    parser.add_argument("--model-revision", default="main", help="Model revision")
    parser.add_argument("--model-max-context", type=int, default=32768, help="Maximum context length")
    parser.add_argument("--system-prompt", default=None, help="Optional system prompt")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code in model repo")

    # Parallelism settings (adjust based on available GPUs and model size)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism")
    parser.add_argument("--dp", type=int, default=1, help="Data parallelism")

    # vLLM-specific optimizations and settings
    parser.add_argument("--max-concurrent-generations", type=int, default=500)
    parser.add_argument("--max-concurrent-documents", type=int, default=500)
    parser.add_argument("--max-num-seqs", type=int, default=256, help="Max sequences in batch (reduce if OOM)")
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192, help="Chunked-prefill batch size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="Fraction of GPU memory for KV cache")
    parser.add_argument("--block-size", type=int, default=16, help="KV cache block size (16 or 32)")
    parser.add_argument("--speculative-config", default=None, help="Speculative decoding config (JSON)")
    parser.add_argument("--quantization", default=None, help="Quantization method (e.g. bitsandbytes)")
    parser.add_argument("--kv-cache-dtype", default="auto", help="KV cache dtype: auto, fp8_e4m3, fp8_e5m2")
    parser.add_argument("--optimization-level", type=int, default=3, help="0 = fast startup, 3 = best throughput")
    parser.add_argument("--metric-interval", type=int, default=120, help="Metric reporting interval in seconds")

    # Generation settings (overrides model defaults if set)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max output tokens per generation")
    parser.add_argument("--rollouts-per-document", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible generation")

    # Processing settings
    parser.add_argument("--examples-per-chunk", type=int, default=500, help="Documents per checkpoint chunk")
    parser.add_argument("--tasks", type=int, default=1, help="Number of parallel tasks")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Generate synthetic data using vLLM on a single node with local I/O."""
    # Extract arguments as local variables
    input_path = args.input_path
    input_format = args.input_format
    prompt_column = args.prompt_column
    prompt_template = args.prompt_template
    max_examples = args.max_examples
    output_path = args.output_path
    server_type = args.server_type
    model_name_or_path = args.model_name_or_path
    model_revision = args.model_revision
    model_max_context = args.model_max_context
    system_prompt = args.system_prompt
    trust_remote_code = args.trust_remote_code
    tp = args.tp
    pp = args.pp
    dp = args.dp
    max_concurrent_generations = args.max_concurrent_generations
    max_concurrent_documents = args.max_concurrent_documents
    max_num_seqs = args.max_num_seqs
    max_num_batched_tokens = args.max_num_batched_tokens
    gpu_memory_utilization = args.gpu_memory_utilization
    block_size = args.block_size
    speculative_config = args.speculative_config
    quantization = args.quantization
    kv_cache_dtype = args.kv_cache_dtype
    optimization_level = args.optimization_level
    metric_interval = args.metric_interval
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    max_tokens = args.max_tokens
    rollouts_per_document = args.rollouts_per_document
    seed = args.seed
    examples_per_chunk = args.examples_per_chunk
    tasks = args.tasks
    workers = args.workers

    # Check for available GPUs and adjust DP accordingly (vLLM requires at least 1 GPU)
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise ValueError("At least one CUDA GPU is required.")
    tp = min(tp, available_gpus)
    logger.info(f"Running locally on {available_gpus} GPU(s)")

    # Validate model config
    model_config = AutoConfig.from_pretrained(
        model_name_or_path, revision=model_revision, trust_remote_code=trust_remote_code
    )
    validate_config(tp=tp, pp=pp, dp=dp, config=model_config, prompt_template=prompt_template)

    # Detect input format if set to auto, and initialize reader
    if input_format == "auto":
        input_format = _detect_input_format(input_path)
        logger.info(f"Auto-detected input format: {input_format}")

    reader_limit = _compute_reader_limit(max_examples=max_examples, tasks=tasks)

    if input_format == "parquet":
        reader = ParquetReader(data_folder=input_path, text_key=prompt_column, limit=reader_limit)
    elif input_format == "jsonl":
        reader = JsonlReader(data_folder=input_path, text_key=prompt_column, limit=reader_limit)
    else:
        raise ValueError(f"Unsupported input format: {input_format}. Use 'jsonl' or 'parquet'.")

    # Resolve output / checkpoint / log paths
    output_dir = Path(output_path)
    checkpoints_dir = str(output_dir / ".checkpoints")
    logs_dir = str(output_dir / ".logs")

    # Normalize optional configs
    spec_raw = speculative_config
    if isinstance(spec_raw, str) and spec_raw.strip().lower() in ("none", "null", ""):
        spec_raw = None
    normalized_spec = normalize_speculative(spec_raw)
    normalized_quant = normalize_quantization(quantization)
    normalized_kv_dtype = normalize_kvc_dtype(kv_cache_dtype)

    # Resolve generation settings, falling back to model defaults if not set
    generation_config = GenerationConfig.from_pretrained(
        model_name_or_path, revision=model_revision, trust_remote_code=trust_remote_code
    )
    temperature = temperature if temperature is not None else getattr(generation_config, "temperature", 1.0)
    top_p = top_p if top_p is not None else getattr(generation_config, "top_p", 1.0)
    top_k = top_k if top_k is not None else getattr(generation_config, "top_k", -1)

    # Rollout function for a single document
    async def simple_rollout(
        document: Document,
        generate: Callable[[dict[str, Any]], Awaitable[InferenceResult]],
    ) -> InferenceResult:
        """Send a single request per document and return the result."""
        messages = [] if system_prompt is None else [{"role": "system", "content": system_prompt}]

        if isinstance(document.text, list) and all(isinstance(msg, dict) for msg in document.text):
            if prompt_template:
                raise ValueError("Prompt template is not supported for message lists")
            messages.extend(document.text)
        else:
            content = (
                prompt_template.replace("[[DOCUMENT]]", document.text) if prompt_template else document.text
            )

            # Truncate if content exceeds the context budget (~3 chars/token)
            char_budget = (model_max_context - max_tokens) * 3
            if len(content) > char_budget:
                original_len = len(content)
                last_newline = content.rfind("\n", 0, char_budget)
                content = content[:last_newline] if last_newline != -1 else content[:char_budget]

                logger.info(
                    f"Truncated content from {original_len} to {len(content)} chars "
                    f"(budget: {char_budget} chars)"
                )

            messages.append({"role": "user", "content": content})

        return await generate(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                **({"seed": seed} if seed is not None else {}),
            }
        )

    # Build model kwargs with normalized configs
    quant_kwargs: dict[str, Any] = {}
    if normalized_quant == "bitsandbytes":
        quant_kwargs["quantization"] = "bitsandbytes"

    kv_cache_kwargs: dict[str, Any] = {}
    if normalized_kv_dtype != "auto":
        kv_cache_kwargs["kv_cache_dtype"] = normalized_kv_dtype
        kv_cache_kwargs["calculate_kv_scales"] = True

    model_kwargs = {
        "revision": model_revision,
        "dtype": "bfloat16",
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "block-size": block_size,
        "gpu-memory-utilization": gpu_memory_utilization,
        **({"speculative_config": normalized_spec} if normalized_spec else {}),
        **quant_kwargs,
        **kv_cache_kwargs,
        "optimization-level": optimization_level,
    }

    # Inference configuration for the runner
    inference_config = InferenceConfig(
        server_type=server_type,
        model_name_or_path=model_name_or_path,
        model_kwargs=model_kwargs,
        model_max_context=model_max_context,
        rollouts_per_document=rollouts_per_document,
        max_concurrent_generations=max_concurrent_generations,
        max_concurrent_documents=max_concurrent_documents,
        metric_interval=metric_interval,
        tp=tp,
        dp=dp,
        pp=pp,
        server_log_folder=str(Path(logs_dir) / "server_logs"),
    )

    # Pipeline: reader -> inference -> JSONL writer
    pipeline = [
        reader,
        InferenceRunner(
            rollout_fn=simple_rollout,
            config=inference_config,
            records_per_chunk=examples_per_chunk,
            checkpoints_local_dir=checkpoints_dir,
            output_writer=JsonlWriter(
                output_folder=str(output_dir),
                output_filename="${rank}_${chunk_index}.jsonl",
                compression=None,
                expand_metadata=True,
            ),
        ),
    ]

    # Execute the pipeline
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=logs_dir,
        tasks=tasks,
        workers=workers,
    )
    executor.run()
    logger.info(f"Done. Output written to {output_dir}")


if __name__ == "__main__":
    main(parse_args())
