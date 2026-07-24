"""
Generate agentic traces using smolagents' CodeAgent. This script allows you to
run a CodeAgent over a dataset of prompts and record the agent's reasoning, tool calls,
and final answers as structured execution traces.

Usage examples:
    # Remote model (LiteLLM — OpenAI-compatible endpoint)
    python generate_agent_traces.py \\
        --model-type litellm \\
        --model-id deepseek/deepseek-v4-flash \\
        --system-prompt-file SYSTEM.yaml \\
        --dataset data.jsonl \\
        --prompt-column prompt \\
        --ground-truth-column ground_truth \\
        --max-steps 20 \\
        --output-dir traces/

    # Local model (Transformers) with thinking mode
    python generate_agent_traces.py \\
        --model-type transformers \\
        --model-id Qwen/Qwen3-8B \\
        --enable-thinking \\
        --dataset data.jsonl \\
        --max-steps 15 \\
        --output-dir traces/

    # Local model (vLLM)
    python generate_agent_traces.py \\
        --model-type vllm \\
        --model-id Qwen/Qwen3-8B \\
        --dataset data.jsonl \\
        --max-steps 15 \\
        --output-dir traces/

    # Resume a previous run (skip already-processed traces)
    python generate_agent_traces.py \\
        --model-type litellm \\
        --model-id deepseek/deepseek-v4-flash \\
        --dataset data.jsonl \\
        --output-dir traces/  # reads traces/metadata.jsonl automatically
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from utils import (
    TRACE_STATUS_FAIL,
    TRACE_STATUS_SUCCESS,
    OutputManager,
    TraceRecord,
    append_metadata_entry,
    build_model,
    execute_single_trace,
    load_dataset,
    load_processed_ids,
    load_system_prompt,
    logger,
)

# Marker strings identifying an unrecoverable inference-engine crash (e.g. a
# dead vLLM EngineCore). Once this happens every subsequent trace will fail
# instantly too, so we abort the whole run instead of burning through the
# remaining rows.
FATAL_ERROR_MARKERS = (
    "EngineDeadError",
    "EngineCore encountered an issue",
)

# Secondary, error-message-agnostic guardrail: a dead/unresponsive engine
# typically makes every subsequent trace fail near-instantly (no real
# generation happens), e.g. duration_seconds ~ 0.01s, regardless of the exact
# error text (which can change between vLLM/smolagents versions). If we see
# this many consecutive failures that each complete faster than the
# threshold below, treat it as a systemic failure and abort.
FAST_FAIL_DURATION_THRESHOLD_SECONDS = 2.0
MAX_CONSECUTIVE_FAST_FAILURES = 5


def main(args) -> None:
    # Ensure only our own logger.info() messages are visible on stdout.
    # Silence noisy third-party loggers.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    # Robust catch-all: third-party libraries (e.g. the search/Wikipedia
    # HTTP clients that emit "response: <url> <status>") propagate their
    # records up to the root handler.  Rather than chase every individual
    # logger name, drop anything below WARNING that does not originate from
    # our own logger so that chatter never reaches the console.
    class _OwnLoggerOrWarning(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.name.startswith(logger.name) or record.levelno >= logging.WARNING

    for _handler in logging.getLogger().handlers:
        _handler.addFilter(_OwnLoggerOrWarning())

    for _name in (
        "LiteLLM",
        "httpx",
        "urllib3",
        "wikipedia",
        "wikipediaapi",
        "datasets",
        "huggingface_hub",
        "requests",
        "vllm",
        # Torch inductor emits per-rank Triton bundler WARNINGs
        # ("Directory ... is not empty - skipping!") that flood the logs.
        "torch",
        "torch._inductor",
        "torch._inductor.triton_bundler",
    ):
        logging.getLogger(_name).setLevel(logging.ERROR)
    # smolagents internals log per-step code-execution errors and model-
    # generation failures at ERROR level; these are already captured in
    # the trace record, so silence them entirely.
    for _name in (
        "smolagents",
        "smolagents.local_python_executor",
        "smolagents.models",
        "smolagents.agents",
        "smolagents.memory",
        "smolagents.tools",
        "smolagents.monitoring",
    ):
        logging.getLogger(_name).setLevel(logging.CRITICAL)

    # Load the environment variables from .env file (if present)
    load_dotenv()

    # Basic validation
    id_column = args.id_column if args.id_column else None
    executor_timeout = args.executor_timeout if args.executor_timeout > 0 else None

    # Load dataset
    logger.info("📂 Loading dataset: %s", args.dataset)
    rows = load_dataset(
        path=args.dataset,
        prompt_column=args.prompt_column,
        id_column=id_column,
        ground_truth_column=args.ground_truth_column,
    )
    logger.info("   %d example(s) loaded.", len(rows))

    # Build model
    logger.info(
        "🤖 Building model: %s / %s",
        args.model_type,
        os.getenv("MODEL_ID") if not args.model_id else args.model_id,
    )
    model = build_model(
        model_type=args.model_type,
        model_id=args.model_id,
        api_key=args.api_key,
        api_base=args.api_base,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        apply_chat_template_kwargs={"enable_thinking": args.enable_thinking},
        model_kwargs={"max_model_len": args.max_new_tokens} if args.model_type == "vllm" else None,
    )

    # Load system prompt
    prompt_templates = load_system_prompt(args.system_prompt_file)

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip traces already present in metadata.jsonl
    if not args.no_resume:
        processed_ids = load_processed_ids(output_dir)
        if processed_ids:
            before = len(rows)
            rows = [r for r in rows if r["_trace_id"] not in processed_ids]
            skipped = before - len(rows)
            if skipped:
                logger.info(
                    "📋 Resume: skipping %d already-processed trace(s).  %d remaining.",
                    skipped,
                    len(rows),
                )
        else:
            logger.info("📋 Resume: no previous traces found — starting fresh.")
    else:
        logger.info("📋 Resume disabled (--no-resume).")

    if not rows:
        logger.info("🏁 All traces already processed. Nothing to do.")
        return

    # Create the consolidated output manager
    output_mgr = OutputManager(
        output_dir,
        max_entries_per_file=args.max_entries_per_file,
    )

    # Execute traces
    success_count = 0
    fail_count = 0
    consecutive_fast_failures = 0
    total = len(rows)

    logger.info("─" * 72)

    for i, row in enumerate(rows):
        idx = i + 1
        trace_id = row.get("_trace_id", f"row_{i}")
        prompt_snippet = str(row[args.prompt_column])[:100].replace("\n", " ")

        try:
            trace = execute_single_trace(
                row=row,
                model=model,
                prompt_templates=prompt_templates,
                max_steps=args.max_steps,
                executor_timeout=executor_timeout,
                prompt_column=args.prompt_column,
                language=args.language,
                enable_planning=args.enable_planning,
                code_block_opening_tag=args.code_block_opening_tag,
                code_block_closing_tag=args.code_block_closing_tag,
            )
        except Exception as e:
            # Create a minimal failure trace
            prompt = str(row[args.prompt_column])
            trace = TraceRecord(
                trace_id=trace_id,
                prompt=prompt,
                ground_truth=str(row.get("_ground_truth"))
                if row.get("_ground_truth") is not None
                else None,
                status=TRACE_STATUS_FAIL,
                error=f"{type(e).__name__}: {e}",
            )

        # Save formatted trace (only if successful and formatting not disabled).
        # NOTE: save_formatted_trace may downgrade trace.status to
        # TRACE_STATUS_FAIL if unclosed <think> tags are detected in the
        # formatted conversation — a guard rail against malformed traces.
        if trace.status == TRACE_STATUS_SUCCESS and not args.disable_formatting:
            output_mgr.save_formatted_trace(
                trace,
                code_block_opening_tag=args.code_block_opening_tag,
                code_block_closing_tag=args.code_block_closing_tag,
                language=args.language,
            )

        # Save raw trace (always, with final status — may have been
        # downgraded by save_formatted_trace above).
        output_mgr.save_raw_trace(trace)

        # Append to metadata
        append_metadata_entry(trace, output_dir)

        # Single-line status
        if trace.status == TRACE_STATUS_SUCCESS:
            success_count += 1
            consecutive_fast_failures = 0
            logger.info(
                "[%d/%d] %s | ✅ %d steps %.1fs | %s",
                idx,
                total,
                trace_id[:12],
                trace.num_steps,
                trace.duration_seconds,
                prompt_snippet,
            )
        else:
            fail_count += 1
            # Collapse multi-line errors into a single line for clean display.
            raw_err = trace.error or ""
            # Take the first meaningful line; collapse whitespace; truncate.
            err_first_line = raw_err.split("\n")[0].strip()
            # Collapse runs of whitespace and limit length
            err_clean = " ".join(err_first_line.split())[:120]
            logger.warning(
                "[%d/%d] %s | ❌ %d steps %.1fs | %s  -> %s",
                idx,
                total,
                trace_id[:12],
                trace.num_steps,
                trace.duration_seconds,
                prompt_snippet,
                err_clean,
            )

            fatal_marker = (
                next((m for m in FATAL_ERROR_MARKERS if m in trace.error), None)
                if trace.error
                else None
            )
            if fatal_marker:
                logger.error(
                    "💀 Fatal inference-engine error detected (%s) — aborting run at "
                    "[%d/%d] %s. The engine is dead and every remaining trace would "
                    "fail too; fix/restart the engine and resume this run later.",
                    fatal_marker,
                    idx,
                    total,
                    trace_id[:12],
                )
                sys.exit(1)

            # Duration-agnostic fallback: catches the same systemic breakdown
            # even when the error text doesn't match any known marker (e.g. a
            # new/unseen exception class wrapping the engine crash).
            if trace.duration_seconds < FAST_FAIL_DURATION_THRESHOLD_SECONDS:
                consecutive_fast_failures += 1
            else:
                consecutive_fast_failures = 0

            if consecutive_fast_failures >= MAX_CONSECUTIVE_FAST_FAILURES:
                logger.error(
                    "💀 %d consecutive near-instant failures (<%.1fs each) detected "
                    "ending at [%d/%d] %s — this looks like a dead/unresponsive "
                    "inference engine rather than genuine task failures. Aborting "
                    "run; fix/restart the engine and resume this run later.",
                    consecutive_fast_failures,
                    FAST_FAIL_DURATION_THRESHOLD_SECONDS,
                    idx,
                    total,
                    trace_id[:12],
                )
                sys.exit(1)

    # Summary
    logger.info("\n%s", "=" * 60)
    logger.info("🏁 Done. %d trace(s) processed.", len(rows))
    logger.info("   ✅ Success: %d", success_count)
    logger.info("   ❌ Failed:  %d", fail_count)
    logger.info("   📂 Output:  %s", output_dir.resolve())
    logger.info("      - raw_traces/       : full trace JSON arrays")
    if not args.disable_formatting:
        logger.info("      - formatted_traces/ : conversation-format JSON arrays")
    logger.info("      - metadata.jsonl    : summary of all traces")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Model arguments
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["litellm", "transformers", "vllm"],
        help="Type of model to use.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Model identifier (e.g. 'deepseek/deepseek-v4-flash', 'Qwen/Qwen3-32B'). "
        "Falls back to the MODEL_ID environment variable if not set.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for LiteLLM models (overrides API_KEY env var).",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="API base URL for LiteLLM models (overrides API_BASE env var).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=10000,
        help="Max new tokens to generate for local models (default: 10000).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for local models (transformers/vllm). "
        "If unset, the model's own generation defaults are used.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus-sampling (top-p) value for local models "
        "(transformers/vllm). If unset, the model's own generation "
        "defaults are used.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling cutoff for local models (transformers/vllm). "
        "If unset, the model's own generation defaults are used.",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the dataset file (.json, .jsonl, .parquet).",
    )
    parser.add_argument(
        "--prompt-column",
        default="prompt",
        help="Name of the column containing the prompt (default: 'prompt').",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="Name of the column containing trace IDs (default: 'id'). "
        "If the column exists in the dataset, its values are used; "
        "otherwise prompt hashes are generated. "
        "Set to empty string to always use prompt hashes.",
    )
    parser.add_argument(
        "--ground-truth-column",
        default=None,
        help="Name of the column containing ground-truth answers (optional).",
    )

    # Agent arguments
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum number of agent steps per example (default: 20).",
    )
    parser.add_argument(
        "--executor-timeout",
        type=int,
        default=120,
        help="Max seconds per tool execution step (default: 120). Use 0 or negative to disable.",
    )

    # System prompt arguments
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Path to a YAML file with custom prompt templates "
        "(same structure as smolagents' code_agent.yaml).",
    )
    parser.add_argument(
        "--code-block-opening-tag",
        default="<code>",
        help="Opening tag for code blocks (default: '<code>').",
    )
    parser.add_argument(
        "--code-block-closing-tag",
        default="</code>",
        help="Closing tag for code blocks (default: '</code>').",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        default="./traces/",
        help="Base directory for saving traces (default: './traces/').",
    )

    # Other arguments
    parser.add_argument(
        "--language",
        default="en",
        choices=["en", "pt"],
        help="Language for tools: 'en' (default) or 'pt' "
        "(Portuguese Wikipedia + DDG region pt-br).",
    )
    parser.add_argument(
        "--disable-formatting",
        action="store_true",
        help="Skip saving formatted conversation traces (only save raw traces and metadata).",
    )

    # Local-model arguments
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable thinking/reasoning mode for local models "
        "(Transformers / vLLM).  Passes "
        "apply_chat_template_kwargs={'enable_thinking': True} to "
        "the model.  Ignored for LiteLLM.",
    )

    # Planning argument
    parser.add_argument(
        "--enable-planning",
        action="store_true",
        default=False,
        help="Run a single planning step at the beginning of each "
        "trace before the agent starts executing.",
    )

    # Resume / output arguments
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Disable automatic resume.  When set, previously processed "
        "traces (listed in metadata.jsonl) are NOT skipped.",
    )
    parser.add_argument(
        "--max-entries-per-file",
        type=int,
        default=50_000,
        help="Maximum number of JSON objects per consolidated output "
        "file before rotating to a new file (default: 50000).",
    )

    args = parser.parse_args()
    main(args)
