"""
Utilities for generating and saving traces of CodeAgent executions.
"""

import contextlib
import glob
import hashlib
import importlib.resources
import io
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import datasets
import yaml
from patches import (
    _ensure_vllm_tokenizer_compat,
    _patch_smolagents_binop_guard,
    _patch_smolagents_execution_timeout,
    _PatchedVLLMModel,
)
from smolagents import CodeAgent
from smolagents.agents import EMPTY_PROMPT_TEMPLATES, PromptTemplates, populate_template
from smolagents.default_tools import (
    DuckDuckGoSearchTool,
    FinalAnswerTool,
    PythonInterpreterTool,
    VisitWebpageTool,
    WikipediaSearchTool,
)
from smolagents.local_python_executor import BASE_BUILTIN_MODULES
from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep, SystemPromptStep, TaskStep
from smolagents.models import (
    LiteLLMModel,
    TransformersModel,
    VLLMModel,
)
from smolagents.monitoring import LogLevel
from smolagents.tools import Tool
from tools import get_custom_tools

# Logger for generate_agent_traces.py and utils.py
logger = logging.getLogger("AgentTraceRecorder")

# Constants for trace completion status
TRACE_STATUS_SUCCESS = "success"
TRACE_STATUS_FAIL = "fail"

# Supported file formats for dataset loading
_FILE_FORMATS = {".jsonl": "json", ".json": "json", ".parquet": "parquet"}

# Time threshold for cleaning up old Triton cache files (in seconds)
TRITON_CACHE_CLEANUP_AGE = 3600

# Maximum number of JSON objects per consolidated output file before rotation
DEFAULT_MAX_ENTRIES_PER_FILE = 50_000

# Lines matching this pattern are progress-bar / framework noise written
# directly to stderr (tqdm bars, weight-loading shards, Triton bundler
# spam, ...).
_STDERR_NOISE_RE = re.compile(
    r"it/s\]|it/s,|s/it\]|\d+%\s*\|"  # tqdm progress bars
    r"|Processed prompts|Adding requests|Adding lora"  # vLLM generate bars
    r"|Loading .*shards|Loading safetensors|Capturing CUDA"  # weight loading
    r"|triton_bundler|is not empty - skipping"  # Triton bundler warnings
)


@dataclass
class TraceRecord:
    """All information recorded for a single dataset example."""

    trace_id: str
    prompt: str
    ground_truth: str | None = None
    status: str = TRACE_STATUS_FAIL
    final_answer: Any = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""
    started_at: str = ""
    ended_at: str = ""
    duration_seconds: float = 0.0
    num_steps: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class OutputManager:
    """Manages consolidated trace output files with automatic rotation.

    Traces are written as proper JSON arrays (`[{…}, {…}]`).

    When a file reaches *max_entries_per_file* entries a new file is
    started with an incremented index.

    File naming pattern:
        `raw_traces/raw_traces_00001.json`, `raw_traces/raw_traces_00002.json`, …
        `formatted_traces_00001.json`, `formatted_traces_00002.json`, …

    On initialization the manager scans existing files in *output_dir*
    so that subsequent runs can resume appending to the correct file.

    Args:
        output_dir: Base directory for trace output.
        max_entries_per_file: Max JSON objects per consolidated file
            (default 50,000).
    """

    def __init__(
        self,
        output_dir: Path,
        max_entries_per_file: int = DEFAULT_MAX_ENTRIES_PER_FILE,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.max_entries = max_entries_per_file
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize counters by scanning existing files (for resume)
        self._raw_idx, self._raw_count = self._scan_existing("raw_traces")
        self._fmt_idx, self._fmt_count = self._scan_existing("formatted_traces")

    @staticmethod
    def _count_entries(filepath: Path) -> int:
        """Count top-level entries in a JSON array file.

        Parses the whole file — acceptable for a one-time startup cost.
        """
        try:
            with open(filepath, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return len(data)
            return 0
        except (FileNotFoundError, json.JSONDecodeError):
            return 0

    def _scan_existing(self, prefix: str) -> tuple[int, int]:
        """Scan existing consolidated files to determine resume position.

        Returns `(file_index, entry_count_in_last_file)`.
        """
        target_dir = self.output_dir / prefix
        if not target_dir.is_dir():
            return 1, 0  # Start fresh

        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.json$")
        max_idx = 0
        for p in target_dir.iterdir():
            m = pattern.match(p.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))

        if max_idx == 0:
            return 1, 0  # Start fresh

        last_file = target_dir / f"{prefix}_{max_idx:05d}.json"
        return max_idx, self._count_entries(last_file)

    @staticmethod
    def _rotate_if_needed(count: int, idx: int, limit: int) -> tuple[int, int]:
        """Return `(new_idx, new_count)`, rotating if *count* reached *limit*."""
        if count >= limit:
            return idx + 1, 0
        return idx, count

    def _raw_path(self) -> Path:
        return self.output_dir / "raw_traces" / f"raw_traces_{self._raw_idx:05d}.json"

    def _fmt_path(self) -> Path:
        return self.output_dir / "formatted_traces" / f"formatted_traces_{self._fmt_idx:05d}.json"

    @staticmethod
    def _append_to_json_array(filepath: Path, obj: dict[str, Any], is_first: bool) -> None:
        """Append a JSON object to a JSON-array file using file seeking.

        The file is always kept as a valid `[{…}, {…}]` array.  On the
        first write the file is created with `[\n{obj}\n]`.  Subsequent
        writes seek to just before the closing `\n]` and insert a comma
        plus the new object.

        Args:
            filepath:  Path to the JSON array file.
            obj:       Dictionary to serialise and append.
            is_first:  `True` if this is the first entry in a new file.
        """
        obj_json = json.dumps(obj, indent=2, ensure_ascii=False, default=str)

        # Ensure parent directory exists (e.g. raw_traces/ or formatted_traces/)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if is_first:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("[\n")
                f.write(obj_json)
                f.write("\n]\n")
        else:
            with open(filepath, "r+", encoding="utf-8") as f:
                f.seek(0, 2)  # end of file
                end_pos = f.tell()
                # File always ends with "\n]\n" — step back 3 bytes
                f.seek(max(0, end_pos - 3))
                f.write(",\n")
                f.write(obj_json)
                f.write("\n]\n")

    def save_raw_trace(self, trace: TraceRecord) -> Path:
        """Append a raw trace to the current consolidated JSON array file.

        Args:
            trace: The trace record to save.

        Returns:
            Path to the file that was written to.
        """
        self._raw_idx, self._raw_count = self._rotate_if_needed(
            self._raw_count, self._raw_idx, self.max_entries
        )
        is_first = self._raw_count == 0
        path = self._raw_path()

        data: dict[str, Any] = {
            "trace_id": trace.trace_id,
            "prompt": trace.prompt,
            "ground_truth": trace.ground_truth,
            "status": trace.status,
            "final_answer": trace.final_answer,
            "system_prompt": trace.system_prompt,
            "started_at": trace.started_at,
            "ended_at": trace.ended_at,
            "duration_seconds": trace.duration_seconds,
            "num_steps": trace.num_steps,
            "error": trace.error,
            "steps": trace.steps,
            "metadata": trace.metadata,
        }

        self._append_to_json_array(path, data, is_first)
        self._raw_count += 1
        return path

    def save_formatted_trace(
        self,
        trace: TraceRecord,
        code_block_opening_tag: str = "<code>",
        code_block_closing_tag: str = "</code>",
        language: str = "en",
    ) -> Path | None:
        """Append a formatted conversation trace to the current JSON array file.

        Only successful traces are saved.  Returns `None` if the trace
        status is not `TRACE_STATUS_SUCCESS`.

        Args:
            trace:                  The trace record.
            code_block_opening_tag: Opening tag for tool calls.
            code_block_closing_tag: Closing tag for tool calls.
            language:               Language code ("en" or "pt").

        Returns:
            Path to the file, or `None` if the trace was skipped.
        """
        if trace.status != TRACE_STATUS_SUCCESS:
            return None

        self._fmt_idx, self._fmt_count = self._rotate_if_needed(
            self._fmt_count, self._fmt_idx, self.max_entries
        )
        is_first = self._fmt_count == 0
        path = self._fmt_path()

        conversation = format_trace_as_conversation(
            trace,
            code_block_opening_tag=code_block_opening_tag,
            code_block_closing_tag=code_block_closing_tag,
            language=language,
        )

        # Guard rail: discard traces whose formatted conversation contains
        # unclosed <think> tags.  These are malformed and would break
        # downstream training / extraction pipelines.
        if _conversation_has_unclosed_think(conversation):
            trace.status = TRACE_STATUS_FAIL
            trace.error = "Unclosed <think> tags detected in formatted conversation"
            return None

        entry: dict[str, Any] = {
            "trace_id": trace.trace_id,
            "messages": conversation,  # we use the key "messages" to be compatible with the TRL expected chat format
        }

        self._append_to_json_array(path, entry, is_first)
        self._fmt_count += 1
        return path


def setup_triton_cache() -> None:
    """Setup a per-rank Triton cache directory with stale-file cleanup.

    Determines the local rank using (in order of preference):
    1. torch.distributed.get_rank() if distributed is initialized
    2. LOCAL_RANK environment variable
    3. CUDA_VISIBLE_DEVICES environment variable (comma-joined values
       replaced with hyphens)
    4. Falls back to "0"

    Creates a directory scoped to the SLURM job (if running under SLURM)
    and the determined rank, then cleans up cache files older than
    TRITON_CACHE_CLEANUP_AGE seconds.
    """
    cache_dir = os.environ.get("TRITON_CACHE_DIR", "./.cache/triton_cache")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")

    # Determine rank: prefer torch.distributed, then env vars, then default.
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

    # Clean up stale cache files
    try:
        current_time = time.time()
        for root, _, files in os.walk(rank_cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if os.path.getmtime(file_path) < current_time - TRITON_CACHE_CLEANUP_AGE:
                        os.remove(file_path)
                except OSError:
                    pass
    except Exception:
        pass


def load_dataset(
    path: str | Path,
    prompt_column: str = "prompt",
    id_column: str | None = "id",
    ground_truth_column: str | None = "ground_truth",
    cache_dir: str | None = "./.cache",
) -> list[dict[str, Any]]:
    """Load a dataset from a local file, directory, or HuggingFace Hub.

    Source type is detected automatically:
    - Directory  -> all .jsonl or .parquet files inside are loaded.
    - Local file -> .jsonl or .parquet are supported.
    - Anything else is treated as a HuggingFace Hub dataset identifier.

    Args:
        path:               Path to the dataset file/directory, or HF Hub ID.
        prompt_column:      Name of the column containing the prompt (default "prompt").
        id_column:          Name of the column containing the ID (default "id").
                            If None, IDs are generated from prompt hashes.
        ground_truth_column: Name of the column containing ground-truth answers.
        cache_dir:          Optional directory for caching dataset files.

    Returns:
        A list of dicts, each with at least {"_trace_id", "prompt"}.

    Raises:
        ValueError: If the required prompt column is missing.
        FileNotFoundError: If the file does not exist.
    """
    path_str = str(path)

    # Load the dataset
    if os.path.isdir(path_str):
        # Directory: pick first matching format
        for ext, fmt in _FILE_FORMATS.items():
            files = sorted(glob.glob(os.path.join(path_str, f"*{ext}")))
            if files:
                dataset = datasets.load_dataset(
                    fmt, data_files=files, split="train", num_proc=len(files), cache_dir=cache_dir
                )
                break
        else:
            raise ValueError(f"No .jsonl or .parquet files found in '{path_str}'.")

    elif os.path.isfile(path_str):
        ext = Path(path_str).suffix.lower()
        fmt = _FILE_FORMATS.get(ext)
        if fmt is None:
            raise ValueError(f"Unsupported file format '{ext}'. Expected .jsonl or .parquet.")
        dataset = datasets.load_dataset(
            fmt, data_files=path_str, split="train", cache_dir=cache_dir
        )
    else:
        # HuggingFace Hub
        dataset = datasets.load_dataset(path_str, split="train", cache_dir=cache_dir)

    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    # Validate prompt column
    if prompt_column not in dataset.column_names:
        raise ValueError(
            f"Column '{prompt_column}' not found in dataset. "
            f"Available columns: {dataset.column_names}"
        )

    # Convert to list of dicts
    rows = list(dataset)

    # Add ground-truth if configured
    if ground_truth_column:
        for r in rows:
            r["_ground_truth"] = r.get(ground_truth_column)

    # Add trace IDs
    for r in rows:
        if id_column and id_column in r:
            r["_trace_id"] = str(r[id_column])
        else:
            r["_trace_id"] = hashlib.sha256(str(r[prompt_column]).encode()).hexdigest()

    return rows


def _detect_device() -> str:
    """Return 'cuda' if GPU is available, otherwise 'cpu'."""
    try:
        import torch  # noqa: F401

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def build_model(
    model_type: str,
    model_id: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    max_new_tokens: int = 16384,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> LiteLLMModel | TransformersModel | VLLMModel:
    """Build a model instance based on the requested type.

    Args:
        model_type:  One of "litellm", "transformers", "vllm".
        model_id:    The model identifier.  Falls back to MODEL_ID env var
                     if not provided.
        api_key:     API key (for litellm).
        api_base:    API base URL (for litellm).
        max_new_tokens:  Max tokens to generate (for transformers/vllm).
        temperature: Sampling temperature (transformers/vllm).  When None the
                     model's own generation defaults are used.
        top_p:       Nucleus-sampling probability (transformers/vllm).  When
                     None the model's own generation defaults are used.
        top_k:       Top-k sampling cutoff (transformers/vllm).  When None the
                     model's own generation defaults are used.
        apply_chat_template_kwargs: Extra kwargs for tokenizer apply_chat_template.
        model_kwargs:       Extra kwargs passed to the model constructor.
                           For TransformersModel these are unpacked as **kwargs.
                           For VLLMModel these are merged into model_kwargs.

    Returns:
        A model instance.

    Raises:
        ValueError: If the model type is unknown.
    """
    model_type = model_type.lower()

    # Collect only the explicitly-requested sampling parameters so that, when
    # left unset, the model's own generation defaults remain in effect.
    sampling_kwargs: dict[str, Any] = {}
    if temperature is not None:
        sampling_kwargs["temperature"] = temperature
    if top_p is not None:
        sampling_kwargs["top_p"] = top_p
    if top_k is not None:
        sampling_kwargs["top_k"] = top_k

    resolved_model_id = model_id or os.getenv("MODEL_ID")
    if not resolved_model_id:
        raise RuntimeError(
            "No model ID provided. Set --model-id or the MODEL_ID environment variable."
        )

    if model_type == "litellm":
        if not api_key:
            api_key = os.getenv("API_KEY")
            if not api_key:
                raise RuntimeError(
                    "No API key provided. Set --api-key or the API_KEY environment variable."
                )
        return LiteLLMModel(
            model_id=resolved_model_id,
            api_key=api_key,
            api_base=api_base or os.getenv("API_BASE"),
        )

    elif model_type == "transformers":
        device = _detect_device()
        dtype = "float16" if "awq" in resolved_model_id.lower() else "bfloat16"
        # Forward sampling params into the constructor **kwargs; they become
        # `self.kwargs` and are passed through to HF `model.generate()`.
        transformers_kwargs: dict[str, Any] = dict(model_kwargs or {})
        transformers_kwargs.update(sampling_kwargs)
        # HF only applies temperature/top_p/top_k when sampling is enabled.
        if sampling_kwargs and (temperature is None or temperature > 0):
            transformers_kwargs.setdefault("do_sample", True)
        logger.info(
            "📦 TransformersModel("
            "model_id=%s, torch_dtype=%s, "
            "max_new_tokens=%s, apply_chat_template_kwargs=%s, "
            "device_map=%s, model_kwargs=%s)",
            resolved_model_id,
            dtype,
            max_new_tokens,
            apply_chat_template_kwargs or {},
            device,
            transformers_kwargs,
        )
        return TransformersModel(
            model_id=resolved_model_id,
            device_map=device,
            torch_dtype=dtype,
            max_new_tokens=max_new_tokens,
            apply_chat_template_kwargs=apply_chat_template_kwargs or {},
            **transformers_kwargs,
        )

    elif model_type == "vllm":
        setup_triton_cache()
        _ensure_vllm_tokenizer_compat()
        device = _detect_device()
        dtype = "float16" if "awq" in resolved_model_id.lower() else "bfloat16"
        vlm_kwargs: dict[str, Any] = {
            "tensor_parallel_size": 1,
            "dtype": dtype,
            # Set to False to disable the "Loading safetensors checkpoint shards" tqdm bar.
            "use_tqdm_on_load": True,
        }
        if device == "cuda":
            try:
                import torch  # noqa: F401

                gpu_count = torch.cuda.device_count()
                vlm_kwargs["tensor_parallel_size"] = gpu_count
            except Exception:
                pass
        if model_kwargs:
            vlm_kwargs.update(model_kwargs)
        logger.info(
            "📦 VLLMModel("
            "model_id=%s, torch_dtype=%s, "
            "max_new_tokens=%s, apply_chat_template_kwargs=%s, "
            "device_map=%s, model_kwargs=%s, sampling=%s)",
            resolved_model_id,
            dtype,
            max_new_tokens,
            apply_chat_template_kwargs or {},
            device,
            vlm_kwargs,
            sampling_kwargs,
        )
        return _PatchedVLLMModel(
            model_id=resolved_model_id,
            model_kwargs=vlm_kwargs,
            apply_chat_template_kwargs=apply_chat_template_kwargs or {},
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. Choose from: litellm, transformers, vllm"
        )


def _load_smolagents_default_prompt() -> dict[str, Any]:
    """Load the original code_agent.yaml prompt templates from the smolagents library.

    This is the same default that CodeAgent uses internally when no
    prompt_templates are provided.

    Returns:
        A PromptTemplates-compatible dict.
    """
    text = importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text()
    return yaml.safe_load(text)


def load_system_prompt(path: str | Path | None) -> dict[str, Any]:
    """Load prompt templates, with a two-tier fallback:

    1. **User-provided file** (`path` argument).
    2. **Library default** — smolagents' built-in `code_agent.yaml`,
       loaded via importlib.resources.

    Args:
        path: Path to a YAML file with prompt templates, or None to use
              the smolagents library default.

    Returns:
        A PromptTemplates-compatible dict.

    Raises:
        FileNotFoundError: If `path` is given but does not exist.
    """
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"System prompt file not found: {path}")
        with open(path, encoding="utf-8") as f:
            templates = yaml.safe_load(f)
        logger.info(f"📋 System prompt loaded from: {path}")
        return templates

    # Fall back to smolagents library default
    logger.info("📋 System prompt loaded from: smolagents.prompts/code_agent.yaml")
    return _load_smolagents_default_prompt()


def normalize_answer(text: str) -> str:
    """Normalize an answer string for comparison.

    Steps:
      1. Lowercase
      2. Collapse all whitespace to single spaces
      3. Strip leading/trailing whitespace
      4. Strip trailing punctuation common in answer formatting
      5. Try to normalize numeric values (42, 42.0, 42.00 -> 42)

    Args:
        text: Raw answer string.

    Returns:
        Normalized string.
    """
    if not isinstance(text, str):
        text = str(text)

    # Lowercase
    text = text.lower()

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    # Strip
    text = text.strip()

    # Remove common trailing punctuation
    text = re.sub(r"[.!?。！？]+$", "", text)

    # Normalize comma-formatted numbers before numeric matching.
    # Handles:
    #   - Thousands separators:  "40,000"    -> "40000"
    #   - Decimal comma:         "42,5"      -> "42.5"
    #   - Mixed (US):            "1,234.56"  -> "1234.56"
    #   - Mixed (EU):            "1.234,56"  -> "1234.56"
    if "," in text and re.search(r"\d", text):
        if "." in text:
            # Both separators present: whichever comes last is the decimal mark.
            if text.rfind(",") > text.rfind("."):
                # EU style — dot is thousands, comma is decimal
                text = text.replace(".", "").replace(",", ".")
            else:
                # US/UK style — comma is thousands, dot is decimal
                text = text.replace(",", "")
        else:
            # Only commas present
            if re.search(r",\d{1,2}$", text):
                # Trailing 1-2 digits after comma -> decimal comma
                text = text.replace(",", ".")
            else:
                # Otherwise treat as thousands separator
                text = text.replace(",", "")

    # Numeric normalization: if the entire string looks like a number,
    # convert to a canonical form.
    # Matches integers, decimals (42, +42, -42, 42.0, 42.00, .5, 5.)
    numeric_pattern = r"^[+-]?(?:\d+\.?\d*|\.\d+)$"
    if re.match(numeric_pattern, text):
        try:
            # If it's an integer-equivalent float, show as int
            val = float(text)
            # Round to avoid floating-point noise when not an integer-equivalent
            text = str(int(val)) if val == int(val) else f"{val:.10g}"
        except (ValueError, OverflowError):
            pass

    return text


def compare_answer(final_answer: Any, ground_truth: str) -> bool:
    """Compare the agent's final answer against ground truth.

    For the comparison, we normalize both sides and check whether
    the ground truth appears within the final answer (or vice versa).

    Args:
        final_answer: The output of final_answer().
        ground_truth: The expected answer string from the dataset.

    Returns:
        True if the answers match, False otherwise.
    """
    if final_answer is None:
        return False

    gt = normalize_answer(str(ground_truth))
    fa = normalize_answer(str(final_answer))

    if not gt or not fa:
        return False

    # Exact match after normalization
    if gt == fa:
        return True

    # Containment: ground truth is a substring of the final answer or vice versa
    return bool(gt in fa or fa in gt)


def _extract_model_output_text(step: dict[str, Any]) -> str:
    """Extract the text content from a model_output_message dict.

    Handles both the ChatMessage dict format and raw string format.
    """
    mom = step.get("model_output_message")
    if mom is None:
        return ""
    if isinstance(mom, str):
        return mom
    # ChatMessage dict format
    content = mom.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "\n".join(texts)
    return str(content)


def _extract_code_action(step: dict[str, Any]) -> str:
    """Extract the code action from a step dict."""
    return step.get("code_action", "") or ""


def _extract_observations(step: dict[str, Any]) -> str:
    """Extract observations from a step dict."""
    return step.get("observations", "") or ""


def _extract_error(step: dict[str, Any]) -> str:
    """Extract the error message from a step dict, if any.

    Returns an empty string when no error is present. Otherwise returns
    the error text (mirroring what smolagents itself appends to the
    conversation so the model can react to it).
    """
    err = step.get("error")
    if not err:
        return ""
    if isinstance(err, dict):
        return str(err.get("message", "")) or ""
    return str(err)


def _conversation_has_unclosed_think(conversation: list[dict[str, str]]) -> bool:
    """Return True if any message in the conversation has an unclosed <think> tag.

    An unclosed <think> tag means the number of `<think>` openings does not
    equal the number of `</think>` closings within that message's content.
    """
    for msg in conversation:
        content = msg.get("content", "")
        if not content:
            continue
        if content.count("<think>") != content.count("</think>"):
            return True
    return False


def _make_tool_call_xml(code: str, opening_tag: str, closing_tag: str) -> str:
    """Wrap code in the configured XML-like tool call tags."""
    return f"{opening_tag}\n{code}\n{closing_tag}"


def _make_tool_response_xml(observation: str) -> str:
    """Wrap observation in XML-like tool response tags."""
    return f"<tool_response>\n{observation}\n</tool_response>"


def format_trace_as_conversation(
    trace: TraceRecord,
    code_block_opening_tag: str = "<code>",
    code_block_closing_tag: str = "</code>",
    language: str = "en",
) -> list[dict[str, str]]:
    """Convert a raw trace into a structured conversation format.

    The conversation includes:
      1. System prompt
      2. Initial user prompt
      3. Assistant reasoning / text outside code blocks
      4. Tool invocations (wrapped in configured tags)
      5. Tool responses (wrapped in <tool_response> tags)
      6. Final answer call

    Args:
        trace:                    The raw trace record.
        code_block_opening_tag:   Opening tag for tool calls (default "<code>").
        code_block_closing_tag:   Closing tag for tool calls (default "</code>").

    Returns:
        A list of {"role": "...", "content": "..."} dicts.
    """
    conversation: list[dict[str, str]] = []

    # 1. System prompt
    if trace.system_prompt:
        conversation.append({"role": "system", "content": trace.system_prompt})

    # 2. Initial user prompt
    conversation.append({"role": "user", "content": trace.prompt})

    for step in trace.steps:
        step_type = step.get("_step_type", "")

        if step_type == "TaskStep":
            # Usually the task is the same as the prompt. Skip if it's a duplicate.
            task = step.get("task", "")
            if task and task.strip() != trace.prompt.strip():
                conversation.append({"role": "user", "content": task.strip()})

        elif step_type == "PlanningStep":
            # Planning steps: emit the plan as assistant text
            plan = step.get("plan", "") or ""
            if plan:
                conversation.append({"role": "assistant", "content": plan.strip()})

        elif step_type == "ActionStep":
            model_output = _extract_model_output_text(step)
            code_action = _extract_code_action(step)
            observations = _extract_observations(step)
            error_text = _extract_error(step)

            # Build combined assistant content: thought + code in a single message.
            assistant_parts = []
            if model_output:
                # Strip the code block from the model output to get just the "thought" part
                # The code block itself will be emitted together below.
                thought = model_output
                if code_action and code_action in thought:
                    thought = thought.replace(code_action, "").strip()

                # Also strip raw code block tags from the thought
                thought = re.sub(
                    re.escape(code_block_opening_tag) + r".*?" + re.escape(code_block_closing_tag),
                    "",
                    thought,
                    flags=re.DOTALL,
                ).strip()

                if thought:
                    assistant_parts.append(thought)

            # Tool call (the code block)
            if code_action:
                tool_call_content = _make_tool_call_xml(
                    code_action, code_block_opening_tag, code_block_closing_tag
                )
                assistant_parts.append(tool_call_content)

            if assistant_parts:
                conversation.append({"role": "assistant", "content": "\n".join(assistant_parts)})

            # Tool response: observation (success) and/or error (failure).
            # Combine both when present so the model sees exactly what it
            # received as feedback in the next turn.
            response_parts = []
            if observations:
                response_parts.append(observations)
            if error_text:
                response_parts.append(f"Error:\n{error_text}")
            if response_parts:
                conversation.append(
                    {
                        "role": "user",
                        "content": _make_tool_response_xml("\n".join(response_parts)),
                    }
                )

            # If this is the final answer action, note that we've already
            # captured it via the tool call mechanism above.
            # No extra final entry is needed,i.e., the final_answer() call
            # is already in the code action.

        elif step_type == "FinalAnswerStep":
            output = step.get("output")
            if output is not None:
                conversation.append(
                    {
                        "role": "assistant",
                        "content": f"The final answer is: {output}",
                    }
                )

    # Cleanup!
    # Strip the "Pensamento: " / "Thought: " prefix from all assistant
    # messages — this is a formatting artifact from the model's output.
    _THOUGHT_PREFIX_RE = re.compile(r"^(?:Pensamento|Thought):\s*")
    for msg in conversation:
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            msg["content"] = _THOUGHT_PREFIX_RE.sub("", msg["content"]).strip()

    # Translate hardcoded English strings when language != "en".
    # smolagents injects planning prefixes and tool-response labels in
    # English regardless of the user's language.
    if language != "en":
        _PLAN_TRANSLATIONS = {
            "pt": {
                # Planning prefixes (injected by smolagents._generate_planning_step)
                "Here are the facts I know and the plan of action that I will follow to solve the task:": "Aqui estão os fatos que conheço e o plano de ação que seguirei para resolver a tarefa:",
                "I still need to solve the task I was given:": "Ainda preciso resolver a tarefa que me foi dada:",
                "Here are the facts I know and my new/updated plan of action to solve the task:": "Aqui estão os fatos que conheço e o meu plano de ação novo/atualizado para resolver a tarefa:",
                # Tool-response labels (injected by smolagents.LocalPythonExecutor)
                "Execution logs:": "Registros de execução:",
                "Last output from code snippet:": "Última saída do trecho de código:",
                # Error label (emitted by this formatter for failed code steps)
                "Error:": "Erro:",
            },
        }
        translations = _PLAN_TRANSLATIONS.get(language, {})
        for msg in conversation:
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                for en_text, translated in translations.items():
                    if msg["content"].startswith(en_text):
                        msg["content"] = msg["content"].replace(en_text, translated, 1)
                        break
        # Also translate tool_response messages (role="user" containing
        # <tool_response>).
        for msg in conversation:
            if (
                msg.get("role") == "user"
                and isinstance(msg.get("content"), str)
                and "<tool_response>" in msg["content"]
            ):
                for en_text, translated in translations.items():
                    # Only apply to the labels, not the planning prefixes
                    # (those are already handled above for assistant messages).
                    if "\n" in en_text or en_text.endswith(":"):
                        msg["content"] = msg["content"].replace(en_text, translated)

    # Discard the last message if it is the tool_response that
    # follows the final_answer() code block.
    if len(conversation) >= 2:
        last = conversation[-1]
        second_last = conversation[-2]
        if (
            last.get("role") == "user"
            and isinstance(last.get("content"), str)
            and last["content"].startswith("<tool_response>")
            and second_last.get("role") == "assistant"
            and isinstance(second_last.get("content"), str)
            and "final_answer(" in second_last["content"]
        ):
            conversation.pop()

    # Discard empty tool_response messages. Those where no
    # actual output appeared between "Execution logs:" (or
    # translated equivalent) and "Last output from code snippet:".
    _EMPTY_TR = re.compile(
        r"^<tool_response>\n(?:Execution logs|Registros de execução):\n"
        r"(?:Last output from code snippet|Última saída do trecho de código):\nNone\n"
        r"</tool_response>$"
    )
    conversation = [
        msg
        for msg in conversation
        if not (
            msg.get("role") == "user"
            and isinstance(msg.get("content"), str)
            and _EMPTY_TR.match(msg["content"])
        )
    ]

    return conversation


def save_raw_trace(trace: TraceRecord, output_dir: Path) -> Path:
    """Save a raw trace as a JSON file.

    Args:
        trace:      The trace record to save.
        output_dir: Base output directory.

    Returns:
        Path to the saved file.
    """
    raw_dir = output_dir / "raw_traces"
    raw_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "trace_id": trace.trace_id,
        "prompt": trace.prompt,
        "ground_truth": trace.ground_truth,
        "status": trace.status,
        "final_answer": trace.final_answer,
        "system_prompt": trace.system_prompt,
        "started_at": trace.started_at,
        "ended_at": trace.ended_at,
        "duration_seconds": trace.duration_seconds,
        "num_steps": trace.num_steps,
        "error": trace.error,
        "steps": trace.steps,
        "metadata": trace.metadata,
    }

    filepath = raw_dir / f"trace_{trace.trace_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    return filepath


def save_formatted_trace(
    trace: TraceRecord,
    output_dir: Path,
    code_block_opening_tag: str = "<code>",
    code_block_closing_tag: str = "</code>",
    language: str = "en",
) -> Path:
    """Format a successful trace and save it as a conversation JSON file.

    Args:
        trace:                  The trace record (must have status "success").
        output_dir:             Base output directory.
        code_block_opening_tag: Opening tag for tool calls.
        code_block_closing_tag: Closing tag for tool calls.

    Returns:
        Path to the saved file.
    """
    formatted_dir = output_dir / "formatted_traces"
    formatted_dir.mkdir(parents=True, exist_ok=True)

    conversation = format_trace_as_conversation(
        trace,
        code_block_opening_tag=code_block_opening_tag,
        code_block_closing_tag=code_block_closing_tag,
        language=language,
    )

    filepath = formatted_dir / f"trace_{trace.trace_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(conversation, f, indent=2, ensure_ascii=False)

    return filepath


def append_metadata_entry(
    trace: TraceRecord,
    output_dir: Path,
) -> None:
    """Append a one-line JSON record to metadata.jsonl.

    Args:
        trace:      The trace record.
        output_dir: Base output directory.
    """
    metadata_path = output_dir / "metadata.jsonl"
    entry: dict[str, Any] = {
        "trace_id": trace.trace_id,
        "status": trace.status,
        "num_steps": trace.num_steps,
        "duration_seconds": trace.duration_seconds,
        "started_at": trace.started_at,
        "has_ground_truth": trace.ground_truth is not None,
        "error": trace.error,
    }
    with open(metadata_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_processed_ids(output_dir: Path) -> set[str]:
    """Load already-processed trace IDs from `metadata.jsonl`.

    This enables resume functionality: on a subsequent run, the script can
    skip traces whose IDs already appear in the metadata file.

    Args:
        output_dir: Base output directory containing `metadata.jsonl`.

    Returns:
        Set of trace IDs that have already been processed (may be empty).
    """
    metadata_path = output_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return set()

    ids: set[str] = set()
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                trace_id = entry.get("trace_id")
                if trace_id:
                    ids.add(trace_id)
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def _extract_step_type(step: Any) -> str:
    """Extract a string label for the type of memory step."""
    if isinstance(step, TaskStep):
        return "TaskStep"
    elif isinstance(step, PlanningStep):
        return "PlanningStep"
    elif isinstance(step, ActionStep):
        return "ActionStep"
    elif isinstance(step, FinalAnswerStep):
        return "FinalAnswerStep"
    elif isinstance(step, SystemPromptStep):
        return "SystemPromptStep"
    else:
        return type(step).__name__


def _build_default_tools(timeout_seconds: int | None = None, language: str = "en") -> list[Tool]:
    """Build the standard set of tools for a CodeAgent.

    Includes tools available in smolagents.default_tools:
      - PythonInterpreterTool  (sandboxed Python execution)
      - FinalAnswerTool        (return the final answer)
      - DuckDuckGoSearchTool   (web search via DuckDuckGo)
      - VisitWebpageTool       (fetch & convert webpage to Markdown)
      - WikipediaSearchTool    (search Wikipedia)

    Args:
        timeout_seconds: Max execution time per code snippet.
        language:        Language code for tool configuration.
                         `"en"` -> default English tools.
                         `"pt"` -> Portuguese Wikipedia + DDG region pt-br.

    Returns:
        List of Tool instances.
    """
    _patch_smolagents_execution_timeout()
    _patch_smolagents_binop_guard()

    if language == "pt":
        from tools import RegionDuckDuckGoSearchTool

        search_tool: Tool = RegionDuckDuckGoSearchTool(region="pt-br")
        wiki_tool = WikipediaSearchTool(language="pt")
    else:
        search_tool = DuckDuckGoSearchTool()
        wiki_tool = WikipediaSearchTool(language="en")

    tools: list[Tool] = [
        PythonInterpreterTool(timeout_seconds=timeout_seconds or 120),
        FinalAnswerTool(),
        search_tool,
        VisitWebpageTool(),
        wiki_tool,
    ]

    try:
        tools.extend(get_custom_tools())
    except Exception as exc:
        logger.warning("Could not load custom tools: %s", exc)

    return tools


def execute_single_trace(
    row: dict[str, Any],
    model: LiteLLMModel | TransformersModel | VLLMModel,
    prompt_templates: dict[str, Any],
    max_steps: int,
    executor_timeout: int | None,
    prompt_column: str = "prompt",
    language: str = "en",
    additional_authorized_imports: list[str] | None = None,
    extra_tools: list[Tool] | None = None,
    enable_planning: bool = False,
    code_block_opening_tag: str = "<code>",
    code_block_closing_tag: str = "</code>",
) -> TraceRecord:
    """Create a fresh CodeAgent, run one prompt, and return a TraceRecord.

    Args:
        row:                          Dataset row with at least 'prompt'.
        model:                        The model instance.
        prompt_templates:             Prompt template dict (system_prompt + optional
                                      planning/managed_agent/final_answer).
        max_steps:                    Maximum agent steps.
        executor_timeout:             Timeout per code execution (seconds).
        prompt_column:                Name of the prompt column.
        language:                     Language code for tools ("en" or "pt").
        additional_authorized_imports: Extra imports to allow.
        extra_tools:                  Additional tools beyond defaults.
        enable_planning:              If True, run a single planning step at the
                                      start before executing actions.
        code_block_opening_tag:       Opening tag for code blocks.
        code_block_closing_tag:       Closing tag for code blocks.

    Returns:
        A populated TraceRecord.
    """
    prompt = str(row[prompt_column])
    trace_id = str(row.get("_trace_id", hashlib.sha256(prompt.encode()).hexdigest()[:16]))
    ground_truth = row.get("_ground_truth")

    # Build tools
    tools = _build_default_tools(timeout_seconds=executor_timeout, language=language)
    if extra_tools:
        tools.extend(extra_tools)

    # Build authorized imports
    authorized_imports = list(BASE_BUILTIN_MODULES)
    # BASE_BUILTIN_MODULES is a set of standard library modules that are safe to import.
    # These are:
    # - "collections"
    # - "datetime"
    # - "itertools"
    # - "math"
    # - "queue"
    # - "random"
    # - "re"
    # - "stat"
    # - "statistics"
    # - "time"
    # - "unicodedata"
    if additional_authorized_imports:
        authorized_imports.extend(additional_authorized_imports)

    # Build full prompt templates
    # If the prompt_templates dict somehow lacks a system_prompt key,
    # fall back to the smolagents library default.
    _default_sp = prompt_templates.get(
        "system_prompt",
        _load_smolagents_default_prompt().get("system_prompt", ""),
    )
    full_templates: PromptTemplates = {
        **EMPTY_PROMPT_TEMPLATES,
        "system_prompt": _default_sp,
    }
    if "planning" in prompt_templates:
        full_templates["planning"] = prompt_templates["planning"]
    if "managed_agent" in prompt_templates:
        full_templates["managed_agent"] = prompt_templates["managed_agent"]
    if "final_answer" in prompt_templates:
        full_templates["final_answer"] = prompt_templates["final_answer"]

    # Build the agent
    agent = CodeAgent(
        tools=tools,
        model=model,
        prompt_templates=full_templates,
        max_steps=max_steps,
        additional_authorized_imports=authorized_imports,
        code_block_tags=(code_block_opening_tag, code_block_closing_tag),
        planning_interval=(max_steps + 1) if enable_planning else None,
        verbosity_level=LogLevel.OFF,  # Suppress Rich console output
    )

    # Redirect stderr to suppress smolagents/vLLM internal print()
    # noise which bypasses Python's logging framework entirely.
    started_at = datetime.now(UTC)
    error_msg: str | None = None
    final_answer: Any = None
    state: str = TRACE_STATUS_FAIL
    _stderr_capture = io.StringIO()

    try:
        sys.stderr = _stderr_capture
        result = agent.run(prompt, return_full_result=True)
        final_answer = result.output
        state = TRACE_STATUS_SUCCESS if result.state == "success" else TRACE_STATUS_FAIL
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        # Try to extract any partial result from memory
        try:
            for step in reversed(agent.memory.steps):
                if isinstance(step, FinalAnswerStep):
                    final_answer = step.output
                    break
                elif isinstance(step, ActionStep) and step.is_final_answer:
                    final_answer = step.action_output
                    break
        except Exception:
            pass
    finally:
        sys.stderr = sys.__stderr__
        # Merge any stderr noise into the error message (one-liner, truncated).
        # tqdm/progress bars use carriage returns and flood the buffer, so we
        # split on both \r and \n and drop framework-noise lines; otherwise a
        # failed trace would record the vLLM progress bar instead of the real
        # error.
        stderr_text = _stderr_capture.getvalue().strip()
        if stderr_text:
            stderr_lines = [
                ln.strip()
                for ln in re.split(r"[\r\n]+", stderr_text)
                if ln.strip() and not _STDERR_NOISE_RE.search(ln)
            ]
            if stderr_lines:
                stderr_summary = " | ".join(stderr_lines[-3:])  # last 3 real lines
                error_msg = f"{error_msg} [{stderr_summary}]" if error_msg else stderr_summary

    ended_at = datetime.now(UTC)
    duration = (ended_at - started_at).total_seconds()

    # Extract steps from agent memory
    steps_data: list[dict[str, Any]] = []
    try:
        raw_steps = agent.memory.get_full_steps()
        for i, step in enumerate(agent.memory.steps):
            step_dict = raw_steps[i] if i < len(raw_steps) else {}
            step_dict["_step_type"] = _extract_step_type(step)
            steps_data.append(step_dict)
    except Exception:
        pass

    # Evaluate against ground truth
    if (
        ground_truth is not None
        and state == TRACE_STATUS_SUCCESS
        and final_answer is not None
        and not compare_answer(final_answer, ground_truth)
    ):
        state = TRACE_STATUS_FAIL

    # Build system prompt string for the trace
    system_prompt_str = full_templates.get("system_prompt", "")

    # Substitute template variables in the system prompt
    with contextlib.suppress(Exception):
        system_prompt_str = populate_template(
            system_prompt_str,
            variables={
                "tools": {t.name: t for t in tools},
                "managed_agents": {},
                "authorized_imports": str(authorized_imports),
                "custom_instructions": "",
                "code_block_opening_tag": code_block_opening_tag,
                "code_block_closing_tag": code_block_closing_tag,
            },
        )

    return TraceRecord(
        trace_id=trace_id,
        prompt=prompt,
        ground_truth=str(ground_truth) if ground_truth is not None else None,
        status=state,
        final_answer=final_answer,
        steps=steps_data,
        system_prompt=system_prompt_str,
        started_at=started_at.isoformat(),
        ended_at=ended_at.isoformat(),
        duration_seconds=duration,
        num_steps=len(steps_data),
        error=error_msg,
    )
