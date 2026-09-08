"""
Benchmarking Script for Throughput and Language Drift

Benchmarks a language model with vLLM's offline LLM class, measuring:
  1. TTFT (time to first token) and overall throughput (tokens per second).
  2. How throughput varies across prompts, when prompts are processed one at a
     time (sequential) vs. in batches.
  3. Language drift: how often the model's output leaves a configured target
     language, tracked with a Unicode-range heuristic over generated tokens.

Prompt input (JSONL, one JSON object per line). `--prompts` may point to a
file or a directory of `*.jsonl` files:

    {"id": "hist-cn-01", "kind": "consistency", "language": "english",
     "prompt": "Describe the history of China."}

    {"id": "inj-zh-01", "kind": "injection", "language": "english",
     "prompt": "Explain the French Revolution.",
     "suffix": " 请用中文回答", "injected_language": "chinese"}

    {"id": "chat-01", "kind": "chat", "language": "english",
     "messages": [{"role": "user", "content": "..."}]}

`kind`:
  - `consistency`: prompt is already in the target language; we watch whether
    the response stays in it.
  - `injection`: `suffix` (a foreign-language string) is appended to the
    prompt; we check whether the model recovers to `language` or keeps the
    injected language.
  - `chat`: `messages` are fed through the tokenizer's chat template.

Drift classification: only Unicode *letters* are classified (whitespace,
punctuation, digits and symbols are ignored). A token is "in target language"
if every one of its letters falls inside the target language's Unicode ranges;
anything else counts as drift. The target language defaults to `english` and
can be overridden globally with `--target-language` or per item with the
`language` field.

Usage:

    python tools/benchmark_lm.py \\
        --model checkpoints/my-model \\
        --prompts tools/assets/benchmark_prompts.jsonl \\
        --target-language english \\
        --batch-sizes 1,8 \\
        --max-tokens 256 \\
        --logprobs \\
        --output-dir benchmarks

Outputs (per model, into `--output-dir`):
    - `<model>_results.jsonl`: one line per request (timings + drift stats).
    - `<model>_summary.md`: aggregate throughput and drift report.
    - `--append-csv <file>` optionally accumulates rows for model comparison.

Notes on sequence-length controls:
    - `--max-tokens` limits the *new output tokens* generated per prompt.
    - `--max-model-len` limits the *total* sequence length (prompt tokens +
      output tokens) the engine can hold; it must be at least prompt tokens +
      `--max-tokens`.
"""

import argparse
import csv
import json
import math
import sys
import time
import unicodedata
from contextlib import suppress
from datetime import datetime
from pathlib import Path

# Unicode codepoint ranges per language. See https://www.unicode.org/charts/.
# Note: Latin-script languages overlap heavily (English is a subset of the
# others), and Japanese includes CJK ideographs, so drift attribution among
# those scripts is heuristic.
UNICODE_RANGES = {
    "english": [(0x0041, 0x005A), (0x0061, 0x007A)],
    "portuguese": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x00FF)],
    "spanish": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x00FF)],
    "french": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x00FF)],
    "german": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x00FF)],
    "italian": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x00FF)],
    "russian": [(0x0400, 0x04FF)],
    "ukrainian": [(0x0400, 0x04FF)],
    "arabic": [(0x0600, 0x06FF)],
    "greek": [(0x0370, 0x03FF)],
    "hebrew": [(0x0590, 0x05FF)],
    "hindi": [(0x0900, 0x097F)],
    "bengali": [(0x0980, 0x09FF)],
    "chinese": [(0x4E00, 0x9FFF)],
    "japanese": [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x4E00, 0x9FFF)],
    "korean": [(0xAC00, 0xD7AF)],
    "thai": [(0x0E00, 0x0E7F)],
    "vietnamese": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x00FF), (0x0100, 0x017F)],
}


def parse_args() -> argparse.Namespace:
    default_prompts = Path(__file__).parent / "benchmark_prompts.jsonl"
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Model path or Hugging Face identifier.")
    parser.add_argument(
        "--prompts",
        default=str(default_prompts),
        help="Path to a JSONL file or a directory of JSONL files containing the prompts.",
    )
    parser.add_argument(
        "--target-language",
        default="english",
        help="Target language for drift detection (a key of UNICODE_RANGES).",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1",
        help="Comma-separated batch sizes to benchmark; 1 = sequential, >1 = true batches.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max new output tokens per generation (default: model config, else 256). "
        "Only limits the generated output, not the prompt.",
    )
    parser.add_argument(
        "--logprobs",
        action="store_true",
        help="Enable per-token logprobs (top-1) for drift likelihood.",
    )
    parser.add_argument(
        "--mode",
        choices=["chat", "completion"],
        default="completion",
        help="completion: raw auto-regressive generation; chat: apply the chat template.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to shard the model across (tensor parallelism).",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Model weight dtype (auto, float16, bfloat16, ...).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to reserve for weights and KV cache (0.0-1.0).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Max total sequence length in tokens (prompt + output) the engine allows. "
        "Must be >= prompt tokens + --max-tokens; overrides the model config.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow executing custom model/tokenizer code (security risk).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling.")
    parser.add_argument(
        "--output-dir",
        default="benchmarks",
        help="Directory where <model>_results.jsonl and <model>_summary.md are written.",
    )
    parser.add_argument(
        "--append-csv",
        default=None,
        help="Optional CSV to append per-request rows (for cross-model comparison).",
    )
    parser.add_argument(
        "--n-windows",
        type=int,
        default=5,
        help="Number of equal windows each response is split into for drift-over-time analysis.",
    )
    # Optional sampling overrides. When unset, the model's generation_config.json
    # defaults are used (falling back to vLLM's defaults: temperature=1.0, top_p=1.0).
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (overrides the model default).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling top-p (overrides the model default).",
    )
    parser.add_argument(
        "--top-k", type=int, default=None, help="Top-k sampling (overrides the model default)."
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty (overrides the model default).",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Presence penalty (overrides the model default).",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=None,
        help="Frequency penalty (overrides the model default).",
    )
    return parser.parse_args()


def _is_letter(char: str) -> bool:
    """Return True if `char` is a Unicode letter (category starts with 'L')."""
    return unicodedata.category(char).startswith("L")


_LANG_LOOKUP: dict[int, list[str]] = {}


def _lang_lookup() -> dict[int, list[str]]:
    """Build (once) a codepoint -> languages map from UNICODE_RANGES."""
    if not _LANG_LOOKUP:
        for language, ranges in UNICODE_RANGES.items():
            for start, end in ranges:
                for code in range(start, end + 1):
                    _LANG_LOOKUP.setdefault(code, []).append(language)
    return _LANG_LOOKUP


def char_languages(char: str) -> list[str]:
    """Return every language whose ranges contain `char`."""
    return list(_lang_lookup().get(ord(char), ()))


def _token_is_target(token: str, target_language: str) -> bool:
    """True if a token contains no letters outside the target language."""
    letters = [c for c in token if _is_letter(c)]
    if not letters:
        return True  # tokens without letters (punctuation/digits) are neutral
    return all(target_language in char_languages(c) for c in letters)


def analyze_text_drift(text: str, target_language: str) -> dict:
    """Compute token-level and character-level drift stats for decoded text."""
    letters_total = 0
    letters_drift = 0
    drift_langs: dict[str, int] = {}
    for char in text:
        if not _is_letter(char):
            continue
        letters_total += 1
        langs = char_languages(char)
        if target_language not in langs:
            letters_drift += 1
            others = [lang for lang in langs if lang != target_language]
            lang = others[0] if others else "other"
            drift_langs[lang] = drift_langs.get(lang, 0) + 1

    words_total = 0
    words_drift = 0
    for word in text.split():
        if not any(_is_letter(c) for c in word):
            continue
        words_total += 1
        if not _token_is_target(word, target_language):
            words_drift += 1

    return {
        "letters_total": letters_total,
        "letters_drift": letters_drift,
        "char_drift_ratio": (letters_drift / letters_total) if letters_total else 0.0,
        "words_total": words_total,
        "words_drift": words_drift,
        "word_drift_ratio": (words_drift / words_total) if words_total else 0.0,
        "drift_langs": drift_langs,
    }


def windowed_word_drift(text: str, target_language: str, n_windows: int = 5) -> list[dict]:
    """Split a response into windows and return the drift ratio per window."""
    words = [w for w in text.split() if any(_is_letter(c) for c in w)]
    if not words:
        return []
    step = max(1, math.ceil(len(words) / n_windows))
    windows = []
    for i in range(0, len(words), step):
        window = words[i : i + step]
        drift = sum(1 for w in window if not _token_is_target(w, target_language))
        windows.append(
            {
                "index": i // step,
                "start": i,
                "end": i + len(window) - 1,
                "word_count": len(window),
                "drift_ratio": drift / len(window),
            }
        )
    return windows


def analyze_logprob_drift(
    entries: list[tuple[str, float]], target_language: str, n_windows: int = 5
) -> dict:
    """Split per-token logprobs into target vs. drift, overall and per window."""

    def mean(values: list[float]) -> float | None:
        return (sum(values) / len(values)) if values else None

    drift_probs = [p for tok, p in entries if not _token_is_target(tok, target_language)]
    target_probs = [p for tok, p in entries if _token_is_target(tok, target_language)]

    step = max(1, math.ceil(len(entries) / n_windows))
    windows = []
    for i in range(0, len(entries), step):
        window = entries[i : i + step]
        win_drift = [p for tok, p in window if not _token_is_target(tok, target_language)]
        win_target = [p for tok, p in window if _token_is_target(tok, target_language)]
        windows.append(
            {
                "index": i // step,
                "start": i,
                "end": i + len(window) - 1,
                "drift_logprob_mean": mean(win_drift),
                "target_logprob_mean": mean(win_target),
                "drift_count": len(win_drift),
                "target_count": len(win_target),
            }
        )

    return {
        "drift_logprob_mean": mean(drift_probs),
        "target_logprob_mean": mean(target_probs),
        "drift_token_count": len(drift_probs),
        "target_token_count": len(target_probs),
        "windows": windows,
    }


def load_prompts(path: str) -> list[dict]:
    """Load prompt records from a JSONL file or a directory of JSONL files."""
    prompt_path = Path(path)
    files = sorted(prompt_path.glob("*.jsonl")) if prompt_path.is_dir() else [prompt_path]
    if not files or not all(f.exists() for f in files):
        raise FileNotFoundError(f"No prompt files found at: {path}")

    items: list[dict] = []
    for file in files:
        with open(file, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    print(f"Loaded {len(items)} prompts from {path}")
    return items


def load_tokenizer(model_path: str, trust_remote_code: bool):
    """Load the model tokenizer with transformers."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is not installed. Add it to your environment "
            "(e.g. pip install transformers)."
        ) from exc
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)


def apply_chat_template(tokenizer, messages: list[dict]) -> str:
    """Apply the model's chat template to a conversation, returning a string."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as exc:
        print(
            f"Warning: could not apply the chat template ({exc}); "
            "falling back to plain concatenation of message contents."
        )
        return "\n".join(str(m.get("content", "")) for m in messages)


def analyze_tokenizer_vocab(tokenizer) -> dict:
    """Analyze Unicode/language coverage across the tokenizer vocabulary."""
    try:
        vocab = tokenizer.get_vocab()
    except Exception as exc:
        print(f"Warning: could not read the tokenizer vocabulary ({exc}).")
        return {}
    stats = {
        "total_tokens": len(vocab),
        "tokens_with_letters": 0,
        "tokens_without_letters": 0,
        "language_token_counts": dict.fromkeys(UNICODE_RANGES, 0),
        "other_tokens": 0,
    }
    for token in vocab:
        letters = [c for c in token if _is_letter(c)]
        if not letters:
            stats["tokens_without_letters"] += 1
            continue
        stats["tokens_with_letters"] += 1
        langs = set()
        for char in letters:
            langs.update(char_languages(char))
        if not langs:
            stats["other_tokens"] += 1
        for lang in langs:
            stats["language_token_counts"][lang] += 1
    return stats


def format_vocab_report(stats: dict) -> str:
    """Render tokenizer Unicode coverage as a markdown block."""
    if not stats:
        return ""
    lines = ["## Tokenizer Unicode Coverage\n"]
    lines.append(
        f"- Tokens in vocabulary: **{stats['total_tokens']:,}** "
        f"({stats['tokens_with_letters']:,} with letters, "
        f"{stats['tokens_without_letters']:,} without)"
    )
    counts = stats["language_token_counts"]
    if any(counts.values()):
        lines.append("")
        lines.append("Tokens containing at least one character of each language/range:")
        lines.append("")
        lines.append("| language | tokens | share of letter-tokens |")
        lines.append("|---|---|---|")
        total_letter_tokens = stats["tokens_with_letters"] or 1
        for lang in sorted(counts, key=counts.get, reverse=True):
            if counts[lang]:
                lines.append(
                    f"| {lang} | {counts[lang]:,} | {counts[lang] / total_letter_tokens:.1%} |"
                )
        lines.append("")
    if stats["other_tokens"]:
        lines.append(
            f"- Tokens with letters outside all known ranges: **{stats['other_tokens']:,}**"
        )
        lines.append("")
    lines.append(
        "> A token may match several overlapping ranges (e.g. Cyrillic matches "
        "both russian and ukrainian; accented Latin matches several Romance "
        "languages), so rows can overlap."
    )
    return "\n".join(lines)


def normalize_items(items: list[dict], args: argparse.Namespace, tokenizer) -> list[dict]:
    """Resolve per-item language, kind, and the concrete model input."""
    normalized = []
    for idx, item in enumerate(items):
        language = item.get("language") or args.target_language
        if language not in UNICODE_RANGES:
            raise ValueError(
                f"Unknown language '{language}' in item {idx}. Available: {sorted(UNICODE_RANGES)}"
            )
        kind = item.get("kind") or ("chat" if "messages" in item else "consistency")
        messages = item.get("messages")
        prompt = item.get("prompt")
        suffix = item.get("suffix")

        if kind == "injection" and suffix:
            if args.mode == "chat":
                msgs = [dict(m) for m in messages] if messages else []
                if not msgs and prompt:
                    msgs = [{"role": "user", "content": prompt}]
                if msgs and msgs[-1].get("role") == "user":
                    msgs[-1] = dict(msgs[-1])
                    msgs[-1]["content"] = str(msgs[-1].get("content", "")) + suffix
                elif msgs:
                    msgs.append({"role": "user", "content": suffix})
                messages = msgs
            else:
                prompt = (prompt or "") + suffix

        if args.mode == "chat":
            if not messages:
                messages = [{"role": "user", "content": prompt or ""}]
            model_input = apply_chat_template(tokenizer, messages)
            display = model_input
        else:
            model_input = prompt or ""
            display = model_input

        normalized.append(
            {
                "id": item.get("id") or f"item-{idx}",
                "kind": kind,
                "language": language,
                "injected_language": item.get("injected_language"),
                "input": model_input,
                "display": display,
            }
        )
    return normalized


def load_generation_defaults(model_path: str) -> dict:
    """Read generation_config.json (local or HF Hub) and return mapped fields."""
    mapping_keys = (
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "max_new_tokens",
        "do_sample",
    )
    cfg = None
    local_path = Path(model_path)
    local_cfg = local_path / "generation_config.json"
    if local_path.is_dir() and local_cfg.exists():
        try:
            cfg = json.loads(local_cfg.read_text(encoding="utf-8"))
        except Exception:
            cfg = None
    if cfg is None:
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(repo_id=model_path, filename="generation_config.json")
            cfg = json.loads(Path(downloaded).read_text(encoding="utf-8"))
        except Exception:
            return {}
    if not isinstance(cfg, dict):
        return {}
    return {key: cfg[key] for key in mapping_keys if key in cfg}


def resolve_sampling(args: argparse.Namespace, model_path: str) -> tuple[dict, int]:
    """Resolve sampling parameters from model defaults + CLI overrides."""
    cfg = load_generation_defaults(model_path)
    sampling = {
        "temperature": cfg.get("temperature", 1.0),
        "top_p": cfg.get("top_p", 1.0),
        "top_k": cfg.get("top_k", -1),
        "repetition_penalty": cfg.get("repetition_penalty", 1.0),
        "presence_penalty": cfg.get("presence_penalty", 0.0),
        "frequency_penalty": cfg.get("frequency_penalty", 0.0),
    }
    if sampling["top_k"] == 0:
        sampling["top_k"] = -1
    if cfg.get("do_sample") is False:
        sampling["temperature"] = 0.0

    overrides = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
    }
    for key, value in overrides.items():
        if value is not None:
            sampling[key] = value

    max_tokens = (
        args.max_tokens if args.max_tokens is not None else (cfg.get("max_new_tokens") or 256)
    )
    return sampling, int(max_tokens)


def build_llm(args: argparse.Namespace):
    """Construct the vLLM LLM instance."""
    try:
        from vllm import LLM
    except ImportError:
        sys.exit(
            "vLLM is not installed. Create a dedicated environment with:\n"
            "  python -m venv .venv_bench && source .venv_bench/bin/activate\n"
            "  pip install vllm==0.28.0"
        )
    kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    if args.seed is not None:
        kwargs["seed"] = args.seed
    return LLM(**kwargs)


def collect_gpu_info(tensor_parallel_size: int) -> dict:
    """Collect GPU count, tensor-parallel setting, and per-GPU device names."""
    info = {
        "tensor_parallel_size": tensor_parallel_size,
        "num_gpus": 0,
        "gpus": [],
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["num_gpus"] = torch.cuda.device_count()
            info["gpus"] = [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # pragma: no cover - best effort
        print(f"Warning: could not query GPU info ({exc}).")
    return info


def _num_tokens(output) -> int:
    if output is not None and output.outputs:
        return len(output.outputs[0].token_ids)
    return 0


def _output_text(output) -> str:
    if output is not None and output.outputs:
        return output.outputs[0].text or ""
    return ""


def _prompt_token_count(output) -> int:
    ids = getattr(output, "prompt_token_ids", None)
    return len(ids) if ids else 0


def _logprob_entries(output) -> list[tuple[str, float]]:
    """Extract (decoded_token, logprob) pairs from a RequestOutput."""
    entries = []
    if output is None or not output.outputs:
        return entries
    for logprob in output.outputs[0].logprobs or []:
        if isinstance(logprob, dict):
            token, prob = logprob.get("decoded_token"), logprob.get("logprob")
        else:
            token = getattr(logprob, "decoded_token", None)
            prob = getattr(logprob, "logprob", None)
        if prob is not None:
            entries.append((token or "", float(prob)))
    return entries


def _metrics_info(output) -> dict:
    """Extract vLLM per-request metrics (TTFT, TPOT, ...)."""
    info = {}
    if output is None:
        return info
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return info
    for attr in (
        "arrival_time",
        "first_token_time",
        "last_token_time",
        "decode_time_per_output_token",
        "num_prompt_tokens",
        "num_generation_tokens",
        "decode_time",
        "prefill_time",
    ):
        with suppress(Exception):
            info[attr] = getattr(metrics, attr)
    if "time_to_first_token" not in info:
        try:
            info["time_to_first_token"] = metrics.time_to_first_token
        except Exception:
            arrival = info.get("arrival_time")
            first = info.get("first_token_time")
            if arrival is not None and first is not None:
                info["time_to_first_token"] = first - arrival
    return info


def _stream_one(llm, model_input, sampling_params) -> tuple:
    """Stream a single request. Returns (final_output, first, last, submit, done)."""
    submit = time.perf_counter()
    first = None
    last = None
    count = 0
    final = None
    try:
        stream = llm.generate(model_input, sampling_params, use_tqdm=False, stream=True)
    except TypeError:
        stream = None

    if stream is None:
        # Older vLLM without offline streaming.
        output = llm.generate(model_input, sampling_params, use_tqdm=False)
        if isinstance(output, list):
            output = output[0]
        done = time.perf_counter()
        return output, None, None, submit, done

    for output in stream:
        if isinstance(output, list):
            output = output[0]
        final = output
        num_tokens = _num_tokens(output)
        if num_tokens > count:
            now = time.perf_counter()
            if first is None:
                first = now
            count = num_tokens
            last = now
    return final, first, last, submit, time.perf_counter()


def _assemble_result(
    item: dict,
    batch_size: int,
    args: argparse.Namespace,
    num_tokens: int,
    text: str,
    metrics: dict,
    ttft: float | None,
    ttft_source: str | None,
    total_time: float,
    itl: float | None,
    prompt_tokens: int,
    logprob_entries: list[tuple[str, float]],
    sampling: dict,
    gpu_info: dict,
) -> dict:
    """Build the per-request result record."""
    result = {
        "id": item["id"],
        "batch_size": batch_size,
        "kind": item["kind"],
        "language": item["language"],
        "injected_language": item.get("injected_language"),
        "mode": args.mode,
        "prompt": item["display"],
        "prompt_tokens": prompt_tokens,
        "generated_tokens": num_tokens,
        "ttft_s": ttft,
        "ttft_source": ttft_source,
        "total_time_s": total_time,
        "tokens_per_sec": (num_tokens / total_time) if total_time > 0 else 0.0,
        "inter_token_latency_s": itl,
        "metrics": metrics,
        "drift": analyze_text_drift(text, item["language"]),
        "drift_windows": windowed_word_drift(text, item["language"], args.n_windows),
        "sampling": sampling,
        "gpu_info": gpu_info,
    }
    if args.logprobs:
        result["logprob_drift"] = analyze_logprob_drift(
            logprob_entries, item["language"], args.n_windows
        )
    result["text"] = text
    return result


def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def run_sequential(
    llm, items: list[dict], sampling_params, args, sampling: dict, gpu_info: dict
) -> tuple[list[dict], int, float]:
    """Run prompts one at a time (batch size 1), streaming for per-token timing."""
    results = []
    tokens_total = 0
    start = time.perf_counter()
    for item in items:
        final, first, last, submit, done = _stream_one(llm, item["input"], sampling_params)
        num_tokens = _num_tokens(final)
        text = _output_text(final)
        metrics = _metrics_info(final)
        metric_ttft = metrics.get("time_to_first_token")
        if metric_ttft is not None:
            ttft, ttft_source = float(metric_ttft), "metrics"
        elif first is not None:
            ttft, ttft_source = first - submit, "stream"
        else:
            ttft, ttft_source = None, None

        itl = None
        if num_tokens > 1 and first is not None and last is not None:
            itl = (last - first) / (num_tokens - 1)
        elif metrics.get("decode_time_per_output_token") is not None:
            itl = float(metrics["decode_time_per_output_token"])

        results.append(
            _assemble_result(
                item,
                batch_size=1,
                args=args,
                num_tokens=num_tokens,
                text=text,
                metrics=metrics,
                ttft=ttft,
                ttft_source=ttft_source,
                total_time=done - submit,
                itl=itl,
                prompt_tokens=_prompt_token_count(final),
                logprob_entries=_logprob_entries(final),
                sampling=sampling,
                gpu_info=gpu_info,
            )
        )
        tokens_total += num_tokens
    return results, tokens_total, time.perf_counter() - start


def run_batch(
    llm, items: list[dict], sampling_params, batch_size: int, args, sampling: dict, gpu_info: dict
) -> tuple[list[dict], int, float]:
    """Run prompts in batches; timing relies on vLLM per-request metrics."""
    results = []
    tokens_total = 0
    start = time.perf_counter()
    for chunk in _chunks(items, batch_size):
        submit = time.perf_counter()
        inputs = [item["input"] for item in chunk]
        outputs = llm.generate(inputs, sampling_params, use_tqdm=False)
        done = time.perf_counter()
        if not isinstance(outputs, list):
            outputs = [outputs]

        for item, output in zip(chunk, outputs, strict=True):
            num_tokens = _num_tokens(output)
            metrics = _metrics_info(output)
            metric_ttft = metrics.get("time_to_first_token")
            ttft = float(metric_ttft) if metric_ttft is not None else None
            results.append(
                _assemble_result(
                    item,
                    batch_size=batch_size,
                    args=args,
                    num_tokens=num_tokens,
                    text=_output_text(output),
                    metrics=metrics,
                    ttft=ttft,
                    ttft_source="metrics" if ttft is not None else None,
                    total_time=done - submit,
                    itl=metrics.get("decode_time_per_output_token"),
                    prompt_tokens=_prompt_token_count(output),
                    logprob_entries=_logprob_entries(output),
                    sampling=sampling,
                    gpu_info=gpu_info,
                )
            )
            tokens_total += num_tokens
    return results, tokens_total, time.perf_counter() - start


def _mean(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (p / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _fmt_ms(value: float | None) -> str:
    return f"{value * 1000:.2f}" if value is not None else "n/a"


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def aggregate_windows(results: list[dict], n_windows: int) -> list[tuple[float, int]]:
    """Average drift ratios across requests, aligned by window index."""
    sums = [0.0] * n_windows
    counts = [0] * n_windows
    for result in results:
        for window in result.get("drift_windows", []):
            index = window.get("index")
            if index is not None and 0 <= index < n_windows:
                sums[index] += window["drift_ratio"]
                counts[index] += 1
    return [(sums[i] / counts[i] if counts[i] else 0.0, counts[i]) for i in range(n_windows)]


def aggregate_logprob_windows(results: list[dict], n_windows: int) -> list[dict]:
    """Average logprob drift across requests, aligned by window index."""
    drift_sums = [0.0] * n_windows
    target_sums = [0.0] * n_windows
    counts = [0] * n_windows
    for result in results:
        for window in result.get("logprob_drift", {}).get("windows", []):
            index = window.get("index")
            if index is None or not 0 <= index < n_windows:
                continue
            if window.get("drift_logprob_mean") is not None:
                drift_sums[index] += window["drift_logprob_mean"]
            if window.get("target_logprob_mean") is not None:
                target_sums[index] += window["target_logprob_mean"]
            counts[index] += 1
    return [
        {
            "drift_logprob_mean": (drift_sums[i] / counts[i]) if counts[i] else None,
            "target_logprob_mean": (target_sums[i] / counts[i]) if counts[i] else None,
        }
        for i in range(n_windows)
    ]


def drift_by_position(results: list[dict], n_buckets: int = 5) -> list[tuple[int, int, float]]:
    """Mean word drift ratio per run-order bucket (drift over the run)."""
    if not results:
        return []
    step = max(1, math.ceil(len(results) / n_buckets))
    buckets = []
    for i in range(0, len(results), step):
        chunk = results[i : i + step]
        ratios = [r["drift"]["word_drift_ratio"] for r in chunk]
        buckets.append((i, min(i + step - 1, len(results) - 1), _mean(ratios)))
    return buckets


def build_summary(
    args: argparse.Namespace,
    items: list[dict],
    sampling: dict,
    max_tokens: int,
    pass_agg: dict[int, dict],
    vocab_stats: dict,
    gpu_info: dict,
) -> str:
    """Render the markdown summary report."""
    lines = []
    lines.append("# LM Benchmark Summary\n")
    lines.append(f"- **Model:** `{args.model}`")
    lines.append(f"- **Target language:** `{args.target_language}`")
    lines.append(f"- **Mode:** `{args.mode}`")
    lines.append(f"- **Prompts:** {len(items)}")
    lines.append(f"- **Batch sizes:** {', '.join(str(b) for b in args.batch_sizes)}")
    lines.append(f"- **Max tokens:** {max_tokens}")
    lines.append(
        f"- **Sampling (resolved):** temperature={sampling['temperature']}, "
        f"top_p={sampling['top_p']}, top_k={sampling['top_k']}, "
        f"repetition_penalty={sampling['repetition_penalty']}, "
        f"presence_penalty={sampling['presence_penalty']}, "
        f"frequency_penalty={sampling['frequency_penalty']}"
    )
    if args.logprobs:
        lines.append("- **Logprobs:** enabled")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("\n---\n")
    lines.append("\n## Hardware\n")
    lines.append(f"- **GPUs:** {gpu_info.get('num_gpus', 0)}")
    lines.append(f"- **Tensor parallel size:** {gpu_info.get('tensor_parallel_size')}")
    gpu_names = gpu_info.get("gpus") or []
    if gpu_names:
        lines.append(f"- **GPU model(s):** {', '.join(gpu_names)}")
    lines.append(f"- **Dtype:** `{args.dtype}`")
    lines.append(f"- **GPU memory utilization:** {args.gpu_memory_utilization}")
    if args.max_model_len is not None:
        lines.append(f"- **Max model len:** {args.max_model_len}")
    lines.append("")

    vocab_report = format_vocab_report(vocab_stats)
    if vocab_report:
        lines.append("\n---\n")
        lines.append(vocab_report)

    lines.append("\n---\n")
    lines.append("\n## Throughput\n")
    lines.append(
        "\n| batch_size | requests | generated_tokens | elapsed_s | tokens/s | TTFT p50 (ms) | TTFT p95 (ms) | TPOT mean (ms) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for batch_size in args.batch_sizes:
        agg = pass_agg[batch_size]
        res = agg["results"]
        ttf = [r["ttft_s"] for r in res if r["ttft_s"] is not None]
        tpots = [r["inter_token_latency_s"] for r in res if r["inter_token_latency_s"] is not None]
        tokens_per_sec = agg["tokens"] / agg["elapsed"] if agg["elapsed"] > 0 else 0.0
        lines.append(
            f"| {batch_size} | {len(res)} | {agg['tokens']} | {agg['elapsed']:.2f} | "
            f"{tokens_per_sec:.2f} | {_fmt_ms(_median(ttf))} | {_fmt_ms(_percentile(ttf, 95))} | "
            f"{_fmt_ms(_mean(tpots))} |"
        )
    lines.append("")

    # Drift statistics are most meaningful for the sequential pass.
    drift_results = pass_agg.get(1, {}).get("results") or list(
        pass_agg[args.batch_sizes[0]]["results"]
    )
    letters_total = sum(r["drift"]["letters_total"] for r in drift_results)
    letters_drift = sum(r["drift"]["letters_drift"] for r in drift_results)
    words_total = sum(r["drift"]["words_total"] for r in drift_results)
    words_drift = sum(r["drift"]["words_drift"] for r in drift_results)
    drift_langs: dict[str, int] = {}
    for r in drift_results:
        for lang, count in r["drift"]["drift_langs"].items():
            drift_langs[lang] = drift_langs.get(lang, 0) + count

    lines.append("\n## Language Drift\n")
    lines.append(
        f"- Character drift ratio: **{(letters_drift / letters_total) if letters_total else 0.0:.4f}**"
    )
    lines.append(
        f"- Word drift ratio: **{(words_drift / words_total) if words_total else 0.0:.4f}**"
    )
    lines.append(
        f"- Letters outside `{args.target_language}`: {letters_drift:,} / {letters_total:,}\n"
    )
    if drift_langs:
        lines.append("Drift by language (characters):\n")
        lines.append("| language | chars |")
        lines.append("|---|---|")
        for lang in sorted(drift_langs, key=drift_langs.get, reverse=True):
            lines.append(f"| {lang} | {drift_langs[lang]:,} |")
        lines.append("")

    lines.append("### Drift over time (within responses, windowed)\n")
    lines.append("| window | mean word drift ratio |")
    lines.append("|---|---|")
    for index, (ratio, _count) in enumerate(aggregate_windows(drift_results, args.n_windows)):
        lines.append(f"| {index + 1} | {ratio:.4f} |")
    lines.append("")

    lines.append("### Drift over the run (by prompt position)\n")
    lines.append("| prompt range | mean word drift ratio |")
    lines.append("|---|---|")
    for start_idx, end_idx, ratio in drift_by_position(drift_results):
        lines.append(f"| {start_idx + 1}-{end_idx + 1} | {ratio:.4f} |")
    lines.append("")

    if args.logprobs:
        drift_means = [r["logprob_drift"]["drift_logprob_mean"] for r in drift_results]
        target_means = [r["logprob_drift"]["target_logprob_mean"] for r in drift_results]
        drift_means = [v for v in drift_means if v is not None]
        target_means = [v for v in target_means if v is not None]
        lines.append("### Log-likelihood drift\n")
        lines.append(f"- Mean logprob of **drift** tokens: {_fmt(_mean(drift_means))}")
        lines.append(f"- Mean logprob of **target-language** tokens: {_fmt(_mean(target_means))}\n")
        lines.append("| window | drift token logprob | target token logprob |")
        lines.append("|---|---|---|")
        for index, window in enumerate(aggregate_logprob_windows(drift_results, args.n_windows)):
            lines.append(
                f"| {index + 1} | {_fmt(window['drift_logprob_mean'])} | {_fmt(window['target_logprob_mean'])} |"
            )
        lines.append("")

    lines.append(
        "> Drift attribution is heuristic: Latin-script languages overlap, "
        "Japanese includes CJK ideographs, and characters outside every known "
        "range are bucketed as `other`."
    )
    return "\n".join(lines)


def write_results(results: list[dict], path: Path) -> None:
    """Write one JSON line per request."""
    with open(path, "w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")


def write_csv(results: list[dict], path: str, model: str) -> None:
    """Append per-request rows to a CSV for cross-model comparison."""
    fieldnames = [
        "model",
        "id",
        "batch_size",
        "kind",
        "language",
        "injected_language",
        "prompt_tokens",
        "generated_tokens",
        "ttft_s",
        "tokens_per_sec",
        "inter_token_latency_s",
        "word_drift_ratio",
        "char_drift_ratio",
    ]
    write_header = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model": model,
                    "id": result["id"],
                    "batch_size": result["batch_size"],
                    "kind": result["kind"],
                    "language": result["language"],
                    "injected_language": result.get("injected_language"),
                    "prompt_tokens": result["prompt_tokens"],
                    "generated_tokens": result["generated_tokens"],
                    "ttft_s": result["ttft_s"],
                    "tokens_per_sec": result["tokens_per_sec"],
                    "inter_token_latency_s": result["inter_token_latency_s"],
                    "word_drift_ratio": result["drift"]["word_drift_ratio"],
                    "char_drift_ratio": result["drift"]["char_drift_ratio"],
                }
            )


def sanitize(name: str) -> str:
    """Turn a model identifier into a filesystem-safe tag."""
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def main() -> None:
    args = parse_args()
    # Convert comma-separated batch sizes string to list of integers
    args.batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]
    if args.target_language not in UNICODE_RANGES:
        sys.exit(
            f"Unknown target language '{args.target_language}'. Available: {sorted(UNICODE_RANGES)}"
        )

    items = load_prompts(args.prompts)
    if not items:
        sys.exit("No prompts loaded.")

    sampling, max_tokens = resolve_sampling(args, args.model)
    sampling_record = dict(sampling)
    sampling_record["max_tokens"] = max_tokens

    try:
        from vllm import SamplingParams
    except ImportError:
        sys.exit(
            "vLLM is not installed. Create a dedicated environment with:\n"
            "  python -m venv .venv-bench && source .venv-bench/bin/activate\n"
            "  pip install -r tools/benchmark_requirements.txt"
        )

    params_kwargs = dict(sampling)
    params_kwargs["max_tokens"] = max_tokens
    params_kwargs["detokenize"] = True
    if args.logprobs:
        params_kwargs["logprobs"] = 1
    sampling_params = SamplingParams(**params_kwargs)

    llm = build_llm(args)

    gpu_info = collect_gpu_info(args.tensor_parallel_size)

    tokenizer = load_tokenizer(args.model, args.trust_remote_code)

    # Analyze the tokenizer vocabulary's Unicode coverage before benchmarking.
    vocab_stats = analyze_tokenizer_vocab(tokenizer)
    vocab_report = format_vocab_report(vocab_stats)
    if vocab_report:
        print(vocab_report)

    normalized = normalize_items(items, args, tokenizer)

    all_results = []
    pass_agg: dict[int, dict] = {}
    for batch_size in args.batch_sizes:
        print(f"Running pass with batch size {batch_size} ...")
        if batch_size <= 1:
            results, tokens, elapsed = run_sequential(
                llm, normalized, sampling_params, args, sampling_record, gpu_info
            )
        else:
            results, tokens, elapsed = run_batch(
                llm, normalized, sampling_params, batch_size, args, sampling_record, gpu_info
            )
        all_results.extend(results)
        pass_agg[batch_size] = {"tokens": tokens, "elapsed": elapsed, "results": results}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = sanitize(args.model)
    results_path = output_dir / f"{model_tag}_results.jsonl"
    summary_path = output_dir / f"{model_tag}_summary.md"

    write_results(all_results, results_path)
    summary_text = build_summary(
        args, normalized, sampling, max_tokens, pass_agg, vocab_stats, gpu_info
    )
    summary_path.write_text(summary_text, encoding="utf-8")
    if args.append_csv:
        write_csv(all_results, args.append_csv, args.model)

    print(f"\nWrote results to {results_path}")
    print(f"Wrote summary to {summary_path}")
    if args.append_csv:
        print(f"Appended CSV rows to {args.append_csv}")
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
