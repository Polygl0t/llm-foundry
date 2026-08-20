"""
Extract all <think>...</think> reasoning traces from formatted trace files.

Reads a JSON file containing trace objects (each with a trace_id and messages array),
extracts the text inside <think>...</think> tags from all assistant messages, and writes
a JSONL output where each line is {"id": trace_id, "text": extracted_think_text}.

Usage:
    python extract_reasoning_traces.py <input_file.json>

The output file will be named <input_file>_reasoning_traces.jsonl.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def get_trace_id(obj: dict, id_column: str) -> str:
    """
    Resolve a trace identifier for obj.

    1. Use the value stored under *id_column* if present and non-empty.
    2. Fall back to a SHA-256 hash of the first user message's content.
    3. If neither is available, raise ValueError.
    """
    # 1. explicit id column
    raw = obj.get(id_column)
    if raw is not None and str(raw).strip():
        return str(raw).strip()

    # 2. fallback: hash of first user message
    messages = obj.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if content:
                return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # 3. nothing worked
    raise ValueError(
        f"Object has no usable '{id_column}' and no user message to hash. "
        f"Keys present: {list(obj.keys())}"
    )


def extract_think_blocks(text: str) -> list[str]:
    """
    Extract all text between <think> and </think> tags from a string.

    Uses a non-greedy regex with DOTALL so that multi-line think blocks
    are captured correctly.  Returns a list of matched contents (the tags
    themselves are stripped).
    """
    # re.DOTALL so that '.' matches newlines; non-greedy so that multiple
    # <think>...</think> blocks in the same message are captured individually.
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    return pattern.findall(text)


def main(args) -> None:
    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Insert the suffix before the final extension
        output_path = input_path.with_suffix("").with_suffix(
            input_path.suffix + "_reasoning_traces.jsonl"
        )
        # The above is fragile with double extensions like ".concat.json".
        # Handle more robustly by using string replace on the stem + suffix.
        full_name = input_path.name
        if full_name.endswith(".json"):
            output_name = full_name[: -len(".json")] + "_reasoning_traces.jsonl"
        else:
            output_name = full_name + "_reasoning_traces.jsonl"
        output_path = input_path.with_name(output_name)

    print(f"Reading: {input_path}")
    with open(input_path, encoding=args.encoding) as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        print(
            f"Error: expected a JSON array at the top level, got {type(data).__name__}",
            file=sys.stderr,
        )
        sys.exit(1)

    total_traces = len(data)
    total_think_blocks = 0
    traces_with_think = 0

    print(f"Writing: {output_path}")
    with open(output_path, "w", encoding=args.encoding) as out_fh:
        for obj in data:
            try:
                trace_id = get_trace_id(obj, args.id_column)
            except ValueError as exc:
                print(
                    f"Error: skipping object — {exc}",
                    file=sys.stderr,
                )
                continue

            messages = obj.get("messages", [])
            found_in_this_trace = False

            for msg in messages:
                if msg.get("role") != "assistant":
                    continue

                content = msg.get("content", "")
                if not content:
                    continue

                think_texts = extract_think_blocks(content)
                for text in think_texts:
                    # Strip leading/trailing whitespace from each block
                    text = text.strip()
                    if not text:
                        continue

                    out_fh.write(
                        json.dumps({"id": trace_id, "text": text}, ensure_ascii=False) + "\n"
                    )
                    total_think_blocks += 1
                    found_in_this_trace = True

            if found_in_this_trace:
                traces_with_think += 1

    print(f"Processed {total_traces} traces.")
    print(f"  Traces with at least one <think> block: {traces_with_think}")
    print(f"  Total <think> blocks extracted:         {total_think_blocks}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input JSON file containing trace objects.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=("Path to the output JSONL file.  Defaults to '<input>_reasoning_traces.jsonl'."),
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="trace_id",
        help=(
            "Name of the JSON field used as the trace identifier "
            "(default: 'trace_id').  If absent or empty, falls back "
            "to a SHA-256 hash of the first user message's content."
        ),
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="File encoding (default: utf-8).",
    )

    args = parser.parse_args()
    main(args)
