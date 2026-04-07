"""
Math dataset generator for the gym.

Downloads math QA datasets from HuggingFace, parses them into
(question, answer) pairs, and outputs samples in the standardized
gym format with the "math:answer_check" verifier.

Sources:
  - Polygl0t/gsm8k-pt: GSM8K translated to Portuguese.
  - Polygl0t/gigaverbo-v2-sft (math split): Math problems with
    answers in the pattern "A resposta é: $answer$".

Usage:
    python generate_from_math_dataset.py \
        --output_file math_tasks.jsonl \
        --num_samples 20000

    python generate_from_math_dataset.py \
        --output_file math_tasks.jsonl \
        --num_samples 20000 \
        --sources gsm8k math_sft \
        --seed 42 \
        --cache_dir ./.cache

    # Only GSM8K:
    python generate_from_math_dataset.py \
        --output_file gsm8k_tasks.jsonl \
        --sources gsm8k

    # Only gigaverbo math SFT:
    python generate_from_math_dataset.py \
        --output_file math_sft_tasks.jsonl \
        --sources math_sft
"""

import json
import random
import hashlib
import argparse
from pathlib import Path
import datasets


VERIFIER_ID = "math:answer_check"

# Dataset loading & parsing
def load_gsm8k(cache_dir):
    """Load and parse Polygl0t/gsm8k-pt. Returns list of (question, answer) tuples."""
    ds = datasets.load_dataset(
        "Polygl0t/gsm8k-pt", split="train", cache_dir=cache_dir,
    )
    pairs = []
    for row in ds:
        answer = row["answer"]
        if "####" not in answer:
            continue
        answer = answer.split("####")[1].strip()
        if _is_valid_number(answer):
            pairs.append((row["question"], answer))
    return pairs


def load_gigaverbo_math(cache_dir):
    """Load and parse Polygl0t/gigaverbo-v2-sft math split. Returns list of (question, answer) tuples."""
    ds = datasets.load_dataset(
        "Polygl0t/gigaverbo-v2-sft", "math", split="train", cache_dir=cache_dir,
    )
    pairs = []
    for row in ds:
        messages = row["messages"]
        user_message = None
        assistant_message = None
        for msg in messages:
            if msg["role"] == "user" and user_message is None:
                user_message = msg["content"]
            elif msg["role"] == "assistant" and assistant_message is None:
                assistant_message = msg["content"]
        if assistant_message is None or "A resposta é:" not in assistant_message:
            continue
        answer = assistant_message.split("A resposta é:")[1].strip()
        if answer.startswith("$") and answer.endswith("$"):
            answer = answer[1:-1]
        if user_message and _is_valid_number(answer):
            pairs.append((user_message, answer))
    return pairs


def _is_valid_number(s):
    """Check if a string represents a valid number."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# Sample construction

def build_sample(question, answer):
    """Build one gym sample from a math (question, answer) pair."""
    sample_id = hashlib.md5(question.encode()).hexdigest()
    return {
        "id": sample_id,
        "prompt": question,
        "verifier_id_list": [VERIFIER_ID],
        "kwargs": [json.dumps({"expected_answer": answer}, ensure_ascii=False)],
    }


def validate_sample(sample):
    """Return a list of issues (empty means valid)."""
    issues = []
    if not sample.get("prompt", "").strip():
        issues.append("Empty prompt")
    if not sample.get("verifier_id_list"):
        issues.append("Empty verifier_id_list")
    if len(sample.get("verifier_id_list", [])) != len(sample.get("kwargs", [])):
        issues.append("verifier_id_list and kwargs length mismatch")
    first_kw = sample.get("kwargs", ["{}"])[0]
    if isinstance(first_kw, str):
        first_kw = json.loads(first_kw)
    answer = first_kw.get("expected_answer", "")
    if not answer:
        issues.append("Missing expected_answer")
    return issues

# Main
def main(args):
    output_path = Path(args.output_file)
    seed = args.seed
    cache_dir = args.cache_dir

    sources = args.sources
    print(f"Loading datasets from HuggingFace (sources: {sources})...")

    all_pairs = []
    if "gsm8k" in sources:
        gsm8k_pairs = load_gsm8k(cache_dir)
        print(f"  gsm8k-pt: {len(gsm8k_pairs)} valid pairs")
        all_pairs.extend(gsm8k_pairs)

    if "math_sft" in sources:
        gigaverbo_pairs = load_gigaverbo_math(cache_dir)
        print(f"  gigaverbo-v2-sft/math: {len(gigaverbo_pairs)} valid pairs")
        all_pairs.extend(gigaverbo_pairs)

    print(f"  Total: {len(all_pairs)} pairs")

    if not all_pairs:
        raise RuntimeError("No valid math pairs found in the datasets.")

    random.seed(seed)
    random.shuffle(all_pairs)

    if args.num_samples > len(all_pairs):
        raise ValueError(
            f"--num_samples ({args.num_samples}) exceeds the total number of "
            f"available pairs ({len(all_pairs)}). Use at most {len(all_pairs)}."
        )

    selected = all_pairs[:args.num_samples]

    samples = []
    seen = set()
    total_issues = 0

    for question, answer in selected:
        sample = build_sample(question, answer)
        sid = sample["id"]
        if sid in seen:
            continue
        seen.add(sid)
        issues = validate_sample(sample)
        if issues:
            total_issues += len(issues)
            if args.verbose:
                print(f"  ID {sid}: {issues}")
            continue
        samples.append(sample)

    print(f"\nResults:")
    print(f"  Generated samples:  {len(samples)}")
    print(f"  Validation issues:  {total_issues}")

    assert total_issues == 0, f"FAIL: {total_issues} validation issues found"

    # Write output
    use_jsonl = output_path.suffix.lower() == ".jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        if use_jsonl:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        else:
            json.dump(samples, f, ensure_ascii=False, indent=2)

    fmt = "JSONL" if use_jsonl else "JSON"
    print(f"\nOutput written to: {output_path} ({fmt})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Math dataset generator for the gym.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output JSON/JSONL file.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Maximum number of samples to generate (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["gsm8k", "math_sft"],
        default=["gsm8k", "math_sft"],
        help="Which dataset sources to use (default: both). "
             "Options: gsm8k, math_sft.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./.cache",
        help="HuggingFace datasets cache directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed validation warnings.",
    )
    args = parser.parse_args()

    main(args)