"""
Math dataset generator for the gym.

Loads verified math QA problems from assets/math-problems.jsonl and optionally
generates additional synthetic problems using the built-in generator.

Dataset-based problems use exact match validation.
Synthetic problems use relaxed validation (exact match OR integer part of a float).

Usage:
    # Use all 12481 problems from the dataset (no synthetic): 
    python generate_from_math_dataset.py \
        --output_file math_tasks.jsonl \
        --num_samples 12481

    # With synthetic problems:
    python generate_from_math_dataset.py \
        --output_file math_tasks.jsonl \
        --num_samples 12481 \
        --num_synthetic 10000 \
        --seed 42

    # Only synthetic:
    python generate_from_math_dataset.py \
        --output_file math_tasks.jsonl \
        --num_samples 0 \
        --num_synthetic 500 \
        --seed 42

"""

import json
import random
import operator
import hashlib
import argparse
from pathlib import Path

# The math-problems.jsonl file is expected to be in the same directory as this script, under 
# an "assets" subdirectory.
ASSETS_DIR = Path(__file__).parent / "assets"
MATH_PROBLEMS_JSONL = ASSETS_DIR / "math-problems.jsonl"

# The verifier ID for math answer checking (defined in gym/verifiers.py).
VERIFIER_ID = "math:answer_check"

# Synthetic math problem generator
_OPS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
}

# A variety of preambles to make the synthetic problems more natural and diverse.
_PREAMBLE = [
    "Resolva a seguinte expressão matemática:",
    "Como eu posso resolver esta expressão matemática?",
    "Qual é o resultado desta expressão matemática?",
    "Resolva o seguinte problema matemático:",
    "Resolva isto:",
    "Qual é a resposta para esta expressão?"
]

# To keep evaluation simple and fast, we limit to numbers up to 999 (3 digits).
_MAX_DIGIT = 999  # up to 3-digit numbers


def _generate_expression(depth, rng):
    """Recursively generate a parenthesized math expression."""
    if depth == 0:
        return str(rng.randint(0, _MAX_DIGIT))

    left = _generate_expression(depth - 1, rng)
    right = _generate_expression(depth - 1, rng)
    op = rng.choice(list(_OPS.keys()))

    left_str = f"({left})" if " " in left else left
    right_str = f"({right})" if " " in right else right

    return f"{left_str} {op} {right_str}"


def _evaluate_expression(expr):
    try:
        return eval(expr)
    except ZeroDivisionError:
        return None


def generate_math_problems(n, max_depth=3, seed=None):
    """
    Generate *n* synthetic math problems as (question, answer) pairs.

    Uses up to 3-digit numbers (0-999) and expression trees up to *max_depth*.
    Division-by-zero cases are silently skipped and regenerated.

    Args:
        n: Number of problems to generate.
        max_depth: Maximum expression tree depth (default: 3).
        seed: Random seed for reproducibility.

    Returns:
        List of (question, answer) tuples where answer is a string.
    """
    rng = random.Random(seed)
    problems = []

    while len(problems) < n:
        depth = rng.randint(1, max_depth)
        expr = _generate_expression(depth, rng)
        answer = _evaluate_expression(expr)
        if answer is None:
            continue  # skip division by zero
        preamble = rng.choice(_PREAMBLE)
        question = f"{preamble}\n{expr}"
        problems.append((question, str(answer)))

    return problems


# Dataset loading
def load_math_problems(jsonl_path=None):
    """
    Load (question, answer) pairs from math-problems.jsonl.

    Args:
        jsonl_path: Path to the JSONL file. Defaults to MATH_PROBLEMS_JSONL.

    Returns:
        List of (question, answer) string tuples.
    """
    if jsonl_path is None:
        jsonl_path = MATH_PROBLEMS_JSONL
    pairs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt", "").strip()
            answer = str(row.get("expected_answer", "")).strip()
            if prompt and answer:
                pairs.append((prompt, answer))
    return pairs


# Sample construction
def build_sample(question, answer, relaxed=False):
    """
    Build one gym sample from a math (question, answer) pair.

    Args:
        question: The problem text.
        answer: The expected answer string.
        relaxed: If True, use relaxed validation (exact match OR integer part
            of a float answer). Used for synthetic problems.
    """
    sample_id = hashlib.md5(question.encode()).hexdigest()
    kwargs = {"expected_answer": answer}
    if relaxed:
        kwargs["relaxed"] = True
    return {
        "id": sample_id,
        "prompt": question,
        "verifier_id_list": [VERIFIER_ID],
        "kwargs": [json.dumps(kwargs, ensure_ascii=False)],
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


# Main generation loop
def main(args):
    output_path = Path(args.output_file)
    seed = args.seed
    num_samples = args.num_samples
    num_synthetic = args.num_synthetic

    samples = []
    seen = set()
    total_issues = 0

    # Dataset-based problems
    if num_samples > 0:
        print(f"Loading dataset from {MATH_PROBLEMS_JSONL}...")
        all_pairs = load_math_problems()
        print(f"  Loaded: {len(all_pairs)} pairs")

        if num_samples > len(all_pairs):
            raise ValueError(
                f"--num_samples ({num_samples}) exceeds the total number of "
                f"available pairs ({len(all_pairs)}). Use at most {len(all_pairs)}."
            )

        rng = random.Random(seed)
        rng.shuffle(all_pairs)
        selected = all_pairs[:num_samples]

        for question, answer in selected:
            sample = build_sample(question, answer, relaxed=False)
            sid = sample["id"]
            if sid in seen:
                continue
            seen.add(sid)
            issues = validate_sample(sample)
            if issues:
                total_issues += len(issues)
                if args.verbose:
                    print(f"  Dataset ID {sid}: {issues}")
                continue
            samples.append(sample)

        print(f"  Dataset samples added: {len(samples)}")

    # Synthetic problems
    if num_synthetic > 0:
        print(f"Generating {num_synthetic} synthetic problems...")
        synth_seed = seed + 1 if seed is not None else None
        synth_pairs = generate_math_problems(
            n=num_synthetic,
            max_depth=3,
            seed=synth_seed,
        )
        before = len(samples)
        for question, answer in synth_pairs:
            sample = build_sample(question, answer, relaxed=True)
            sid = sample["id"]
            if sid in seen:
                continue
            seen.add(sid)
            issues = validate_sample(sample)
            if issues:
                total_issues += len(issues)
                if args.verbose:
                    print(f"  Synthetic ID {sid}: {issues}")
                continue
            samples.append(sample)
        print(f"  Synthetic samples added: {len(samples) - before}")

    if not samples:
        raise RuntimeError("No samples generated. Use --num_samples and/or --num_synthetic.")

    print(f"\nResults:")
    print(f"  Total samples:      {len(samples)}")
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
        help="Number of dataset samples from math-problems.jsonl (default: 500). "
            "Set to 0 to skip dataset. "
            "Max number of samples in the dataset is: 12481."
    )
    parser.add_argument(
        "--num_synthetic",
        type=int,
        default=0,
        help="Number of synthetic problems to generate via math_generator (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed validation warnings.",
    )
    args = parser.parse_args()

    main(args)
