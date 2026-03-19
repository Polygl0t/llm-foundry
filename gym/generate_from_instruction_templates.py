"""
Template-based generation of instruction samples.

This generator constructs each sample from scratch:

  1. A template provides the base task prompt (with variable slots).
  2. Modifiers (instruction constraints) are selected from an
     instruction set and appended to the prompt.

Usage:
    python generate_from_templates.py \
        --output_file output.json \
        --num_samples 500

    python generate_from_templates.py \
        --output_file output.json \
        --num_samples 50000 \
        --min_modifiers 2 --max_modifiers 5 \
        --seed 123 --verbose
"""

import json
import random
import argparse
from pathlib import Path

from instruction_metadata import (
    ALL_VERIFIER_IDS,
    get_addable_verifiers,
    is_combination_valid,
    generate_kwargs_for_verifier,
    generate_description_for_verifier,
)
from instruction_templates import TEMPLATES

# Constants
# Verifiers excluded from automatic modifier selection
_EXCLUDED_MODIFIERS = {
    # constrained_response conflicts with everything - not useful as a modifier
    "detectable_format:constrained_response",
}

# All verifier IDs available as modifiers
MODIFIER_IDS = sorted(
    iid for iid in ALL_VERIFIER_IDS if iid not in _EXCLUDED_MODIFIERS
)

# Must be appended last (its kwargs depend on the final prompt text)
_REPEAT_PROMPT_ID = "combination:repeat_prompt"

# Template Filling
def fill_template(template):
    """Pick a random prompt variant and fill its slots.

    Returns the filled prompt string.
    """
    prompt_fmt = random.choice(template["prompts"])
    filled_slots = {
        slot: random.choice(values)
        for slot, values in template["slots"].items()
    }
    return prompt_fmt.format(**filled_slots)


# Modifier Selection
def select_modifier_ids(count):
    """Greedily select count mutually-compatible modifier verifier IDs.

    At each step the conflict matrix is consulted to ensure the new ID
    is compatible with all previously selected ones.

    Returns a list of verifier IDs (may be shorter than count if no
    compatible candidates remain).
    """
    selected = []
    for _ in range(count):
        addable = get_addable_verifiers(selected)
        candidates = [iid for iid in addable if iid in MODIFIER_IDS]
        if not candidates:
            break
        selected.append(random.choice(candidates))
    return selected


# Sample Construction
def _normalize_kwargs(kw):
    """Convert integer kwargs values to float."""
    return {
        k: float(v) if isinstance(v, int) else v
        for k, v in kw.items()
    }

def build_sample(template, key, min_modifiers=1, max_modifiers=4):
    """Build one complete sample from a template + random modifiers.

    The prompt is constructed by concatenating:
        base_prompt + modifier_1_description + ... + modifier_N_description
    and the kwargs list is built in lockstep, so they always match.
    """
    # 1. Fill template → base prompt
    base_prompt = fill_template(template)

    # 2. Select modifier verifier IDs
    num_modifiers = random.randint(min_modifiers, max_modifiers)
    modifier_ids = select_modifier_ids(num_modifiers)

    # 3. Separate repeat_prompt (must be applied last)
    regular_ids = [iid for iid in modifier_ids if iid != _REPEAT_PROMPT_ID]
    has_repeat = _REPEAT_PROMPT_ID in modifier_ids

    # 4. Generate descriptions + kwargs for regular modifiers
    prompt_parts = [base_prompt.rstrip()]
    verifier_ids = []
    kwargs_list = []

    for iid in regular_ids:
        kw = generate_kwargs_for_verifier(iid)
        desc = generate_description_for_verifier(iid, kw)
        prompt_parts.append(desc)
        verifier_ids.append(iid)
        kwargs_list.append(_normalize_kwargs(kw))

    # 5. Handle repeat_prompt last (needs the full prompt so far)
    if has_repeat:
        prompt_before_repeat = " ".join(prompt_parts)
        kw = generate_kwargs_for_verifier(
            _REPEAT_PROMPT_ID, prompt_before_repeat
        )
        desc = generate_description_for_verifier(_REPEAT_PROMPT_ID, kw)
        verifier_ids.append(_REPEAT_PROMPT_ID)
        kwargs_list.append(_normalize_kwargs(kw))
        final_prompt = prompt_before_repeat + "\n" + desc
    else:
        final_prompt = " ".join(prompt_parts)

    return {
        "key": key,
        "prompt": final_prompt,
        "verifier_id_list": verifier_ids,
        "kwargs": kwargs_list,
    }


# Validation
def sample_fingerprint(sample):
    """Hashable fingerprint for uniqueness checking."""
    return (sample["prompt"], tuple(sorted(sample["verifier_id_list"])))

def validate_sample(sample):
    """Return a list of issues (empty means valid)."""
    issues = []

    n_ids = len(sample.get("verifier_id_list", []))
    n_kw = len(sample.get("kwargs", []))
    if n_ids != n_kw:
        issues.append(
            f"verifier_id_list length ({n_ids}) != kwargs length ({n_kw})"
        )

    for iid in sample.get("verifier_id_list", []):
        if iid not in ALL_VERIFIER_IDS:
            issues.append(f"Unknown instruction ID: {iid}")

    if not is_combination_valid(sample.get("verifier_id_list", [])):
        issues.append("Instruction combination has conflicts")

    if not sample.get("prompt", "").strip():
        issues.append("Empty prompt")

    # Verify each modifier description appears in the prompt
    for i, iid in enumerate(sample.get("verifier_id_list", [])):
        kw = sample["kwargs"][i]
        desc = generate_description_for_verifier(iid, kw)
        if desc and desc not in sample.get("prompt", ""):
            issues.append(f"Description for {iid} not found in prompt")

    return issues

def main(args):
    output_path = Path(args.output_file)
    num_samples = args.num_samples
    seed = args.seed
    max_retries = 20

    print(
        f"Generating {num_samples} samples from {len(TEMPLATES)} templates "
        f"(seed={seed}, modifiers={args.min_modifiers}-{args.max_modifiers})"
    )

    samples = []
    seen = set()
    key_counter = args.start_key
    retries_used = 0

    for i in range(num_samples):
        sample = None

        for attempt in range(max_retries):
            # Deterministic seed per (sample_index, attempt)
            random.seed(seed + i * max_retries + attempt)

            template = random.choice(TEMPLATES)
            candidate = build_sample(
                template,
                key=key_counter,
                min_modifiers=args.min_modifiers,
                max_modifiers=args.max_modifiers,
            )

            fp = sample_fingerprint(candidate)
            if fp not in seen:
                sample = candidate
                seen.add(fp)
                break

            retries_used += 1

        if sample is None:
            # Extremely unlikely with enough templates, but be safe
            print(f"  Warning: could not produce unique sample #{i+1}")
            continue

        samples.append(sample)
        key_counter += 1

    # Validate
    total_issues = 0
    for s in samples:
        issues = validate_sample(s)
        if issues:
            total_issues += len(issues)
            if args.verbose:
                print(f"  Key {s['key']}: {issues}")

    conflict_free = sum(
        1 for s in samples if is_combination_valid(s["verifier_id_list"])
    )
    unique_fps = len({sample_fingerprint(s) for s in samples})

    print(f"\nResults:")
    print(f"  Generated samples:  {len(samples)}")
    print(
        f"  Conflict-free:      {conflict_free}/{len(samples)}"
        f"  ({100 * conflict_free / max(len(samples), 1):.1f}%)"
    )
    print(
        f"  Unique:             {unique_fps}/{len(samples)}"
        f"  ({100 * unique_fps / max(len(samples), 1):.1f}%)"
    )
    print(f"  Validation issues:  {total_issues}")
    if retries_used:
        print(f"  Uniqueness retries: {retries_used}")

    #  Hard assertions 
    assert conflict_free == len(samples), (
        f"FAIL: {len(samples) - conflict_free} samples have instruction conflicts"
    )
    assert unique_fps == len(samples), (
        f"FAIL: {len(samples) - unique_fps} duplicate samples"
    )
    assert total_issues == 0, (
        f"FAIL: {total_issues} validation issues found"
    )

    #  Write output 
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"\nOutput written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Template-based Instruction generation.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Total number of samples to generate (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--start_key",
        type=int,
        default=10000,
        help="Starting key value for generated samples (default: 10000).",
    )
    parser.add_argument(
        "--min_modifiers",
        type=int,
        default=1,
        help="Minimum number of instruction modifiers per sample (default: 1).",
    )
    parser.add_argument(
        "--max_modifiers",
        type=int,
        default=4,
        help="Maximum number of instruction modifiers per sample (default: 4).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed validation warnings.",
    )
    args = parser.parse_args()
    main(args)
