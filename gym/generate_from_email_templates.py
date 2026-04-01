"""
Email structured-extraction task generator.

Reads a JSONL file of emails (one JSON object per line with an "email" key)
and produces gym-format samples that ask a model to extract specific fields
from each email as a structured JSON object.

Two categories of fields are supported:

  Direct fields (extracted from email content):
      subject, sender, receiver, intent, summary

  Injected fields (deterministic values added to the email context):
      date, attachments, spam, sender_email, telephone_number

For injected fields an explicit metadata header is prepended to the email
text, and a corresponding ``email:field_value`` verifier is added so the
extraction can be verified exactly.

Usage:
    python generate_from_email_templates.py \
        --emails_file ./data/emails.jsonl \
        --output_file email_tasks.jsonl \
        --num_samples 500

    python generate_from_email_templates.py \
        --emails_file ./data/emails.jsonl \
        --output_file email_tasks.json \
        --num_samples 1000 \
        --min_fields 3 --max_fields 7 \
        --seed 42 --start_key 0 --verbose
"""

import json
import random
import argparse
from pathlib import Path

from tasks_metadata import (
    EMAIL_ALL_FIELDS,
    EMAIL_DIRECT_FIELDS,
    EMAIL_INJECTED_FIELDS,
    EMAIL_FIELD_LABELS,
    EMAIL_TASK_IDS,
    EMAIL_DEFAULTS,
)


# Prompt templates
_PROMPT_PREAMBLES = [
    "Leia o e-mail abaixo e extraia as informações solicitadas em formato JSON.",
    "Com base no e-mail a seguir, extraia os campos indicados como objeto JSON.",
    "A partir do e-mail abaixo, produza um objeto JSON com as informações solicitadas.",
    "Analise o e-mail a seguir e extraia as informações pedidas em JSON.",
    "Dado o e-mail abaixo, extraia e organize as informações em um objeto JSON.",
    "Processe o e-mail a seguir e retorne os dados solicitados como um objeto JSON.",
    "Leia com atenção o e-mail abaixo e forneça as informações pedidas em JSON.",
]

_FORMAT_INSTRUCTION = (
    "Formate sua resposta EXATAMENTE como um bloco JSON markdown, "
    "sem nenhum texto antes ou depois:\n"
    "```json\n"
    "{\n"
    "  ...\n"
    "}\n"
    "```\n"
    "O JSON deve conter SOMENTE as chaves solicitadas, sem campos adicionais."
)


# ---------------------------------------------------------------------------
# Random value generators for injected fields
# ---------------------------------------------------------------------------

_EMAIL_DOMAINS = [
    "gmail.com",
    "yahoo.com.br",
    "hotmail.com",
    "outlook.com",
    "empresa.com.br",
    "trabalho.com.br",
    "usp.br",
    "ufmg.br",
    "mail.com",
    "protonmail.com",
]

_FIRST_NAMES = [
    "carlos", "ana", "pedro", "maria", "joao", "julia",
    "lucas", "fernanda", "rafael", "camila", "bruno", "mariana",
    "rodrigo", "patricia", "gustavo", "larissa", "felipe", "sophia",
    "nicholas", "isabela", "diego", "carolina", "vinicius", "amanda",
    "gabriel", "bianca", "matheus", "juliana", "ricardo", "marina"
]

_LAST_NAMES = [
    "silva", "souza", "oliveira", "santos", "lima",
    "pereira", "costa", "rodrigues", "alves", "martins",
    "ferreira", "gomes", "ribeiro", "carvalho", "almeida",
    "nascimento", "lopes", "machado", "barbosa", "rocha", 
    "dias", "freitas", "araujo", "melo", "cardoso"
]


def _random_date(rng):
    """Generate a random ISO 8601 timestamp between 2020 and 2026."""
    year = rng.randint(2020, 2026)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    return f"{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00"


def _random_sender_email(rng):
    """Generate a plausible Brazilian sender email address."""
    first = rng.choice(_FIRST_NAMES)
    last = rng.choice(_LAST_NAMES)
    domain = rng.choice(_EMAIL_DOMAINS)
    sep = rng.choice([".", "_", ""])
    num_suffix = str(rng.randint(1, 99)) if rng.random() < 0.3 else ""
    return f"{first}{sep}{last}{num_suffix}@{domain}"


def _random_phone(rng):
    """Generate a plausible Brazilian phone number in one of several formats."""
    ddd = rng.choice(["11", "21", "31", "41", "51", "61", "71", "81", "91"])
    prefix = rng.choice(["98", "99", "97", "96"])
    part1 = f"{rng.randint(1000, 9999)}"
    part2 = f"{rng.randint(1000, 9999)}"
    fmt = rng.choice([
        f"({ddd}) {prefix}{part1}-{part2}",
        f"+55 {ddd} {prefix}{part1}-{part2}",
        f"+55 ({ddd}) {prefix}{part1}-{part2}",
        f"{ddd} {prefix}{part1}-{part2}",
    ])
    return fmt


def generate_injected_values(rng):
    """Generate random values for all five injected fields.

    Returns a dict with keys: date, attachments, spam, sender_email,
    telephone_number.
    """
    return {
        "date": _random_date(rng),
        "attachments": rng.random() < 0.3,
        "spam": rng.random() < 0.1,
        "sender_email": _random_sender_email(rng),
        "telephone_number": _random_phone(rng),
    }



# Email context construction
def _injected_label(field, value):
    """Return the Portuguese header line for a single injected field."""
    if field == "date":
        return f"Data de recebimento: {value}"
    if field == "attachments":
        return f"Anexos: {'Sim' if value else 'Não'}"
    if field == "spam":
        return f"Classificação de spam: {'Sim' if value else 'Não'}"
    if field == "sender_email":
        return f"E-mail do remetente: {value}"
    if field == "telephone_number":
        return f"Telefone de contato: {value}"
    return f"{field}: {value}"


def build_email_context(email_text, injected_values):
    """Prepend a structured metadata header to the email text.

    The header contains all five injected fields so the model can read
    them directly, regardless of which subset is requested.
    """
    header_lines = ["--- Metadados do E-mail ---"]
    for field in EMAIL_INJECTED_FIELDS:
        header_lines.append(_injected_label(field, injected_values[field]))
    header_lines.append("---")
    return "\n".join(header_lines) + "\n\n" + email_text


# Sample construction
def build_email_sample(email_text, key, fields, injected_values, rng):
    """Build one email extraction gym sample.

    Args:
        email_text: Raw email content (will have metadata header prepended).
        key: Integer identifier for this sample.
        fields: List of field names to request in the output JSON.
        injected_values: Dict of all five injected field values.
        rng: ``random.Random`` instance for reproducibility.

    Returns:
        Dict with keys: key, prompt, verifier_id_list, kwargs.
    """
    email_with_meta = build_email_context(email_text, injected_values)

    # Describe requested fields
    field_details = "\n".join(
        f"  - {EMAIL_FIELD_LABELS[f]}" for f in fields
    )
    fields_str = ", ".join(fields)

    preamble = rng.choice(_PROMPT_PREAMBLES)
    prompt = (
        f"{preamble}\n\n"
        f"Campos solicitados:\n{field_details}\n\n"
        f"E-mail:\n{email_with_meta}\n\n"
        f"{_FORMAT_INSTRUCTION}\n"
        f"Chaves obrigatórias: {fields_str}"
    )

    # Always include the two format verifiers
    verifier_id_list = ["email:json_format", "email:schema_keys"]
    kwargs_list = [
        {},                                        # email:json_format — no kwargs
        {"required_keys": sorted(fields)},         # email:schema_keys
    ]

    # Exact-match verifiers for every injected field that was requested
    for field in fields:
        if field in EMAIL_INJECTED_FIELDS:
            verifier_id_list.append("email:field_value")
            kwargs_list.append({
                "field_name": field,
                "expected_value": injected_values[field],
            })

    return {
        "key": key,
        "prompt": prompt,
        "verifier_id_list": verifier_id_list,
        "kwargs": kwargs_list,
    }


# Validation
def validate_email_sample(sample):
    """Return a list of validation issues for an email sample (empty = valid)."""
    issues = []

    n_ids = len(sample.get("verifier_id_list", []))
    n_kw = len(sample.get("kwargs", []))
    if n_ids != n_kw:
        issues.append(
            f"verifier_id_list length ({n_ids}) != kwargs length ({n_kw})"
        )

    for iid in sample.get("verifier_id_list", []):
        if iid not in EMAIL_TASK_IDS:
            issues.append(f"Unknown email task ID: {iid}")

    if not sample.get("prompt", "").strip():
        issues.append("Empty prompt")

    vids = sample.get("verifier_id_list", [])
    if "email:json_format" not in vids:
        issues.append("Missing email:json_format verifier")
    if "email:schema_keys" not in vids:
        issues.append("Missing email:schema_keys verifier")

    for i, iid in enumerate(vids):
        kw = sample["kwargs"][i] if i < len(sample.get("kwargs", [])) else {}
        if iid == "email:schema_keys":
            keys = kw.get("required_keys", [])
            if not keys:
                issues.append("email:schema_keys has empty required_keys")
            else:
                for k in keys:
                    if k not in EMAIL_ALL_FIELDS:
                        issues.append(f"Unknown field in required_keys: {k}")
        elif iid == "email:field_value":
            if not kw.get("field_name"):
                issues.append("email:field_value missing field_name")
            if "expected_value" not in kw:
                issues.append("email:field_value missing expected_value")

    return issues


def sample_fingerprint(sample):
    """Hashable fingerprint for deduplication."""
    return sample["prompt"]


# Data loading
def load_emails(emails_file):
    """Load emails from a JSONL file.

    Each line must be a JSON object with an ``"email"`` key.
    Returns a list of email strings.
    """
    path = Path(emails_file)
    if not path.exists():
        raise FileNotFoundError(f"Emails file not found: {emails_file}")

    emails = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {emails_file}"
                ) from exc
            if "email" not in obj:
                raise ValueError(
                    f"Line {line_no} in {emails_file} has no 'email' key"
                )
            text = obj["email"].strip()
            if text:
                emails.append(text)
    return emails


# Main generation loop
def main(args):
    output_path = Path(args.output_file)

    emails = load_emails(args.emails_file)
    if not emails:
        print("Error: no usable emails found in the input file.")
        return

    min_fields = max(1, args.min_fields)
    max_fields = min(args.max_fields, len(EMAIL_ALL_FIELDS))
    if min_fields > max_fields:
        raise ValueError(
            f"--min_fields ({min_fields}) > --max_fields ({max_fields})"
        )

    print(
        f"Loaded {len(emails)} emails. "
        f"Generating {args.num_samples} samples "
        f"(fields={min_fields}-{max_fields}, seed={args.seed})"
    )

    samples = []
    seen = set()
    key_counter = args.start_key
    retries_used = 0
    max_retries = 20

    for i in range(args.num_samples):
        sample = None

        for attempt in range(max_retries):
            rng = random.Random(args.seed + i * max_retries + attempt)

            email_text = rng.choice(emails)
            injected = generate_injected_values(rng)

            num_fields = rng.randint(min_fields, max_fields)
            fields = rng.sample(EMAIL_ALL_FIELDS, num_fields)

            candidate = build_email_sample(
                email_text=email_text,
                key=key_counter,
                fields=fields,
                injected_values=injected,
                rng=rng,
            )

            fp = sample_fingerprint(candidate)
            if fp not in seen:
                sample = candidate
                seen.add(fp)
                break

            retries_used += 1

        if sample is None:
            print(f"  Warning: could not produce unique sample #{i + 1}")
            continue

        samples.append(sample)
        key_counter += 1

    # Validate all samples
    total_issues = 0
    for s in samples:
        issues = validate_email_sample(s)
        if issues:
            total_issues += len(issues)
            if args.verbose:
                print(f"  Key {s['key']}: {issues}")

    unique_fps = len({sample_fingerprint(s) for s in samples})

    print(f"\nResults:")
    print(f"  Generated samples:  {len(samples)}")
    print(
        f"  Unique:             {unique_fps}/{len(samples)}"
        f"  ({100 * unique_fps / max(len(samples), 1):.1f}%)"
    )
    print(f"  Validation issues:  {total_issues}")
    if retries_used:
        print(f"  Retries used:       {retries_used}")

    assert total_issues == 0, f"FAIL: {total_issues} validation issues found"
    assert unique_fps == len(samples), (
        f"FAIL: {len(samples) - unique_fps} duplicate samples"
    )

    use_jsonl = output_path.suffix.lower() == ".jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        if use_jsonl:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        else:
            json.dump(samples, f, ensure_ascii=False, indent=2)

    fmt = "JSONL" if use_jsonl else "JSON"
    print(f"\nOutput written to: {output_path} ({fmt})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Email structured-extraction task generator.",
    )
    parser.add_argument(
        "--emails_file",
        type=str,
        required=True,
        help="Path to the JSONL file containing emails (one per line, 'email' key).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path for the output JSON or JSONL file.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=EMAIL_DEFAULTS["min_fields"] * 100,
        help="Number of samples to generate (default: 300).",
    )
    parser.add_argument(
        "--min_fields",
        type=int,
        default=EMAIL_DEFAULTS["min_fields"],
        help=f"Minimum number of JSON fields to request per sample "
             f"(default: {EMAIL_DEFAULTS['min_fields']}).",
    )
    parser.add_argument(
        "--max_fields",
        type=int,
        default=EMAIL_DEFAULTS["max_fields"],
        help=f"Maximum number of JSON fields to request per sample "
             f"(default: {EMAIL_DEFAULTS['max_fields']}).",
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
        default=0,
        help="Starting integer key for generated samples (default: 0).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample validation warnings.",
    )
    args = parser.parse_args()
    main(args)
