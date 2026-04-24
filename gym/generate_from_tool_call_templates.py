"""
Procedural generator for tool-calling tasks.

Reads `tools.json` (tool definitions with per-tool input_samples and
request_samples) and produces gym-format samples covering:

  1. Valid tool-call tasks: model should produce `<tool_call>...</tool_call>`
  2. Refusal tasks: model should explain why no tool applies

Each valid prompt is self-contained: argument values are sampled directly
from the tool's `input_samples` and embedded in the user request using the
tool's own `request_samples` as templates.

Usage:
    python generate_from_tool_call_templates.py \
        --output_file tool_call_tasks.jsonl \
        --num_samples 500

    python generate_from_tool_call_templates.py \
        --output_file tool_call_tasks.jsonl \
        --num_samples 20000 \
        --min_tools 1 --max_tools 3 \
        --refusal_ratio 0.2 \
        --seed 42 --verbose
"""

import json
import random
import hashlib
import argparse
from pathlib import Path

from tasks_metadata import (
    TOOL_CALL_TASK_IDS,
)

_DATA_DIR = Path(__file__).resolve().parent / "assets"


# Prompt system preambles (Portuguese, varied styles)
_SYSTEM_PREAMBLES = [
    (
        "Você pode chamar uma ou mais das funções a seguir para ajudar "
        "a responder à solicitação do usuário."
    ),
    (
        "As funções abaixo estão disponíveis para uso. Utilize-as conforme "
        "necessário para responder às perguntas do usuário."
    ),
    (
        "Você tem acesso a ferramentas que podem ser chamadas para auxiliar "
        "nas respostas. As funções disponíveis são:"
    ),
    (
        "## Ferramentas\n\n"
        "Existem funções disponíveis que você pode usar para ajudar a "
        "responder às perguntas do usuário."
    ),
    (
        "## Tools\n\nAs ferramentas a seguir estão disponíveis para você "
        "usar ao responder às consultas do usuário."
    ),
    (
        "As seguintes funções estão disponíveis para uso. Chame-as quando "
        "precisar para responder ao usuário."
    ),
    (
        "# Ferramentas disponíveis\n\n"
        "Você pode chamar as funções listadas abaixo para auxiliar "
        "na consulta do usuário."
    ),
    (
        "## Funções\n\nAs seguintes funções estão à sua disposição para "
        "responder às solicitações do usuário."
    ),
    (
        "Estão disponíveis as seguintes funções, que você pode invocar "
        "para responder ao usuário."
    ),
    (
        "# Ferramentas\n\nVocê pode usar as ferramentas listadas abaixo "
        "para auxiliar na consulta do usuário."
    ),
    (
        "Quando necessário, você pode chamar as seguintes funções para "
        "responder às perguntas do usuário."
    ),
    (
        "## Funções disponíveis\n\nUtilize as funções abaixo conforme "
        "necessário para responder às consultas do usuário."
    ),
    (
        "# Tools / Ferramentas\n\nAs funções listadas abaixo podem ser "
        "invocadas para ajudar a responder às perguntas do usuário."
    ),
    (
        "# Tools / Ferramentas\n\nVocê tem acesso a uma série de funções "
        "que podem ser usadas para ajudar a responder às perguntas."
    ),
]


# Instruction for how the model should format tool calls in its response.
_TOOL_CALL_INSTRUCTION = (
    "Para cada chamada de função, retorne um objeto json com o nome da "
    "função e os argumentos dentro das tags XML <tool_call></tool_call>:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)


# User query templates for REFUSAL tasks (no tool applies)
_REFUSAL_QUERY_TEMPLATES = [
    "Você pode pedir uma pizza para mim, por favor?",
    "Pode me recomendar um bom restaurante?",
    "Gostaria que você marcasse uma consulta médica para mim.",
    "Pode ligar para meu amigo e dar um recado?",
    "Preciso que você compre passagens aéreas para mim.",
    "Gostaria que você fizesse uma reserva em um hotel.",
    "Pode me contar uma piada?",
    "Gostaria de ouvir uma música. Pode tocar algo?",
    "Pode me dar um abraço virtual?",
    "Preciso de ajuda para arrumar minha casa.",
    "Você pode cozinhar algo para mim?",
    "Gostaria que você me ajudasse a dormir melhor.",
    "Pode me ensinar a dirigir?",
    "Preciso que você cuide do meu cachorro.",
    "Pode me ajudar a mudar de casa?",
    "Gostaria que você fosse ao supermercado para mim.",
    "Pode me dar conselhos sobre relacionamentos?",
    "Preciso que você conserte meu computador.",
    "Pode me ajudar a pintar a parede da sala?",
    "Gostaria que você me ensinasse a nadar.",
    "Pode me ajudar a escolher um presente de aniversário?",
    "Preciso que você lave meu carro.",
    "Pode me ajudar a organizar minha festa?",
    "Gostaria que você me dissesse o futuro.",
    "Pode me ajudar a encontrar um emprego?",
    "Preciso que você faça uma entrega para mim.",
    "Pode me ajudar a decorar minha casa?",
    "Gostaria que você me preparasse para uma entrevista.",
    "Pode me ajudar com minha mudança de visual?",
    "Quanto custa um apartamento no centro de São Paulo?",
    "Qual é o melhor time de futebol do Brasil?",
    "Pode me dizer como está o trânsito agora?",
    "Preciso de ajuda para montar um móvel.",
    "Pode me recomendar um filme para hoje à noite?",
    "Gostaria que você me acordasse amanhã às 7h.",
    "Pode me ajudar a aprender a tocar violão?",
    "Preciso que você me ajude a traduzir uma conversa ao vivo.",
    "Pode me ajudar a criar um perfil de namoro?",
    "Gostaria que você me ajudasse a treinar para uma maratona.",
]


# Data loading
def load_tools(path):
    """
    Load tool definitions from a JSON file.

    Accepts both a list at the top level and the `{"tools": [...]}` format.
    Only tools with a valid function name are returned.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data["tools"] if isinstance(data, dict) and "tools" in data else data
    return [t for t in items if t.get("function", {}).get("name")]


def load_tool_call_data(path):
    """
    Load tools from a JSON file.

    Accepts both a list at the top level and the `{"tools": [...]}` format.
    Only tools with a valid function name are returned.
    """
    return load_tools(path)



# Prompt formatting (turns tool definitions into the string format expected in the prompt)
def _format_tools_block(tools):
    """
    Serialize tools for inclusion in a prompt.

    Strips metadata-only fields (`input_samples`, `request_samples`) so
    the model sees only the clean function schema.
    """
    _METADATA_KEYS = {"input_samples", "request_samples"}
    result = "<tools>"
    for tool in tools:
        clean = {k: v for k, v in tool.items() if k not in _METADATA_KEYS}
        result += "\n" + json.dumps(clean, ensure_ascii=False)
    result += "\n</tools>"
    return result


# Argument sampling from input_samples
def _fallback_arg_value(schema, rng):
    """Return a plausible value for *schema* when no `input_samples` exist."""
    ptype = schema.get("type", "string")
    if ptype == "string":
        return rng.choice(["exemplo", "teste", "Porto Alegre", "valor"])
    if ptype == "number":
        return rng.choice([10.0, 25.5, 50.0, 100.0, 250.0])
    if ptype == "integer":
        return rng.choice([1, 5, 10, 42, 100])
    if ptype == "boolean":
        return rng.choice([True, False])
    if ptype == "array":
        return rng.choice([["item1", "item2"], ["a", "b", "c"]])
    if ptype == "object":
        return {"chave": "valor"}
    return "valor_exemplo"


def _sample_args_from_inputs(tool, rng):
    """
    Sample argument values for *tool* from its `input_samples`.

    All required parameters are always included.  Optional parameters are
    included with 40 % probability.  When `input_samples` has no entry for a
    parameter its schema type is used to produce a fallback value.

    Returns a `{name: value}` dict.
    """
    params = tool["function"].get("parameters", {})
    properties = params.get("properties", {})
    required = set(params.get("required", []))
    input_samples = tool.get("input_samples", {})

    args = {}

    for name in required:
        if name in input_samples:
            samples = input_samples[name]
            args[name] = rng.choice(samples) if isinstance(samples, list) else samples
        else:
            args[name] = _fallback_arg_value(properties.get(name, {}), rng)

    for name, schema in properties.items():
        if name in args:
            continue
        if rng.random() < 0.4:
            if name in input_samples:
                samples = input_samples[name]
                args[name] = (
                    rng.choice(samples) if isinstance(samples, list) else samples
                )
            else:
                args[name] = _fallback_arg_value(schema, rng)

    return args


def get_tool_arg_types(tool):
    """Return `{param_name: json_type}` for all parameters in *tool*."""
    params = tool["function"].get("parameters", {})
    properties = params.get("properties", {})
    return {name: schema.get("type", "string") for name, schema in properties.items()}



# User request construction
def _build_user_request(tool, args, rng):
    """
    Construct a self-contained user request for *tool* with *args* embedded.

    The request starts with a sentence drawn from the tool's own
    `request_samples` and appends the argument values inline so the prompt
    is fully self-contained (no clarification needed from the model).
    """
    request_samples = tool.get("request_samples", [])
    properties = tool["function"].get("parameters", {}).get("properties", {})

    if request_samples:
        base = rng.choice(request_samples)
    else:
        desc = tool["function"].get("description", tool["function"]["name"])
        base = f"Preciso de ajuda com: {desc.lower().rstrip('.')}."

    if not args:
        return base

    phrases = []
    for name, value in args.items():
        schema = properties.get(name, {})
        param_desc = schema.get("description", name).rstrip(".")
        if isinstance(value, bool):
            val_str = "sim" if value else "não"
        elif isinstance(value, list):
            val_str = ", ".join(str(v) for v in value)
        else:
            val_str = str(value)
        phrases.append((param_desc, val_str))

    if base.rstrip().endswith(":") and len(phrases) == 1:
        # The base already introduces the single parameter; just use the value
        # to avoid "O título do filme é: O título do filme: Interstellar."
        inline = phrases[0][1]
    else:
        inline = "; ".join(f"{d}: {v}" for d, v in phrases)

    return f"{base} {inline}."


# Full prompt assembly
def build_valid_prompt(tool, user_query, rng, extra_tools=None):
    """
    Assemble a complete prompt for a valid tool-call task.

    Args:
        tool: The target tool that should be called.
        user_query: A self-contained user request string (with arg values
            already embedded).
        rng: `random.Random` instance.
        extra_tools: Optional list of distractor tools to include alongside
            the target in the tools block.

    Returns:
        `(prompt_string, prompt_tools_list)`
    """
    preamble = rng.choice(_SYSTEM_PREAMBLES)

    prompt_tools = [tool] + (extra_tools or [])
    rng.shuffle(prompt_tools)

    tools_block = _format_tools_block(prompt_tools)
    prompt = (
        f"{preamble}\n\n"
        f"Você recebe assinaturas de funções dentro de tags XML "
        f"<tools></tools>:\n{tools_block}\n\n"
        f"{_TOOL_CALL_INSTRUCTION}\n\n"
        f"{user_query}"
    )
    return prompt, prompt_tools


def build_refusal_prompt(distractor_tools, rng):
    """
    Assemble a complete prompt for a refusal task.

    The user asks for something that none of the provided tools can handle.

    Args:
        distractor_tools: Tools to include (none will match the refusal query).
        rng: `random.Random` instance.

    Returns:
        The complete prompt string.
    """
    preamble = rng.choice(_SYSTEM_PREAMBLES)
    tools_block = _format_tools_block(distractor_tools)
    user_query = rng.choice(_REFUSAL_QUERY_TEMPLATES)

    prompt = (
        f"{preamble}\n\n"
        f"Você recebe assinaturas de funções dentro de tags XML "
        f"<tools></tools>:\n{tools_block}\n\n"
        f"{_TOOL_CALL_INSTRUCTION}\n\n"
        f"{user_query}"
    )
    return prompt


def build_valid_completion(tool_name, args):
    """Return the expected `<tool_call>…</tool_call>` completion string."""
    call_obj = {"name": tool_name, "arguments": args}
    return (
        "<tool_call>\n"
        + json.dumps(call_obj, ensure_ascii=False)
        + "\n</tool_call>"
    )


# Sample construction
def build_tool_call_sample(
    tool,
    all_tools,
    rng,
    min_tools=1,
    max_tools=3,
    is_valid=True,
    min_refusal_words=5,
):
    """
    Build one complete tool-call gym sample.

    Args:
        tool: Target tool for valid tasks; ignored for refusal tasks.
        all_tools: Full pool of available tools.
        rng: `random.Random` instance.
        min_tools: Minimum number of tools included in the prompt (1-3).
        max_tools: Maximum number of tools included in the prompt (1-3).
        is_valid: `True` → valid tool-call task; `False` → refusal task.
        min_refusal_words: Minimum word count expected in a refusal response.

    Returns:
        `{"id": str, "prompt": str, "verifier_id_list": list, "kwargs": list}`
    """
    if is_valid:
        # Sample arguments from the tool's own input_samples
        args = _sample_args_from_inputs(tool, rng)
        user_query = _build_user_request(tool, args, rng)

        # Select distractor tools to pad the tools block (1-3 total)
        num_extra = rng.randint(max(0, min_tools - 1), max(0, max_tools - 1))
        other_tools = [
            t for t in all_tools
            if t["function"]["name"] != tool["function"]["name"]
        ]
        extra = rng.sample(other_tools, min(num_extra, len(other_tools)))

        prompt, _ = build_valid_prompt(tool, user_query, rng, extra)

        params = tool["function"].get("parameters", {})
        required_keys = list(params.get("required", []))
        arg_types = get_tool_arg_types(tool)
        checked_types = {k: arg_types[k] for k in args if k in arg_types}

        verifier_ids = [
            "tool_call:format",
            "tool_call:name",
            "tool_call:args_keys",
            "tool_call:args_types",
        ]
        kwargs_list = [
            {"expect_call": True},
            {"expected_name": tool["function"]["name"]},
            {"required_arg_keys": required_keys},
            {"expected_arg_types": checked_types},
        ]

    else:
        num_distractors = rng.randint(min_tools, max_tools)
        distractors = rng.sample(all_tools, min(num_distractors, len(all_tools)))
        prompt = build_refusal_prompt(distractors, rng)

        verifier_ids = [
            "tool_call:format",
            "tool_call:refusal",
        ]
        kwargs_list = [
            {"expect_call": False},
            {"min_refusal_words": min_refusal_words},
        ]

    sample_id = hashlib.md5(prompt.encode()).hexdigest()
    return {
        "id": sample_id,
        "prompt": prompt,
        "verifier_id_list": verifier_ids,
        "kwargs": [json.dumps(kw, ensure_ascii=False) for kw in kwargs_list],
    }



# Validation
def validate_tool_call_sample(sample):
    """Return a list of validation issue strings (empty list = valid)."""
    issues = []

    n_ids = len(sample.get("verifier_id_list", []))
    n_kw = len(sample.get("kwargs", []))
    if n_ids != n_kw:
        issues.append(
            f"verifier_id_list length ({n_ids}) != kwargs length ({n_kw})"
        )

    for iid in sample.get("verifier_id_list", []):
        if iid not in TOOL_CALL_TASK_IDS:
            issues.append(f"Unknown tool-call task ID: {iid}")

    if not sample.get("prompt", "").strip():
        issues.append("Empty prompt")

    allowed_keys = {"id", "prompt", "verifier_id_list", "kwargs"}
    extra_keys = set(sample.keys()) - allowed_keys
    if extra_keys:
        issues.append(f"Unexpected keys in sample: {extra_keys}")

    return issues


def sample_fingerprint(sample):
    """Return a hashable fingerprint for deduplication (the prompt string)."""
    return sample.get("prompt", "")


# Main generation loop
def main(args):
    output_path = Path(args.output_file)
    data_path = args.data_file or str(_DATA_DIR / "tools.json")

    all_tools = load_tool_call_data(data_path)
    if not all_tools:
        print("Error: no tools loaded.")
        return

    num_samples = args.num_samples
    num_refusals = int(num_samples * args.refusal_ratio)
    num_valid = num_samples - num_refusals

    print(
        f"Loaded {len(all_tools)} tools.\n"
        f"Generating {num_samples} samples "
        f"({num_valid} valid + {num_refusals} refusal, seed={args.seed})"
    )

    samples = []
    seen = set()
    retries_used = 0
    max_retries = 20

    for i in range(num_valid):
        sample = None
        for attempt in range(max_retries):
            rng = random.Random(args.seed + i * max_retries + attempt)
            tool = rng.choice(all_tools)
            candidate = build_tool_call_sample(
                tool=tool,
                all_tools=all_tools,
                rng=rng,
                min_tools=args.min_tools,
                max_tools=args.max_tools,
                is_valid=True,
                min_refusal_words=args.min_refusal_words,
            )
            sid = candidate["id"]
            if sid not in seen:
                sample = candidate
                seen.add(sid)
                break
            retries_used += 1
        if sample is None:
            print(f"  Warning: could not produce unique valid sample #{i+1}")
            continue
        samples.append(sample)

    offset = num_valid * max_retries
    for i in range(num_refusals):
        sample = None
        for attempt in range(max_retries):
            rng = random.Random(args.seed + offset + i * max_retries + attempt)
            candidate = build_tool_call_sample(
                tool=None,
                all_tools=all_tools,
                rng=rng,
                min_tools=args.min_tools,
                max_tools=args.max_tools,
                is_valid=False,
                min_refusal_words=args.min_refusal_words,
            )
            sid = candidate["id"]
            if sid not in seen:
                sample = candidate
                seen.add(sid)
                break
            retries_used += 1
        if sample is None:
            print(f"  Warning: could not produce unique refusal sample #{i+1}")
            continue
        samples.append(sample)

    rng_shuffle = random.Random(args.seed)
    rng_shuffle.shuffle(samples)

    total_issues = 0
    for s in samples:
        issues = validate_tool_call_sample(s)
        if issues:
            total_issues += len(issues)
            if args.verbose:
                print(f"  ID {s['id']}: {issues}")

    n_valid = sum(
        1 for s in samples
        if any(json.loads(kw).get("expect_call") is True for kw in s["kwargs"])
    )
    n_refusal = len(samples) - n_valid
    unique_ids = len({s["id"] for s in samples})

    print(f"\nResults:")
    print(f"  Generated samples:  {len(samples)}")
    print(f"  Valid tool-calls:   {n_valid}")
    print(f"  Refusals:           {n_refusal}")
    print(
        f"  Unique:             {unique_ids}/{len(samples)}"
        f"  ({100 * unique_ids / max(len(samples), 1):.1f}%)"
    )
    print(f"  Validation issues:  {total_issues}")
    if retries_used:
        print(f"  Uniqueness retries: {retries_used}")

    assert unique_ids == len(samples), (
        f"FAIL: {len(samples) - unique_ids} duplicate samples"
    )
    assert total_issues == 0, f"FAIL: {total_issues} validation issues found"

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
        description="Procedural tool-calling task generator.",
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
        help="Total number of samples to generate (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--min_tools",
        type=int,
        default=1,
        help="Minimum number of tools in prompt (default: 1).",
    )
    parser.add_argument(
        "--max_tools",
        type=int,
        default=3,
        help="Maximum number of tools in prompt (default: 3).",
    )
    parser.add_argument(
        "--refusal_ratio",
        type=float,
        default=0.5,
        help="Fraction of samples that are refusal tasks (default: 0.5).",
    )
    parser.add_argument(
        "--min_refusal_words",
        type=int,
        default=5,
        help="Minimum words expected in refusal responses (default: 5).",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Path to tools.json (default: assets/tools.json).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed validation warnings.",
    )
    main(parser.parse_args())
