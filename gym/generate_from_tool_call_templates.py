"""
Procedural generator for tool-calling tasks.

Reads `tools.json` (tool definitions and examples)
and produces new gym-format samples covering:

  1. Valid tool-call tasks: model should produce `<tool_call>...</tool_call>`
  2. Refusal tasks: model should explain why no tool applies

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

_DATA_DIR = Path(__file__).resolve().parent / "data"


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

_TOOL_CALL_INSTRUCTION = (
    "Para cada chamada de função, retorne um objeto json com o nome da "
    "função e os argumentos dentro das tags XML <tool_call></tool_call>:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)


# User query templates for VALID tasks (tool is applicable)
#
# Each template group targets a tool category.  `{slots}` are filled
# from the tool's parameter descriptions at generation time.

_VALID_QUERY_TEMPLATES = {
    "calculate": [
        "Oi, preciso calcular {description_hint}. Pode me ajudar?",
        "Olá! Gostaria que você calculasse {description_hint} para mim.",
        "Preciso de ajuda com um cálculo: {description_hint}.",
        "Você poderia me ajudar a calcular {description_hint}?",
        "Pode fazer o cálculo de {description_hint}?",
    ],
    "get": [
        "Oi, você pode me dar informações sobre {description_hint}?",
        "Gostaria de saber sobre {description_hint}.",
        "Preciso de informações sobre {description_hint}. Pode me ajudar?",
        "Me diga sobre {description_hint}, por favor.",
        "Olá! Você poderia buscar {description_hint} para mim?",
    ],
    "search": [
        "Preciso pesquisar {description_hint}. Pode me ajudar?",
        "Gostaria de pesquisar {description_hint}.",
        "Pode buscar informações sobre {description_hint}?",
        "Me ajude a encontrar {description_hint}, por favor.",
        "Olá! Quero buscar {description_hint}.",
    ],
    "analyze": [
        "Preciso analisar {description_hint}. Pode me ajudar?",
        "Gostaria de fazer uma análise de {description_hint}.",
        "Você poderia analisar {description_hint} para mim?",
        "Preciso de uma análise sobre {description_hint}.",
        "Me ajude a analisar {description_hint}, por favor.",
    ],
    "create": [
        "Preciso criar {description_hint}. Pode me ajudar?",
        "Gostaria de gerar {description_hint}.",
        "Você poderia criar {description_hint} para mim?",
        "Me ajude a criar {description_hint}, por favor.",
        "Oi, preciso que você crie {description_hint}.",
    ],
    "convert": [
        "Preciso converter {description_hint}. Pode me ajudar?",
        "Gostaria de fazer uma conversão de {description_hint}.",
        "Você poderia converter {description_hint} para mim?",
        "Me ajude a converter {description_hint}, por favor.",
        "Oi, preciso de uma conversão: {description_hint}.",
    ],
    "generic": [
        "Oi, {description_hint}. Pode me ajudar?",
        "Olá! Preciso de ajuda com o seguinte: {description_hint}.",
        "Gostaria de {description_hint}.",
        "Você poderia me ajudar com {description_hint}?",
        "Preciso de ajuda: {description_hint}.",
        "Oi, estou precisando de {description_hint}. Pode me ajudar?",
        "Olá, preciso que você faça {description_hint} para mim.",
    ],
}


# User query templates for REFUSAL tasks (no tool applies)

_REFUSAL_QUERY_TEMPLATES = [
    "Você pode pedir uma pizza para mim, por favor?",
    "Pode me recomendar um bom restaurante?",
    "Gostaria que você marcasse uma consulta médica para mim.",
    "Pode ligar para meu amigo e dar um recado?",
    "Preciso que você compre passagens aéreas para mim.",
    "Pode escrever e enviar um e-mail para meu chefe?",
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
def load_tool_call_data(path):
    """Load the merged tool-call data file.

    Returns (tools, examples) where tools is a list of tool-schema dicts
    and examples is a list of example dicts.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    tools = [t for t in data["tools"] if t.get("function", {}).get("name")]
    examples = data.get("examples", [])
    return tools, examples


# Keep for backwards compat / tests
def load_tools(path):
    """Load tool definitions from a JSON file (legacy helper)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "tools" in data:
        items = data["tools"]
    else:
        items = data
    return [t for t in items if t.get("function", {}).get("name")]


def _tools_by_name(tools):
    """Build a name→tool dict for quick lookup."""
    return {t["function"]["name"]: t for t in tools}



# Argument generation helpers
_EXAMPLE_VALUES = {
    "string": [
        "exemplo", "teste", "São Paulo", "Rio de Janeiro",
        "Nova York", "hoje", "amanhã", "importante", "urgente",
        "relatório mensal", "João", "Maria", "2024-01-15",
    ],
    "number": [10, 25, 50, 100, 3.14, 9.99, 42, 0.5, 1000, 250],
    "integer": [1, 2, 3, 5, 10, 15, 20, 25, 50, 100],
    "boolean": [True, False],
    "array": [
        ["item1", "item2"],
        ["a", "b", "c"],
        ["primeiro", "segundo", "terceiro"],
    ],
}


def _generate_arg_value(param_schema, rng):
    """Generate a plausible argument value based on the JSON schema type."""
    ptype = param_schema.get("type", "string")

    if ptype == "string":
        desc = param_schema.get("description", "").lower()
        # Try to generate contextual values based on description
        if any(w in desc for w in ["data", "date", "nascimento"]):
            return f"{rng.randint(1980, 2005)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        if any(w in desc for w in ["nome", "name", "título", "title"]):
            names = ["João Silva", "Maria Santos", "O Poderoso Chefão",
                     "Ana Costa", "Pedro Lima", "Interstellar"]
            return rng.choice(names)
        if any(w in desc for w in ["local", "cidade", "location", "destino", "origem"]):
            places = ["São Paulo", "Rio de Janeiro", "Nova York",
                      "Londres", "Tokyo", "Lisboa", "Paris"]
            return rng.choice(places)
        if any(w in desc for w in ["moeda", "currency"]):
            currencies = ["USD", "BRL", "EUR", "GBP", "JPY"]
            return rng.choice(currencies)
        if any(w in desc for w in ["idioma", "language", "língua"]):
            langs = ["Português", "Inglês", "Espanhol", "Francês"]
            return rng.choice(langs)
        return rng.choice(_EXAMPLE_VALUES["string"])
    elif ptype in ("number", "integer"):
        return rng.choice(_EXAMPLE_VALUES.get(ptype, [42]))
    elif ptype == "boolean":
        return rng.choice([True, False])
    elif ptype == "array":
        return rng.choice(_EXAMPLE_VALUES["array"])
    elif ptype == "object":
        return {"chave": "valor"}
    return "valor_exemplo"


def generate_tool_arguments(tool, rng):
    """Generate argument values for all required + some optional params.

    Returns dict of argument name → value.
    """
    params = tool["function"].get("parameters", {})
    properties = params.get("properties", {})
    required = set(params.get("required", []))
    args = {}
    for name, schema in properties.items():
        if name in required or rng.random() < 0.5:
            args[name] = _generate_arg_value(schema, rng)
    # Always include at least required params
    for name in required:
        if name not in args:
            schema = properties.get(name, {"type": "string"})
            args[name] = _generate_arg_value(schema, rng)
    return args


def get_tool_arg_types(tool):
    """Extract expected JSON types for each parameter from the tool schema."""
    params = tool["function"].get("parameters", {})
    properties = params.get("properties", {})
    return {name: schema.get("type", "string") for name, schema in properties.items()}



# Prompt construction
def _format_tools_block(tools):
    """Serialize tools into the <tools>...</tools> XML block for the prompt."""
    lines = []
    for t in tools:
        lines.append(json.dumps(t, ensure_ascii=False))
    return "\n".join(lines)


def _categorize_tool(tool_name):
    """Return the best query-template category for a tool name."""
    name_lower = tool_name.lower()
    for cat in ("calculate", "get", "search", "analyze", "create", "convert"):
        if cat in name_lower:
            return cat
    return "generic"


def _build_description_hint(tool):
    """Create a natural-language hint from the tool's description and params."""
    desc = tool["function"].get("description", "")
    if desc:
        # Lower-case the first letter for natural embedding in a sentence
        hint = desc[0].lower() + desc[1:] if len(desc) > 1 else desc.lower()
        # Strip trailing period
        hint = hint.rstrip(".")
        return hint
    return tool["function"]["name"].replace("_", " ")


def build_valid_prompt(tool, rng, extra_tools=None):
    """Build a full prompt for a valid tool-call task.

    Args:
        tool: The target tool that should be called.
        rng: Random instance.
        extra_tools: Optional list of additional distractor tools.

    Returns:
        The complete prompt string.
    """
    preamble = rng.choice(_SYSTEM_PREAMBLES)

    # Build the tool set: target + optional distractors
    prompt_tools = [tool]
    if extra_tools:
        prompt_tools.extend(extra_tools)
    rng.shuffle(prompt_tools)

    tools_block = _format_tools_block(prompt_tools)
    category = _categorize_tool(tool["function"]["name"])
    templates = _VALID_QUERY_TEMPLATES.get(category, _VALID_QUERY_TEMPLATES["generic"])
    hint = _build_description_hint(tool)
    user_query = rng.choice(templates).format(description_hint=hint)

    prompt = (
        f"{preamble}\n\n"
        f"Você recebe assinaturas de funções dentro de tags XML "
        f"<tools></tools>:\n<tools>\n{tools_block}\n</tools>\n\n"
        f"{_TOOL_CALL_INSTRUCTION}\n\n"
        f"{user_query}"
    )
    return prompt, prompt_tools


def build_refusal_prompt(distractor_tools, rng):
    """Build a full prompt for a refusal task.

    The user asks something that no provided tool can handle.

    Args:
        distractor_tools: Tools to include (none will match the query).
        rng: Random instance.

    Returns:
        The complete prompt string.
    """
    preamble = rng.choice(_SYSTEM_PREAMBLES)
    tools_block = _format_tools_block(distractor_tools)
    user_query = rng.choice(_REFUSAL_QUERY_TEMPLATES)

    prompt = (
        f"{preamble}\n\n"
        f"Você recebe assinaturas de funções dentro de tags XML "
        f"<tools></tools>:\n<tools>\n{tools_block}\n</tools>\n\n"
        f"{_TOOL_CALL_INSTRUCTION}\n\n"
        f"{user_query}"
    )
    return prompt


def build_valid_completion(tool_name, args):
    """Build the expected completion for a valid tool call."""
    call_obj = {"name": tool_name, "arguments": args}
    return (
        "<tool_call>\n"
        + json.dumps(call_obj, ensure_ascii=False)
        + "\n</tool_call>"
    )



# Sample construction
def build_tool_call_sample(
    tool, all_tools, rng,
    min_tools=1, max_tools=3,
    is_valid=True, min_refusal_words=5,
):
    """Build one complete tool-call gym sample.

    Args:
        tool: The target tool (used for valid tasks; ignored for refusal).
        all_tools: Full pool of available tools.
        rng: `random.Random` instance.
        min_tools: Minimum tools to include in prompt.
        max_tools: Maximum tools to include in prompt.
        is_valid: If True, build a valid tool-call task; else a refusal task.
        min_refusal_words: Minimum words expected in a refusal response.

    Returns:
        Dict with keys: id, prompt, verifier_id_list, kwargs.
    """
    if is_valid:
        # Select distractor tools
        num_extra = rng.randint(max(0, min_tools - 1), max(0, max_tools - 1))
        other_tools = [t for t in all_tools if t["function"]["name"] != tool["function"]["name"]]
        extra = rng.sample(other_tools, min(num_extra, len(other_tools)))

        args = generate_tool_arguments(tool, rng)
        prompt, _prompt_tools = build_valid_prompt(tool, rng, extra)

        # Required arg keys from tool schema
        params = tool["function"].get("parameters", {})
        required_keys = list(params.get("required", []))
        arg_types = get_tool_arg_types(tool)
        # Only check types for keys we actually generated
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
        # Refusal task
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
    """Return a list of validation issues (empty = valid)."""
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

    # Must contain only the canonical keys
    allowed_keys = {"id", "prompt", "verifier_id_list", "kwargs"}
    extra_keys = set(sample.keys()) - allowed_keys
    if extra_keys:
        issues.append(f"Unexpected keys in sample: {extra_keys}")

    return issues


def sample_fingerprint(sample):
    """Hashable fingerprint for deduplication."""
    return sample.get("prompt", "")



# Main generation loop
def main(args):
    output_path = Path(args.output_file)

    data_path = args.data_file or str(_DATA_DIR / "tools.json")

    all_tools, examples = load_tool_call_data(data_path)

    if not all_tools:
        print("Error: no tools loaded.")
        return

    num_samples = args.num_samples
    num_refusals = int(num_samples * args.refusal_ratio)
    num_valid = num_samples - num_refusals

    print(
        f"Loaded {len(all_tools)} tools, {len(examples)} existing examples.\n"
        f"Generating {num_samples} samples "
        f"({num_valid} valid + {num_refusals} refusal, seed={args.seed})"
    )

    samples = []
    seen = set()
    retries_used = 0
    max_retries = 20

    # Generate valid samples
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

    # Generate refusal samples
    for i in range(num_refusals):
        sample = None
        offset = num_valid * max_retries
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

    # Shuffle to interleave valid and refusal
    rng_shuffle = random.Random(args.seed)
    rng_shuffle.shuffle(samples)

    # Validate all samples
    total_issues = 0
    for s in samples:
        issues = validate_tool_call_sample(s)
        if issues:
            total_issues += len(issues)
            if args.verbose:
                print(f"  ID {s['id']}: {issues}")

    n_valid = sum(
        1 for s in samples
        if any(
            json.loads(kw).get("expect_call") is True
            for kw in s["kwargs"]
        )
    )
    n_refusal = len(samples) - n_valid
    unique_ids = len({s['id'] for s in samples})

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

    # Hard assertions
    assert unique_ids == len(samples), (
        f"FAIL: {len(samples) - unique_ids} duplicate samples"
    )
    assert total_issues == 0, (
        f"FAIL: {total_issues} validation issues found"
    )

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
        help="Path to tools.json (default: data/tools.json).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed validation warnings.",
    )
    args = parser.parse_args()

    main(args)
