"""
Metadata for the procedural generation of instructions in the Portuguese gym environment. This file
contains conflict sets, kwargs templates, and description/kwargs generators.
"""

import random
import copy

# Verifier Category Prefixes
_KEYWORD = "keywords:"
_LANGUAGE = "language:"
_LENGTH = "length_constraints:"
_CONTENT = "detectable_content:"
_FORMAT = "detectable_format:"
_COMBINATION = "combination:"
_STARTEND = "startend:"
_CHANGE_CASES = "change_case:"
_PUNCTUATION = "punctuation:"

# All Verifier IDs
ALL_VERIFIER_IDS = [
    _KEYWORD + "existence",
    _KEYWORD + "frequency",
    _KEYWORD + "forbidden_words",
    _KEYWORD + "letter_frequency",
    _LANGUAGE + "response_language",
    _LENGTH + "number_sentences",
    _LENGTH + "number_paragraphs",
    _LENGTH + "number_words",
    _LENGTH + "nth_paragraph_first_word",
    _CONTENT + "number_placeholders",
    _CONTENT + "postscript",
    _FORMAT + "number_bullet_lists",
    _FORMAT + "constrained_response",
    _FORMAT + "number_highlighted_sections",
    _FORMAT + "multiple_sections",
    _FORMAT + "json_format",
    _FORMAT + "title",
    _COMBINATION + "two_responses",
    _COMBINATION + "repeat_prompt",
    _STARTEND + "end_checker",
    _CHANGE_CASES + "capital_word_frequency",
    _CHANGE_CASES + "portuguese_capital",
    _CHANGE_CASES + "portuguese_lowercase",
    _PUNCTUATION + "no_comma",
    _STARTEND + "quotation",
]

# Conflict Matrix
_RAW_CONFLICTS = {
    _KEYWORD + "existence": {_KEYWORD + "existence"},
    _KEYWORD + "frequency": {_KEYWORD + "frequency"},
    _KEYWORD + "forbidden_words": {_KEYWORD + "forbidden_words"},
    _KEYWORD + "letter_frequency": {_KEYWORD + "letter_frequency"},
    _LANGUAGE + "response_language": {
        _LANGUAGE + "response_language",
        _FORMAT + "multiple_sections",
        _KEYWORD + "existence",
        _KEYWORD + "frequency",
        _KEYWORD + "forbidden_words",
        _STARTEND + "end_checker",
        _CHANGE_CASES + "portuguese_capital",
        _CHANGE_CASES + "portuguese_lowercase",
    },
    _LENGTH + "number_sentences": {_LENGTH + "number_sentences"},
    _LENGTH + "number_paragraphs": {
        _LENGTH + "number_paragraphs",
        _LENGTH + "nth_paragraph_first_word",
        _LENGTH + "number_sentences",
        _LENGTH + "nth_paragraph_first_word",
    },
    _LENGTH + "number_words": {_LENGTH + "number_words"},
    _LENGTH + "nth_paragraph_first_word": {
        _LENGTH + "nth_paragraph_first_word",
        _LENGTH + "number_paragraphs",
    },
    _CONTENT + "number_placeholders": {_CONTENT + "number_placeholders"},
    _CONTENT + "postscript": {_CONTENT + "postscript"},
    _FORMAT + "number_bullet_lists": {_FORMAT + "number_bullet_lists"},
    _FORMAT + "constrained_response": set(ALL_VERIFIER_IDS),
    _FORMAT + "number_highlighted_sections": {
        _FORMAT + "number_highlighted_sections"
    },
    _FORMAT + "multiple_sections": {
        _FORMAT + "multiple_sections",
        _LANGUAGE + "response_language",
        _FORMAT + "number_highlighted_sections",
    },
    _FORMAT + "json_format": set(ALL_VERIFIER_IDS)
    - {_KEYWORD + "forbidden_words", _KEYWORD + "existence"},
    _FORMAT + "title": {_FORMAT + "title"},
    _COMBINATION + "two_responses": set(ALL_VERIFIER_IDS)
    - {
        _KEYWORD + "forbidden_words",
        _KEYWORD + "existence",
        _LANGUAGE + "response_language",
        _FORMAT + "title",
        _PUNCTUATION + "no_comma",
    },
    _COMBINATION + "repeat_prompt": set(ALL_VERIFIER_IDS)
    - {
        _KEYWORD + "existence",
        _FORMAT + "title",
        _PUNCTUATION + "no_comma",
    },
    _STARTEND + "end_checker": {_STARTEND + "end_checker"},
    _CHANGE_CASES + "capital_word_frequency": {
        _CHANGE_CASES + "capital_word_frequency",
        _CHANGE_CASES + "portuguese_lowercase",
        _CHANGE_CASES + "portuguese_capital",
    },
    _CHANGE_CASES + "portuguese_capital": {_CHANGE_CASES + "portuguese_capital"},
    _CHANGE_CASES + "portuguese_lowercase": {
        _CHANGE_CASES + "portuguese_lowercase",
        _CHANGE_CASES + "portuguese_capital",
    },
    _PUNCTUATION + "no_comma": {_PUNCTUATION + "no_comma"},
    _STARTEND + "quotation": {_STARTEND + "quotation", _FORMAT + "title"},
}


def conflict_make(conflicts):
    """Make conflicts symmetric: if A conflicts with B, B conflicts with A."""
    conflicts = {k: set(v) for k, v in conflicts.items()}
    for key in list(conflicts.keys()):
        for k in list(conflicts[key]):
            if k not in conflicts:
                conflicts[k] = set()
            conflicts[k].add(key)
        conflicts[key].add(key)
    return conflicts


# Build the symmetric conflict matrix once at import time
VERIFIER_CONFLICTS = conflict_make(_RAW_CONFLICTS)

# Empty Kwargs Template
EMPTY_KWARGS_TEMPLATE = {
    "capital_frequency": None,
    "capital_relation": None,
    "end_phrase": None,
    "first_word": None,
    "forbidden_words": None,
    "frequency": None,
    "keyword": None,
    "keywords": None,
    "language": None,
    "let_frequency": None,
    "let_relation": None,
    "letter": None,
    "nth_paragraph": None,
    "num_bullets": None,
    "num_highlights": None,
    "num_paragraphs": None,
    "num_placeholders": None,
    "num_sections": None,
    "num_sentences": None,
    "num_words": None,
    "postscript_marker": None,
    "prompt_to_repeat": None,
    "relation": None,
    "section_spliter": None,
}

# Word Banks
KEYWORDS_PT = [
    "fundamental",
    "consequência",
    "perspectiva",
    "inovação",
    "sustentável",
    "paradigma",
    "contexto",
    "relevante",
    "impacto",
    "estratégia",
    "primordial",
    "essencial",
    "significativo",
    "correlação",
    "evidência",
    "tecnologia",
    "diversidade",
    "criatividade",
    "transformação",
    "evolução",
    "desenvolvimento",
    "sociedade",
    "comunicação",
    "aprendizado",
    "experiência",
    "oportunidade",
    "desafio",
    "progresso",
    "qualidade",
    "eficiência",
]

FORBIDDEN_WORDS_PT = [
    "entretanto",
    "porém",
    "contudo",
    "todavia",
    "obviamente",
    "claramente",
    "simplesmente",
    "basicamente",
    "literalmente",
    "apenas",
    "absolutamente",
    "definitivamente",
    "certamente",
    "provavelmente",
    "possivelmente",
]

ENDING_OPTIONS_PT = [
    "Isso faz sentido?",
    "Há algo mais que eu possa ajudar?",
    "Há algo mais em que eu possa ajudar?",
    "Posso ajudar com algo mais?",
    "Espero que isso ajude!",
]

POSTSCRIPT_MARKERS = ["P.S.", "P.P.S"]
SECTION_SPLITTERS = ["Seção", "SEÇÃO"]

FIRST_WORDS_PT = [
    "primeiramente",
    "inicialmente",
    "considerando",
    "antes",
    "certamente",
    "naturalmente",
    "basicamente",
    "atualmente",
    "historicamente",
    "geralmente",
    "normalmente",
    "recentemente",
]

LANGUAGE_CODES = {
    "pt": "Português",
    "en": "Inglês",
    "es": "Espanhol",
    "fr": "Francês",
    "de": "Alemão",
    "it": "Italiano",
}

COMPARISON_RELATIONS = ("less than", "at least")

LETTERS_PT = list("aeiorlsnpqmtbc")


# Conflict Resolution
def get_conflict_set(verifier_ids):
    """Get all verifiers that conflict with any of the given verifier IDs."""
    conflicting = set()
    for vid in verifier_ids:
        if vid in VERIFIER_CONFLICTS:
            conflicting |= VERIFIER_CONFLICTS[vid]
    return conflicting


def get_addable_verifiers(current_ids):
    """Get verifier IDs that can be added without creating conflicts."""
    conflicting = get_conflict_set(current_ids)
    all_ids = set(ALL_VERIFIER_IDS)
    return sorted(all_ids - conflicting)


def is_combination_valid(verifier_ids):
    """Check if a list of verifier IDs has no internal conflicts."""
    for i, vid in enumerate(verifier_ids):
        conflicts = VERIFIER_CONFLICTS.get(vid, set())
        for j, other_id in enumerate(verifier_ids):
            if i != j and other_id in conflicts:
                return False
    return True


# Kwargs Generation
def make_empty_kwargs():
    """Return a kwargs dict with all keys set to None."""
    return copy.deepcopy(EMPTY_KWARGS_TEMPLATE)


def generate_kwargs_for_verifier(verifier_id, prompt_text=""):
    """Generate random kwargs for a specific verifier type."""
    kw = make_empty_kwargs()

    if verifier_id == _KEYWORD + "existence":
        kw["keywords"] = sorted(random.sample(KEYWORDS_PT, k=random.randint(1, 3)))

    elif verifier_id == _KEYWORD + "frequency":
        kw["keyword"] = random.choice(KEYWORDS_PT)
        kw["frequency"] = random.randint(1, 3)
        kw["relation"] = random.choice(COMPARISON_RELATIONS)

    elif verifier_id == _KEYWORD + "forbidden_words":
        kw["forbidden_words"] = sorted(
            random.sample(FORBIDDEN_WORDS_PT, k=random.randint(1, 3))
        )

    elif verifier_id == _KEYWORD + "letter_frequency":
        kw["letter"] = random.choice(LETTERS_PT)
        kw["let_frequency"] = random.randint(3, 10)
        kw["let_relation"] = random.choice(COMPARISON_RELATIONS)

    elif verifier_id == _LANGUAGE + "response_language":
        kw["language"] = random.choice(list(LANGUAGE_CODES.keys()))

    elif verifier_id == _LENGTH + "number_sentences":
        kw["num_sentences"] = random.randint(3, 20)
        kw["relation"] = random.choice(COMPARISON_RELATIONS)

    elif verifier_id == _LENGTH + "number_paragraphs":
        kw["num_paragraphs"] = random.randint(2, 5)

    elif verifier_id == _LENGTH + "number_words":
        kw["num_words"] = random.choice([50, 100, 150, 200, 250, 300, 400, 500])
        kw["relation"] = random.choice(COMPARISON_RELATIONS)

    elif verifier_id == _LENGTH + "nth_paragraph_first_word":
        n_para = random.randint(2, 5)
        kw["num_paragraphs"] = n_para
        kw["nth_paragraph"] = random.randint(1, n_para)
        kw["first_word"] = random.choice(FIRST_WORDS_PT)

    elif verifier_id == _CONTENT + "number_placeholders":
        kw["num_placeholders"] = random.randint(2, 8)

    elif verifier_id == _CONTENT + "postscript":
        kw["postscript_marker"] = random.choice(POSTSCRIPT_MARKERS)

    elif verifier_id == _FORMAT + "number_bullet_lists":
        kw["num_bullets"] = random.randint(2, 8)

    elif verifier_id == _FORMAT + "number_highlighted_sections":
        kw["num_highlights"] = random.randint(2, 5)

    elif verifier_id == _FORMAT + "multiple_sections":
        kw["section_spliter"] = random.choice(SECTION_SPLITTERS)
        kw["num_sections"] = random.randint(2, 5)

    elif verifier_id == _STARTEND + "end_checker":
        kw["end_phrase"] = random.choice(ENDING_OPTIONS_PT)

    elif verifier_id == _CHANGE_CASES + "capital_word_frequency":
        kw["capital_frequency"] = random.randint(3, 20)
        kw["capital_relation"] = random.choice(COMPARISON_RELATIONS)

    elif verifier_id == _COMBINATION + "repeat_prompt":
        kw["prompt_to_repeat"] = prompt_text

    # These instructions have no kwargs:
    # constrained_response, json_format, title, two_responses,
    # portuguese_capital, portuguese_lowercase, no_comma, quotation

    return kw


# Description Generation
def _relation_pt(relation):
    """Convert English relation string to Portuguese."""
    if relation == "at least":
        return "pelo menos"
    return "menos de"


def generate_description_for_verifier(verifier_id, kwargs):
    """Generate Portuguese description text for a verifier with given kwargs."""

    if verifier_id == _KEYWORD + "existence":
        keywords = kwargs.get("keywords") or ["exemplo"]
        return f"Inclua as palavras-chave {keywords} na resposta."

    elif verifier_id == _KEYWORD + "frequency":
        keyword = kwargs.get("keyword") or "exemplo"
        frequency = int(kwargs.get("frequency") or 2)
        relation = kwargs.get("relation") or "at least"
        return (
            f"Na sua resposta, a palavra {keyword} deve aparecer"
            f" {_relation_pt(relation)} {frequency} vezes."
        )

    elif verifier_id == _KEYWORD + "forbidden_words":
        words = kwargs.get("forbidden_words") or ["exemplo"]
        return f"Não inclua as palavras-chave {words} na resposta."

    elif verifier_id == _KEYWORD + "letter_frequency":
        letter = kwargs.get("letter") or "a"
        freq = int(kwargs.get("let_frequency") or 5)
        rel = kwargs.get("let_relation") or "at least"
        return (
            f"Em sua resposta, a letra {letter} deve aparecer"
            f" {_relation_pt(rel)} {freq} vezes."
        )

    elif verifier_id == _LANGUAGE + "response_language":
        lang = kwargs.get("language") or "pt"
        lang_name = LANGUAGE_CODES.get(lang, lang)
        return (
            f"Toda a sua resposta deve estar em {lang_name},"
            " nenhuma outra linguagem é permitida."
        )

    elif verifier_id == _LENGTH + "number_sentences":
        num = int(kwargs.get("num_sentences") or 5)
        relation = kwargs.get("relation") or "at least"
        return f"Sua resposta deve conter {_relation_pt(relation)} {num} sentenças."

    elif verifier_id == _LENGTH + "number_paragraphs":
        num = int(kwargs.get("num_paragraphs") or 3)
        return (
            f"Sua resposta deve ter {num} parágrafos."
            " Os parágrafos são separados pelo divisor markdown: ***"
        )

    elif verifier_id == _LENGTH + "number_words":
        num = int(kwargs.get("num_words") or 200)
        relation = kwargs.get("relation") or "at least"
        return f"Responda com {_relation_pt(relation)} {num} palavras."

    elif verifier_id == _LENGTH + "nth_paragraph_first_word":
        n_para = int(kwargs.get("num_paragraphs") or 3)
        nth = int(kwargs.get("nth_paragraph") or 1)
        word = kwargs.get("first_word") or "primeiramente"
        return (
            f"Deve haver {n_para} parágrafos."
            f" O parágrafo {nth} deve começar com a palavra {word}."
        )

    elif verifier_id == _CONTENT + "number_placeholders":
        num = int(kwargs.get("num_placeholders") or 3)
        return (
            f"A resposta deve conter pelo menos {num} espaços reservados"
            " representados por colchetes, como [endereço]."
        )

    elif verifier_id == _CONTENT + "postscript":
        marker = kwargs.get("postscript_marker") or "P.S."
        return (
            "No final da sua resposta, por favor adicione explicitamente"
            f" um posfácio começando com {marker}"
        )

    elif verifier_id == _FORMAT + "number_bullet_lists":
        num = int(kwargs.get("num_bullets") or 3)
        return (
            f"Sua resposta deve conter exatamente {num} itens."
            " Use os marcadores markdown, como:\n"
            "* Este é o ponto 1.\n* Este é o ponto 2"
        )

    elif verifier_id == _FORMAT + "constrained_response":
        return (
            "Responda com uma das seguintes opções: Minha resposta é sim.,"
            " Minha resposta é não., Minha resposta é talvez."
        )

    elif verifier_id == _FORMAT + "number_highlighted_sections":
        num = int(kwargs.get("num_highlights") or 3)
        return (
            f"Destaque pelo menos {num} seções em sua resposta com"
            " markdown, ou seja, *seção destacada*."
        )

    elif verifier_id == _FORMAT + "multiple_sections":
        spliter = kwargs.get("section_spliter") or "Seção"
        num = int(kwargs.get("num_sections") or 3)
        return (
            f"Sua resposta deve ter {num} seções."
            f" Marque o início de cada seção com {spliter} X."
        )

    elif verifier_id == _FORMAT + "json_format":
        return (
            "Todo o output deve estar em formato JSON."
            " Você pode usar marcadores markdown como ```."
        )

    elif verifier_id == _FORMAT + "title":
        return (
            "Sua resposta deve conter um título, envolto em duplas setas"
            " angulares, como <<poema de alegria>>."
        )

    elif verifier_id == _COMBINATION + "two_responses":
        return (
            "Dê duas respostas diferentes. As respostas e somente as respostas"
            " devem ser separadas por 6 símbolos de asterisco: ******."
        )

    elif verifier_id == _COMBINATION + "repeat_prompt":
        return (
            "Primeiro repita o pedido palavra por palavra sem alterações,"
            " depois dê sua resposta (1. não diga nenhuma palavra ou caractere"
            " antes de repetir o pedido; 2. o pedido que você precisa repetir"
            " não inclui esta frase)"
        )

    elif verifier_id == _STARTEND + "end_checker":
        phrase = kwargs.get("end_phrase") or "Há algo mais que eu possa ajudar?"
        return (
            f"Termine sua resposta com esta frase exata {phrase}."
            " Nenhuma outra palavra deve seguir esta frase."
        )

    elif verifier_id == _CHANGE_CASES + "capital_word_frequency":
        freq = int(kwargs.get("capital_frequency") or 10)
        rel = kwargs.get("capital_relation") or "less than"
        return (
            "Em sua resposta, palavras com todas as letras maiúsculas devem"
            f" aparecer {_relation_pt(rel)} {freq} vezes."
        )

    elif verifier_id == _CHANGE_CASES + "portuguese_capital":
        return (
            "Sua resposta inteira deve estar em português e em todas"
            " as letras maiúsculas."
        )

    elif verifier_id == _CHANGE_CASES + "portuguese_lowercase":
        return (
            "Sua resposta inteira deve estar em português e em todas"
            " as letras minúsculas. Nenhuma letra maiúscula é permitida."
        )

    elif verifier_id == _PUNCTUATION + "no_comma":
        return "Em sua resposta, evite o uso de vírgulas."

    elif verifier_id == _STARTEND + "quotation":
        return "Envolva toda a sua resposta com aspas duplas."

    return ""


# Verifiers that are straightforward to add (no complex setup needed)
# and that work well as additional constraints
SAFE_ADDABLE_VERIFIERS = [
    _PUNCTUATION + "no_comma",
    _STARTEND + "quotation",
    _FORMAT + "title",
    _CONTENT + "postscript",
    _STARTEND + "end_checker",
    _LENGTH + "number_words",
    _LENGTH + "number_sentences",
    _KEYWORD + "existence",
    _KEYWORD + "forbidden_words",
    _KEYWORD + "letter_frequency",
    _FORMAT + "number_highlighted_sections",
    _CONTENT + "number_placeholders",
    _CHANGE_CASES + "portuguese_lowercase",
    _CHANGE_CASES + "portuguese_capital",
    _FORMAT + "number_bullet_lists",
    _CHANGE_CASES + "capital_word_frequency",
]


# Long Context Retrieval Tasks
_LONG_CONTEXT = "long_context:"

LONG_CONTEXT_TASK_IDS = [
    _LONG_CONTEXT + "common_words",
    _LONG_CONTEXT + "rare_words",
    _LONG_CONTEXT + "count_word",
    _LONG_CONTEXT + "word_at_position",
    _LONG_CONTEXT + "frequency_comparison",
]

LONG_CONTEXT_DEFAULTS = {
    "top_k_range": (5, 10),
    "common_repeats_range": (15, 40),
    "uncommon_repeats_range": (1, 5),
    "common_nums_range": (5, 15),
    "tokens_to_generate": {
        "common_words": 250,
        "rare_words": 250,
        "count_word": 50,
        "word_at_position": 50,
        "frequency_comparison": 100,
    },
}


def generate_long_context_task_description(task_id):
    """Return a brief Portuguese description for a long context task ID."""
    descriptions = {
        _LONG_CONTEXT + "common_words":
            "Identificar as palavras mais frequentes em uma lista longa.",
        _LONG_CONTEXT + "rare_words":
            "Identificar as palavras menos frequentes em uma lista longa.",
        _LONG_CONTEXT + "count_word":
            "Contar o número de ocorrências de uma palavra específica.",
        _LONG_CONTEXT + "word_at_position":
            "Identificar a palavra em uma posição específica da lista.",
        _LONG_CONTEXT + "frequency_comparison":
            "Comparar a frequência de duas palavras na lista.",
    }
    return descriptions.get(task_id, "")


# Haystack Retrieval Tasks
_HAYSTACK = "haystack:"

HAYSTACK_TASK_IDS = [
    _HAYSTACK + "needle_single_number",
    _HAYSTACK + "needle_multi_number_same_key",
    _HAYSTACK + "needle_multi_number_diff_keys",
    _HAYSTACK + "needle_uuid",
]

HAYSTACK_DEFAULTS = {
    "num_needles_range": (2, 6),
    "num_keys_range": (2, 6),
    "num_uuid_pairs_range": (8, 20),
    "tokens_to_generate": {
        "needle_single_number": 50,
        "needle_multi_number_same_key": 200,
        "needle_multi_number_diff_keys": 200,
        "needle_uuid": 80,
    },
}


def generate_haystack_task_description(task_id):
    """Return a brief Portuguese description for a haystack task ID."""
    descriptions = {
        _HAYSTACK + "needle_single_number":
            "Encontrar um número especial escondido em um documento longo.",
        _HAYSTACK + "needle_multi_number_same_key":
            "Encontrar múltiplos números especiais para a mesma chave em um documento.",
        _HAYSTACK + "needle_multi_number_diff_keys":
            "Encontrar números especiais para chaves diferentes em um documento.",
        _HAYSTACK + "needle_uuid":
            "Encontrar o código UUID associado a uma chave específica.",
    }
    return descriptions.get(task_id, "")


# Math Tasks
_MATH = "math:"

MATH_TASK_IDS = [
    _MATH + "answer_check",
]


def generate_math_task_description(task_id):
    """Return a brief Portuguese description for a math task ID."""
    descriptions = {
        _MATH + "answer_check":
            "Resolver um problema matemático e fornecer a resposta numérica correta.",
    }
    return descriptions.get(task_id, "")


# Email Extraction Tasks
_EMAIL = "email:"

# Fields extractable directly from email content (no injection needed)
EMAIL_DIRECT_FIELDS = [
    "subject",          # Assunto do e-mail
    "sender",           # Nome do remetente
    "receiver",         # Nome do destinatário
    "intent",           # Intenção/propósito do e-mail
    "summary",          # Resumo breve do conteúdo
]

# Fields synthetically injected into the email context at generation time,
# enabling deterministic verification via exact-match.
EMAIL_INJECTED_FIELDS = [
    "date",             # Data/hora de recebimento (ISO 8601)
    "attachments",      # Presença de anexos (booleano)
    "spam",             # Classificação de spam (booleano)
    "sender_email",     # Endereço de e-mail do remetente
    "telephone_number", # Número de telefone mencionado
]

EMAIL_ALL_FIELDS = EMAIL_DIRECT_FIELDS + EMAIL_INJECTED_FIELDS

# Portuguese field-description labels used inside prompts
EMAIL_FIELD_LABELS = {
    "subject":          "subject (assunto do e-mail)",
    "sender":           "sender (nome do remetente)",
    "receiver":         "receiver (nome do destinatário)",
    "intent":           "intent (intenção/propósito principal do e-mail)",
    "summary":          "summary (resumo breve do conteúdo)",
    "date":             "date (data de recebimento conforme cabeçalho, formato ISO 8601)",
    "attachments":      "attachments (true se há anexos, false caso contrário)",
    "spam":             "spam (true se é spam, false caso contrário)",
    "sender_email":     "sender_email (endereço de e-mail do remetente)",
    "telephone_number": "telephone_number (número de telefone mencionado no e-mail)",
}

EMAIL_TASK_IDS = [
    _EMAIL + "json_format",
    _EMAIL + "schema_keys",
    _EMAIL + "field_value",
]

EMAIL_DEFAULTS = {
    "min_fields": 3,
    "max_fields": 7,
    "tokens_to_generate": 300,
}


def generate_email_task_description(task_id):
    """Return a brief Portuguese description for an email extraction task ID."""
    descriptions = {
        _EMAIL + "json_format":
            "Formatar a resposta como objeto JSON válido dentro de bloco markdown.",
        _EMAIL + "schema_keys":
            "O JSON deve conter exatamente as chaves solicitadas.",
        _EMAIL + "field_value":
            "Um campo específico deve ter o valor exato esperado.",
    }
    return descriptions.get(task_id, "")
