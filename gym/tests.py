# %%
#######################################
# 1. Imports & Setup
#######################################
import random
import string as _string

from tasks_metadata import (
    ALL_VERIFIER_IDS,
    VERIFIER_CONFLICTS,
    EMPTY_KWARGS_TEMPLATE,
    get_addable_verifiers,
    is_combination_valid,
    make_empty_kwargs,
    generate_kwargs_for_verifier,
    generate_description_for_verifier,
    LONG_CONTEXT_TASK_IDS,
    HAYSTACK_TASK_IDS,
)
from instruction_templates import TEMPLATES
from verifier import Verifier, VERIFICATION_REGISTRY
from generate_from_instruction_templates import (
    fill_template,
    select_modifier_ids,
    build_sample,
    validate_sample,
    sample_fingerprint,
    MODIFIER_IDS,
)

print("All imports OK ✓")

# %%
#######################################
# 2. Multi-constraint verifier — pass and partial failure
#    (covers: title, no_comma, end_checker, keywords, letter_frequency)
#######################################
# All five pass
v = Verifier(
    verifier_id_list=[
        "detectable_format:title",
        "punctuation:no_comma",
        "startend:end_checker",
        "keywords:existence",
        "keywords:letter_frequency",
    ],
    kwargs=[
        {},
        {},
        {"end_phrase": "Obrigado!"},
        {"keywords": ["inovação", "estratégia"]},
        {"letter": "a", "let_frequency": 3, "let_relation": "at least"},
    ],
    completion=(
        "<<Inovação e Estratégia>>\n"
        "A inovação é a base de toda estratégia empresarial. Obrigado!"
    ),
)
results = v.verify()
assert results == [True, True, True, True, True], f"Expected all True, got {results}"

# Title missing + comma present → first two fail, rest pass
v2 = Verifier(
    verifier_id_list=[
        "detectable_format:title",
        "punctuation:no_comma",
        "startend:end_checker",
    ],
    kwargs=[
        {},
        {},
        {"end_phrase": "Obrigado!"},
    ],
    completion="Resposta sem título, com vírgula. Obrigado!",
)
r2 = v2.verify()
assert r2 == [False, False, True], f"Expected [False, False, True], got {r2}"
print("Test 2 — multi-constraint pass + partial failure: OK ✓")

# %%
#######################################
# 3. Keywords + forbidden words + frequency (pass & fail)
#######################################
v = Verifier(
    verifier_id_list=[
        "keywords:existence",
        "keywords:forbidden_words",
        "keywords:frequency",
    ],
    kwargs=[
        {"keywords": ["inovação", "estratégia"]},
        {"forbidden_words": ["entretanto", "porém"]},
        {"keyword": "teste", "frequency": 3, "relation": "at least"},
    ],
    completion="A inovação e estratégia exigem teste teste teste de qualidade.",
)
assert v.verify() == [True, True, True]

# Keywords missing + forbidden word present + frequency too low
v2 = Verifier(
    verifier_id_list=[
        "keywords:existence",
        "keywords:forbidden_words",
        "keywords:frequency",
    ],
    kwargs=[
        {"keywords": ["inovação", "estratégia"]},
        {"forbidden_words": ["entretanto", "porém"]},
        {"keyword": "teste", "frequency": 3, "relation": "at least"},
    ],
    completion="Entretanto nada de especial aqui.",
)
assert v2.verify() == [False, False, False]
print("Test 3 — keywords + forbidden + frequency: OK ✓")

# %%
#######################################
# 4. Length constraints + detectable content
#    (sentences, paragraphs, words, placeholders, postscript)
#######################################
v = Verifier(
    verifier_id_list=[
        "length_constraints:number_sentences",
        "length_constraints:number_paragraphs",
        "length_constraints:number_words",
        "detectable_content:number_placeholders",
        "detectable_content:postscript",
    ],
    kwargs=[
        {"num_sentences": 3, "relation": "at least"},
        {"num_paragraphs": 2},
        {"num_words": 5, "relation": "at least"},
        {"num_placeholders": 2},
        {"postscript_marker": "P.S."},
    ],
    completion=(
        "Envie para [endereço] no dia [data]. "
        "Segunda frase aqui. Terceira frase aqui.\n"
        "***\n"
        "Segundo parágrafo com mais texto.\n"
        "P.S. Não esqueça."
    ),
)
assert v.verify() == [True, True, True, True, True]

# All fail: one sentence, no separator, few words, no placeholders, no P.S.
v2 = Verifier(
    verifier_id_list=[
        "length_constraints:number_sentences",
        "length_constraints:number_paragraphs",
        "length_constraints:number_words",
        "detectable_content:number_placeholders",
        "detectable_content:postscript",
    ],
    kwargs=[
        {"num_sentences": 5, "relation": "at least"},
        {"num_paragraphs": 3},
        {"num_words": 100, "relation": "at least"},
        {"num_placeholders": 3},
        {"postscript_marker": "P.S."},
    ],
    completion="Curto.",
)
assert v2.verify() == [False, False, False, False, False]
print("Test 4 — length constraints + detectable content: OK ✓")

# %%
#######################################
# 5. Detectable format verifiers
#    (bullets, constrained_response, highlights, sections, json)
#######################################
# Bullet list pass & fail
v = Verifier(
    verifier_id_list=["detectable_format:number_bullet_lists"],
    kwargs=[{"num_bullets": 3}],
    completion="Pontos:\n* Ponto 1\n* Ponto 2\n* Ponto 3",
)
assert v.verify() == [True]
v2 = Verifier(
    verifier_id_list=["detectable_format:number_bullet_lists"],
    kwargs=[{"num_bullets": 3}],
    completion="* Ponto 1\n* Ponto 2",
)
assert v2.verify() == [False]

# Constrained response pass & fail
v3 = Verifier(
    verifier_id_list=["detectable_format:constrained_response"],
    kwargs=[{}],
    completion="Minha resposta é sim.",
)
assert v3.verify() == [True]
v4 = Verifier(
    verifier_id_list=["detectable_format:constrained_response"],
    kwargs=[{}],
    completion="Não sei o que dizer.",
)
assert v4.verify() == [False]

# Highlighted sections
v5 = Verifier(
    verifier_id_list=["detectable_format:number_highlighted_sections"],
    kwargs=[{"num_highlights": 2}],
    completion="Observe o *primeiro destaque* e o *segundo destaque* nesta resposta.",
)
assert v5.verify() == [True]
v6 = Verifier(
    verifier_id_list=["detectable_format:number_highlighted_sections"],
    kwargs=[{"num_highlights": 5}],
    completion="Sem destaques aqui.",
)
assert v6.verify() == [False]

# Multiple sections
v7 = Verifier(
    verifier_id_list=["detectable_format:multiple_sections"],
    kwargs=[{"section_spliter": "Seção", "num_sections": 2}],
    completion="Seção 1\nConteúdo.\nSeção 2\nMais conteúdo.",
)
assert v7.verify() == [True]

# JSON format
v8 = Verifier(
    verifier_id_list=["detectable_format:json_format"],
    kwargs=[{}],
    completion='```json\n{"chave": "valor"}\n```',
)
assert v8.verify() == [True]
v9 = Verifier(
    verifier_id_list=["detectable_format:json_format"],
    kwargs=[{}],
    completion="Isso não é JSON.",
)
assert v9.verify() == [False]
print("Test 5 — detectable format verifiers: OK ✓")

# %%
#######################################
# 6. Combination + startend verifiers
#    (two_responses, repeat_prompt, quotation)
#######################################
# Two responses
v = Verifier(
    verifier_id_list=["combination:two_responses"],
    kwargs=[{}],
    completion="Primeira resposta.******Segunda resposta diferente.",
)
assert v.verify() == [True]
v2 = Verifier(
    verifier_id_list=["combination:two_responses"],
    kwargs=[{}],
    completion="Resposta única sem separador.",
)
assert v2.verify() == [False]

# Repeat prompt
prompt = "Escreva uma carta para meu amigo."
v3 = Verifier(
    verifier_id_list=["combination:repeat_prompt"],
    kwargs=[{"prompt_to_repeat": prompt}],
    completion=prompt + "\n\nQuerido amigo, espero que esteja bem.",
)
assert v3.verify() == [True]
v4 = Verifier(
    verifier_id_list=["combination:repeat_prompt"],
    kwargs=[{"prompt_to_repeat": prompt}],
    completion="Querido amigo, espero que esteja bem.",
)
assert v4.verify() == [False]

# Quotation
v5 = Verifier(
    verifier_id_list=["startend:quotation"],
    kwargs=[{}],
    completion='"Esta resposta está entre aspas."',
)
assert v5.verify() == [True]
v6 = Verifier(
    verifier_id_list=["startend:quotation"],
    kwargs=[{}],
    completion="Esta resposta não está entre aspas.",
)
assert v6.verify() == [False]
print("Test 6 — combination + startend verifiers: OK ✓")

# %%
#######################################
# 7. Unknown verifier ID raises error
#######################################
try:
    v = Verifier(
        verifier_id_list=["fake_category:nonexistent"],
        kwargs=[{}],
        completion="Teste.",
    )
    v.verify()
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "Unknown verifier ID" in str(e)
print("Test 7 — unknown verifier raises error: OK ✓")

# %%
#######################################
# 8. Metadata integrity — registry, conflict symmetry, self-conflict
#######################################
# All verifier IDs exist in registry
for vid in ALL_VERIFIER_IDS:
    assert vid in VERIFICATION_REGISTRY, f"Missing registry entry for {vid}"

# Conflict matrix is symmetric
for vid_a, conflicts in VERIFIER_CONFLICTS.items():
    for vid_b in conflicts:
        assert vid_a in VERIFIER_CONFLICTS.get(vid_b, set()), (
            f"Conflict asymmetry: {vid_a} conflicts with {vid_b} but not vice-versa"
        )

# Every verifier self-conflicts
for vid in ALL_VERIFIER_IDS:
    assert vid in VERIFIER_CONFLICTS.get(vid, set()), (
        f"{vid} should self-conflict"
    )

# Long context + haystack IDs also in registry
for tid in LONG_CONTEXT_TASK_IDS:
    assert tid in VERIFICATION_REGISTRY, f"Missing registry entry: {tid}"
for tid in HAYSTACK_TASK_IDS:
    assert tid in VERIFICATION_REGISTRY, f"Missing registry entry: {tid}"
print("Test 8 — metadata integrity (registry + conflicts): OK ✓")

# %%
#######################################
# 9. Metadata helpers — is_combination_valid, get_addable, make_empty_kwargs
#######################################
assert is_combination_valid(["detectable_format:title", "punctuation:no_comma"])
assert not is_combination_valid([
    "detectable_format:constrained_response",
    "detectable_format:title",
])
assert is_combination_valid([])
assert is_combination_valid(["punctuation:no_comma"])

addable = get_addable_verifiers(["detectable_format:constrained_response"])
assert len(addable) == 0, f"Expected empty, got {addable}"
addable2 = get_addable_verifiers([])
assert set(addable2) == set(ALL_VERIFIER_IDS)

kw = make_empty_kwargs()
assert all(val is None for val in kw.values()), "All empty kwargs should be None"
kw["language"] = "pt"
kw2 = make_empty_kwargs()
assert kw2["language"] is None, "make_empty_kwargs should return independent copies"
print("Test 9 — metadata helpers: OK ✓")

# %%
#######################################
# 10. Generation pipeline — kwargs, descriptions, templates, fill
#######################################
random.seed(42)
# generate_kwargs_for_verifier + generate_description_for_verifier for all IDs
for vid in ALL_VERIFIER_IDS:
    kw = generate_kwargs_for_verifier(vid, prompt_text="Texto de teste.")
    assert isinstance(kw, dict), f"Expected dict for {vid}"
    assert set(kw.keys()) == set(EMPTY_KWARGS_TEMPLATE.keys()), (
        f"Kwargs keys mismatch for {vid}"
    )
    desc = generate_description_for_verifier(vid, kw)
    assert isinstance(desc, str) and len(desc) > 0, (
        f"Description for {vid} should be a non-empty string"
    )

# Templates structure
assert len(TEMPLATES) > 0, "TEMPLATES should not be empty"
for t in TEMPLATES:
    assert "id" in t, "Template missing 'id'"
    assert "prompts" in t and len(t["prompts"]) > 0, "Template missing 'prompts'"
    assert "slots" in t, "Template missing 'slots'"
    for prompt_fmt in t["prompts"]:
        field_names = [
            fname for _, fname, _, _ in _string.Formatter().parse(prompt_fmt)
            if fname is not None
        ]
        for fname in field_names:
            assert fname in t["slots"], (
                f"Template {t['id']}: slot '{fname}' in prompt but not in slots dict"
            )

# fill_template
for t in TEMPLATES:
    filled = fill_template(t)
    assert isinstance(filled, str) and len(filled) > 0
    assert "{" not in filled, f"Unfilled slot in template {t['id']}: {filled}"
print(f"Test 10 — generation pipeline ({len(TEMPLATES)} templates, {len(ALL_VERIFIER_IDS)} verifiers): OK ✓")

# %%
#######################################
# 11. Sample building + validation + fingerprint uniqueness
#######################################
random.seed(42)
# select_modifier_ids
ids = select_modifier_ids(5)
assert len(ids) <= 5
assert is_combination_valid(ids), "Selected modifiers should be conflict-free"
for iid in ids:
    assert iid in MODIFIER_IDS, f"Unexpected modifier ID: {iid}"

# build_sample
template = TEMPLATES[0]
sample = build_sample(template, key=99, min_modifiers=1, max_modifiers=3)
assert "key" in sample and sample["key"] == 99
assert "prompt" in sample and len(sample["prompt"]) > 0
assert len(sample["verifier_id_list"]) == len(sample["kwargs"])
assert is_combination_valid(sample["verifier_id_list"])

# validate_sample — valid
sample2 = build_sample(TEMPLATES[0], key=1, min_modifiers=1, max_modifiers=2)
assert validate_sample(sample2) == [], f"Expected no issues, got {validate_sample(sample2)}"

# validate_sample — bad (unknown verifier + empty prompt)
bad_sample = {
    "key": 0, "prompt": "Test",
    "verifier_id_list": ["fake_category:nonexistent"], "kwargs": [{}],
}
issues = validate_sample(bad_sample)
assert len(issues) > 0 and any("Unknown" in i for i in issues)
bad_sample2 = {
    "key": 0, "prompt": "   ",
    "verifier_id_list": [], "kwargs": [],
}
assert any("Empty prompt" in i for i in validate_sample(bad_sample2))

# fingerprint uniqueness
fps = set()
for i in range(20):
    random.seed(i)
    s = build_sample(random.choice(TEMPLATES), key=i, min_modifiers=1, max_modifiers=3)
    fps.add(sample_fingerprint(s))
assert len(fps) > 10, f"Expected >10 unique fingerprints, got {len(fps)}"
print(f"Test 11 — sample building + validation + fingerprints ({len(fps)}/20 unique): OK ✓")

# %%
#######################################
# 12. End-to-end: generate + verify
#######################################
random.seed(123)
template = TEMPLATES[0]
sample = build_sample(template, key=1, min_modifiers=1, max_modifiers=1)
assert validate_sample(sample) == [], f"Sample has validation issues: {validate_sample(sample)}"

v = Verifier(
    verifier_id_list=sample["verifier_id_list"],
    kwargs=sample["kwargs"],
    completion="Dummy completion text.",
)
results = v.verify()
assert isinstance(results, list)
assert len(results) == len(sample["verifier_id_list"])
assert all(isinstance(r, bool) for r in results)
print("Test 12 — end-to-end generate + verify: OK ✓")

# %%
#######################################
# 13. Long context verifiers — pass, fail, partial, edge cases
#######################################
# CommonWordsChecker — pass (all), partial (50%), fail (none)
v = Verifier(
    verifier_id_list=["long_context:common_words"],
    kwargs=[{"expected_words": ["gato", "cachorro", "pássaro"]}],
    completion="As palavras mais comuns são: gato, cachorro e pássaro.",
)
assert v.verify() == [True]

v2 = Verifier(
    verifier_id_list=["long_context:common_words"],
    kwargs=[{"expected_words": ["gato", "cachorro", "pássaro", "elefante"]}],
    completion="As palavras mais comuns são gato e cachorro.",
)
assert v2.verify() == [True], "Should pass with 2/4 = 50%"

v3 = Verifier(
    verifier_id_list=["long_context:common_words"],
    kwargs=[{"expected_words": ["gato", "cachorro", "pássaro", "elefante"]}],
    completion="Não sei quais são as palavras.",
)
assert v3.verify() == [False]

# Empty expected_words edge case
v4 = Verifier(
    verifier_id_list=["long_context:common_words"],
    kwargs=[{"expected_words": []}],
    completion="Qualquer resposta.",
)
assert v4.verify() == [True], "Empty expected_words should pass"

# RareWordsChecker — pass & fail
v5 = Verifier(
    verifier_id_list=["long_context:rare_words"],
    kwargs=[{"expected_words": ["hipopótamo", "rinoceronte"]}],
    completion="As palavras mais raras são hipopótamo e rinoceronte.",
)
assert v5.verify() == [True]
v6 = Verifier(
    verifier_id_list=["long_context:rare_words"],
    kwargs=[{"expected_words": ["hipopótamo", "rinoceronte", "camaleão"]}],
    completion="Não encontrei nenhuma palavra rara.",
)
assert v6.verify() == [False]

# CountWordChecker — pass & fail
v7 = Verifier(
    verifier_id_list=["long_context:count_word"],
    kwargs=[{"target_word": "gato", "expected_count": 7}],
    completion="A palavra \"gato\" aparece 7 vezes na lista.",
)
assert v7.verify() == [True]
v8 = Verifier(
    verifier_id_list=["long_context:count_word"],
    kwargs=[{"target_word": "gato", "expected_count": 7}],
    completion="A palavra \"gato\" aparece 5 vezes na lista.",
)
assert v8.verify() == [False]

# WordAtPositionChecker — pass & fail
v9 = Verifier(
    verifier_id_list=["long_context:word_at_position"],
    kwargs=[{"position": 42, "expected_word": "cachorro"}],
    completion="A palavra na posição 42 é \"cachorro\".",
)
assert v9.verify() == [True]
v10 = Verifier(
    verifier_id_list=["long_context:word_at_position"],
    kwargs=[{"position": 42, "expected_word": "cachorro"}],
    completion="A palavra na posição 42 é \"gato\".",
)
assert v10.verify() == [False]

# FrequencyComparisonChecker — pass & fail
v11 = Verifier(
    verifier_id_list=["long_context:frequency_comparison"],
    kwargs=[{"word_a": "gato", "word_b": "cachorro", "expected_winner": "gato"}],
    completion="\"gato\" aparece 15 vezes e \"cachorro\" 8 vezes. Portanto, \"gato\" é mais frequente.",
)
assert v11.verify() == [True]
v12 = Verifier(
    verifier_id_list=["long_context:frequency_comparison"],
    kwargs=[{"word_a": "gato", "word_b": "cachorro", "expected_winner": "gato"}],
    completion="A palavra \"cachorro\" é mais frequente na lista.",
)
assert v12.verify() == [False]
print("Test 13 — long context verifiers (pass/fail/partial/edge): OK ✓")

# %%
#######################################
# 14. Long context — end-to-end generate + verify
#######################################
from generate_from_long_context_templates import (
    build_sample as build_lc_sample,
    validate_sample as validate_lc_sample,
    LONG_CONTEXT_TEMPLATES,
)

for idx, template in enumerate(LONG_CONTEXT_TEMPLATES[:5]):
    random.seed(42 + idx)
    tt = template["task_type"]
    # Word-list tasks use num_words; haystack tasks use documents/num_chars/rng
    if tt.startswith("needle_"):
        continue  # tested separately in haystack tests
    sample = build_lc_sample(template, key=idx, num_words=30)
    issues = validate_lc_sample(sample)
    assert issues == [], f"Template {idx} has issues: {issues}"
    assert "completion" not in sample

    # Build a correct completion from kwargs
    kw = sample["kwargs"][0]
    if "expected_words" in kw and kw["expected_words"] is not None:
        completion = "Palavras: " + ", ".join(kw["expected_words"])
    elif "target_word" in kw and kw["target_word"] is not None:
        completion = f'A palavra "{kw["target_word"]}" aparece {kw["expected_count"]} vezes.'
    elif "position" in kw and kw["position"] is not None:
        completion = f'A palavra na posição {kw["position"]} é "{kw["expected_word"]}".'
    elif "expected_winner" in kw and kw["expected_winner"] is not None:
        completion = f'"{kw["expected_winner"]}" é mais frequente.'
    else:
        completion = "Resposta genérica."

    v = Verifier(
        verifier_id_list=sample["verifier_id_list"],
        kwargs=sample["kwargs"],
        completion=completion,
    )
    results = v.verify()
    assert results == [True], f"Template {idx}: expected [True], got {results}"
print("Test 14 — long context end-to-end (all templates): OK ✓")

# %%
#######################################
# 15. Haystack verifiers — pass, fail, partial, edge cases
#######################################
# NeedleSingleNumberChecker
v = Verifier(
    verifier_id_list=["haystack:needle_single_number"],
    kwargs=[{"key": "seco-dia", "expected_values": {"seco-dia": ["7777458"]}}],
    completion="O número especial para seco-dia é: **7777458**",
)
assert v.verify() == [True]
v2 = Verifier(
    verifier_id_list=["haystack:needle_single_number"],
    kwargs=[{"key": "seco-dia", "expected_values": {"seco-dia": ["7777458"]}}],
    completion="O número especial para seco-dia é: **1234567**",
)
assert v2.verify() == [False]

# NeedleMultiNumberSameKeyChecker — all found, partial ≥50%, fail <50%
v3 = Verifier(
    verifier_id_list=["haystack:needle_multi_number_same_key"],
    kwargs=[{"key": "seco-dia", "expected_values": {"seco-dia": ["7777458", "8855030"]}}],
    completion="1. **7777458**\n2. **8855030**",
)
assert v3.verify() == [True]
v4 = Verifier(
    verifier_id_list=["haystack:needle_multi_number_same_key"],
    kwargs=[{"key": "seco-dia", "expected_values": {"seco-dia": ["7777458", "8855030", "1111111", "2222222"]}}],
    completion="1. **7777458**\n2. **8855030**",
)
assert v4.verify() == [True], "Partial ≥50% should pass"
v5 = Verifier(
    verifier_id_list=["haystack:needle_multi_number_same_key"],
    kwargs=[{"key": "seco-dia", "expected_values": {"seco-dia": ["7777458", "8855030", "1111111", "2222222"]}}],
    completion="Nenhum número encontrado.",
)
assert v5.verify() == [False]

# NeedleMultiNumberDiffKeysChecker
v6 = Verifier(
    verifier_id_list=["haystack:needle_multi_number_diff_keys"],
    kwargs=[{"expected_values": {"úmido-educação": ["9339304"], "frio-temporão": ["2770262"]}}],
    completion="1. **Para úmido-educação**: **9339304**\n2. **Para frio-temporão**: **2770262**",
)
assert v6.verify() == [True]
v7 = Verifier(
    verifier_id_list=["haystack:needle_multi_number_diff_keys"],
    kwargs=[{"expected_values": {"úmido-educação": ["9339304"], "frio-temporão": ["2770262"]}}],
    completion="Não encontrei os números.",
)
assert v7.verify() == [False]

# NeedleUUIDChecker — pass, fail, case-insensitive
v8 = Verifier(
    verifier_id_list=["haystack:needle_uuid"],
    kwargs=[{
        "query_key": "9384e37b-7e27-40ad-8cdd-3646065df267",
        "expected_values": {"9384e37b-7e27-40ad-8cdd-3646065df267": ["64807347-1142-483e-8f2d-bafcce9f112d"]},
    }],
    completion="O código UUID é: **64807347-1142-483e-8f2d-bafcce9f112d**",
)
assert v8.verify() == [True]
v9 = Verifier(
    verifier_id_list=["haystack:needle_uuid"],
    kwargs=[{
        "query_key": "9384e37b-7e27-40ad-8cdd-3646065df267",
        "expected_values": {"9384e37b-7e27-40ad-8cdd-3646065df267": ["64807347-1142-483e-8f2d-bafcce9f112d"]},
    }],
    completion="O código UUID é: **aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee**",
)
assert v9.verify() == [False]
v10 = Verifier(
    verifier_id_list=["haystack:needle_uuid"],
    kwargs=[{
        "query_key": "abc",
        "expected_values": {"abc": ["64807347-1142-483E-8F2D-BAFCCE9F112D"]},
    }],
    completion="O UUID é: 64807347-1142-483e-8f2d-bafcce9f112d.",
)
assert v10.verify() == [True], "UUID check should be case-insensitive"

# Empty expected_values edge case
v11 = Verifier(
    verifier_id_list=["haystack:needle_multi_number_same_key"],
    kwargs=[{"key": "", "expected_values": {}}],
    completion="Qualquer resposta.",
)
assert v11.verify() == [True], "Empty expected_values should pass"
print("Test 15 — haystack verifiers (pass/fail/partial/edge): OK ✓")

# %%
#######################################
# 16. Haystack — end-to-end generate + verify (all templates)
#######################################
from generate_from_long_context_templates import (
    build_sample as build_hs_sample,
    validate_sample as validate_hs_sample,
    load_documents,
)
HAYSTACK_TEMPLATES = [
    t for t in LONG_CONTEXT_TEMPLATES if t["task_type"].startswith("needle_")
]
_hs_docs = load_documents("./data")

for idx, hs_template in enumerate(HAYSTACK_TEMPLATES):
    random.seed(42 + idx)
    hs_sample = build_hs_sample(
        hs_template, key=idx, documents=_hs_docs,
        num_chars=2000, rng=random.Random(42 + idx),
    )
    hs_issues = validate_hs_sample(hs_sample)
    assert hs_issues == [], f"Haystack template {idx} has issues: {hs_issues}"
    assert "completion" not in hs_sample

    kw = hs_sample["kwargs"][0]
    values = list(kw["expected_values"].values())[0] if kw["expected_values"] else []
    # diff_keys needs key names in the completion; uuid needs the value only
    tt = hs_template["task_type"]
    if tt == "needle_multi_number_diff_keys":
        parts = []
        for k, vs in kw["expected_values"].items():
            parts.append(f"Para {k}: **{vs[0]}**")
        completion = "\n".join(parts) if parts else "Nenhum."
    else:
        completion = " ".join(f"**{val}**" for val in values) if values else "Nenhum."
    v = Verifier(
        verifier_id_list=hs_sample["verifier_id_list"],
        kwargs=hs_sample["kwargs"],
        completion=completion,
    )
    results = v.verify()
    assert results == [True], f"Haystack template {idx}: expected [True], got {results}"
print(f"Test 16 — haystack end-to-end ({len(HAYSTACK_TEMPLATES)} templates): OK ✓")

# %%
#######################################
# 17. Math verifier — pass, fail, edge cases
#######################################
# Pass — answer present in completion
v = Verifier(
    verifier_id_list=["math:answer_check"],
    kwargs=[{"expected_answer": "42"}],
    completion="A resposta é 42.",
)
assert v.verify() == [True]

# Pass — answer embedded in longer text
v2 = Verifier(
    verifier_id_list=["math:answer_check"],
    kwargs=[{"expected_answer": "3.14"}],
    completion="O valor de pi é aproximadamente 3.14 radianos.",
)
assert v2.verify() == [True]

# Fail — wrong answer
v3 = Verifier(
    verifier_id_list=["math:answer_check"],
    kwargs=[{"expected_answer": "42"}],
    completion="A resposta é 43.",
)
assert v3.verify() == [False]

# Fail — answer completely absent
v4 = Verifier(
    verifier_id_list=["math:answer_check"],
    kwargs=[{"expected_answer": "100"}],
    completion="Não sei a resposta.",
)
assert v4.verify() == [False]

# Fail — empty expected answer
v5 = Verifier(
    verifier_id_list=["math:answer_check"],
    kwargs=[{"expected_answer": ""}],
    completion="Qualquer coisa.",
)
assert v5.verify() == [False]

# Pass — negative number
v6 = Verifier(
    verifier_id_list=["math:answer_check"],
    kwargs=[{"expected_answer": "-5"}],
    completion="O resultado é -5 unidades.",
)
assert v6.verify() == [True]

print("Test 17 — math verifier (pass/fail/edge): OK ✓")

# %%
#######################################
# 18. Math — build_sample + validate + verify end-to-end
#######################################
from generate_from_math_dataset import (
    build_sample as build_math_sample,
    validate_sample as validate_math_sample,
)

sample = build_math_sample("Quanto é 2 + 2?", "4", key=0)
assert sample["verifier_id_list"] == ["math:answer_check"]
assert sample["kwargs"] == [{"expected_answer": "4"}]
assert validate_math_sample(sample) == []

# Verify it passes with a correct completion
v = Verifier(
    verifier_id_list=sample["verifier_id_list"],
    kwargs=sample["kwargs"],
    completion="A soma de 2 + 2 é 4.",
)
assert v.verify() == [True]

# Verify it fails with an incorrect completion
v2 = Verifier(
    verifier_id_list=sample["verifier_id_list"],
    kwargs=sample["kwargs"],
    completion="A soma de 2 + 2 é 5.",
)
assert v2.verify() == [False]

print("Test 18 — math end-to-end (build + validate + verify): OK ✓")

# %%
#######################################
# Summary
#######################################
print("\n" + "=" * 40)
print("All tests passed!")
print("=" * 40)