# %%
#######################################
# 1. Imports & Setup
#######################################
import sys, random, copy, re, json

from instruction_metadata import (
    ALL_VERIFIER_IDS,
    VERIFIER_CONFLICTS,
    EMPTY_KWARGS_TEMPLATE,
    get_conflict_set,
    get_addable_verifiers,
    is_combination_valid,
    make_empty_kwargs,
    generate_kwargs_for_verifier,
    generate_description_for_verifier,
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
# 2. Verifier — title + no_comma
#######################################
v = Verifier(
    verifier_id_list=["detectable_format:title", "punctuation:no_comma"],
    kwargs=[
        {"capital_frequency": None},
        {"capital_frequency": None},
    ],
    completion="<<Meu Título>>\nEsta é a resposta sem vírgulas.",
)
results = v.verify()
assert results == [True, True], f"Expected [True, True], got {results}"
print("Test 2 — title + no_comma (pass): OK ✓")

# %%
#######################################
# 3. Verifier — title missing
#######################################
v = Verifier(
    verifier_id_list=["detectable_format:title"],
    kwargs=[{}],
    completion="Resposta sem título algum.",
)
results = v.verify()
assert results == [False], f"Expected [False], got {results}"
print("Test 3 — title missing (fail): OK ✓")

# %%
#######################################
# 4. Verifier — keyword existence
#######################################
v = Verifier(
    verifier_id_list=["keywords:existence"],
    kwargs=[{"keywords": ["inovação", "estratégia"]}],
    completion="A inovação é a base de toda estratégia empresarial.",
)
assert v.verify() == [True]

v2 = Verifier(
    verifier_id_list=["keywords:existence"],
    kwargs=[{"keywords": ["inovação", "estratégia"]}],
    completion="Esta resposta não contém nenhuma das palavras esperadas.",
)
assert v2.verify() == [False]
print("Test 4 — keyword existence: OK ✓")

# %%
#######################################
# 5. Verifier — keyword frequency
#######################################
v = Verifier(
    verifier_id_list=["keywords:frequency"],
    kwargs=[{"keyword": "teste", "frequency": 3, "relation": "at least"}],
    completion="teste teste teste — aqui está.",
)
assert v.verify() == [True]

v2 = Verifier(
    verifier_id_list=["keywords:frequency"],
    kwargs=[{"keyword": "teste", "frequency": 3, "relation": "at least"}],
    completion="teste aqui aparece só uma vez.",
)
assert v2.verify() == [False]
print("Test 5 — keyword frequency: OK ✓")

# %%
#######################################
# 6. Verifier — forbidden words
#######################################
v = Verifier(
    verifier_id_list=["keywords:forbidden_words"],
    kwargs=[{"forbidden_words": ["entretanto", "porém"]}],
    completion="Esta frase é perfeitamente válida.",
)
assert v.verify() == [True]

v2 = Verifier(
    verifier_id_list=["keywords:forbidden_words"],
    kwargs=[{"forbidden_words": ["entretanto", "porém"]}],
    completion="Entretanto a frase usou uma palavra proibida.",
)
assert v2.verify() == [False]
print("Test 6 — forbidden words: OK ✓")

# %%
#######################################
# 7. Verifier — number of sentences
#######################################
v = Verifier(
    verifier_id_list=["length_constraints:number_sentences"],
    kwargs=[{"num_sentences": 3, "relation": "at least"}],
    completion="Primeira frase. Segunda frase. Terceira frase.",
)
assert v.verify() == [True]

v2 = Verifier(
    verifier_id_list=["length_constraints:number_sentences"],
    kwargs=[{"num_sentences": 5, "relation": "at least"}],
    completion="Apenas uma frase.",
)
assert v2.verify() == [False]
print("Test 7 — number of sentences: OK ✓")

# %%
#######################################
# 8. Verifier — number of paragraphs
#######################################
v = Verifier(
    verifier_id_list=["length_constraints:number_paragraphs"],
    kwargs=[{"num_paragraphs": 3}],
    completion="Parágrafo um.\n***\nParágrafo dois.\n***\nParágrafo três.",
)
assert v.verify() == [True]

v2 = Verifier(
    verifier_id_list=["length_constraints:number_paragraphs"],
    kwargs=[{"num_paragraphs": 3}],
    completion="Parágrafo único sem separador.",
)
assert v2.verify() == [False]
print("Test 8 — number of paragraphs: OK ✓")

# %%
#######################################
# 9. Verifier — number of words
#######################################
v = Verifier(
    verifier_id_list=["length_constraints:number_words"],
    kwargs=[{"num_words": 5, "relation": "at least"}],
    completion="Esta resposta tem cinco palavras exatamente aqui agora.",
)
assert v.verify() == [True]

v2 = Verifier(
    verifier_id_list=["length_constraints:number_words"],
    kwargs=[{"num_words": 100, "relation": "at least"}],
    completion="Poucas palavras.",
)
assert v2.verify() == [False]
print("Test 9 — number of words: OK ✓")

# %%
#######################################
# 10. Verifier — placeholders
#######################################
v = Verifier(
    verifier_id_list=["detectable_content:number_placeholders"],
    kwargs=[{"num_placeholders": 2}],
    completion="Envie para [endereço] no dia [data].",
)
assert v.verify() == [True]

v2 = Verifier(
    verifier_id_list=["detectable_content:number_placeholders"],
    kwargs=[{"num_placeholders": 3}],
    completion="Sem espaços reservados aqui.",
)
assert v2.verify() == [False]
print("Test 10 — placeholders: OK ✓")

# %%
#######################################
# 11. Verifier — postscript
#######################################
v = Verifier(
    verifier_id_list=["detectable_content:postscript"],
    kwargs=[{"postscript_marker": "P.S."}],
    completion="Obrigado pela atenção.\nP.S. Não se esqueça de responder.",
)
assert v.verify() == [True]

v2 = Verifier(
    verifier_id_list=["detectable_content:postscript"],
    kwargs=[{"postscript_marker": "P.S."}],
    completion="Obrigado pela atenção.",
)
assert v2.verify() == [False]
print("Test 11 — postscript: OK ✓")

# %%
#######################################
# 12. Verifier — bullet list
#######################################
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
print("Test 12 — bullet list: OK ✓")

# %%
#######################################
# 13. Verifier — constrained response
#######################################
v = Verifier(
    verifier_id_list=["detectable_format:constrained_response"],
    kwargs=[{}],
    completion="Minha resposta é sim.",
)
assert v.verify() == [True]

v2 = Verifier(
    verifier_id_list=["detectable_format:constrained_response"],
    kwargs=[{}],
    completion="Não sei o que dizer.",
)
assert v2.verify() == [False]
print("Test 13 — constrained response: OK ✓")

# %%
#######################################
# 14. Verifier — highlighted sections
#######################################
v = Verifier(
    verifier_id_list=["detectable_format:number_highlighted_sections"],
    kwargs=[{"num_highlights": 2}],
    completion="Observe o *primeiro destaque* e o *segundo destaque* nesta resposta.",
)
assert v.verify() == [True]

v2 = Verifier(
    verifier_id_list=["detectable_format:number_highlighted_sections"],
    kwargs=[{"num_highlights": 5}],
    completion="Sem destaques aqui.",
)
assert v2.verify() == [False]
print("Test 14 — highlighted sections: OK ✓")

# %%
#######################################
# 15. Verifier — multiple sections
#######################################
v = Verifier(
    verifier_id_list=["detectable_format:multiple_sections"],
    kwargs=[{"section_spliter": "Seção", "num_sections": 2}],
    completion="Seção 1\nConteúdo.\nSeção 2\nMais conteúdo.",
)
assert v.verify() == [True]

v2 = Verifier(
    verifier_id_list=["detectable_format:multiple_sections"],
    kwargs=[{"section_spliter": "Seção", "num_sections": 3}],
    completion="Sem seções nenhuma.",
)
assert v2.verify() == [False]
print("Test 15 — multiple sections: OK ✓")

# %%
#######################################
# 16. Verifier — JSON format
#######################################
v = Verifier(
    verifier_id_list=["detectable_format:json_format"],
    kwargs=[{}],
    completion='```json\n{"chave": "valor"}\n```',
)
assert v.verify() == [True]

v2 = Verifier(
    verifier_id_list=["detectable_format:json_format"],
    kwargs=[{}],
    completion="Isso não é JSON.",
)
assert v2.verify() == [False]
print("Test 16 — JSON format: OK ✓")

# %%
#######################################
# 17. Verifier — two responses
#######################################
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
print("Test 17 — two responses: OK ✓")

# %%
#######################################
# 18. Verifier — end checker
#######################################
v = Verifier(
    verifier_id_list=["startend:end_checker"],
    kwargs=[{"end_phrase": "Há algo mais que eu possa ajudar?"}],
    completion="Aqui está tudo. Há algo mais que eu possa ajudar?",
)
assert v.verify() == [True]

v2 = Verifier(
    verifier_id_list=["startend:end_checker"],
    kwargs=[{"end_phrase": "Há algo mais que eu possa ajudar?"}],
    completion="Aqui está tudo. Obrigado!",
)
assert v2.verify() == [False]
print("Test 18 — end checker: OK ✓")

# %%
#######################################
# 19. Verifier — no comma
#######################################
v = Verifier(
    verifier_id_list=["punctuation:no_comma"],
    kwargs=[{}],
    completion="Esta frase não tem vírgula nenhuma.",
)
assert v.verify() == [True]

v2 = Verifier(
    verifier_id_list=["punctuation:no_comma"],
    kwargs=[{}],
    completion="Esta frase, por outro lado, tem vírgula.",
)
assert v2.verify() == [False]
print("Test 19 — no comma: OK ✓")

# %%
#######################################
# 20. Verifier — quotation
#######################################
v = Verifier(
    verifier_id_list=["startend:quotation"],
    kwargs=[{}],
    completion='"Esta resposta está entre aspas."',
)
assert v.verify() == [True]

v2 = Verifier(
    verifier_id_list=["startend:quotation"],
    kwargs=[{}],
    completion="Esta resposta não está entre aspas.",
)
assert v2.verify() == [False]
print("Test 20 — quotation: OK ✓")

# %%
#######################################
# 21. Verifier — repeat prompt
#######################################
prompt = "Escreva uma carta para meu amigo."
v = Verifier(
    verifier_id_list=["combination:repeat_prompt"],
    kwargs=[{"prompt_to_repeat": prompt}],
    completion=prompt + "\n\nQuerido amigo, espero que esteja bem.",
)
assert v.verify() == [True]

v2 = Verifier(
    verifier_id_list=["combination:repeat_prompt"],
    kwargs=[{"prompt_to_repeat": prompt}],
    completion="Querido amigo, espero que esteja bem.",
)
assert v2.verify() == [False]
print("Test 21 — repeat prompt: OK ✓")

# %%
#######################################
# 22. Verifier — letter frequency
#######################################
v = Verifier(
    verifier_id_list=["keywords:letter_frequency"],
    kwargs=[{"letter": "a", "let_frequency": 3, "let_relation": "at least"}],
    completion="Banana abacaxi amanhã.",
)
assert v.verify() == [True]

v2 = Verifier(
    verifier_id_list=["keywords:letter_frequency"],
    kwargs=[{"letter": "z", "let_frequency": 5, "let_relation": "at least"}],
    completion="Sem muitos z por aqui.",
)
assert v2.verify() == [False]
print("Test 22 — letter frequency: OK ✓")

# %%
#######################################
# 23. Verifier — multiple constraints
#######################################
v = Verifier(
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
    completion="<<Meu Título>>\nEsta resposta não tem vírgula. Obrigado!",
)
results = v.verify()
assert results == [True, True, True], f"Expected [True, True, True], got {results}"
print("Test 23 — multiple constraints (all pass): OK ✓")

# %%
#######################################
# 24. Verifier — unknown verifier ID
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
print("Test 24 — unknown verifier raises error: OK ✓")

# %%
#######################################
# 25. Registry coverage — all IDs mapped
#######################################
for vid in ALL_VERIFIER_IDS:
    assert vid in VERIFICATION_REGISTRY, f"Missing registry entry for {vid}"
print("Test 25 — all verifier IDs in registry: OK ✓")

# %%
#######################################
# 26. Conflict matrix — symmetry
#######################################
for vid_a, conflicts in VERIFIER_CONFLICTS.items():
    for vid_b in conflicts:
        assert vid_a in VERIFIER_CONFLICTS.get(vid_b, set()), (
            f"Conflict asymmetry: {vid_a} conflicts with {vid_b} but not vice-versa"
        )
print("Test 26 — conflict matrix symmetry: OK ✓")

# %%
#######################################
# 27. Conflict matrix — self-conflict
#######################################
for vid in ALL_VERIFIER_IDS:
    assert vid in VERIFIER_CONFLICTS.get(vid, set()), (
        f"{vid} should self-conflict"
    )
print("Test 27 — every verifier self-conflicts: OK ✓")

# %%
#######################################
# 28. is_combination_valid
#######################################
assert is_combination_valid(["detectable_format:title", "punctuation:no_comma"])
assert not is_combination_valid([
    "detectable_format:constrained_response",
    "detectable_format:title",
])
assert is_combination_valid([])
assert is_combination_valid(["punctuation:no_comma"])
print("Test 28 — is_combination_valid: OK ✓")

# %%
#######################################
# 29. get_addable_verifiers
#######################################
addable = get_addable_verifiers(["detectable_format:constrained_response"])
# constrained_response conflicts with everything
assert len(addable) == 0, f"Expected empty, got {addable}"

addable2 = get_addable_verifiers([])
assert set(addable2) == set(ALL_VERIFIER_IDS)
print("Test 29 — get_addable_verifiers: OK ✓")

# %%
#######################################
# 30. make_empty_kwargs
#######################################
kw = make_empty_kwargs()
assert all(v is None for v in kw.values()), "All empty kwargs should be None"
# Ensure it's a deep copy
kw["language"] = "pt"
kw2 = make_empty_kwargs()
assert kw2["language"] is None, "make_empty_kwargs should return independent copies"
print("Test 30 — make_empty_kwargs: OK ✓")

# %%
#######################################
# 31. generate_kwargs_for_verifier
#######################################
random.seed(42)
for vid in ALL_VERIFIER_IDS:
    kw = generate_kwargs_for_verifier(vid, prompt_text="Texto de teste.")
    assert isinstance(kw, dict), f"Expected dict for {vid}"
    # Should contain all template keys
    assert set(kw.keys()) == set(EMPTY_KWARGS_TEMPLATE.keys()), (
        f"Kwargs keys mismatch for {vid}"
    )
print("Test 31 — generate_kwargs_for_verifier returns full template: OK ✓")

# %%
#######################################
# 32. generate_description_for_verifier
#######################################
random.seed(42)
for vid in ALL_VERIFIER_IDS:
    kw = generate_kwargs_for_verifier(vid, prompt_text="Prompt de teste.")
    desc = generate_description_for_verifier(vid, kw)
    assert isinstance(desc, str), f"Description for {vid} should be a string"
    assert len(desc) > 0, f"Description for {vid} should not be empty"
print("Test 32 — generate_description_for_verifier: OK ✓")

# %%
#######################################
# 33. Templates structure
#######################################
assert len(TEMPLATES) > 0, "TEMPLATES should not be empty"
for t in TEMPLATES:
    assert "id" in t, "Template missing 'id'"
    assert "prompts" in t and len(t["prompts"]) > 0, "Template missing 'prompts'"
    assert "slots" in t, "Template missing 'slots'"
    # All slot names used in prompts should exist in slots dict
    for prompt_fmt in t["prompts"]:
        import string as _string
        field_names = [
            fname for _, fname, _, _ in _string.Formatter().parse(prompt_fmt)
            if fname is not None
        ]
        for fname in field_names:
            assert fname in t["slots"], (
                f"Template {t['id']}: slot '{fname}' in prompt but not in slots dict"
            )
print(f"Test 33 — {len(TEMPLATES)} templates structurally valid: OK ✓")

# %%
#######################################
# 34. fill_template
#######################################
random.seed(42)
for t in TEMPLATES:
    filled = fill_template(t)
    assert isinstance(filled, str) and len(filled) > 0
    # Should not contain unfilled {slot} placeholders
    assert "{" not in filled, f"Unfilled slot in template {t['id']}: {filled}"
print("Test 34 — fill_template fills all slots: OK ✓")

# %%
#######################################
# 35. select_modifier_ids
#######################################
random.seed(42)
ids = select_modifier_ids(5)
assert len(ids) <= 5
assert is_combination_valid(ids), "Selected modifiers should be conflict-free"
# All returned IDs should be valid modifier IDs
for iid in ids:
    assert iid in MODIFIER_IDS, f"Unexpected modifier ID: {iid}"
print(f"Test 35 — select_modifier_ids returned {len(ids)} conflict-free IDs: OK ✓")

# %%
#######################################
# 36. build_sample
#######################################
random.seed(42)
template = TEMPLATES[0]
sample = build_sample(template, key=99, min_modifiers=1, max_modifiers=3)

assert "key" in sample and sample["key"] == 99
assert "prompt" in sample and len(sample["prompt"]) > 0
assert "verifier_id_list" in sample
assert "kwargs" in sample
assert len(sample["verifier_id_list"]) == len(sample["kwargs"])
assert is_combination_valid(sample["verifier_id_list"])
print("Test 36 — build_sample structure: OK ✓")

# %%
#######################################
# 37. validate_sample — valid sample
#######################################
random.seed(42)
sample = build_sample(TEMPLATES[0], key=1, min_modifiers=1, max_modifiers=2)
issues = validate_sample(sample)
assert issues == [], f"Expected no issues, got {issues}"
print("Test 37 — validate_sample (valid): OK ✓")

# %%
#######################################
# 38. validate_sample — bad sample
#######################################
# Use an unknown verifier ID — validate_sample can handle this without crashing
bad_sample = {
    "key": 0,
    "prompt": "Test",
    "verifier_id_list": ["fake_category:nonexistent"],
    "kwargs": [{}],
}
issues = validate_sample(bad_sample)
assert len(issues) > 0, "Should detect unknown verifier ID"
assert any("Unknown" in i for i in issues)

# Also test empty prompt
bad_sample2 = {
    "key": 0,
    "prompt": "   ",
    "verifier_id_list": [],
    "kwargs": [],
}
issues2 = validate_sample(bad_sample2)
assert any("Empty prompt" in i for i in issues2)
print("Test 38 — validate_sample (bad): OK ✓")

# %%
#######################################
# 39. sample_fingerprint uniqueness
#######################################
random.seed(42)
fps = set()
for i in range(20):
    random.seed(i)
    s = build_sample(random.choice(TEMPLATES), key=i, min_modifiers=1, max_modifiers=3)
    fps.add(sample_fingerprint(s))
# With 20 different seeds we expect mostly unique samples
assert len(fps) > 10, f"Expected >10 unique fingerprints, got {len(fps)}"
print(f"Test 39 — fingerprint uniqueness ({len(fps)}/20): OK ✓")

# %%
#######################################
# 40. End-to-end: generate + verify
#######################################
random.seed(123)
template = TEMPLATES[0]
sample = build_sample(template, key=1, min_modifiers=1, max_modifiers=1)

# Check that validation passes
issues = validate_sample(sample)
assert issues == [], f"Sample has validation issues: {issues}"

# Verifier can be instantiated with generated data
v = Verifier(
    verifier_id_list=sample["verifier_id_list"],
    kwargs=sample["kwargs"],
    completion="Dummy completion text.",
)
results = v.verify()
assert isinstance(results, list)
assert len(results) == len(sample["verifier_id_list"])
assert all(isinstance(r, bool) for r in results)
print("Test 40 — end-to-end generate + verify: OK ✓")

# %%
#######################################
# Summary
#######################################
print("\n" + "=" * 40)
print("All tests passed!")
print("=" * 40)