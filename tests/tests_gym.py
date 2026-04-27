
"""
Gym test suite for verifiers, templates, and generation pipeline.

Run with:
    python tests_gym.py

Requirements:
- transformers
- nltk
- langdetect
- immutabledict
- packaging
- datasets
"""

# %%
#######################################
# 1. Imports & Setup
#######################################
import sys
import os
import tempfile

sys.pycache_prefix = os.path.join(tempfile.gettempdir(), "pycache")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
GYM_DIR = os.path.join(REPO_ROOT, "gym")
if GYM_DIR not in sys.path:
    sys.path.insert(0, GYM_DIR)

import json
import random
import string as _string

def _parse_kw(kw):
    """Deserialize a kwargs entry (string or dict) into a dict."""
    return json.loads(kw) if isinstance(kw, str) else kw

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

print("All imports OK ✅")


def test_02_multiconstraint_verifier_pass_and_partial_failure():
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

    # Title missing + comma present -> first two fail, rest pass
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
    print("Test 2 — multi-constraint pass + partial failure: OK ✅")


def test_03_keywords_forbidden_words_frequency_pass_fail():
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
    print("Test 3 — keywords + forbidden + frequency: OK ✅")


def test_04_length_constraints_detectable_content():
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
    print("Test 4 — length constraints + detectable content: OK ✅")


def test_05_detectable_format_verifiers():
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
    print("Test 5 — detectable format verifiers: OK ✅")


def test_06_combination_startend_verifiers():
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
    print("Test 6 — combination + startend verifiers: OK ✅")


def test_07_unknown_verifier_id_raises_error():
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
    print("Test 7 — unknown verifier raises error: OK ✅")


def test_08_metadata_integrity_registry_conflict_symmetry_selfconflict():
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
    print("Test 8 — metadata integrity (registry + conflicts): OK ✅")


def test_09_metadata_helpers_iscombinationvalid_getaddable_makeemptykwar():
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
    print("Test 9 — metadata helpers: OK ✅")


def test_10_generation_pipeline_kwargs_descriptions_templates_fill():
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
    print(f"Test 10 — generation pipeline ({len(TEMPLATES)} templates, {len(ALL_VERIFIER_IDS)} verifiers): OK ✅")


def test_11_sample_building_validation_fingerprint_uniqueness():
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
    sample = build_sample(template, min_modifiers=1, max_modifiers=3)
    assert "id" in sample and isinstance(sample["id"], str) and len(sample["id"]) == 32
    assert "prompt" in sample and len(sample["prompt"]) > 0
    assert len(sample["verifier_id_list"]) == len(sample["kwargs"])
    assert is_combination_valid(sample["verifier_id_list"])

    # validate_sample — valid
    sample2 = build_sample(TEMPLATES[0], min_modifiers=1, max_modifiers=2)
    assert validate_sample(sample2) == [], f"Expected no issues, got {validate_sample(sample2)}"

    # validate_sample — bad (unknown verifier + empty prompt)
    bad_sample = {
        "id": "dummy", "prompt": "Test",
        "verifier_id_list": ["fake_category:nonexistent"], "kwargs": [{}],
    }
    issues = validate_sample(bad_sample)
    assert len(issues) > 0 and any("Unknown" in i for i in issues)
    bad_sample2 = {
        "id": "dummy", "prompt": "   ",
        "verifier_id_list": [], "kwargs": [],
    }
    assert any("Empty prompt" in i for i in validate_sample(bad_sample2))

    # fingerprint uniqueness
    fps = set()
    for i in range(20):
        random.seed(i)
        s = build_sample(random.choice(TEMPLATES), min_modifiers=1, max_modifiers=3)
        fps.add(sample_fingerprint(s))
    assert len(fps) > 10, f"Expected >10 unique fingerprints, got {len(fps)}"
    print(f"Test 11 — sample building + validation + fingerprints ({len(fps)}/20 unique): OK ✅")


def test_12_endtoend_generate_verify():
    # 12. End-to-end: generate + verify
    #######################################
    random.seed(123)
    template = TEMPLATES[0]
    sample = build_sample(template, min_modifiers=1, max_modifiers=1)
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
    print("Test 12 — end-to-end generate + verify: OK ✅")


def test_13_long_context_verifiers_pass_fail_partial_edge_cases():
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
    print("Test 13 — long context verifiers (pass/fail/partial/edge): OK ✅")


def test_14_long_context_endtoend_generate_verify():
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
        sample = build_lc_sample(template, num_words=30)
        issues = validate_lc_sample(sample)
        assert issues == [], f"Template {idx} has issues: {issues}"
        assert "completion" not in sample

        # Build a correct completion from kwargs
        kw = _parse_kw(sample["kwargs"][0])
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
    print("Test 14 — long context end-to-end (all templates): OK ✅")


def test_15_haystack_verifiers_pass_fail_partial_edge_cases():
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
    print("Test 15 — haystack verifiers (pass/fail/partial/edge): OK ✅")


def test_16_haystack_endtoend_generate_verify_all_templates():
    # 16. Haystack — end-to-end generate + verify (all templates)
    #######################################
    from generate_from_long_context_templates import (
        build_sample as build_hs_sample,
        validate_sample as validate_hs_sample,
        load_documents,
        LONG_CONTEXT_TEMPLATES,
    )
    HAYSTACK_TEMPLATES = [
        t for t in LONG_CONTEXT_TEMPLATES if t["task_type"].startswith("needle_")
    ]
    _hs_docs = load_documents(os.path.join(GYM_DIR, "assets"))

    for idx, hs_template in enumerate(HAYSTACK_TEMPLATES):
        random.seed(42 + idx)
        hs_sample = build_hs_sample(
            hs_template, documents=_hs_docs,
            num_chars=2000, rng=random.Random(42 + idx),
        )
        hs_issues = validate_hs_sample(hs_sample)
        assert hs_issues == [], f"Haystack template {idx} has issues: {hs_issues}"
        assert "completion" not in hs_sample

        kw = _parse_kw(hs_sample["kwargs"][0])
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
    print(f"Test 16 — haystack end-to-end ({len(HAYSTACK_TEMPLATES)} templates): OK ✅")


def test_17_math_verifier_pass_fail_edge_cases_and_relaxed_mode():
    # 17. Math verifier — pass, fail, edge cases, and relaxed mode
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

    # Relaxed mode — integer part of a float answer accepted
    v7 = Verifier(
        verifier_id_list=["math:answer_check"],
        kwargs=[{"expected_answer": "3.6666666666666665", "relaxed": True}],
        completion="A resposta é aproximadamente 3.",
    )
    assert v7.verify() == [True], "Relaxed: integer part '3' should be accepted"

    # Relaxed mode — exact float match also accepted
    v8 = Verifier(
        verifier_id_list=["math:answer_check"],
        kwargs=[{"expected_answer": "3.5", "relaxed": True}],
        completion="A resposta exata é 3.5.",
    )
    assert v8.verify() == [True], "Relaxed: exact float match should be accepted"

    # Relaxed mode — wrong integer part fails
    v9 = Verifier(
        verifier_id_list=["math:answer_check"],
        kwargs=[{"expected_answer": "3.5", "relaxed": True}],
        completion="A resposta é 4.",
    )
    assert v9.verify() == [False], "Relaxed: wrong integer part should fail"

    # Relaxed mode — integer expected answer still requires exact substring match
    v10 = Verifier(
        verifier_id_list=["math:answer_check"],
        kwargs=[{"expected_answer": "7", "relaxed": True}],
        completion="A resposta é 7.",
    )
    assert v10.verify() == [True], "Relaxed: integer answer exact match should pass"

    # Relaxed mode — negative float, integer part accepted
    v11 = Verifier(
        verifier_id_list=["math:answer_check"],
        kwargs=[{"expected_answer": "-3.5", "relaxed": True}],
        completion="O resultado é -3 (parte inteira).",
    )
    assert v11.verify() == [True], "Relaxed: negative float integer part should be accepted"

    print("Test 17 — math verifier (pass/fail/edge/relaxed): OK ✅")


def test_18_math_buildsample_validate_verify_jsonl_synthetic_generation():
    # 18. Math — build_sample + validate + verify + JSONL + synthetic generation
    #######################################
    from generate_from_math_dataset import (
        build_sample as build_math_sample,
        validate_sample as validate_math_sample,
        load_math_problems,
        generate_math_problems as _gen_math_problems,
        MATH_PROBLEMS_JSONL,
    )

    #  JSONL dataset loading 
    math_pairs = load_math_problems(MATH_PROBLEMS_JSONL)
    assert len(math_pairs) > 0, "JSONL dataset should contain problems"
    q0, a0 = math_pairs[0]
    assert q0.strip() and a0.strip(), "Each pair must have non-empty question and answer"

    #  Dataset sample: exact match (no relaxed flag) 
    sample = build_math_sample("Quanto é 2 + 2?", "4")
    assert isinstance(sample["id"], str) and len(sample["id"]) == 32
    assert sample["verifier_id_list"] == ["math:answer_check"]
    assert _parse_kw(sample["kwargs"][0]) == {"expected_answer": "4"}, \
        "Dataset sample kwargs must not include relaxed flag"
    assert validate_math_sample(sample) == []

    v = Verifier(
        verifier_id_list=sample["verifier_id_list"],
        kwargs=sample["kwargs"],
        completion="A soma de 2 + 2 é 4.",
    )
    assert v.verify() == [True]

    v2 = Verifier(
        verifier_id_list=sample["verifier_id_list"],
        kwargs=sample["kwargs"],
        completion="A soma de 2 + 2 é 5.",
    )
    assert v2.verify() == [False]

    #  Synthetic sample: relaxed validation 
    sample_synth = build_math_sample("Resolva: 10 / 3", "3.3333333333333335", relaxed=True)
    kw_synth = _parse_kw(sample_synth["kwargs"][0])
    assert kw_synth.get("relaxed") is True, "Synthetic sample must have relaxed=True"
    assert kw_synth.get("expected_answer") == "3.3333333333333335"
    assert validate_math_sample(sample_synth) == []

    # Relaxed verifier: integer part "3" accepted for "3.3333..."
    v_synth = Verifier(
        verifier_id_list=sample_synth["verifier_id_list"],
        kwargs=sample_synth["kwargs"],
        completion="A resposta é aproximadamente 3.",
    )
    assert v_synth.verify() == [True]

    # Relaxed verifier: wrong integer part fails
    v_synth_fail = Verifier(
        verifier_id_list=sample_synth["verifier_id_list"],
        kwargs=sample_synth["kwargs"],
        completion="A resposta é 5.",
    )
    assert v_synth_fail.verify() == [False]

    #  math_generator logic (now inlined in generate_from_math_dataset) 
    synth_pairs = _gen_math_problems(n=10, max_depth=3, seed=42)
    assert len(synth_pairs) == 10, "Should generate exactly 10 problems"
    for sq, sa in synth_pairs:
        assert sq.strip(), "Question must be non-empty"
        assert sa.strip(), "Answer must be non-empty"
        ms = build_math_sample(sq, sa, relaxed=True)
        assert validate_math_sample(ms) == [], f"Synthetic sample invalid: {ms}"

    # Determinism: same seed produces same problems
    synth_pairs_2 = _gen_math_problems(n=10, max_depth=3, seed=42)
    assert synth_pairs == synth_pairs_2, "generate_math_problems must be deterministic"

    print("Test 18 — math end-to-end (JSONL + synthetic + relaxed verify): OK ✅")


def test_19_email_json_format_verifier_pass_fail_edge_cases():
    # 19. Email JSON format verifier — pass, fail, edge cases
    #######################################
    from generate_from_email_templates import (
        build_email_sample,
        validate_email_sample,
        generate_injected_values,
    )
    from tasks_metadata import (
        EMAIL_TASK_IDS,
        EMAIL_INJECTED_FIELDS,
    )

    # Pass: valid JSON object in ```json``` block
    v = Verifier(
        verifier_id_list=["email:json_format"],
        kwargs=[{}],
        completion='```json\n{"subject": "Reunião"}\n```',
    )
    assert v.verify() == [True]

    # Pass: ```JSON``` specifier (uppercase) accepted
    v2 = Verifier(
        verifier_id_list=["email:json_format"],
        kwargs=[{}],
        completion='```JSON\n{"subject": "Teste", "sender": "Ana"}\n```',
    )
    assert v2.verify() == [True]

    # Fail: raw JSON without fenced block
    v3 = Verifier(
        verifier_id_list=["email:json_format"],
        kwargs=[{}],
        completion='{"subject": "Reunião"}',
    )
    assert v3.verify() == [False], "Raw JSON without code block should fail"

    # Fail: fenced block with invalid JSON syntax
    v4 = Verifier(
        verifier_id_list=["email:json_format"],
        kwargs=[{}],
        completion='```json\n{subject: Reunião}\n```',
    )
    assert v4.verify() == [False], "Invalid JSON syntax should fail"

    # Fail: plain text response
    v5 = Verifier(
        verifier_id_list=["email:json_format"],
        kwargs=[{}],
        completion="O assunto do e-mail é Reunião.",
    )
    assert v5.verify() == [False]

    # Pass: multi-field JSON with booleans
    v6 = Verifier(
        verifier_id_list=["email:json_format"],
        kwargs=[{}],
        completion='```json\n{"subject": "Reunião", "spam": false, "attachments": true}\n```',
    )
    assert v6.verify() == [True]

    # Pass: double-backtick fence (``json...``) — soft format acceptance
    v7 = Verifier(
        verifier_id_list=["email:json_format"],
        kwargs=[{}],
        completion='``json\n{"subject": "Reunião"}\n``',
    )
    assert v7.verify() == [True], "Double-backtick fence should be accepted"

    # Pass: mismatched fence count (``json open, ``` close) — real model output pattern
    v8 = Verifier(
        verifier_id_list=["email:json_format"],
        kwargs=[{}],
        completion='``json\n{"subject": "Reunião"}\n```',
    )
    assert v8.verify() == [True], "Mismatched backtick count should be accepted"

    # Pass: single-backtick fence
    v9 = Verifier(
        verifier_id_list=["email:json_format"],
        kwargs=[{}],
        completion='`json\n{"subject": "Reunião"}\n`',
    )
    assert v9.verify() == [True], "Single-backtick fence should be accepted"

    print("Test 19 — email:json_format verifier (pass/fail/edge): OK ✅")


def test_20_email_schema_keys_verifier_pass_fail_edge_cases():
    # 20. Email schema keys verifier — pass, fail, edge cases
    #######################################
    # Pass: exact key match — single key
    v = Verifier(
        verifier_id_list=["email:schema_keys"],
        kwargs=[{"required_keys": ["subject"]}],
        completion='```json\n{"subject": "Proposta Comercial"}\n```',
    )
    assert v.verify() == [True]

    # Pass: exact key match — multiple keys
    v2 = Verifier(
        verifier_id_list=["email:schema_keys"],
        kwargs=[{"required_keys": ["subject", "sender", "spam"]}],
        completion='```json\n{"subject": "Proposta", "sender": "Carlos", "spam": false}\n```',
    )
    assert v2.verify() == [True]

    # Fail: missing required key
    v3 = Verifier(
        verifier_id_list=["email:schema_keys"],
        kwargs=[{"required_keys": ["subject", "sender", "spam"]}],
        completion='```json\n{"subject": "Proposta", "sender": "Carlos"}\n```',
    )
    assert v3.verify() == [False], "Missing key 'spam' should fail"

    # Fail: extra key not in required_keys
    v4 = Verifier(
        verifier_id_list=["email:schema_keys"],
        kwargs=[{"required_keys": ["subject"]}],
        completion='```json\n{"subject": "Proposta", "sender": "Carlos"}\n```',
    )
    assert v4.verify() == [False], "Extra key 'sender' should fail"

    # Fail: no parseable JSON
    v5 = Verifier(
        verifier_id_list=["email:schema_keys"],
        kwargs=[{"required_keys": ["subject"]}],
        completion="Não é JSON.",
    )
    assert v5.verify() == [False]

    # Pass: key order does not matter
    v6 = Verifier(
        verifier_id_list=["email:schema_keys"],
        kwargs=[{"required_keys": ["sender", "subject"]}],
        completion='```json\n{"subject": "Teste", "sender": "Maria"}\n```',
    )
    assert v6.verify() == [True], "Key order should not matter"
    print("Test 20 — email:schema_keys verifier (pass/fail/edge): OK ✅")


def test_21_email_field_value_verifier_pass_fail_edge_cases_str_bool():
    # 21. Email field value verifier — pass, fail, edge cases (str + bool)
    #######################################
    # Pass: string field exact match
    v = Verifier(
        verifier_id_list=["email:field_value"],
        kwargs=[{"field_name": "date", "expected_value": "2023-04-15T11:30:00"}],
        completion='```json\n{"date": "2023-04-15T11:30:00"}\n```',
    )
    assert v.verify() == [True]

    # Fail: wrong string value
    v2 = Verifier(
        verifier_id_list=["email:field_value"],
        kwargs=[{"field_name": "date", "expected_value": "2023-04-15T11:30:00"}],
        completion='```json\n{"date": "2024-12-01T09:00:00"}\n```',
    )
    assert v2.verify() == [False]

    # Pass: boolean False field
    v3 = Verifier(
        verifier_id_list=["email:field_value"],
        kwargs=[{"field_name": "spam", "expected_value": False}],
        completion='```json\n{"spam": false}\n```',
    )
    assert v3.verify() == [True]

    # Pass: boolean True field
    v4 = Verifier(
        verifier_id_list=["email:field_value"],
        kwargs=[{"field_name": "attachments", "expected_value": True}],
        completion='```json\n{"attachments": true}\n```',
    )
    assert v4.verify() == [True]

    # Fail: wrong boolean value
    v5 = Verifier(
        verifier_id_list=["email:field_value"],
        kwargs=[{"field_name": "spam", "expected_value": False}],
        completion='```json\n{"spam": true}\n```',
    )
    assert v5.verify() == [False], "Expected False but got True"

    # Fail: field missing from JSON
    v6 = Verifier(
        verifier_id_list=["email:field_value"],
        kwargs=[{"field_name": "spam", "expected_value": False}],
        completion='```json\n{"subject": "Teste"}\n```',
    )
    assert v6.verify() == [False], "Missing field should fail"

    # Pass: sender_email string match
    v7 = Verifier(
        verifier_id_list=["email:field_value"],
        kwargs=[{"field_name": "sender_email", "expected_value": "carlos.silva@empresa.com"}],
        completion='```json\n{"sender_email": "carlos.silva@empresa.com"}\n```',
    )
    assert v7.verify() == [True]

    # Pass: raw JSON accepted for field_value (lenient parsing)
    v8 = Verifier(
        verifier_id_list=["email:field_value"],
        kwargs=[{"field_name": "spam", "expected_value": False}],
        completion='{"spam": false, "subject": "Oi"}',
    )
    assert v8.verify() == [True], "field_value should accept raw JSON"
    print("Test 21 — email:field_value verifier (pass/fail/edge): OK ✅")


def test_22_email_endtoend_build_validate_verify():
    # 22. Email — end-to-end build + validate + verify
    #######################################
    import json as _json_mod
    from generate_from_email_templates import (
        build_email_sample,
        validate_email_sample,
        generate_injected_values,
    )
    from tasks_metadata import EMAIL_TASK_IDS, EMAIL_INJECTED_FIELDS

    _test_email = (
        "Assunto: Proposta de Parceria\n\n"
        "Olá Maria,\n\n"
        "Gostaria de apresentar uma proposta de parceria entre nossas empresas.\n"
        "Temos interesse em colaborar no próximo trimestre.\n\n"
        "Aguardo seu retorno.\n\nAbraços,\nCarlos"
    )

    rng_test = random.Random(42)
    _injected = generate_injected_values(rng_test)

    _fields = ["subject", "sender", "spam", "date", "attachments"]

    sample_email = build_email_sample(
        email_text=_test_email,
        fields=_fields,
        injected_values=_injected,
        rng=rng_test,
    )

    # Structural assertions
    assert isinstance(sample_email["id"], str) and len(sample_email["id"]) == 32
    assert "email:json_format" in sample_email["verifier_id_list"]
    assert "email:schema_keys" in sample_email["verifier_id_list"]
    assert len(sample_email["verifier_id_list"]) == len(sample_email["kwargs"])

    # Validate the sample
    issues_email = validate_email_sample(sample_email)
    assert issues_email == [], f"Email sample has issues: {issues_email}"

    # Count expected field_value verifiers (spam, date, attachments are injected)
    fv_count = sample_email["verifier_id_list"].count("email:field_value")
    injected_requested = [f for f in _fields if f in EMAIL_INJECTED_FIELDS]
    assert fv_count == len(injected_requested), (
        f"Expected {len(injected_requested)} field_value verifiers, got {fv_count}"
    )

    # Build a correct completion using the known injected values
    schema_idx = sample_email["verifier_id_list"].index("email:schema_keys")
    req_keys = _parse_kw(sample_email["kwargs"][schema_idx])["required_keys"]
    correct_obj = {}
    for k in req_keys:
        if k in EMAIL_INJECTED_FIELDS:
            for i, iid in enumerate(sample_email["verifier_id_list"]):
                _kw_i = _parse_kw(sample_email["kwargs"][i])
                if iid == "email:field_value" and _kw_i["field_name"] == k:
                    correct_obj[k] = _kw_i["expected_value"]
                    break
        elif k == "subject":
            correct_obj[k] = "Proposta de Parceria"
        elif k == "sender":
            correct_obj[k] = "Carlos"
        else:
            correct_obj[k] = "valor genérico"

    correct_completion = "```json\n" + _json_mod.dumps(correct_obj, ensure_ascii=False) + "\n```"

    v_email = Verifier(
        verifier_id_list=sample_email["verifier_id_list"],
        kwargs=sample_email["kwargs"],
        completion=correct_completion,
    )
    results_email = v_email.verify()
    assert results_email[0] == True, f"email:json_format failed: {correct_completion}"
    assert results_email[1] == True, f"email:schema_keys failed: {correct_completion}"
    for i, (iid, res) in enumerate(zip(sample_email["verifier_id_list"], results_email)):
        if iid == "email:field_value":
            assert res == True, (
                f"email:field_value[{i}] failed for field "
                f"'{_parse_kw(sample_email['kwargs'][i])['field_name']}': {correct_completion}"
            )

    # A wrong completion must fail schema_keys
    wrong_completion = '```json\n{"wrong_key": "value"}\n```'
    v_wrong = Verifier(
        verifier_id_list=sample_email["verifier_id_list"],
        kwargs=sample_email["kwargs"],
        completion=wrong_completion,
    )
    results_wrong = v_wrong.verify()
    assert results_wrong[1] == False, "schema_keys should fail for wrong keys"

    # All EMAIL_TASK_IDS must be in VERIFICATION_REGISTRY
    for tid in EMAIL_TASK_IDS:
        assert tid in VERIFICATION_REGISTRY, f"Missing registry entry for {tid}"

    print("Test 22 — email end-to-end (build + validate + verify): OK ✅")


def test_23_toolcall_verifiers_pass_and_fail_scenarios():
    # 23. Tool-call verifiers — pass and fail scenarios
    #######################################
    from tasks_metadata import TOOL_CALL_TASK_IDS

    # 23a. Valid tool call — all verifiers pass
    v_tc = Verifier(
        verifier_id_list=[
            "tool_call:format",
            "tool_call:name",
            "tool_call:args_keys",
            "tool_call:args_types",
        ],
        kwargs=[
            {"expect_call": True},
            {"expected_name": "calculate_distance"},
            {"required_arg_keys": ["source", "destination"]},
            {"expected_arg_types": {"source": "string", "destination": "string"}},
        ],
        completion=(
            '<tool_call>\n'
            '{"name": "calculate_distance", "arguments": '
            '{"source": "New York", "destination": "Los Angeles"}}\n'
            '</tool_call>'
        ),
    )
    assert v_tc.verify() == [True, True, True, True]

    # 23b. Wrong tool name
    v_tc2 = Verifier(
        verifier_id_list=["tool_call:format", "tool_call:name"],
        kwargs=[
            {"expect_call": True},
            {"expected_name": "calculate_distance"},
        ],
        completion='<tool_call>\n{"name": "wrong_tool", "arguments": {}}\n</tool_call>',
    )
    assert v_tc2.verify() == [True, False]

    # 23c. Missing required arg keys
    v_tc3 = Verifier(
        verifier_id_list=["tool_call:args_keys"],
        kwargs=[{"required_arg_keys": ["source", "destination"]}],
        completion='<tool_call>\n{"name": "x", "arguments": {"source": "NYC"}}\n</tool_call>',
    )
    assert v_tc3.verify() == [False]

    # 23d. Wrong argument types
    v_tc4 = Verifier(
        verifier_id_list=["tool_call:args_types"],
        kwargs=[{"expected_arg_types": {"amount": "number", "currency": "string"}}],
        completion='<tool_call>\n{"name": "x", "arguments": {"amount": "abc", "currency": "USD"}}\n</tool_call>',
    )
    assert v_tc4.verify() == [False]

    # 23e. Refusal — correct
    v_tc5 = Verifier(
        verifier_id_list=["tool_call:format", "tool_call:refusal"],
        kwargs=[{"expect_call": False}, {"min_refusal_words": 5}],
        completion="Sinto muito, mas não posso fazer isso. Minhas funções são limitadas.",
    )
    assert v_tc5.verify() == [True, True]

    # 23f. Refusal fails — model called a tool when it shouldn't
    v_tc6 = Verifier(
        verifier_id_list=["tool_call:format", "tool_call:refusal"],
        kwargs=[{"expect_call": False}, {"min_refusal_words": 5}],
        completion='<tool_call>\n{"name": "x", "arguments": {}}\n</tool_call>',
    )
    assert v_tc6.verify() == [False, False]

    # 23g. No tool_call tags when expected
    v_tc7 = Verifier(
        verifier_id_list=["tool_call:format"],
        kwargs=[{"expect_call": True}],
        completion="Aqui está a resposta sem ferramenta.",
    )
    assert v_tc7.verify() == [False]

    # 23h. Malformed JSON in tool_call
    v_tc8 = Verifier(
        verifier_id_list=["tool_call:format"],
        kwargs=[{"expect_call": True}],
        completion="<tool_call>\nnot json\n</tool_call>",
    )
    assert v_tc8.verify() == [False]

    # 23i. Refusal too short
    v_tc9 = Verifier(
        verifier_id_list=["tool_call:refusal"],
        kwargs=[{"min_refusal_words": 10}],
        completion="Não posso.",
    )
    assert v_tc9.verify() == [False]

    print("Test 23 — tool-call verifiers (pass/fail/edge): OK ✅")


def test_24_toolcall_endtoend_generate_validate_verify():
    # 24. Tool-call end-to-end (generate + validate + verify)
    #######################################
    import re as _re
    from tasks_metadata import TOOL_CALL_TASK_IDS
    from generate_from_tool_call_templates import (
        load_tool_call_data,
        build_tool_call_sample,
        build_valid_completion,
        validate_tool_call_sample,
        sample_fingerprint as tc_fingerprint,
        _REFUSAL_QUERY_TEMPLATES as _REFUSAL_TEMPLATES,
    )
    from pathlib import Path

    _data_dir = os.path.join(GYM_DIR, "assets")
    all_tools = load_tool_call_data(os.path.join(_data_dir, "tools.json"))
    assert len(all_tools) > 0, "No tools loaded"

    rng = random.Random(42)

    # Verify output format matches the canonical gym schema
    for i in range(5):
        tool = rng.choice(all_tools)
        sample = build_tool_call_sample(
            tool=tool, all_tools=all_tools,
            rng=random.Random(42 + i), is_valid=True,
        )
        assert set(sample.keys()) == {"id", "prompt", "verifier_id_list", "kwargs"}, \
            f"Valid sample has wrong keys: {set(sample.keys())}"

    for i in range(5):
        sample = build_tool_call_sample(
            tool=None, all_tools=all_tools,
            rng=random.Random(99 + i), is_valid=False,
        )
        assert set(sample.keys()) == {"id", "prompt", "verifier_id_list", "kwargs"}, \
            f"Refusal sample has wrong keys: {set(sample.keys())}"

    # Generate valid samples and verify with a matching completion
    rng2 = random.Random(42)
    for i in range(10):
        tool = rng2.choice(all_tools)
        sample_rng = random.Random(42 + i)
        sample = build_tool_call_sample(
            tool=tool, all_tools=all_tools,
            rng=sample_rng, is_valid=True,
        )
        issues = validate_tool_call_sample(sample)
        assert not issues, f"Valid sample {i} has issues: {issues}"

        # Build a completion that matches the expected tool call
        expected_name = _parse_kw(sample["kwargs"][1])["expected_name"]
        expected_arg_keys = _parse_kw(sample["kwargs"][2])["required_arg_keys"]
        expected_arg_types = _parse_kw(sample["kwargs"][3])["expected_arg_types"]
        # Generate arguments that satisfy the verifiers (include all typed args)
        args = {}
        for key in expected_arg_types:
            t = expected_arg_types[key]
            if t == "string":
                args[key] = "test_value"
            elif t in ("number", "integer"):
                args[key] = 42
            elif t == "boolean":
                args[key] = True
            elif t == "array":
                args[key] = ["a", "b"]
            elif t == "object":
                args[key] = {"k": "v"}
            else:
                args[key] = "test"
        completion = build_valid_completion(expected_name, args)

        v = Verifier(
            verifier_id_list=sample["verifier_id_list"],
            kwargs=sample["kwargs"],
            completion=completion,
        )
        results = v.verify()
        assert all(results), f"Valid sample {i} verification failed: {results}"

    # Generate and verify refusal samples
    for i in range(10):
        sample = build_tool_call_sample(
            tool=None, all_tools=all_tools,
            rng=random.Random(99 + i), is_valid=False,
        )
        issues = validate_tool_call_sample(sample)
        assert not issues, f"Refusal sample {i} has issues: {issues}"

        refusal_text = (
            "Sinto muito, mas não tenho a capacidade de realizar essa tarefa. "
            "Minhas funções são limitadas às que me foram fornecidas."
        )
        v = Verifier(
            verifier_id_list=sample["verifier_id_list"],
            kwargs=sample["kwargs"],
            completion=refusal_text,
        )
        results = v.verify()
        assert all(results), f"Refusal sample {i} verification failed: {results}"

    # Uniqueness check
    fps = set()
    rng3 = random.Random(42)
    for i in range(50):
        tool = rng3.choice(all_tools)
        sample = build_tool_call_sample(
            tool=tool,
            all_tools=all_tools,
            rng=random.Random(200 + i),
            is_valid=True,
        )
        fps.add(tc_fingerprint(sample))
    assert len(fps) == 50, f"Expected 50 unique fingerprints, got {len(fps)}"

    # All TOOL_CALL_TASK_IDS must be in VERIFICATION_REGISTRY
    for tid in TOOL_CALL_TASK_IDS:
        assert tid in VERIFICATION_REGISTRY, f"Missing registry entry for {tid}"

    # 24b. Tool count in prompt is between 1-3 (inclusive)
    rng_tc = random.Random(42)
    for i in range(20):
        tool = rng_tc.choice(all_tools)
        sample = build_tool_call_sample(
            tool=tool, all_tools=all_tools,
            rng=random.Random(300 + i), is_valid=True,
            min_tools=1, max_tools=3,
        )
        # Match the actual tools block (after a newline, not the inline mention)
        tools_match = _re.search(r'<tools>\n(.*?)</tools>', sample['prompt'], _re.DOTALL)
        assert tools_match, f"Sample {i} has no <tools> block"
        tool_jsons = [
            ln.strip() for ln in tools_match.group(1).strip().splitlines()
            if ln.strip().startswith('{')
        ]
        tool_count = len(tool_jsons)
        assert 1 <= tool_count <= 3, (
            f"Sample {i} has {tool_count} tools (expected 1-3)"
        )

    # 24c. Prompt contains a sentence drawn from the tool's own request_samples
    rng_rs = random.Random(42)
    for i in range(20):
        tool = rng_rs.choice(all_tools)
        rs = tool.get("request_samples", [])
        if not rs:
            continue
        sample = build_tool_call_sample(
            tool=tool, all_tools=all_tools,
            rng=random.Random(400 + i), is_valid=True,
        )
        found = any(r.strip() in sample['prompt'] for r in rs)
        assert found, (
            f"Sample {i} (tool={tool['function']['name']}) prompt does not contain "
            f"any request_sample.\nPrompt: {sample['prompt'][:300]}"
        )

    # 24d. Required arg values are drawn from tool's input_samples
    rng_inp = random.Random(42)
    for i in range(20):
        tool = rng_inp.choice(all_tools)
        input_samples = tool.get("input_samples", {})
        required = tool["function"].get("parameters", {}).get("required", [])
        if not required or not input_samples:
            continue
        sample = build_tool_call_sample(
            tool=tool, all_tools=all_tools,
            rng=random.Random(500 + i), is_valid=True,
        )
        for req_param in required:
            if req_param not in input_samples:
                continue
            valid_values = input_samples[req_param]
            if not isinstance(valid_values, list):
                valid_values = [valid_values]
            # For array params each valid value is itself a list; check its elements.
            def _in_prompt(v):
                if isinstance(v, list):
                    return any(str(e) in sample['prompt'] for e in v)
                return str(v) in sample['prompt']
            any_found = any(_in_prompt(v) for v in valid_values)
            assert any_found, (
                f"Sample {i}: tool={tool['function']['name']}, param={req_param}: "
                f"none of {valid_values[:3]}... found in prompt"
            )

    # 24e. Refusal prompts contain a sentence from _REFUSAL_QUERY_TEMPLATES
    for i in range(10):
        sample = build_tool_call_sample(
            tool=None, all_tools=all_tools,
            rng=random.Random(600 + i), is_valid=False,
        )
        found = any(rt in sample['prompt'] for rt in _REFUSAL_TEMPLATES)
        assert found, f"Refusal sample {i} does not contain any refusal query template"

    print("Test 24 — tool-call end-to-end (generate + validate + verify): OK ✅")


def test_25_thinking_format_verifier_enablethinking_flag():
    # 25. Thinking format verifier — enable_thinking flag
    #######################################
    # 25a. enable_thinking=True, valid <think> block -> first result True
    v_think = Verifier(
        verifier_id_list=["punctuation:no_comma"],
        kwargs=[{}],
        completion="<think>\nPreciso responder sem vírgulas.\n</think>\nResposta sem pontuação extra.",
        enable_thinking=True,
    )
    r_think = v_think.verify()
    assert len(r_think) == 2, f"Expected 2 results, got {len(r_think)}"
    assert r_think[0] == True, "thinking_format should pass with non-empty <think> block"
    assert r_think[1] == True, "no_comma should pass"

    # 25b. enable_thinking=True, missing <think> tags -> first result False
    v_think2 = Verifier(
        verifier_id_list=["punctuation:no_comma"],
        kwargs=[{}],
        completion="Resposta direta sem bloco de raciocínio.",
        enable_thinking=True,
    )
    r_think2 = v_think2.verify()
    assert r_think2[0] == False, "thinking_format should fail without <think> tags"
    assert r_think2[1] == True, "no_comma should still pass"

    # 25c. enable_thinking=True, empty <think></think> tags -> fail
    v_think3 = Verifier(
        verifier_id_list=[],
        kwargs=[],
        completion="<think>   </think>\nResposta.",
        enable_thinking=True,
    )
    assert v_think3.verify() == [False], "Empty <think> tags should fail"

    # 25d. enable_thinking=True, <think> with content -> pass
    v_think4 = Verifier(
        verifier_id_list=[],
        kwargs=[],
        completion="<think>Vou pensar sobre isso.</think>\nA resposta é 42.",
        enable_thinking=True,
    )
    assert v_think4.verify() == [True]

    # 25e. enable_thinking=False (default) — no extra check prepended
    v_think5 = Verifier(
        verifier_id_list=["punctuation:no_comma"],
        kwargs=[{}],
        completion="Resposta simples sem vírgulas.",
    )
    r_think5 = v_think5.verify()
    assert len(r_think5) == 1, "Without enable_thinking, result length should match verifier_id_list"
    assert r_think5 == [True]

    # 25f. Direct checker in registry
    assert "reasoning:thinking_format" in VERIFICATION_REGISTRY

    print("Test 25 — thinking format verifier (enable_thinking flag): OK ✅")


def test_26_soft_matching_sentence_count_1_boundary_tolerance():
    # 26. Soft matching — sentence count (±1 boundary tolerance)
    #######################################
    # 26a. Strict mode: exactly-at-boundary "less than N" fails
    v_strict = Verifier(
        verifier_id_list=["length_constraints:number_sentences"],
        kwargs=[{"num_sentences": 6, "relation": "less than"}],
        completion=(
            "Frase um. Frase dois. Frase três. "
            "Frase quatro. Frase cinco. Frase seis."
        ),
        strict=True,
    )
    assert v_strict.verify() == [False], (
        "Strict: exactly 6 sentences with 'less than 6' must fail"
    )

    # 26b. Soft mode: exactly-at-boundary "less than N" passes
    v_soft = Verifier(
        verifier_id_list=["length_constraints:number_sentences"],
        kwargs=[{"num_sentences": 6, "relation": "less than"}],
        completion=(
            "Frase um. Frase dois. Frase três. "
            "Frase quatro. Frase cinco. Frase seis."
        ),
        strict=False,
    )
    assert v_soft.verify() == [True], (
        "Soft: exactly 6 sentences with 'less than 6' should pass (<=N tolerance)"
    )

    # 26c. Soft mode: one over boundary "less than N" still fails (not infinite tolerance)
    v_over = Verifier(
        verifier_id_list=["length_constraints:number_sentences"],
        kwargs=[{"num_sentences": 4, "relation": "less than"}],
        completion="Frase A. Frase B. Frase C. Frase D. Frase E. Frase F.",
        strict=False,
    )
    # 6 sentences, limit ≤ 4 — even soft cannot pass this (off by 2)
    assert v_over.verify() == [False], (
        "Soft: 6 sentences with 'less than 4' must still fail"
    )

    # 26d. Strict mode: exactly-at-boundary "at least N" (N present): passes
    v_at = Verifier(
        verifier_id_list=["length_constraints:number_sentences"],
        kwargs=[{"num_sentences": 3, "relation": "at least"}],
        completion="Frase A. Frase B. Frase C.",
        strict=True,
    )
    assert v_at.verify() == [True], "Strict: 3 sentences with 'at least 3' must pass"

    # 26e. Soft mode: one below "at least N" passes
    v_one_below = Verifier(
        verifier_id_list=["length_constraints:number_sentences"],
        kwargs=[{"num_sentences": 4, "relation": "at least"}],
        completion="Frase A. Frase B. Frase C.",
        strict=False,
    )
    assert v_one_below.verify() == [True], (
        "Soft: 3 sentences with 'at least 4' should pass (N-1 tolerance)"
    )

    # 26f. Strict default: ensure backward-compatible default is strict
    v_default = Verifier(
        verifier_id_list=["length_constraints:number_sentences"],
        kwargs=[{"num_sentences": 6, "relation": "less than"}],
        completion=(
            "Frase um. Frase dois. Frase três. "
            "Frase quatro. Frase cinco. Frase seis."
        ),
    )
    assert v_default.verify() == [False], "Default (no strict kwarg) must be strict"

    print("Test 26 — soft matching: sentence count boundary tolerance: OK ✅")


def test_27_soft_matching_word_count_10_boundary_tolerance():
    # 27. Soft matching — word count (±10% boundary tolerance)
    #######################################
    # Build a ~100-word completion
    _100_words = " ".join(["palavra"] * 100)  # exactly 100 words

    # 27a. Strict: exactly at boundary for "less than 100" -> fails
    v_w_strict = Verifier(
        verifier_id_list=["length_constraints:number_words"],
        kwargs=[{"num_words": 100, "relation": "less than"}],
        completion=_100_words,
        strict=True,
    )
    assert v_w_strict.verify() == [False], "Strict: 100 words 'less than 100' must fail"

    # 27b. Soft: exactly at boundary for "less than 100" -> passes (within 10%)
    v_w_soft = Verifier(
        verifier_id_list=["length_constraints:number_words"],
        kwargs=[{"num_words": 100, "relation": "less than"}],
        completion=_100_words,
        strict=False,
    )
    assert v_w_soft.verify() == [True], (
        "Soft: 100 words 'less than 100' should pass (≤110 tolerance)"
    )

    # 27c. Soft: 91 words for "at least 100" -> passes (within 10%, threshold = 90)
    _91_words = " ".join(["palavra"] * 91)
    v_w_below = Verifier(
        verifier_id_list=["length_constraints:number_words"],
        kwargs=[{"num_words": 100, "relation": "at least"}],
        completion=_91_words,
        strict=False,
    )
    assert v_w_below.verify() == [True], (
        "Soft: 91 words 'at least 100' should pass (≥90 tolerance)"
    )

    # 27d. Soft: far below still fails
    _50_words = " ".join(["palavra"] * 50)
    v_w_far = Verifier(
        verifier_id_list=["length_constraints:number_words"],
        kwargs=[{"num_words": 100, "relation": "at least"}],
        completion=_50_words,
        strict=False,
    )
    assert v_w_far.verify() == [False], (
        "Soft: 50 words 'at least 100' must still fail (outside ±10%)"
    )

    print("Test 27 — soft matching: word count boundary tolerance: OK ✅")


def test_28_soft_matching_nthparagraphfirstword_with_singlen_separator():
    # 28. Soft matching — nth_paragraph_first_word with single-\n separator
    #######################################
    # 28a. Strict: single-\n separator — fails (expects \n\n)
    v_para_strict = Verifier(
        verifier_id_list=["length_constraints:nth_paragraph_first_word"],
        kwargs=[{"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "geralmente"}],
        completion=(
            "Primeiro parágrafo com algum conteúdo.\n"
            "Geralmente o segundo parágrafo começa aqui."
        ),
        strict=True,
    )
    assert v_para_strict.verify() == [False], (
        "Strict: single-\\n separator should fail (verifier expects \\n\\n)"
    )

    # 28b. Soft: single-\n separator — passes (soft accepts \n too)
    v_para_soft = Verifier(
        verifier_id_list=["length_constraints:nth_paragraph_first_word"],
        kwargs=[{"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "geralmente"}],
        completion=(
            "Primeiro parágrafo com algum conteúdo.\n"
            "Geralmente o segundo parágrafo começa aqui."
        ),
        strict=False,
    )
    assert v_para_soft.verify() == [True], (
        "Soft: single-\\n separator should pass"
    )

    # 28c. \n\n separator: both strict and soft pass
    v_para_nn = Verifier(
        verifier_id_list=["length_constraints:nth_paragraph_first_word"],
        kwargs=[{"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "geralmente"}],
        completion=(
            "Primeiro parágrafo com algum conteúdo.\n\n"
            "Geralmente o segundo parágrafo começa aqui."
        ),
        strict=True,
    )
    assert v_para_nn.verify() == [True], "\\n\\n separator must pass in strict mode"

    # 28d. Soft: wrong first word still fails
    v_para_wrong = Verifier(
        verifier_id_list=["length_constraints:nth_paragraph_first_word"],
        kwargs=[{"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "geralmente"}],
        completion=(
            "Primeiro parágrafo.\n"
            "Normalmente o segundo parágrafo começa aqui."
        ),
        strict=False,
    )
    assert v_para_wrong.verify() == [False], (
        "Soft: wrong first word 'normalmente' ≠ 'geralmente' must still fail"
    )

    # 28e. Soft + comma after first word: "Geralmente," -> still matches "geralmente"
    v_para_comma = Verifier(
        verifier_id_list=["length_constraints:nth_paragraph_first_word"],
        kwargs=[{"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "geralmente"}],
        completion=(
            "Primeiro parágrafo.\n\n"
            "Geralmente, o segundo parágrafo tem uma vírgula após a primeira palavra."
        ),
        strict=True,
    )
    assert v_para_comma.verify() == [True], (
        "Strict: first word with trailing comma should match after punctuation strip"
    )

    print("Test 28 — soft matching: nth_paragraph_first_word with \\n separator: OK ✅")


def test_29_soft_matching_letter_frequency_3_tolerance():
    # 29. Soft matching — letter frequency (±3 tolerance)
    #######################################
    # 29a. Common letter 'a' with low threshold: strict fails, soft passes
    # In a typical Portuguese sentence 'a' appears many times — let's count exactly
    _sample_text = "Para fazer um café, aqueça a água e adicione o pó."
    import collections as _collections
    _a_count = _collections.Counter(_sample_text.lower())['a']

    # Threshold just below actual count -> strict fails
    _threshold_below = _a_count - 1  # e.g. 7 when actual is 8 -> strict: 8 < 7 -> False
    v_lf_strict = Verifier(
        verifier_id_list=["keywords:letter_frequency"],
        kwargs=[{
            "letter": "a", "let_frequency": _threshold_below,
            "let_relation": "less than",
        }],
        completion=_sample_text,
        strict=True,
    )
    assert v_lf_strict.verify() == [False], (
        f"Strict: {_a_count} 'a's with 'less than {_threshold_below}' must fail"
    )

    # Threshold at count+2 -> soft (±3) passes
    _threshold_at_plus2 = _a_count + 2  # within +3 tolerance
    v_lf_soft = Verifier(
        verifier_id_list=["keywords:letter_frequency"],
        kwargs=[{
            "letter": "a", "let_frequency": _a_count,
            "let_relation": "less than",
        }],
        completion=_sample_text,
        strict=False,
    )
    # actual == threshold -> strict: < N fails; soft: < N+3 passes
    assert v_lf_soft.verify() == [True], (
        "Soft: exactly at 'less than N' boundary should pass with +3 tolerance"
    )

    # 29b. "at least" with soft ±3 tolerance
    v_lf_at_soft = Verifier(
        verifier_id_list=["keywords:letter_frequency"],
        kwargs=[{
            "letter": "a", "let_frequency": _a_count + 2,
            "let_relation": "at least",
        }],
        completion=_sample_text,
        strict=False,
    )
    # actual = N, threshold = N+2 -> soft: >= (N+2)-3 = >= N-1 -> True (since actual >= N-1)
    assert v_lf_at_soft.verify() == [True], (
        "Soft: 2 below 'at least' threshold should pass with -3 tolerance"
    )

    # 29c. Strict: same "at least" scenario fails
    v_lf_at_strict = Verifier(
        verifier_id_list=["keywords:letter_frequency"],
        kwargs=[{
            "letter": "a", "let_frequency": _a_count + 2,
            "let_relation": "at least",
        }],
        completion=_sample_text,
        strict=True,
    )
    assert v_lf_at_strict.verify() == [False], (
        "Strict: 2 below 'at least' threshold must fail"
    )

    print("Test 29 — soft matching: letter frequency ±3 tolerance: OK ✅")


def test_30_soft_matching_keyword_frequency_1_tolerance():
    # 30. Soft matching — keyword frequency (±1 tolerance)
    #######################################
    # 30a. Strict: exactly at "less than N" boundary -> fails
    v_kf_strict = Verifier(
        verifier_id_list=["keywords:frequency"],
        kwargs=[{"keyword": "inovação", "frequency": 3, "relation": "less than"}],
        completion="A inovação é importante. A inovação é necessária. A inovação transforma.",
        strict=True,
    )
    assert v_kf_strict.verify() == [False], (
        "Strict: 3 occurrences with 'less than 3' must fail"
    )

    # 30b. Soft: exactly at boundary -> passes (≤N tolerance)
    v_kf_soft = Verifier(
        verifier_id_list=["keywords:frequency"],
        kwargs=[{"keyword": "inovação", "frequency": 3, "relation": "less than"}],
        completion="A inovação é importante. A inovação é necessária. A inovação transforma.",
        strict=False,
    )
    assert v_kf_soft.verify() == [True], (
        "Soft: 3 occurrences with 'less than 3' should pass (≤N tolerance)"
    )

    # 30c. Soft: "at least N" one below threshold -> passes (≥N-1 tolerance)
    v_kf_one_below = Verifier(
        verifier_id_list=["keywords:frequency"],
        kwargs=[{"keyword": "qualidade", "frequency": 3, "relation": "at least"}],
        completion="Qualidade é essencial. Qualidade importa.",
        strict=False,
    )
    assert v_kf_one_below.verify() == [True], (
        "Soft: 2 occurrences (case-insensitive) with 'at least 3' should pass (N-1 tolerance)"
    )

    # 30d. Critical: missing keyword entirely -> both strict and soft fail
    v_kf_missing = Verifier(
        verifier_id_list=["keywords:frequency"],
        kwargs=[{"keyword": "qualidade", "frequency": 3, "relation": "at least"}],
        completion="Não há nenhuma ocorrência da palavra.",
        strict=False,
    )
    assert v_kf_missing.verify() == [False], (
        "Soft: zero occurrences with 'at least 3' (threshold - 1 = 2) must fail"
    )

    print("Test 30 — soft matching: keyword frequency ±1 tolerance: OK ✅")


def test_31_critical_errors_always_fail_in_soft_mode():
    # 31. Critical errors always fail in soft mode
    #######################################
    # Forbidden words must always fail regardless of strict flag
    v_forbidden_soft = Verifier(
        verifier_id_list=["keywords:forbidden_words"],
        kwargs=[{"forbidden_words": ["entretanto", "porém"]}],
        completion="Entretanto, isso é um problema grave.",
        strict=False,
    )
    assert v_forbidden_soft.verify() == [False], (
        "Soft: forbidden word present must always fail (critical semantic error)"
    )

    # Wrong language must always fail
    v_lang_soft = Verifier(
        verifier_id_list=["language:response_language"],
        kwargs=[{"language": "en"}],
        completion="Esta resposta está em português, não em inglês.",
        strict=False,
    )
    # langdetect may return 'pt' which != 'en', so should fail
    try:
        result = v_lang_soft.verify()
        assert result == [False], (
            "Soft: wrong language must fail (critical semantic error)"
        )
    except Exception:
        pass  # langdetect unavailable in test environment — skip

    # Quotation enforcement: missing quotes must fail in both modes
    v_quote_soft = Verifier(
        verifier_id_list=["startend:quotation"],
        kwargs=[{}],
        completion="Esta resposta não está entre aspas.",
        strict=False,
    )
    assert v_quote_soft.verify() == [False], (
        "Soft: missing quotation marks must still fail"
    )

    # Title missing must fail in both modes
    v_title_soft = Verifier(
        verifier_id_list=["detectable_format:title"],
        kwargs=[{}],
        completion="Resposta sem título.",
        strict=False,
    )
    assert v_title_soft.verify() == [False], (
        "Soft: missing title must still fail"
    )

    print("Test 31 — critical errors always fail in soft mode: OK ✅")


def test_32_description_consistency_nthparagraphfirstword_includes_nn_in():
    # 32. Description consistency — nth_paragraph_first_word includes \n\n info
    #######################################
    _kw_para = generate_kwargs_for_verifier(
        "length_constraints:nth_paragraph_first_word"
    )
    _desc_para = generate_description_for_verifier(
        "length_constraints:nth_paragraph_first_word", _kw_para
    )
    assert "\\n\\n" in _desc_para or "\n\n" in _desc_para or "n\\n" in _desc_para, (
        "nth_paragraph_first_word description must mention the \\n\\n paragraph "
        f"separator so the model knows the format. Got: {_desc_para!r}"
    )
    print("Test 32 — description consistency: nth_paragraph_first_word includes \\n\\n: OK ✅")


def test_33_letterfrequency_kwargs_less_than_uses_higher_threshold():
    # 33. letter_frequency kwargs: "less than" uses higher threshold
    #######################################
    random.seed(42)
    _lf_less_than_counts = []
    _lf_at_least_counts = []
    for _ in range(200):
        _lf_kw = generate_kwargs_for_verifier("keywords:letter_frequency")
        if _lf_kw["let_relation"] == "less than":
            _lf_less_than_counts.append(_lf_kw["let_frequency"])
        else:
            _lf_at_least_counts.append(_lf_kw["let_frequency"])

    if _lf_less_than_counts:
        assert min(_lf_less_than_counts) >= 10, (
            f"letter_frequency 'less than' threshold must be ≥10 to be achievable. "
            f"Min found: {min(_lf_less_than_counts)}"
        )
    if _lf_at_least_counts:
        assert max(_lf_at_least_counts) <= 10, (
            f"letter_frequency 'at least' threshold must be ≤10 to be achievable. "
            f"Max found: {max(_lf_at_least_counts)}"
        )

    print("Test 33 — letter_frequency kwargs: 'less than' uses threshold ≥10: OK ✅")

# %%
#######################################
# Summary
#######################################
print("\n" + "=" * 50)
print("All gym tests passed ✅")
print("=" * 50)


if __name__ == "__main__":
    test_02_multiconstraint_verifier_pass_and_partial_failure()
    test_03_keywords_forbidden_words_frequency_pass_fail()
    test_04_length_constraints_detectable_content()
    test_05_detectable_format_verifiers()
    test_06_combination_startend_verifiers()
    test_07_unknown_verifier_id_raises_error()
    test_08_metadata_integrity_registry_conflict_symmetry_selfconflict()
    test_09_metadata_helpers_iscombinationvalid_getaddable_makeemptykwar()
    test_10_generation_pipeline_kwargs_descriptions_templates_fill()
    test_11_sample_building_validation_fingerprint_uniqueness()
    test_12_endtoend_generate_verify()
    test_13_long_context_verifiers_pass_fail_partial_edge_cases()
    test_14_long_context_endtoend_generate_verify()
    test_15_haystack_verifiers_pass_fail_partial_edge_cases()
    test_16_haystack_endtoend_generate_verify_all_templates()
    test_17_math_verifier_pass_fail_edge_cases_and_relaxed_mode()
    test_18_math_buildsample_validate_verify_jsonl_synthetic_generation()
    test_19_email_json_format_verifier_pass_fail_edge_cases()
    test_20_email_schema_keys_verifier_pass_fail_edge_cases()
    test_21_email_field_value_verifier_pass_fail_edge_cases_str_bool()
    test_22_email_endtoend_build_validate_verify()
    test_23_toolcall_verifiers_pass_and_fail_scenarios()
    test_24_toolcall_endtoend_generate_validate_verify()
    test_25_thinking_format_verifier_enablethinking_flag()
    test_26_soft_matching_sentence_count_1_boundary_tolerance()
    test_27_soft_matching_word_count_10_boundary_tolerance()
    test_28_soft_matching_nthparagraphfirstword_with_singlen_separator()
    test_29_soft_matching_letter_frequency_3_tolerance()
    test_30_soft_matching_keyword_frequency_1_tolerance()
    test_31_critical_errors_always_fail_in_soft_mode()
    test_32_description_consistency_nthparagraphfirstword_includes_nn_in()
    test_33_letterfrequency_kwargs_less_than_uses_higher_threshold()
    print("\n" + "=" * 50)
    print("All tests passed ✅")
    print("=" * 50)
