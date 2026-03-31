"""
Verifier for arbitrary instructions.

Given a model completion and a set of verification arguments (verifier IDs and kwargs),
the Verifier evaluates whether the completion satisfies the instruction constraints specified by the verifiers.

Usage:
    from verifier import Verifier

    v = Verifier(
        verifier_id_list=["detectable_format:title", "punctuation:no_comma"],
        kwargs=[
            {"capital_frequency": None},  # full kwargs dict (Nones ignored)
            {"capital_frequency": None},
        ],
        completion="<<Meu Título>>\nEsta é a resposta sem vírgulas.",
    )
    results = v.verify()
    # [True, True]
"""

from verifiers import (
    BulletListChecker,
    CapitalLettersPortugueseChecker,
    CapitalWordFrequencyChecker,
    CommaChecker,
    CommonWordsChecker,
    ConstrainedResponseChecker,
    CountWordChecker,
    EndChecker,
    ForbiddenWords,
    FrequencyComparisonChecker,
    HighlightSectionChecker,
    JsonFormat,
    KeywordChecker,
    KeywordFrequencyChecker,
    LetterFrequencyChecker,
    LowercaseLettersPortugueseChecker,
    MathAnswerChecker,
    NeedleMultiNumberDiffKeysChecker,
    NeedleMultiNumberSameKeyChecker,
    NeedleSingleNumberChecker,
    NeedleUUIDChecker,
    NumberOfSentences,
    NumberOfWords,
    ParagraphChecker,
    ParagraphFirstWordCheck,
    PlaceholderChecker,
    PostscriptChecker,
    QuotationChecker,
    RareWordsChecker,
    RepeatPromptThenAnswer,
    ResponseLanguageChecker,
    SectionChecker,
    TitleChecker,
    TwoResponsesChecker,
    WordAtPositionChecker,
)

# Registry: verifier_id → checker class
# To add a new verifier, add a single entry here.
VERIFICATION_REGISTRY = {
    # General Instruction Constraints
    "keywords:existence": KeywordChecker,
    "keywords:frequency": KeywordFrequencyChecker,
    "keywords:forbidden_words": ForbiddenWords,
    "keywords:letter_frequency": LetterFrequencyChecker,
    "language:response_language": ResponseLanguageChecker,
    "length_constraints:number_sentences": NumberOfSentences,
    "length_constraints:number_paragraphs": ParagraphChecker,
    "length_constraints:number_words": NumberOfWords,
    "length_constraints:nth_paragraph_first_word": ParagraphFirstWordCheck,
    "detectable_content:number_placeholders": PlaceholderChecker,
    "detectable_content:postscript": PostscriptChecker,
    "detectable_format:number_bullet_lists": BulletListChecker,
    "detectable_format:constrained_response": ConstrainedResponseChecker,
    "detectable_format:number_highlighted_sections": HighlightSectionChecker,
    "detectable_format:multiple_sections": SectionChecker,
    "detectable_format:json_format": JsonFormat,
    "detectable_format:title": TitleChecker,
    "combination:two_responses": TwoResponsesChecker,
    "combination:repeat_prompt": RepeatPromptThenAnswer,
    "startend:end_checker": EndChecker,
    "change_case:capital_word_frequency": CapitalWordFrequencyChecker,
    "change_case:portuguese_capital": CapitalLettersPortugueseChecker,
    "change_case:portuguese_lowercase": LowercaseLettersPortugueseChecker,
    "punctuation:no_comma": CommaChecker,
    "startend:quotation": QuotationChecker,
    # Long context retrieval tasks (word-list retrieval, frequency comparison)
    "long_context:common_words": CommonWordsChecker,
    "long_context:rare_words": RareWordsChecker,
    "long_context:count_word": CountWordChecker,
    "long_context:word_at_position": WordAtPositionChecker,
    "long_context:frequency_comparison": FrequencyComparisonChecker,
    # Haystack tasks (retrieving specific numbers or UUIDs from a long document)
    "haystack:needle_single_number": NeedleSingleNumberChecker,
    "haystack:needle_multi_number_same_key": NeedleMultiNumberSameKeyChecker,
    "haystack:needle_multi_number_diff_keys": NeedleMultiNumberDiffKeysChecker,
    "haystack:needle_uuid": NeedleUUIDChecker,
    # Math tasks (verifying correct numerical answer)
    "math:answer_check": MathAnswerChecker,
}


class Verifier:
    """Evaluates whether a model completion satisfies a set of constraints.

    Args:
        verifier_id_list: List of verifier IDs to apply.
        kwargs: List of kwarg dicts (one per verifier), matching the
            format produced by the generator (full template with None values
            for unused keys).
        completion: The model-generated response to evaluate.
    """

    def __init__(self, verifier_id_list, kwargs, completion):
        self.verifier_id_list = verifier_id_list
        self.kwargs = kwargs
        self.completion = completion

    def verify(self):
        """Run all verifiers and return a list of booleans."""
        results = []
        for i, verifier_id in enumerate(self.verifier_id_list):
            passed = self._verify_one(verifier_id, self.kwargs[i])
            results.append(passed)
        return results

    def _verify_one(self, verifier_id, raw_kwargs):
        """Instantiate the checker, build its description, and run verification."""
        cls = VERIFICATION_REGISTRY.get(verifier_id)
        if cls is None:
            raise ValueError(f"Unknown verifier ID: {verifier_id}")

        checker = cls(verifier_id)

        # Filter out None values — build_description only accepts relevant kwargs.
        filtered = {k: v for k, v in raw_kwargs.items() if v is not None}
        checker.build_description(**filtered)

        return checker.check_following(self.completion)

