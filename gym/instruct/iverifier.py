"""
Verifier for procedurally generated instructions.

Given a model completion and a set of instructions with parameters,
the Verifier evaluates whether the completion satisfies each instruction.

Usage:
    from iverifier import InstructVerifier

    v = InstructVerifier(
        instruction_id_list=["detectable_format:title", "punctuation:no_comma"],
        kwargs=[
            {"capital_frequency": None},  # full kwargs dict (Nones ignored)
            {"capital_frequency": None},
        ],
        completion="<<Meu Título>>\nEsta é a resposta sem vírgulas.",
    )
    results = v.verify()
    # [True, True]
"""

from instructions import (
    BulletListChecker,
    CapitalLettersPortugueseChecker,
    CapitalWordFrequencyChecker,
    CommaChecker,
    ConstrainedResponseChecker,
    EndChecker,
    ForbiddenWords,
    HighlightSectionChecker,
    JsonFormat,
    KeywordChecker,
    KeywordFrequencyChecker,
    LetterFrequencyChecker,
    LowercaseLettersPortugueseChecker,
    NumberOfSentences,
    NumberOfWords,
    ParagraphChecker,
    ParagraphFirstWordCheck,
    PlaceholderChecker,
    PostscriptChecker,
    QuotationChecker,
    RepeatPromptThenAnswer,
    ResponseLanguageChecker,
    SectionChecker,
    TitleChecker,
    TwoResponsesChecker,
)

# Registry: instruction_id → checker class
# To add a new verifier, add a single entry here.
INSTRUCTION_REGISTRY = {
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
}


class InstructVerifier:
    """Evaluates whether a model completion satisfies a set of instructions.

    Args:
        instruction_id_list: List of instruction IDs to verify.
        kwargs: List of kwarg dicts (one per instruction), matching the
            format produced by the generator (full template with None values
            for unused keys).
        completion: The model-generated response to evaluate.
    """

    def __init__(self, instruction_id_list, kwargs, completion):
        self.instruction_id_list = instruction_id_list
        self.kwargs = kwargs
        self.completion = completion

    def verify(self):
        """Run all verifiers and return a list of booleans."""
        results = []
        for i, instruction_id in enumerate(self.instruction_id_list):
            passed = self._verify_one(instruction_id, self.kwargs[i])
            results.append(passed)
        return results

    def _verify_one(self, instruction_id, raw_kwargs):
        """Instantiate the checker, build its description, and run verification."""
        cls = INSTRUCTION_REGISTRY.get(instruction_id)
        if cls is None:
            raise ValueError(f"Unknown instruction ID: {instruction_id}")

        checker = cls(instruction_id)

        # Filter out None values — build_description only accepts relevant kwargs.
        filtered = {k: v for k, v in raw_kwargs.items() if v is not None}
        checker.build_description(**filtered)

        return checker.check_following(self.completion)

