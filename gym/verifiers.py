"""A collection of verifiers for procedurally generated tasks."""
import collections
import json
import logging
import random
import re
import string
import langdetect
import utils

logger = logging.getLogger(__name__)

_LANGUAGES = utils.LANGUAGE_CODES

# The relational operation for comparison.
_COMPARISON_RELATION = ("less than", "at least")

# The maximum number of sentences.
_MAX_NUM_SENTENCES = 20

# The number of placeholders.
_NUM_PLACEHOLDERS = 4

# The number of bullet lists.
_NUM_BULLETS = 5

# The options of constrained response.
_CONSTRAINED_RESPONSE_OPTIONS = (
    "Minha resposta é sim.",
    "Minha resposta é não.",
    "Minha resposta é talvez.",
)

# The options of starter keywords.
_STARTER_OPTIONS = (
    "Eu diria",
    "Minha resposta é",
    "Acredito",
    "Na minha opinião",
    "Acho que",
    "Eu suponho",
    "Sinto que",
    "Do meu ponto de vista",
    "Como eu vejo",
    "Para mim",
    "No que me diz respeito",
    "Pelo que entendo",
    "Ao meu ver",
    "Minha opinião sobre isso é",
    "Conforme a minha percepção",
)

# The options of ending keywords.
_ENDING_OPTIONS = ("Isso faz sentido?", "Há algo mais que eu possa ajudar?")

# The number of highlighted sections.
_NUM_HIGHLIGHTED_SECTIONS = 4

# The section splitter.
_SECTION_SPLITER = ("Seção", "SEÇÃO")

# The number of sections.
_NUM_SECTIONS = 5

# The number of paragraphs.
_NUM_PARAGRAPHS = 5

# The postscript marker.
_POSTSCRIPT_MARKER = ("P.S.", "P.P.S")

# The number of keywords.
_NUM_KEYWORDS = 2

# The occurrences of a single keyword.
_KEYWORD_FREQUENCY = 3

# The occurrences of a single letter.
_LETTER_FREQUENCY = 10

# The occurrences of words with all capital letters.
_ALL_CAPITAL_WORD_FREQUENCY = 20

# The number of words in the response.
_NUM_WORDS_LOWER_LIMIT = 50
_NUM_WORDS_UPPER_LIMIT = 300


class TaskVerifier:
    """An instruction template."""

    def __init__(self, instruction_id):
        self.id = instruction_id

    def build_description(self, **kwargs):
        raise NotImplementedError("`build_description` not implemented.")

    def get_instruction_args(self):
        raise NotImplementedError("`get_instruction_args` not implemented.")

    def get_instruction_args_keys(self):
        raise NotImplementedError("`get_instruction_args_keys` not implemented.")

    def check_following(self, value):
        raise NotImplementedError("`check_following` not implemented.")


class ResponseLanguageChecker(TaskVerifier):
    """Check the language of the entire response."""

    def build_description(self, *, language=None):
        """Build the instruction description.

        Args:
          language: A string representing the expected language of the response. The
            language has to comply to the 97 types defined in
            `langid.py` (https://pypi.org/project/langid/1.1.5/), which follows
            ISO 639-1 codes (https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes);
            for example, `en` for English, `zh` for Chinese, `fr` for French.

        Returns:
          A string representing the instruction description.
        """
        self._language = language
        if self._language is None:
            self._language = random.choice(list(_LANGUAGES.keys()))
        # TODO(tianjianlu): opens the description generation to more choices.
        self._description_pattern = (
            "Toda a sua resposta deve estar em {language}, nenhuma outra "
            + "linguagem é permitida."
        )
        return self._description_pattern.format(language=_LANGUAGES[self._language])

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"language": self._language}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["language"]

    def check_following(self, value):
        """Check if the language of the entire response follows the instruction.

        Args:
          value: A string representing the response.

        Returns:
          True if the language of `value` follows instruction; otherwise False.
        """
        assert isinstance(value, str)

        try:
            return langdetect.detect(value) == self._language
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )  # refex: disable=pytotw.037
            return True


class NumberOfSentences(TaskVerifier):
    """Check the number of sentences."""

    def build_description(self, *, num_sentences=None, relation=None):
        """Build the instruction description.

        Args:
          num_sentences: An integer specifying the number of sentences as a
            threshold.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of sentences < the threshold;
            if 'at least', the actual number of sentences >= the threshold.

        Returns:
          A string representing the instruction description.
        """
        # The number of sentences as a threshold for comparison.
        self._num_sentences_threshold = num_sentences
        if self._num_sentences_threshold is None or self._num_sentences_threshold < 0:
            self._num_sentences_threshold = random.randint(1, _MAX_NUM_SENTENCES)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                "Os tipos de relação suportados para comparação devem estar em " \
                "{_COMPARISON_RELATION}, mas {relation} foi fornecida."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = (
            "Sua resposta deve conter {relation} {num_sentences} sentenças."
        )
        return self._description_pattern.format(
            relation=self._comparison_relation,
            num_sentences=self._num_sentences_threshold,
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "num_sentences": self._num_sentences_threshold,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_sentences", "relation"]

    def check_following(self, value):
        """Check if the number of sentences follows the instruction.

        Args:
          value: A string representing the response.

        Returns:
          True if the response follows the instruction.

        Raise:
            ValueError if the string in `instruction_args` is not in
            [`less_than`, `at_least`].
        """
        num_sentences = utils.count_sentences(value)
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_sentences < self._num_sentences_threshold
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_sentences >= self._num_sentences_threshold


class PlaceholderChecker(TaskVerifier):
    """Check the placeholders in template writing."""

    def build_description(self, *, num_placeholders=None):
        """Build the instruction description.

        Args:
          num_placeholders: An integer denoting the minimum number of
            placeholders required in the response.

        Returns:
          A string representing the instruction description.
        """
        self._num_placeholders = num_placeholders
        if self._num_placeholders is None or self._num_placeholders < 0:
            self._num_placeholders = random.randint(1, _NUM_PLACEHOLDERS)
        self._description_pattern = (
            "A resposta deve conter pelo menos {num_placeholders} espaços reservados "
            "representados por colchetes, como [endereço]."
        )
        return self._description_pattern.format(num_placeholders=self._num_placeholders)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_placeholders": self._num_placeholders}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_placeholders"]

    def check_following(self, value):
        """Check if the number of placeholders follows the instruction.

        Args:
          value: A string representing the response.

        Returns:
          True if the actual number of placeholders in the response is greater than
          or equal to `num_placeholders`; otherwise, False.
        """
        placeholders = re.findall(r"\[.*?\]", value)
        num_placeholders = len(placeholders)
        return num_placeholders >= self._num_placeholders


class BulletListChecker(TaskVerifier):
    """Checks the bullet list in the prompt."""

    def build_description(self, *, num_bullets=None):
        """Build the instruction description.

        Args:
          num_bullets: An integer specifying the exact number of bullet lists
            that is required to appear in the response.

        Returns:
          A string representing the instruction description.
        """
        self._num_bullets = num_bullets
        if self._num_bullets is None or self._num_bullets < 0:
            self._num_bullets = random.randint(1, _NUM_BULLETS)
        self._description_pattern = (
            "Sua resposta deve conter exatamente {num_bullets} itens. "
            "Use os marcadores markdown, como:\n"
            + "* Este é o ponto 1. \n"
            + "* Este é o ponto 2"
        )
        return self._description_pattern.format(num_bullets=self._num_bullets)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_bullets": self._num_bullets}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_bullets"]

    def check_following(self, value):
        r"""Check if the number of bullet lists meets the requirement.

        Args:
          value: A string representing the response. The response is expected to
            contain some bullet lists that start with `\*`.

        Returns:
          True if the actual number of bullet lists in the response meets the
          requirement.
        """
        bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
        bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
        num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
        return num_bullet_lists == self._num_bullets


class ConstrainedResponseChecker(TaskVerifier):
    """Checks the constrained response."""

    def build_description(self):
        """Build the instruction description."""
        # A sequence of string(s) representing the options of the expected response.
        self._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS
        self._description_pattern = (
            "Responda com uma das seguintes opções: {response_options}"
        )
        return self._description_pattern.format(
            response_options=self._constrained_responses
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response matches the constrained options.

        Args:
          value: A string representing the response.

        Returns:
          True if the actual response contains one of the options in the constrained
          responses; otherwise False.
        """
        value = value.strip()
        for constrained_response in self._constrained_responses:
            if constrained_response in value:
                return True
        return False


class ConstrainedStartChecker(TaskVerifier):
    """Checks the response start."""

    def build_description(self, *, starter=None):
        """Build the instruction description.

        Args:
          starter: A string representing the keyword that the response should start
            with.

        Returns:
          A string representing the instruction description.
        """
        self._starter = starter.strip() if isinstance(starter, str) else starter
        if self._starter is None:
            self._starter = random.choice(_STARTER_OPTIONS)
        self._description_pattern = (
            "Durante a conversa, quando for sua vez, "
            + "por favor, comece sempre com {starter}"
        )
        return self._description_pattern.format(starter=self._starter)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"starter": self._starter}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["starter"]

    def check_following(self, value):
        """Checks if the response starts with the constrained keyword or phrase.

        Args:
          value: A string representing the response.

        Returns:
          True if the response starts with the given phrase or keyword that is
          contained in `instruction_args`; otherwise, False.
        """
        response_pattern = r"^\s*" + self._starter + r".*$"
        response_with_constrained_start = re.search(
            response_pattern, value, flags=re.MULTILINE
        )
        return True if response_with_constrained_start else False


class HighlightSectionChecker(TaskVerifier):
    """Checks the highlighted section."""

    def build_description(self, *, num_highlights=None):
        """Build the instruction description.

        Args:
          num_highlights: An integer specifying the minimum number of highlighted
            sections.

        Returns:
          A string representing the instruction description.
        """
        self._num_highlights = num_highlights
        if self._num_highlights is None or self._num_highlights < 0:
            self._num_highlights = random.randint(1, _NUM_HIGHLIGHTED_SECTIONS)

        self._description_pattern = (
            "Destaque pelo menos {num_highlights} seções em sua resposta com "
            + "markdown, ou seja, *seção destacada*."
        )

        return self._description_pattern.format(num_highlights=self._num_highlights)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_highlights": self._num_highlights}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_highlights"]

    def check_following(self, value):
        """Checks if the number of highlighted sections meets the requirement.

        Args:
          value: a string representing the response. The response is expected to
            contain highlighted sections in the format of *highlighted*.

        Returns:
          True if the actual number of highlighted sections in the format of
          *highlighted sections* meets the minimum requirement; otherwise False.
        """
        num_highlights = 0
        highlights = re.findall(r"\*[^\n\*]*\*", value)
        double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", value)
        for highlight in highlights:
            if highlight.strip("*").strip():
                num_highlights += 1
        for highlight in double_highlights:
            if highlight.removeprefix("**").removesuffix("**").strip():
                num_highlights += 1

        return num_highlights >= self._num_highlights


class SectionChecker(TaskVerifier):
    """Checks the sections."""

    def build_description(self, *, section_spliter=None, num_sections=None):
        """Build the instruction description.

        Args:
          section_spliter: A string represents the section spliter keyword that
            marks a new section, i.e., `Section` or `SECTION`.
          num_sections: An integer specifying the number of sections.

        Returns:
          A string representing the instruction description.
        """
        self._section_spliter = (
            section_spliter.strip()
            if isinstance(section_spliter, str)
            else section_spliter
        )
        if self._section_spliter is None:
            self._section_spliter = random.choice(_SECTION_SPLITER)

        self._num_sections = num_sections
        if self._num_sections is None or self._num_sections < 0:
            self._num_sections = random.randint(1, _NUM_SECTIONS)

        self._description_pattern = (
            "Sua resposta deve ter {num_sections} seções. Marque o início "
            + "de cada seção com {section_spliter} X, como:\n"
            + "{section_spliter} 1\n"
            + "[conteúdo da seção 1]\n"
            + "{section_spliter} 2\n"
            + "[conteúdo da seção 2]"
        )

        return self._description_pattern.format(
            num_sections=self._num_sections, section_spliter=self._section_spliter
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "section_spliter": self._section_spliter,
            "num_sections": self._num_sections,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["section_spliter", "num_sections"]

    def check_following(self, value):
        """Checks the response contains multiple sections.

        Args:
          value: A string representing the response. The response is expected
            to contain multiple sections (number of sections is greater than 1).
            A new section starts with `Section 1`, where the number denotes the
            section index.

        Returns:
          True if the number of sections in the response is greater than or equal to
          the minimum number of sections; otherwise, False.
        """
        section_splitter_patten = r"\s?" + self._section_spliter + r"\s?\d+\s?"
        sections = re.split(section_splitter_patten, value)
        num_sections = len(sections) - 1
        return num_sections >= self._num_sections


class ParagraphChecker(TaskVerifier):
    """Checks the paragraphs."""

    def build_description(self, *, num_paragraphs=None):
        """Build the instruction description.

        Args:
          num_paragraphs: An integer specifying the number of paragraphs.

        Returns:
          A string representing the instruction description.
        """
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

        self._description_pattern = (
            "Sua resposta deve ter {num_paragraphs} parágrafos. "
            + "Os parágrafos são separados pelo divisor markdown: ***"
        )

        return self._description_pattern.format(num_paragraphs=self._num_paragraphs)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_paragraphs": self._num_paragraphs}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_paragraphs"]

    def check_following(self, value):
        """Checks the response contains required number of paragraphs.

        Args:
          value: A string representing the response. The response may contain
            paragraphs that are separated by the markdown divider: `***`.

        Returns:
          True if the actual number of paragraphs is the same as required;
          otherwise, False.
        """
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        num_paragraphs = len(paragraphs)

        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    num_paragraphs -= 1
                else:
                    return False

        return num_paragraphs == self._num_paragraphs


class PostscriptChecker(TaskVerifier):
    """Checks the postscript."""

    def build_description(self, *, postscript_marker=None):
        """Build the instruction description.

        Args:
          postscript_marker: A string containing the keyword that marks the start
            of the postscript section.

        Returns:
          A string representing the instruction description.
        """
        self._postscript_marker = (
            postscript_marker.strip()
            if isinstance(postscript_marker, str)
            else postscript_marker
        )
        if self._postscript_marker is None:
            self._postscript_marker = random.choice(_POSTSCRIPT_MARKER)

        self._description_pattern = (
            "No final da sua resposta, por favor adicione explicitamente um posfácio "
            + "começando com {postscript}"
        )

        return self._description_pattern.format(postscript=self._postscript_marker)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"postscript_marker": self._postscript_marker}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["postscript_marker"]

    def check_following(self, value):
        """Checks if the response follows the postscript format.

        Args:
          value: a string representing the response. The response is expected to
            contain a postscript section.

        Returns:
          True if the response contains a postscript section starting with
          the keyword containing in the `instruction_args`; otherwise False.
        """
        value = value.lower()
        if self._postscript_marker == "P.P.S":
            postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif self._postscript_marker == "P.S.":
            postscript_pattern = r"\s*p\.\s?s\..*$"
        else:
            postscript_pattern = r"\s*" + self._postscript_marker.lower() + r".*$"
        postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
        return True if postscript else False


class RephraseChecker(TaskVerifier):
    """Checks the rephrase."""

    def build_description(self, *, original_message):
        """Build the instruction description.

        Args:
          original_message: A string representing the original message. The
            rephrased response should only change its words/sentences in between
            its two asterisks, for example, *change me*. Both original and rephrased
            messages should contain the changes in the form of *change me*.

        Returns:
          A string representing the instruction description.
        """
        if not self.is_change(original_message):
            raise ValueError(
                f"Mensagem {original_message} não contém alterações "
                "na forma de *mude-me*."
            )

        self._reference_without_change = original_message
        self._description = (
            "Parafraseando: Sua resposta reformulada deve apenas"
            + "alterar as palavras/frases entre dois asteriscos"
            + "como *mude-me*.\n"
        )
        return self._description

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"original_message": self._reference_without_change}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["original_message"]

    def check_following(self, value):
        r"""Checks if the rephrasing follows the instruction.

        Args:
          value: A string representing the response, which is expected to rephras
            the string of `instruction_args`.

        Returns:
          True if `value` and `instruction_args` only differ by the words/sentences
          in between two asterisks such as *change me*; otherwise, False.
        """

        if not self.is_change(value):
            raise ValueError(
                f"valor {value} não contém alterações na forma de *mude-me*."
            )

        response_without_changes = self.strip_changes(value)
        reference_without_changes = self.strip_changes(self._reference_without_change)

        return response_without_changes == reference_without_changes

    def is_change(self, response):
        """Check if there is change in the response in the form of *change me*."""
        return re.search(r"\*.*\*", response)

    def strip_changes(self, response):
        """Strips off the changes."""
        return re.sub(r"\*.*\*", "", response)


class KeywordChecker(TaskVerifier):
    """Check the existence of certain keywords."""

    def build_description(self, *, keywords=None):
        """Build the instruction description.

        Args:
          keywords: A sequence of strings representing the keywords that are
            expected in the response.

        Returns:
          A string representing the instruction description.
        """

        if not keywords:
            self._keywords = utils.generate_keywords(
                num_keywords=_NUM_KEYWORDS
            )
        else:
            self._keywords = keywords
        self._keywords = sorted(self._keywords)
                                    
        self._description_pattern = "Inclua as palavras-chave {keywords} na resposta."

        return self._description_pattern.format(keywords=self._keywords)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"keywords": self._keywords}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keywords"]

    def check_following(self, value):
        """Check if the response contain the expected keywords."""
        for keyword in self._keywords:
            if not re.search(keyword, value, flags=re.IGNORECASE):
                return False
        return True


class KeywordFrequencyChecker(TaskVerifier):
    """Check the keyword frequency."""

    def build_description(self, *, keyword=None, frequency=None, relation=None):
        """Build the instruction description.

        Args:
          keyword: A string representing a keyword that is expected in the response.
          frequency: An integer specifying the number of times `keyword` is expected
            to appear in the response.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of occurrences < frequency;
            if 'at least', the actual number of occurrences >= frequency.

        Returns:
          A string representing the instruction description.
        """
        if not keyword:
            self._keyword = utils.generate_keywords(num_keywords=1)[0]
        else:
            self._keyword = keyword.strip()

        self._frequency = frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _KEYWORD_FREQUENCY)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                "As tipos de relação suportados para comparação devem estar em "
                f"{_COMPARISON_RELATION}, mas {relation} foi fornecida."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = (
            "Na sua resposta, a palavra {keyword} deve aparecer {relation} "
            + "{frequency} vezes."
        )

        return self._description_pattern.format(
            keyword=self._keyword,
            relation=self._comparison_relation,
            frequency=self._frequency,
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "keyword": self._keyword,
            "frequency": self._frequency,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keyword", "frequency", "relation"]

    def check_following(self, value):
        """Checks if the response contain the keyword with required frequency."""
        actual_occurrences = len(re.findall(self._keyword, value, flags=re.IGNORECASE))

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return actual_occurrences < self._frequency
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return actual_occurrences >= self._frequency


class NumberOfWords(TaskVerifier):
    """Checks the number of words."""

    def build_description(self, *, num_words=None, relation=None):
        """Build the instruction description.

        Args:
          num_words: An integer specifying the number of words contained in the
            response.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of words < num_words;
            if 'at least', the actual number of words >= num_words.

        Returns:
          A string representing the instruction description.
        """

        self._num_words = num_words
        if self._num_words is None or self._num_words < 0:
            self._num_words = random.randint(
                _NUM_WORDS_LOWER_LIMIT, _NUM_WORDS_UPPER_LIMIT
            )

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                "As tipos de relação suportados para comparação devem estar em "
                f"{_COMPARISON_RELATION}, mas {relation} foi fornecida."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = "Responda com {relation} {num_words} palavras."

        return self._description_pattern.format(
            relation=self._comparison_relation, num_words=self._num_words
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_words": self._num_words, "relation": self._comparison_relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_words", "relation"]

    def check_following(self, value):
        """Checks if the response contains the expected number of words."""
        num_words = utils.count_words(value)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_words < self._num_words
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_words >= self._num_words


class JsonFormat(TaskVerifier):
    """Check the Json format."""

    def build_description(self):
        self._description_pattern = (
            "Todo o output deve estar em formato JSON. Você pode usar "
            "marcadores markdown como ```."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        value = (
            value.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(value)
        except ValueError:
            return False
        return True


class ParagraphFirstWordCheck(TaskVerifier):
    """Check the paragraph and the first word of the nth paragraph."""

    def build_description(
        self, num_paragraphs=None, nth_paragraph=None, first_word=None
    ):
        r"""Build the instruction description.

        Args:
          num_paragraphs: An integer indicating the number of paragraphs expected
            in the response. A paragraph is a subset of the string that is
            expected to be separated by '\n\n'.
          nth_paragraph: An integer indicating the paragraph number that we look at.
            Note that n starts from 1.
          first_word: A string that represent the first word of the bth paragraph.

        Returns:
          A string representing the instruction description.
        """
        self._num_paragraphs = int(num_paragraphs) if num_paragraphs is not None else num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

        self._nth_paragraph = int(nth_paragraph) if nth_paragraph is not None else nth_paragraph
        if (
            self._nth_paragraph is None
            or self._nth_paragraph <= 0
            or self._nth_paragraph > self._num_paragraphs
        ):
            self._nth_paragraph = random.randint(1, self._num_paragraphs + 1)

        self._first_word = first_word
        if self._first_word is None:
            self._first_word = utils.generate_keywords(num_keywords=1)[0]
        self._first_word = self._first_word.lower()

        self._description_pattern = (
            "Deve haver {num_paragraphs} parágrafos. "
            +  "Parágrafos e apenas parágrafos são separados entre si por duas "
            + " quebras de linha como se fosse '\\n\\n' em python. "
            + "O parágrafo {nth_paragraph} deve começar com a palavra {first_word}."
        )

        return self._description_pattern.format(
            num_paragraphs=self._num_paragraphs,
            nth_paragraph=self._nth_paragraph,
            first_word=self._first_word,
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_word": self._first_word,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_paragraphs", "nth_paragraph", "first_word"]

    def check_following(self, value):
        """Checks for required number of paragraphs and correct first word.

        Args:
          value: a string representing the response. The response may contain
            paragraphs that are separated by two new lines and the first word of
            the nth paragraph will have to match a specified word.

        Returns:
          True if the number of paragraphs is the same as required and the first
          word of the specified paragraph is the same as required. Otherwise, false.
        """

        paragraphs = re.split(r"\n\n", value)
        num_paragraphs = len(paragraphs)

        for paragraph in paragraphs:
            if not paragraph.strip():
                num_paragraphs -= 1

        # check that index doesn't go out of bounds
        if self._nth_paragraph <= num_paragraphs:
            paragraph = paragraphs[int(self._nth_paragraph) - 1].strip() # cast to int in case of float (5.0 -> 5)
            if not paragraph:
                return False
        else:
            return False

        first_word = ""
        punctuation = {".", ",", "?", "!", "'", '"'}

        # get first word and remove punctuation
        word = paragraph.split()[0].strip()
        # TODO(jeffrey): make more complex?
        word = word.lstrip("'")
        word = word.lstrip('"')

        for letter in word:
            if letter in punctuation:
                break
            first_word += letter.lower()

        return num_paragraphs == self._num_paragraphs and first_word == self._first_word

class KeySentenceChecker(TaskVerifier):
    """Check the existence of certain key sentences."""

    def build_description(self, key_sentences=None, num_sentences=None):
        """Build the instruction description.

        Args:
          key_sentences: A sequences of strings representing the key sentences that
            are expected in the response.
          num_sentences: The number of key sentences that are expected to be seen in
            the response.

        Returns:
          A string representing the instruction description.
        """

        if not key_sentences:
            # TODO(jeffrey) make a generate sentences function? wonderwords package
            self._key_sentences = set(["Por enquanto, isso é o bastante."])
        else:
            self._key_sentences = key_sentences

        if not num_sentences:
            self._num_sentences = random.randint(1, len(self._key_sentences))
        else:
            self._num_sentences = num_sentences

        self._description_pattern = (
            "Inclua {num_sentences} das seguintes frases {key_sentences}"
        )

        return self._description_pattern.format(
            num_sentences=self._num_sentences, key_sentences=self._key_sentences
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "num_sentences": self._num_sentences,
            "key_sentences": list(self._key_sentences),
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_sentences", "key_sentences"]

    def check_following(self, value):
        """Checks if the response contains the expected key sentences."""
        count = 0
        sentences = utils.split_into_sentences(value)
        for sentence in self._key_sentences:
            if sentence in sentences:
                count += 1

        return count == self._num_sentences


class ForbiddenWords(TaskVerifier):
    """Checks that specified words are not used in response."""

    def build_description(self, forbidden_words=None):
        """Build the instruction description.

        Args:
          forbidden_words: A sequences of strings representing words that are not
            allowed in the response.

        Returns:
          A string representing the instruction description.
        """

        if not forbidden_words:
            self._forbidden_words = utils.generate_keywords(
                num_keywords=_NUM_KEYWORDS
            )
        else:
            self._forbidden_words = list(set(forbidden_words))
        self._forbidden_words = sorted(self._forbidden_words)
        self._description_pattern = (
            "Não inclua as palavras-chave {forbidden_words} na resposta."
        )

        return self._description_pattern.format(forbidden_words=self._forbidden_words)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"forbidden_words": self._forbidden_words}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["forbidden_words"]

    def check_following(self, value):
        """Check if the response does not contain the expected keywords."""
        for word in self._forbidden_words:
            if re.search(r"\b" + word + r"\b", value, flags=re.IGNORECASE):
                return False
        return True


class RephraseParagraph(TaskVerifier):
    """Checks that the paragraph is rephrased."""

    def build_description(self, *, original_paragraph, low, high):
        """Builds the instruction description.

        Args:
          original_paragraph: A string presenting the original paragraph. The
            rephrases response should have between low-high words in common.
          low: An integer presenting the lower bound of similar words.
          high: An integer representing the upper bound of similar words.

        Returns:
          A string representing the instruction description.
        """
        # TODO(jeffrey) make more encompassing
        self._original_paragraph = original_paragraph
        self._low = low
        self._high = high

        self._description = (
            "Reescreva o seguinte parágrafo: "
            + "{original_paragraph}\nSua resposta deve conter "
            + "entre {low} e {high} das mesmas palavras. "
            + "Palavras são as mesmas se e somente se todas as "
            + "letras, ignorando maiúsculas e minúsculas, são as mesmas. Por "
            + "exemplo, 'correr' é o mesmo que 'Correr' mas diferente "
            + "de 'correu'."
        )

        return self._description.format(
            original_paragraph=original_paragraph, low=self._low, high=self._high
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "original_paragraph": self._original_paragraph,
            "low": self._low,
            "high": self._high,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["original_paragraph", "low", "high"]

    def check_following(self, value):
        val_words = re.findall(r"\w+", value.lower())
        original_words = re.findall(r"\w+", self._original_paragraph.lower())
        similar_words = 0

        dict_val = collections.Counter(val_words)
        dict_original = collections.Counter(original_words)

        for word in dict_original:
            similar_words += min(dict_original[word], dict_val[word])

        return similar_words >= self._low and similar_words <= self._high


class TwoResponsesChecker(TaskVerifier):
    """Check that two responses were given."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Dê duas respostas diferentes. As respostas e somente as respostas devem"
            " ser separadas por 6 símbolos de asterisco: ******."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response has two different answers.

        Args:
          value: A string representing the response.

        Returns:
          True if two responses are detected and false otherwise.
        """
        valid_responses = list()
        responses = value.split("******")
        for index, response in enumerate(responses):
            if not response.strip():
                if index != 0 and index != len(responses) - 1:
                    return False
            else:
                valid_responses.append(response)
        return (
            len(valid_responses) == 2
            and valid_responses[0].strip() != valid_responses[1].strip()
        )


class RepeatPromptThenAnswer(TaskVerifier):
    """Checks that Prompt is first repeated then answered."""

    def build_description(self, *, prompt_to_repeat=None):
        """Build the instruction description.

        Args:
          prompt_to_repeat: The prompt that is meant to be repeated.

        Returns:
          A string representing the instruction description.
        """
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        else:
            self._prompt_to_repeat = prompt_to_repeat
        self._description_pattern = (
            "Primeiro repita o pedido palavra por palavra sem alterações,"
            " depois dê sua resposta (1. não diga nenhuma palavra ou caractere"
            " antes de repetir o pedido; 2. o pedido que você precisa repetir"
            " não inclui esta frase)"
        )
        return self._description_pattern

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["prompt_to_repeat"]

    def check_following(self, value):
        if value.strip().lower().startswith(self._prompt_to_repeat.strip().lower()):
            return True
        return False


class EndChecker(TaskVerifier):
    """Checks that the prompt ends with a given phrase."""

    def build_description(self, *, end_phrase=None):
        """Build the instruction description.

        Args:
          end_phrase: A string representing the phrase the response should end with.

        Returns:
          A string representing the instruction description.
        """
        self._end_phrase = (
            end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
        )
        if self._end_phrase is None:
            self._end_phrase = random.choice(_ENDING_OPTIONS)
        self._description_pattern = (
            "Termine sua resposta com esta frase exata {ender}. "
            "Nenhuma outra palavra deve seguir esta frase."
        )
        return self._description_pattern.format(ender=self._end_phrase)

    def get_instruction_args(self):
        return {"end_phrase": self._end_phrase}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["end_phrase"]

    def check_following(self, value):
        """Checks if the response ends with the expected phrase."""
        value = value.strip().strip('"').lower()
        self._end_phrase = self._end_phrase.strip().lower()
        return value.endswith(self._end_phrase)


class TitleChecker(TaskVerifier):
    """Checks the response for a title."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Sua resposta deve conter um título, envolto em duplas setas angulares,"
            " como <<poema de alegria>>."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response contains a title."""
        pattern = r"<<[^\n]+>>"
        re_pattern = re.compile(pattern)
        titles = re.findall(re_pattern, value)

        for title in titles:
            if title.lstrip("<").rstrip(">").strip():
                return True
        return False


class LetterFrequencyChecker(TaskVerifier):
    """Checks letter frequency."""

    def build_description(self, *, letter=None, let_frequency=None, let_relation=None):
        """Build the instruction description.

        Args:
          letter: A string representing a letter that is expected in the response.
          let_frequency: An integer specifying the number of times `keyword` is
            expected to appear in the response.
          let_relation: A string in (`less than`, `at least`), defining the
            relational operator for comparison. Two relational comparisons are
            supported for now; if 'less than', the actual number of
            occurrences < frequency; if 'at least', the actual number of
            occurrences >= frequency.

        Returns:
          A string representing the instruction description.
        """
        if (
            not letter
            or len(letter) > 1
            or ord(letter.lower()) < 97
            or ord(letter.lower()) > 122
        ):
            self._letter = random.choice(list(string.ascii_letters))
        else:
            self._letter = letter.strip()
        self._letter = self._letter.lower()

        self._frequency = let_frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _LETTER_FREQUENCY)

        if let_relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif let_relation not in _COMPARISON_RELATION:
            raise ValueError(
                "As tipos de relação suportados para comparação devem estar em "
                f"{_COMPARISON_RELATION}, mas {let_relation} foi fornecida."
            )
        else:
            self._comparison_relation = let_relation

        self._description_pattern = (
            "Em sua resposta, a letra {letter} deve aparecer {let_relation}"
            " {let_frequency} vezes."
        )

        return self._description_pattern.format(
            letter=self._letter,
            let_frequency=self._frequency,
            let_relation=self._comparison_relation,
        )

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {
            "letter": self._letter,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["letter", "let_frequency", "let_relation"]

    def check_following(self, value):
        """Checks that the response contains the letter at the right frequency."""
        value = value.lower()
        letters = collections.Counter(value)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return letters[self._letter] < self._frequency
        else:
            return letters[self._letter] >= self._frequency


class CapitalLettersPortugueseChecker(TaskVerifier):
    """Checks that the response is in portuguese and is in all capital letters."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Sua resposta inteira deve estar em português e em todas as letras maiúsculas."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response is in Portuguese and in all capital letters."""
        assert isinstance(value, str)

        try:
            return value.isupper() and langdetect.detect(value) == "pt"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )  # refex: disable=pytotw.037
            return True


class LowercaseLettersPortugueseChecker(TaskVerifier):
    """Checks that the response is in portuguese and is in all lowercase letters."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Sua resposta inteira deve estar em português e em todas as letras minúsculas."
            " Nenhuma letra maiúscula é permitida."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response is in Portuguese and in all lowercase letters."""
        assert isinstance(value, str)

        try:
            return value.islower() and langdetect.detect(value) == "pt"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )  # refex: disable=pytotw.037
            return True


class CommaChecker(TaskVerifier):
    """Checks the response for no commas."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Em sua resposta, evite o uso de vírgulas."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response does not contain commas."""
        return not re.search(r"\,", value)


class CapitalWordFrequencyChecker(TaskVerifier):
    """Checks frequency of words with all capital letters."""

    def build_description(
        self,
        capital_frequency=None,
        capital_relation=None,
    ):
        """Build the instruction description.

        Args:
          capital_frequency: An integer that represents the number of words that
            should be in all capital letters.
          capital_relation: A string that is 'at least' or 'at most' that refers to
            the frequency.

        Returns:
          A string representing the instruction description.
        """
        self._frequency = capital_frequency
        if self._frequency is None:
            self._frequency = random.randint(1, _ALL_CAPITAL_WORD_FREQUENCY)

        self._comparison_relation = capital_relation
        if capital_relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif capital_relation not in _COMPARISON_RELATION:
            raise ValueError(
                "Os tipos de relação suportados para comparação devem estar em "
                f"{_COMPARISON_RELATION}, mas {capital_relation} foi fornecido."
            )

        self._description_pattern = (
            "Em sua resposta, palavras com todas as letras maiúsculas devem aparecer"
            " {relation} {frequency} vezes."
        )

        return self._description_pattern.format(
            frequency=self._frequency, relation=self._comparison_relation
        )

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {
            "capital_frequency": self._frequency,
            "capital_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["capital_frequency", "capital_relation"]

    def check_following(self, value):
        """Checks the frequency of words with all capital letters."""
        # Hyphenated words will count as one word
        words = utils.nltk.word_tokenize(value)
        capital_words = [word for word in words if word.isupper()]

        capital_words = len(capital_words)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return capital_words < self._frequency
        else:
            return capital_words >= self._frequency


class QuotationChecker(TaskVerifier):
    """Checks response is wrapped with double quotation marks."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Envolva toda a sua resposta com aspas duplas."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response is wrapped with double quotation marks."""
        value = value.strip()
        return len(value) > 1 and value[0] == '"' and value[-1] == '"'


class CommonWordsChecker(TaskVerifier):
    """Checks that the response contains the most frequent words from a list."""

    def build_description(self, *, expected_words=None):
        self._expected_words = expected_words or []
        self._description_pattern = (
            "Identifique as {n} palavras mais frequentes na lista."
        )
        return self._description_pattern.format(n=len(self._expected_words))

    def get_instruction_args(self):
        return {"expected_words": self._expected_words}

    def get_instruction_args_keys(self):
        return ["expected_words"]

    def check_following(self, value):
        """Check if the response mentions all expected common words."""
        if not self._expected_words:
            return True
        value_lower = value.lower()
        found = sum(1 for w in self._expected_words if w.lower() in value_lower)
        return found / len(self._expected_words) >= 0.5


class RareWordsChecker(TaskVerifier):
    """Checks that the response contains the least frequent words from a list."""

    def build_description(self, *, expected_words=None):
        self._expected_words = expected_words or []
        self._description_pattern = (
            "Identifique as {n} palavras menos frequentes na lista."
        )
        return self._description_pattern.format(n=len(self._expected_words))

    def get_instruction_args(self):
        return {"expected_words": self._expected_words}

    def get_instruction_args_keys(self):
        return ["expected_words"]

    def check_following(self, value):
        """Check if the response mentions all expected rare words."""
        if not self._expected_words:
            return True
        value_lower = value.lower()
        found = sum(1 for w in self._expected_words if w.lower() in value_lower)
        return found / len(self._expected_words) >= 0.5


class CountWordChecker(TaskVerifier):
    """Checks that the response contains the correct count for a target word."""

    def build_description(self, *, target_word=None, expected_count=None):
        self._target_word = target_word or ""
        self._expected_count = int(expected_count) if expected_count is not None else 0
        self._description_pattern = (
            "Conte quantas vezes a palavra \"{word}\" aparece na lista."
        )
        return self._description_pattern.format(word=self._target_word)

    def get_instruction_args(self):
        return {
            "target_word": self._target_word,
            "expected_count": self._expected_count,
        }

    def get_instruction_args_keys(self):
        return ["target_word", "expected_count"]

    def check_following(self, value):
        """Check if the response contains the correct count number."""
        return str(self._expected_count) in value


class WordAtPositionChecker(TaskVerifier):
    """Checks that the response contains the word found at a given position."""

    def build_description(self, *, position=None, expected_word=None):
        self._position = int(position) if position is not None else 0
        self._expected_word = expected_word or ""
        self._description_pattern = (
            "Identifique a palavra na posição {pos} da lista."
        )
        return self._description_pattern.format(pos=self._position)

    def get_instruction_args(self):
        return {
            "position": self._position,
            "expected_word": self._expected_word,
        }

    def get_instruction_args_keys(self):
        return ["position", "expected_word"]

    def check_following(self, value):
        """Check if the response contains the expected word."""
        return self._expected_word.lower() in value.lower()


class FrequencyComparisonChecker(TaskVerifier):
    """Checks that the response correctly identifies which word is more frequent."""

    def build_description(self, *, word_a=None, word_b=None, expected_winner=None):
        self._word_a = word_a or ""
        self._word_b = word_b or ""
        self._expected_winner = expected_winner or ""
        self._description_pattern = (
            "Compare a frequência de \"{a}\" e \"{b}\" na lista."
        )
        return self._description_pattern.format(a=self._word_a, b=self._word_b)

    def get_instruction_args(self):
        return {
            "word_a": self._word_a,
            "word_b": self._word_b,
            "expected_winner": self._expected_winner,
        }

    def get_instruction_args_keys(self):
        return ["word_a", "word_b", "expected_winner"]

    def check_following(self, value):
        """Check if the response mentions the expected winner word."""
        return self._expected_winner.lower() in value.lower()


class NeedleSingleNumberChecker(TaskVerifier):
    """Checks that the response contains the single hidden number."""

    def build_description(self, *, key=None, expected_values=None):
        self._key = key or ""
        self._expected_values = expected_values or {}
        self._description_pattern = (
            "Encontre o número especial para {key} no texto."
        )
        return self._description_pattern.format(key=self._key)

    def get_instruction_args(self):
        return {"key": self._key, "expected_values": self._expected_values}

    def get_instruction_args_keys(self):
        return ["key", "expected_values"]

    def check_following(self, value):
        """Check if the response contains all expected numbers for the key."""
        for _key, vals in self._expected_values.items():
            for v in vals:
                if str(v) not in value:
                    return False
        return bool(self._expected_values)


class NeedleMultiNumberSameKeyChecker(TaskVerifier):
    """Checks that the response lists all hidden numbers for a single key."""

    def build_description(self, *, key=None, expected_values=None):
        self._key = key or ""
        self._expected_values = expected_values or {}
        self._description_pattern = (
            "Liste todos os números especiais para {key} no texto."
        )
        return self._description_pattern.format(key=self._key)

    def get_instruction_args(self):
        return {"key": self._key, "expected_values": self._expected_values}

    def get_instruction_args_keys(self):
        return ["key", "expected_values"]

    def check_following(self, value):
        """Check if response contains all expected numbers (≥50% threshold)."""
        all_vals = []
        for _key, vals in self._expected_values.items():
            all_vals.extend(vals)
        if not all_vals:
            return True
        found = sum(1 for v in all_vals if str(v) in value)
        return found / len(all_vals) >= 0.5


class NeedleMultiNumberDiffKeysChecker(TaskVerifier):
    """Checks that the response contains numbers for different keys."""

    def build_description(self, *, expected_values=None):
        self._expected_values = expected_values or {}
        keys_str = ", ".join(self._expected_values.keys()) if self._expected_values else ""
        self._description_pattern = (
            "Liste os números especiais para {keys} no texto."
        )
        return self._description_pattern.format(keys=keys_str)

    def get_instruction_args(self):
        return {"expected_values": self._expected_values}

    def get_instruction_args_keys(self):
        return ["expected_values"]

    def check_following(self, value):
        """Check if response contains numbers for each key (≥50% of all values)."""
        all_vals = []
        for _key, vals in self._expected_values.items():
            all_vals.extend(vals)
        if not all_vals:
            return True
        found = sum(1 for v in all_vals if str(v) in value)
        return found / len(all_vals) >= 0.5


class NeedleUUIDChecker(TaskVerifier):
    """Checks that the response contains the correct UUID value for the queried key."""

    def build_description(self, *, query_key=None, expected_values=None):
        self._query_key = query_key or ""
        self._expected_values = expected_values or {}
        self._description_pattern = (
            "Encontre o código UUID para {key} no texto."
        )
        return self._description_pattern.format(key=self._query_key)

    def get_instruction_args(self):
        return {"query_key": self._query_key, "expected_values": self._expected_values}

    def get_instruction_args_keys(self):
        return ["query_key", "expected_values"]

    def check_following(self, value):
        """Check if the response contains the expected UUID value."""
        for _key, vals in self._expected_values.items():
            for v in vals:
                if str(v).lower() in value.lower():
                    return True
        return not self._expected_values


class MathAnswerChecker(TaskVerifier):
    """Checks that the response contains the expected numerical answer."""

    def build_description(self, *, expected_answer=None):
        self._expected_answer = str(expected_answer) if expected_answer is not None else ""
        self._description_pattern = (
            "Resolva o problema matemático e forneça a resposta correta."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return {"expected_answer": self._expected_answer}

    def get_instruction_args_keys(self):
        return ["expected_answer"]

    def check_following(self, value):
        """Check if the response contains the expected answer."""
        if not self._expected_answer:
            return False
        return self._expected_answer in value


def _extract_email_json(value):
    """Extract and parse a JSON object from a response string.

    Accepts both ```json...``` (or ```) code blocks and raw JSON strings.
    Returns a dict on success, or None if parsing fails.
    """
    stripped = value.strip()
    # Try to extract from a fenced code block first
    m = re.search(r"```(?:json|JSON|Json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except (ValueError, json.JSONDecodeError):
            pass
    # Fall back to parsing the whole string as JSON
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except (ValueError, json.JSONDecodeError):
        pass
    return None


class EmailJsonFormatChecker(TaskVerifier):
    """Checks that the response is a valid JSON object inside a ```json``` block."""

    def build_description(self):
        self._description_pattern = (
            "Formate sua resposta EXATAMENTE como um bloco JSON markdown:\n"
            "```json\n{\n  ...\n}\n```\n"
            "O JSON deve conter apenas as chaves solicitadas, sem texto adicional."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        """Return True only if response has a ```json``` block with a valid JSON object."""
        stripped = value.strip()
        # Require a fenced code block with json/JSON/Json specifier or plain ```
        if not re.search(r"```(?:json|JSON|Json)?", stripped):
            return False
        return _extract_email_json(value) is not None


class EmailSchemaKeysChecker(TaskVerifier):
    """Checks that the JSON response contains exactly the required keys."""

    def build_description(self, *, required_keys=None):
        self._required_keys = sorted(required_keys or [])
        self._description_pattern = (
            "O objeto JSON deve conter EXATAMENTE as chaves: {keys}. "
            "Nenhuma chave adicional ou faltante é permitida."
        )
        return self._description_pattern.format(keys=", ".join(self._required_keys))

    def get_instruction_args(self):
        return {"required_keys": self._required_keys}

    def get_instruction_args_keys(self):
        return ["required_keys"]

    def check_following(self, value):
        """Return True if the JSON object has exactly the required keys."""
        obj = _extract_email_json(value)
        if obj is None:
            return False
        return set(obj.keys()) == set(self._required_keys)


class EmailFieldValueChecker(TaskVerifier):
    """Checks that a specific field in the JSON response matches the expected value.

    Supports both string and boolean expected values.  For booleans the model
    output may be a JSON boolean (True/False) or a string representation.
    """

    def build_description(self, *, field_name=None, expected_value=None):
        self._field_name = field_name or ""
        self._expected_value = expected_value
        if isinstance(self._expected_value, bool):
            val_str = "true" if self._expected_value else "false"
        else:
            val_str = str(self._expected_value)
        self._description_pattern = (
            "O campo \"{field}\" deve ter o valor: {value}."
        )
        return self._description_pattern.format(
            field=self._field_name, value=val_str
        )

    def get_instruction_args(self):
        return {
            "field_name": self._field_name,
            "expected_value": self._expected_value,
        }

    def get_instruction_args_keys(self):
        return ["field_name", "expected_value"]

    def check_following(self, value):
        """Return True if the JSON field matches the expected value."""
        obj = _extract_email_json(value)
        if obj is None or self._field_name not in obj:
            return False
        actual = obj[self._field_name]
        if isinstance(self._expected_value, bool):
            if isinstance(actual, bool):
                return actual == self._expected_value
            if isinstance(actual, str):
                true_strs = {"true", "sim", "verdadeiro", "yes"}
                false_strs = {"false", "não", "nao", "falso", "no"}
                return (
                    actual.lower() in true_strs
                    if self._expected_value
                    else actual.lower() in false_strs
                )
            return False
        return str(actual).strip() == str(self._expected_value).strip()


_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"


def _extract_tool_call_json(text):
    """Extract the JSON object from within <tool_call>...</tool_call> tags.

    Returns the parsed dict, or None if extraction fails.
    """
    start = text.find(_TOOL_CALL_OPEN)
    end = text.find(_TOOL_CALL_CLOSE)
    if start == -1 or end == -1 or end <= start:
        return None
    inner = text[start + len(_TOOL_CALL_OPEN):end].strip()
    try:
        return json.loads(inner)
    except (json.JSONDecodeError, ValueError):
        return None


class ToolCallFormatChecker(TaskVerifier):
    """Check that the response contains a well-formed <tool_call> block.

    For valid tasks (``expect_call=True``) the response must contain a
    properly formatted ``<tool_call>...</tool_call>`` block with parseable
    JSON inside.

    For invalid/refusal tasks (``expect_call=False``) the response must
    NOT contain any ``<tool_call>`` tag.
    """

    def build_description(self, *, expect_call=True):
        self._expect_call = expect_call
        if self._expect_call:
            return (
                "A resposta deve conter uma chamada de ferramenta "
                "formatada dentro de tags <tool_call>...</tool_call>."
            )
        return (
            "A resposta NÃO deve conter nenhuma chamada de ferramenta. "
            "Explique educadamente por que nenhuma ferramenta se aplica."
        )

    def get_instruction_args(self):
        return {"expect_call": self._expect_call}

    def get_instruction_args_keys(self):
        return ["expect_call"]

    def check_following(self, value):
        has_tag = _TOOL_CALL_OPEN in value
        if self._expect_call:
            return _extract_tool_call_json(value) is not None
        return not has_tag


class ToolCallNameChecker(TaskVerifier):
    """Check that the tool_call uses the correct function name."""

    def build_description(self, *, expected_name=None):
        self._expected_name = expected_name or ""
        return (
            f"A chamada de ferramenta deve invocar a função \"{self._expected_name}\"."
        )

    def get_instruction_args(self):
        return {"expected_name": self._expected_name}

    def get_instruction_args_keys(self):
        return ["expected_name"]

    def check_following(self, value):
        obj = _extract_tool_call_json(value)
        if obj is None:
            return False
        return obj.get("name") == self._expected_name


class ToolCallArgsKeysChecker(TaskVerifier):
    """Check that the tool_call arguments contain exactly the required keys."""

    def build_description(self, *, required_arg_keys=None):
        self._required_keys = sorted(required_arg_keys or [])
        keys_str = ", ".join(self._required_keys)
        return (
            f"Os argumentos da chamada devem conter as chaves: {keys_str}."
        )

    def get_instruction_args(self):
        return {"required_arg_keys": self._required_keys}

    def get_instruction_args_keys(self):
        return ["required_arg_keys"]

    def check_following(self, value):
        obj = _extract_tool_call_json(value)
        if obj is None:
            return False
        args = obj.get("arguments", {})
        if not isinstance(args, dict):
            return False
        return set(self._required_keys).issubset(set(args.keys()))


class ToolCallArgsTypesChecker(TaskVerifier):
    """Check that the tool_call argument values match the expected JSON types.

    ``expected_arg_types`` maps argument name → JSON type string
    (``"string"``, ``"number"``, ``"integer"``, ``"boolean"``, ``"array"``,
    ``"object"``).
    """

    _TYPE_MAP = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    def build_description(self, *, expected_arg_types=None):
        self._expected_types = expected_arg_types or {}
        parts = [f"{k}: {v}" for k, v in sorted(self._expected_types.items())]
        return (
            "Os tipos dos argumentos devem ser: "
            + ", ".join(parts) + "."
        )

    def get_instruction_args(self):
        return {"expected_arg_types": self._expected_types}

    def get_instruction_args_keys(self):
        return ["expected_arg_types"]

    def check_following(self, value):
        obj = _extract_tool_call_json(value)
        if obj is None:
            return False
        args = obj.get("arguments", {})
        if not isinstance(args, dict):
            return False
        for key, expected_type_str in self._expected_types.items():
            if key not in args:
                return False
            py_type = self._TYPE_MAP.get(expected_type_str)
            if py_type is None:
                continue
            if not isinstance(args[key], py_type):
                # Allow int where number is expected
                if expected_type_str == "number" and isinstance(args[key], (int, float)):
                    continue
                return False
        return True


class ToolCallRefusalChecker(TaskVerifier):
    """Check that a refusal response does NOT contain tool_call tags
    and provides a minimum-length explanation."""

    def build_description(self, *, min_refusal_words=None):
        self._min_words = min_refusal_words or 5
        return (
            "A resposta deve recusar educadamente sem usar nenhuma ferramenta "
            f"e conter pelo menos {self._min_words} palavras de explicação."
        )

    def get_instruction_args(self):
        return {"min_refusal_words": self._min_words}

    def get_instruction_args_keys(self):
        return ["min_refusal_words"]

    def check_following(self, value):
        if _TOOL_CALL_OPEN in value:
            return False
        word_count = len(value.split())
        return word_count >= self._min_words


class ThinkingFormatChecker(TaskVerifier):
    """Check that the completion contains non-empty <think>...</think> tags."""

    _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    def build_description(self):
        return (
            "A resposta deve incluir um bloco de raciocínio dentro das tags "
            "<think>...</think> antes da resposta final."
        )

    def get_instruction_args(self):
        return {}

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        match = self._THINK_RE.search(value)
        if match is None:
            return False
        return bool(match.group(1).strip())
