"""Typed answer grader for synthetic agent-trace generation.

Grading is driven by a per-item :class:`AnswerSpec` that
declares how the gold answer must be interpreted:

============  =============================================================
`integer`     exact integer match (Roman numerals, thousands separators and
              surrounding unit text are understood)
`float`       numeric match with an explicit per-item tolerance (exact when
              no tolerance is configured)
`quantity`    numeric match *plus* a unit comparison
`entity`      case/accent-insensitive string match with aliases
`date`        calendar-date match across common formats
`list`        ordered or unordered list match
`exact`       normalized string match that also accepts substring containment
              (default for textual answers)
============  =============================================================

Dataset rows may carry optional schema fields — `answer_type`,
`answer_aliases`, `answer_units`, `answer_precision` (absolute
tolerance), `answer_rtol` (relative tolerance) and `answer_ordered`.
When no `answer_type` is present it is inferred from the gold answer, so a
sample that only carries a `ground_truth` column is still graded correctly.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

ANSWER_TYPES = frozenset({"integer", "float", "quantity", "entity", "date", "list", "exact"})


@dataclass(frozen=True)
class AnswerSpec:
    """The grading schema for a single dataset item.

    Args:
        ground_truth: Canonical gold answer (string).
        type:         One of :data:`ANSWER_TYPES` (or None to auto-infer).
        aliases:      Alternative accepted strings (`entity` type).
        units:        Alternative accepted unit spellings (`quantity` type).
        precision:    Absolute numeric tolerance (`float`/`quantity`).
        rtol:         Relative numeric tolerance (`float`/`quantity`).
        ordered:      Whether `list` answers must match in order.
    """

    ground_truth: str
    type: str | None = None
    aliases: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    precision: float | None = None
    rtol: float | None = None
    ordered: bool = True

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> AnswerSpec:
        """Build a spec from a dataset row.

        Reads the optional per-item schema fields while falling back to the
        canonical `ground_truth` string for everything else.

        The trace-generation pipeline normalizes the ground-truth column to
        the `_ground_truth` key, so that key is checked first; the raw
        `ground_truth` key is used as a fallback so the grader also works on
        plain samples that only carry a `ground_truth` field.
        """
        ground_truth = row.get("_ground_truth")
        if ground_truth is None:
            ground_truth = row.get("ground_truth")
        if ground_truth is None:
            raise ValueError("Dataset row is missing 'ground_truth'")
        return cls(
            ground_truth=str(ground_truth),
            type=row.get("answer_type"),
            aliases=tuple(_as_strs(row.get("answer_aliases"))),
            units=tuple(_as_strs(row.get("answer_units"))),
            precision=_as_optional_float(row.get("answer_precision")),
            rtol=_as_optional_float(row.get("answer_rtol")),
            ordered=bool(row.get("answer_ordered", True)),
        )


def _as_strs(value: Any) -> Iterable[str]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple | set):
        return tuple(str(v) for v in value)
    return (str(value),)


def _as_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_text(text: Any) -> str:
    """Light normalization shared by every string comparison.

    Lowercases, strips diacritics, normalizes Unicode quotes, collapses
    whitespace and removes trailing sentence punctuation.  This is the
    transformation applied to `exact` answers; `exact` matching additionally
    accepts substring containment.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[.!?。！？]+$", "", text).strip()
    return text


def _fold_entity(text: Any) -> str:
    """Aggressive normalization for entity matching.

    After :func:`normalize_text`, apostrophes are dropped and the remaining
    punctuation is collapsed to spaces, so `King's College` matches
    `Kings College` and `BRA-Santos Dumont` matches `BRA Santos Dumont`.
    """
    text = normalize_text(text)
    text = text.replace("'", "")
    text = re.sub(r"[^0-9a-z ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_SUPERSCRIPT = str.maketrans({"²": "2", "³": "3", "¹": "1", "⁰": "0"})
_MICRO = str.maketrans({"µ": "u", "μ": "u"})


def normalize_unit(text: Any) -> str:
    """Normalize a unit string for comparison.

    Strips accents and expands Unicode superscripts (`cm²` -> `cm2`),
    maps the micro sign to `u` and removes whitespace and carets.  Case is
    preserved because units are case-sensitive (`Pa` vs `pA`).
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.translate(_SUPERSCRIPT).translate(_MICRO)
    text = text.replace("^", "")
    return re.sub(r"\s+", "", text)


_NUM_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

_ROMAN_RE = re.compile(r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
_ROMAN_VALUES = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}


def _is_roman(token: str) -> bool:
    return _ROMAN_RE.fullmatch(token.strip().upper()) is not None


def _roman_to_int(token: str) -> int:
    total = 0
    prev = 0
    for ch in reversed(token.strip().upper()):
        value = _ROMAN_VALUES[ch]
        total = total - value if value < prev else total + value
        prev = value
    return total


def _coerce_float(text: str) -> float | None:
    """Parse a string as a float, accepting decimal commas (pt-BR/EU style)."""
    t = text.strip()
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        pass
    if "," in t:
        if "." in t:
            # Both separators present: whichever comes last is the decimal mark.
            if t.rfind(",") > t.rfind("."):
                t = t.replace(".", "").replace(",", ".")
            else:
                t = t.replace(",", "")
        elif re.search(r",\d{1,2}$", t):
            t = t.replace(",", ".")
        else:
            t = t.replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


def _single_numeric_token(text: str) -> float | None:
    """Return the value of a string carrying a single leading number.

    Unit text may itself contain digits (`5.4 cm2`, `12.9 m/s`), so a
    second numeric token is tolerated only when it is glued to a letter.
    Strings with several independent numbers (ranges, equations) return None.
    """
    matches = list(_NUM_RE.finditer(text))
    if not matches:
        return None
    if len(matches) == 1:
        return _coerce_float(matches[0].group(0))
    for match in matches[1:]:
        before = text[match.start() - 1] if match.start() > 0 else ""
        after = text[match.end()] if match.end() < len(text) else ""
        if not (before.isalpha() or after.isalpha()):
            return None
    return _coerce_float(matches[0].group(0))


def parse_integer(text: Any) -> int | None:
    """Parse an answer as an exact integer, or return None.

    Accepts plain integers, integer-valued floats (`42.0`), thousand
    separators (`25,000` / `1.234.567`), Roman numerals and a single
    numeric token surrounded by unit text (`29 anos`).
    """
    if text is None:
        return None
    if isinstance(text, bool):
        return None
    if isinstance(text, int | float):
        value = float(text)
        return int(value) if value == int(value) else None
    t = str(text).strip()
    if not t:
        return None

    if _is_roman(t):
        return _roman_to_int(t)

    # Thousands-separated integer, e.g. "1,234" or "1.234.567" (pt-BR).
    if re.fullmatch(r"[+-]?\d{1,3}(?:[.,]\d{3})+", t):
        sign = -1 if t.startswith("-") else 1
        digits = re.sub(r"[.,]", "", t.lstrip("+-"))
        return sign * int(digits)

    value = _coerce_float(t)
    if value is not None and value == int(value):
        return int(value)

    # Single numeric token with surrounding unit text ("29 anos", "5.4 cm2").
    value = _single_numeric_token(t)
    if value is not None and value == int(value):
        return int(value)
    return None


def parse_float(text: Any) -> float | None:
    """Parse an answer as a float, or return None.

    Accepts plain decimals, scientific notation, decimal commas and a single
    numeric token surrounded by unit text.  Simple fractions (`1/2`) are
    also understood.
    """
    if text is None:
        return None
    if isinstance(text, bool):
        return None
    if isinstance(text, int | float):
        return float(text)
    t = str(text).strip()
    if not t:
        return None

    fraction = re.fullmatch(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*/\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
        t,
    )
    if fraction:
        denominator = _coerce_float(fraction.group(2))
        if denominator == 0:
            return None
        return _coerce_float(fraction.group(1)) / denominator

    value = _coerce_float(t)
    if value is not None:
        return value

    return _single_numeric_token(t)


def _compare_numeric(gold: float, answer: float, *, rtol: float | None, atol: float | None) -> bool:
    """Compare two numbers with per-item tolerances (exact by default)."""
    if rtol is None and atol is None:
        return gold == answer
    return math.isclose(gold, answer, rel_tol=rtol or 0.0, abs_tol=atol or 0.0)


_DATE_DMY = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})$")
_DATE_YMD = re.compile(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})$")


def parse_date(text: Any) -> tuple[int, int, int] | None:
    """Parse common numeric date formats into `(year, month, day)`.

    Supports `DD/MM/YYYY` (and `-`/`.` separators) plus ISO
    `YYYY-MM-DD`.  Returns None for anything else.
    """
    if not isinstance(text, str):
        text = str(text)
    t = text.strip()
    match = _DATE_DMY.match(t)
    if match:
        day, month, year = (int(g) for g in match.groups())
    else:
        match = _DATE_YMD.match(t)
        if not match:
            return None
        year, month, day = (int(g) for g in match.groups())
    try:
        date(year, month, day)
    except ValueError:
        return None
    return (year, month, day)


_LIST_DELIM_RE = re.compile(r"\s*[,;|]\s*")


def _split_list(text: Any) -> list[str]:
    parts = _LIST_DELIM_RE.split(str(text))
    return [normalize_text(p) for p in parts if p.strip()]


def _match_integer(final_answer: Any, ground_truth: str) -> bool:
    gold = parse_integer(ground_truth)
    answer = parse_integer(final_answer)
    return gold is not None and answer is not None and gold == answer


def _match_float(
    final_answer: Any,
    ground_truth: str,
    *,
    rtol: float | None,
    atol: float | None,
) -> bool:
    gold = parse_float(ground_truth)
    answer = parse_float(final_answer)
    if gold is None or answer is None:
        return False
    return _compare_numeric(gold, answer, rtol=rtol, atol=atol)


def _split_quantity(text: Any) -> tuple[float | None, str]:
    t = str(text).strip()
    match = _NUM_RE.search(t)
    if not match:
        return None, normalize_unit(t)
    number = _coerce_float(match.group(0))
    unit = (t[: match.start()] + " " + t[match.end() :]).strip()
    return number, normalize_unit(unit)


def _match_quantity(
    final_answer: Any,
    ground_truth: str,
    *,
    units: tuple[str, ...],
    rtol: float | None,
    atol: float | None,
) -> bool:
    gold_value, gold_unit = _split_quantity(ground_truth)
    answer_value, answer_unit = _split_quantity(final_answer)
    if gold_value is None or answer_value is None:
        return False
    if not _compare_numeric(gold_value, answer_value, rtol=rtol, atol=atol):
        return False

    accepted = {gold_unit}
    accepted.update(normalize_unit(u) for u in units)
    if gold_unit and not answer_unit:
        return False  # gold demands a unit, the answer omitted it
    return answer_unit in accepted


def _match_entity(final_answer: Any, ground_truth: str, *, aliases: tuple[str, ...]) -> bool:
    gold = _fold_entity(ground_truth)
    answer = _fold_entity(final_answer)
    if answer == gold:
        return True
    return any(answer == _fold_entity(alias) for alias in aliases)


def _match_date(final_answer: Any, ground_truth: str) -> bool:
    gold = parse_date(ground_truth)
    answer = parse_date(final_answer)
    if gold is not None and answer is not None:
        return gold == answer
    # Unparseable on either side: fall back to a normalized exact match.
    return normalize_text(ground_truth) == normalize_text(final_answer)


def _match_exact(final_answer: Any, ground_truth: str) -> bool:
    """Match textual answers with relaxed substring containment.

    Normalizes both sides and accepts an exact match, or a bidirectional
    substring containment (e.g. "The capital is Paris." matches "Paris").
    """
    gold = normalize_text(ground_truth)
    answer = normalize_text(final_answer)
    if gold == answer:
        return True
    if not gold or not answer:
        return False
    return gold in answer or answer in gold


def _match_list(final_answer: Any, ground_truth: str, *, ordered: bool) -> bool:
    gold_items = _split_list(ground_truth)
    answer_items = _split_list(final_answer)
    if ordered:
        return gold_items == answer_items
    return Counter(gold_items) == Counter(answer_items)


def _infer_type(ground_truth: str) -> str:
    """Infer the answer type from the gold string (no false positives)."""
    if _is_roman(ground_truth.strip()):
        return "integer"
    if parse_integer(ground_truth) is not None:
        return "integer"
    if parse_float(ground_truth) is not None:
        return "float"
    if parse_date(ground_truth) is not None:
        return "date"
    return "exact"


def grade_answer(final_answer: Any, spec: AnswerSpec) -> bool:
    """Grade `final_answer` against a per-item :class:`AnswerSpec`.

    Args:
        final_answer: The agent's final answer (string, number, ...).
        spec:         The answer schema for this dataset item.

    Returns:
        True when the answer matches the schema, False otherwise.

    Raises:
        ValueError: If `spec.type` is not one of :data:`ANSWER_TYPES`.
    """
    if final_answer is None:
        return False

    answer_type = (spec.type or _infer_type(spec.ground_truth)).lower()
    if answer_type == "integer":
        return _match_integer(final_answer, spec.ground_truth)
    if answer_type == "float":
        return _match_float(final_answer, spec.ground_truth, rtol=spec.rtol, atol=spec.precision)
    if answer_type == "quantity":
        return _match_quantity(
            final_answer,
            spec.ground_truth,
            units=spec.units,
            rtol=spec.rtol,
            atol=spec.precision,
        )
    if answer_type == "entity":
        return _match_entity(final_answer, spec.ground_truth, aliases=spec.aliases)
    if answer_type == "date":
        return _match_date(final_answer, spec.ground_truth)
    if answer_type == "list":
        return _match_list(final_answer, spec.ground_truth, ordered=spec.ordered)
    if answer_type == "exact":
        return _match_exact(final_answer, spec.ground_truth)
    raise ValueError(f"Unknown answer_type: {spec.type!r} (expected one of {sorted(ANSWER_TYPES)})")


def compare_answer(
    final_answer: Any,
    ground_truth: str,
    *,
    answer_type: str | None = None,
    aliases: Iterable[str] | None = None,
    units: Iterable[str] | None = None,
    precision: float | None = None,
    rtol: float | None = None,
    ordered: bool = True,
) -> bool:
    """Convenience wrapper around :func:`grade_answer` for direct use/tests."""
    spec = AnswerSpec(
        ground_truth=str(ground_truth),
        type=answer_type,
        aliases=tuple(aliases or ()),
        units=tuple(units or ()),
        precision=precision,
        rtol=rtol,
        ordered=ordered,
    )
    return grade_answer(final_answer, spec)
