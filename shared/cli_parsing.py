"""Utilities for CLI argument parsers.

Provides:
    - num_or_ratio: A function to parse both ints and floats.
"""


def num_or_ratio(v: str) -> int | float:
    try:
        return int(v)
    except ValueError:
        return float(v)
