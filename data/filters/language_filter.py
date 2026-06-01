"""
Language Filter

Filters datasets to keep only samples in specified languages.
Supports two backends:

  langdetect  Probabilistic language detection via the langdetect library.
              More accurate on short/ambiguous text; slower.

  unicode     Heuristic filtering via Unicode character-range matching.
              Deterministic and fast; works even without langdetect installed.

Usage:
    # langdetect backend
    python language_filter.py --backend langdetect \\
        --input_dir data/ --output_dir filtered/ \\
        --languages portuguese --text_column text

    # unicode backend (multi-language)
    python language_filter.py --backend unicode \\
        --input_dir data/ --output_dir filtered/ \\
        --languages english portuguese --num_proc 16

    # Save excluded samples for debugging
    python language_filter.py --backend langdetect \\
        --input_dir data/ --output_dir excluded/ \\
        --languages english --save_excluded
"""
import os
import re
import argparse
from utils import (
    DatasetLoader,
    save_dataset,
    is_messages_column,
    flatten_messages,
)

_TOKENS_PER_CHUNK = 3_000_000  # target ~100MB chunks (adjust as needed)

# Language codes used by the langdetect library.
LANGDETECT_CODES = {
    'english':    'en',
    'portuguese': 'pt',
    'spanish':    'es',
    'french':     'fr',
    'german':     'de',
    'italian':    'it',
    'russian':    'ru',
    'ukrainian':  'uk',
    'arabic':     'ar',
    'greek':      'el',
    'hebrew':     'he',
    'hindi':      'hi',
    'bengali':    'bn',
    'chinese':    'zh-cn',
    'japanese':   'ja',
    'korean':     'ko',
    'thai':       'th',
    'vietnamese': 'vi',
    # Add more as needed. See: https://pypi.org/project/langdetect/
}

# Unicode character ranges for the unicode backend.
UNICODE_RANGES = {
    'english':    r'\u0041-\u005A\u0061-\u007A',
    'portuguese': r'\u0041-\u005A\u0061-\u007A\u00C0-\u00FF',
    'spanish':    r'\u0041-\u005A\u0061-\u007A\u00C0-\u00FF',
    'french':     r'\u0041-\u005A\u0061-\u007A\u00C0-\u00FF',
    'german':     r'\u0041-\u005A\u0061-\u007A\u00C0-\u00FF',
    'italian':    r'\u0041-\u005A\u0061-\u007A\u00C0-\u00FF',
    'russian':    r'\u0400-\u04FF',
    'ukrainian':  r'\u0400-\u04FF',
    'arabic':     r'\u0600-\u06FF',
    'greek':      r'\u0370-\u03FF',
    'hebrew':     r'\u0590-\u05FF',
    'hindi':      r'\u0900-\u097F',
    'bengali':    r'\u0980-\u09FF',
    'chinese':    r'\u4E00-\u9FFF',
    'japanese':   r'\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF',
    'korean':     r'\uAC00-\uD7AF',
    'thai':       r'\u0E00-\u0E7F',
    'vietnamese': r'\u0041-\u005A\u0061-\u007A\u00C0-\u00FF\u0100-\u017F',
    # Add more as needed. See: https://www.unicode.org/charts/
}

SUPPORTED_LANGUAGES = sorted(set(LANGDETECT_CODES) | set(UNICODE_RANGES))

# Backend: langdetect
def _create_langdetect_filter(languages, **_kwargs):
    """Return a filter function backed by the langdetect library."""
    from langdetect import detect, LangDetectException
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0  # ensure deterministic results

    target_codes = set()
    for lang in languages:
        code = LANGDETECT_CODES.get(lang.lower())
        if code:
            target_codes.add(code)
        else:
            print(f"[WARNING] Unknown language '{lang}' for langdetect backend, skipping...")

    if not target_codes:
        raise ValueError(f"No valid languages specified. Available: {sorted(LANGDETECT_CODES)}")

    print(f"[INFO] langdetect target codes: {sorted(target_codes)}")

    def keep(text):
        if not text or len(text.strip()) < 10:
            return False
        try:
            return detect(text) in target_codes
        except LangDetectException:
            return False

    return keep


# Backend: unicode
# Characters always permitted regardless of target language:
# digits, ASCII punctuation, common symbols, and emoji.
_UNICODE_BASE = (
    r'\u0009-\u000D'            # Tab, newline, carriage return, etc.
    r'\u0020-\u0040'            # Space … @  (includes digits 0-9)
    r'\u005B-\u0060'            # [ \ ] ^ _ `
    r'\u007B-\u007E'            # { | } ~
    r'\u2000-\u206F'            # General Punctuation
    r'\u2070-\u209F'            # Superscripts and Subscripts
    r'\u20A0-\u20CF'            # Currency Symbols
    r'\u2100-\u214F'            # Letterlike Symbols
    r'\u2150-\u218F'            # Number Forms
    r'\u2190-\u21FF'            # Arrows
    r'\u2200-\u22FF'            # Mathematical Operators
    r'\u2300-\u23FF'            # Miscellaneous Technical
    r'\u2460-\u24FF'            # Enclosed Alphanumerics
    r'\u2500-\u257F'            # Box Drawing
    r'\u2580-\u259F'            # Block Elements
    r'\u25A0-\u25FF'            # Geometric Shapes
    r'\u2600-\u26FF'            # Miscellaneous Symbols
    r'\u2700-\u27BF'            # Dingbats
    r'\u2B00-\u2BFF'            # Miscellaneous Symbols and Arrows
    r'\U0001F300-\U0001F5FF'    # Miscellaneous Symbols and Pictographs
    r'\U0001F600-\U0001F64F'    # Emoticons
    r'\U0001F680-\U0001F6FF'    # Transport and Map Symbols
    r'\U0001F700-\U0001F77F'    # Alchemical Symbols
    r'\U0001F780-\U0001F7FF'    # Geometric Shapes Extended
    r'\U0001F800-\U0001F8FF'    # Supplemental Arrows-C
    r'\U0001F900-\U0001F9FF'    # Supplemental Symbols and Pictographs
    r'\U0001FA00-\U0001FA6F'    # Chess Symbols
    r'\U0001FA70-\U0001FAFF'    # Symbols and Pictographs Extended-A
)


def _create_unicode_filter(languages, threshold=0.85, **_kwargs):
    """Return a filter function backed by Unicode range heuristics.

    Rather than requiring every character to be in-range (which would reject
    documents containing even a single stray out-of-range character), a sample
    is kept when at least `threshold` fraction of its characters fall inside
    the allowed Unicode ranges (base + language-specific).
    """
    language_ranges = []
    for lang in languages:
        ranges = UNICODE_RANGES.get(lang.lower())
        if ranges:
            language_ranges.append(ranges)
        else:
            print(f"[WARNING] Unknown language '{lang}' for unicode backend, skipping...")

    if not language_ranges:
        raise ValueError(f"No valid languages specified. Available: {sorted(UNICODE_RANGES)}")

    combined = _UNICODE_BASE + ''.join(language_ranges)
    allowed = re.compile(f'[{combined}]')

    def keep(text):
        if not text:
            return False
        total = len(text)
        # Remove all allowed characters; what remains is disallowed.
        disallowed_count = len(allowed.sub('', text))
        return (total - disallowed_count) / total >= threshold

    return keep


_BACKEND_FACTORIES = {
    'langdetect': _create_langdetect_filter,
    'unicode':    _create_unicode_filter,
}


def main(args):

    # Load dataset
    loader = DatasetLoader(path=args.input_dir, cache_dir=args.cache_dir)
    dataset = loader.load()

    if args.text_column not in dataset.column_names:
        raise ValueError(
            f"Column '{args.text_column}' not found. Available: {dataset.column_names}"
        )

    is_messages = is_messages_column(dataset, args.text_column)
    if is_messages:
        print(f"[INFO] Detected messages format in column '{args.text_column}'")
        print("[INFO] Messages will be flattened before filtering")

    original_count = len(dataset)
    original_tokens = None
    if 'token_count' in dataset.column_names:
        original_tokens = sum(dataset['token_count'])
        print(f"[INFO] Original tokens: {original_tokens:,}")

    if args.save_excluded:
        print(f"\n[INFO] Saving EXCLUDED samples (those NOT matching: {', '.join(args.languages)})")
    else:
        print(f"\n[INFO] Filtering samples [{args.backend}] for: {', '.join(args.languages)}")

    language_filter = _BACKEND_FACTORIES[args.backend](
        args.languages,
        threshold=args.unicode_threshold,
    )

    def keep_example(example):
        value = example[args.text_column]
        text = flatten_messages(value) if is_messages else value
        result = language_filter(text)
        return not result if args.save_excluded else result

    filtered_dataset = dataset.filter(
        keep_example,
        num_proc=args.num_proc,
        desc="Filtering dataset...",
    )

    filtered_count = len(filtered_dataset)
    removed_count = original_count - filtered_count
    removed_pct = (removed_count / original_count * 100) if original_count > 0 else 0.0

    print(f"\n[INFO] ===== FILTERING RESULTS =====")
    print(f"[INFO] Original samples: {original_count:,}")
    if args.save_excluded:
        print(f"[INFO] Excluded samples (saved): {filtered_count:,}")
        print(f"[INFO] Matching samples (not saved): {removed_count:,} ({removed_pct:.2f}%)")
    else:
        print(f"[INFO] Filtered samples: {filtered_count:,}")
        print(f"[INFO] Removed samples:  {removed_count:,} ({removed_pct:.2f}%)")

    if original_tokens is not None:
        filtered_tokens = sum(filtered_dataset['token_count'])
        removed_tokens = original_tokens - filtered_tokens
        removed_tokens_pct = (removed_tokens / original_tokens * 100) if original_tokens > 0 else 0.0
        print(f"[INFO] Original tokens: {original_tokens:,}")
        print(f"[INFO] Filtered tokens: {filtered_tokens:,}")
        print(f"[INFO] Removed tokens:  {removed_tokens:,} ({removed_tokens_pct:.2f}%)")
    else:
        filtered_tokens = 0

    if filtered_count == 0:
        print("[WARNING] No samples remaining after filtering. Not saving dataset.")
        return

    save_dataset(filtered_dataset, args.output_dir, args.output_type, _TOKENS_PER_CHUNK, token_count=filtered_tokens)
    print("\n[INFO] Language filtering complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend", choices=list(_BACKEND_FACTORIES), required=True,
        help="Filtering backend: 'langdetect' (probabilistic) or 'unicode' (heuristic)",
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing input dataset files (.jsonl or .parquet)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write filtered dataset files into",
    )
    parser.add_argument(
        "--languages", type=str, nargs='+', required=True,
        help=f"Languages to keep. Available: {', '.join(SUPPORTED_LANGUAGES)}",
    )
    parser.add_argument(
        "--input_type", choices=["jsonl", "parquet"], default="parquet",
        help="Format of input dataset files (default: parquet)",
    )
    parser.add_argument(
        "--output_type", choices=["jsonl", "parquet"], default="parquet",
        help="Format of output dataset files (default: parquet)",
    )
    parser.add_argument(
        "--text_column", type=str, default="text",
        help="Name of the column containing text to filter (default: 'text')",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="./.cache",
        help="Directory to use for caching datasets (default: ./.cache)",
    )
    parser.add_argument(
        "--num_proc", type=int, default=os.cpu_count(),
        help=f"Number of worker processes (default: {os.cpu_count()})",
    )
    parser.add_argument(
        "--save_excluded", action="store_true",
        help="Save samples that do NOT match the specified languages instead of those that do",
    )
    parser.add_argument(
        "--unicode_threshold", type=float, default=0.85,
        help=(
            "Minimum fraction of characters that must fall in the allowed Unicode ranges "
            "(unicode backend only, default: 0.85)"
        ),
    )

    args = parser.parse_args()
    main(args)
