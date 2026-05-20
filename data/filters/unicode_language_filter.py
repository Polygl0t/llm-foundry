"""
Unicode-based Language Filtering

Filters datasets by character set validation using Unicode ranges for 18+ languages.
Complementary to langdetect_language_filter.py - useful when language detection fails.

Usage:
    # Filter for Portuguese only
    python unicode_language_filter.py --input_dir data/ --output_dir filtered/ \\
        --languages portuguese --text_column text
    
    # Multi-language (Latin scripts)
    python unicode_language_filter.py --input_dir data/ --output_dir filtered/ \\
        --languages english portuguese spanish french --num_proc 16
    
    # Save excluded samples for debugging
    python unicode_language_filter.py --input_dir data/ --output_dir excluded/ \\
        --languages english --save_excluded
"""
import argparse
import re
from utils import load_dataset, save_dataset, is_messages_column, flatten_messages

LANGUAGE_RANGES = {
    'english': r'\u0041-\u005A\u0061-\u007A',
    'portuguese': r'\u0041-\u005A\u0061-\u007A\u00C0-\u00FF',
    'spanish': r'\u0041-\u005A\u0061-\u007A\u00C0-\u00FF',
    'french': r'\u0041-\u005A\u0061-\u007A\u00C0-\u00FF',
    'german': r'\u0041-\u005A\u0061-\u007A\u00C0-\u00FF',
    'italian': r'\u0041-\u005A\u0061-\u007A\u00C0-\u00FF',
    'russian': r'\u0400-\u04FF',
    'ukrainian': r'\u0400-\u04FF',
    'arabic': r'\u0600-\u06FF',
    'greek': r'\u0370-\u03FF',
    'hebrew': r'\u0590-\u05FF',
    'hindi': r'\u0900-\u097F',
    'bengali': r'\u0980-\u09FF',
    'chinese': r'\u4E00-\u9FFF',
    'japanese': r'\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF',
    'korean': r'\uAC00-\uD7AF',
    'thai': r'\u0E00-\u0E7F',
    'vietnamese': r'\u0041-\u005A\u0061-\u007A\u00C0-\u00FF\u0100-\u017F',
}

def create_language_filter(languages):
    base_pattern = (
        r'\u0020-\u0040'
        r'\u005B-\u0060'
        r'\u007B-\u007E'
        r'\u0009-\u000D'
        r'\u2000-\u206F'
        r'\u2070-\u209F'
        r'\u20A0-\u20CF'
        r'\u2100-\u214F'
        r'\u2150-\u218F'
        r'\u2190-\u21FF'
        r'\u2200-\u22FF'
        r'\u2300-\u23FF'
        r'\u2460-\u24FF'
        r'\u2500-\u257F'
        r'\u2580-\u259F'
        r'\u25A0-\u25FF'
        r'\u2600-\u26FF'
        r'\u2700-\u27BF'
        r'\u2B00-\u2BFF'
        r'\u1F300-\u1F5FF'
        r'\u1F600-\u1F64F'
        r'\u1F680-\u1F6FF'
        r'\u1F700-\u1F77F'
        r'\u1F780-\u1F7FF'
        r'\u1F800-\u1F8FF'
        r'\u1F900-\u1F9FF'
        r'\u1FA00-\u1FA6F'
        r'\u1FA70-\u1FAFF'
    )

    language_ranges = []
    for lang in languages:
        if lang.lower() in LANGUAGE_RANGES:
            language_ranges.append(LANGUAGE_RANGES[lang.lower()])
        else:
            print(f"[WARNING] Unknown language '{lang}', skipping...")

    if not language_ranges:
        raise ValueError(f"No valid languages specified. Available languages: {list(LANGUAGE_RANGES.keys())}")

    combined_pattern = base_pattern + ''.join(language_ranges)
    allowed_pattern = re.compile(f'^[{combined_pattern}]+$')

    def filter_language(text):
        if not text:
            return False
        return bool(allowed_pattern.match(text))

    return filter_language


def main(args):
    assert args.output_type in ["jsonl", "parquet"], "Output type must be either 'jsonl' or 'parquet'."

    dataset, data_files = load_dataset(args.input_dir, args.input_type, args.cache_dir)

    if args.text_column not in dataset.column_names:
        raise ValueError(f"Column '{args.text_column}' not found in dataset. Available columns: {dataset.column_names}")

    is_messages = is_messages_column(dataset, args.text_column)
    if is_messages:
        print(f"[INFO] Detected messages format in column '{args.text_column}'")
        print("[INFO] Messages will be flattened before filtering")

    original_count = len(dataset)
    if 'token_count' in dataset.column_names:
        original_tokens = sum(dataset['token_count'])
        print(f"[INFO] Original tokens: {original_tokens:,}")

    if args.save_excluded:
        print(f"\n[INFO] Saving EXCLUDED samples (those NOT matching: {', '.join(args.languages)})")
        print("[INFO] (Numbers, punctuation, spaces, emojis, and symbols are always allowed)")
    else:
        print(f"\n[INFO] Filtering samples to keep only: {', '.join(args.languages)}")
        print("[INFO] (Numbers, punctuation, spaces, emojis, and symbols are always allowed)")
    language_filter = create_language_filter(args.languages)

    def filter_with_column(example):
        value = example[args.text_column]
        if is_messages:
            text = flatten_messages(value)
        else:
            text = value
        result = language_filter(text)
        return not result if args.save_excluded else result

    filtered_dataset = dataset.filter(
        filter_with_column,
        num_proc=args.num_proc or 1,
        desc="Filtering by language",
    )

    filtered_count = len(filtered_dataset)
    removed_count = original_count - filtered_count
    removed_percentage = (removed_count / original_count * 100) if original_count > 0 else 0

    print(f"\n[INFO] ===== FILTERING RESULTS =====")
    print(f"[INFO] Original samples: {original_count:,}")
    if args.save_excluded:
        print(f"[INFO] Excluded samples (saved): {filtered_count:,}")
        print(f"[INFO] Matching samples (not saved): {removed_count:,} ({removed_percentage:.2f}%)")
    else:
        print(f"[INFO] Filtered samples: {filtered_count:,}")
        print(f"[INFO] Removed samples: {removed_count:,} ({removed_percentage:.2f}%)")

    if 'token_count' in filtered_dataset.column_names:
        filtered_tokens = sum(filtered_dataset['token_count'])
        removed_tokens = original_tokens - filtered_tokens
        removed_tokens_percentage = (removed_tokens / original_tokens * 100) if original_tokens > 0 else 0
        print(f"[INFO] Original tokens: {original_tokens:,}")
        print(f"[INFO] Filtered tokens: {filtered_tokens:,}")
        print(f"[INFO] Removed tokens: {removed_tokens:,} ({removed_tokens_percentage:.2f}%)")

    if filtered_count == 0:
        print("[WARNING] No samples remaining after filtering. Not saving dataset.")
        return

    save_dataset(filtered_dataset, args.output_dir, args.output_type, n_chunks=len(data_files))
    print("\n[INFO] Language filtering complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--languages", type=str, nargs='+', required=True,
                        help=f"Languages to keep. Available: {', '.join(sorted(LANGUAGE_RANGES.keys()))}")
    parser.add_argument("--input_type", choices=["jsonl", "parquet"], default="parquet")
    parser.add_argument("--output_type", choices=["jsonl", "parquet"], default="parquet")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--save_excluded", action="store_true")

    args = parser.parse_args()
    print(f"Filtering dataset for {', '.join(args.languages)} text only...")
    main(args)
    print("Done! 🎆")
