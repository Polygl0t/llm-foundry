"""
Language Detection Filter using langdetect

Filters datasets to keep only samples in specified languages using the langdetect library.

Input requirements:
- JSONL or Parquet files with a text column (default: "text")
- Optional token_count column for statistics tracking
- Messages format auto-detected and flattened if present

Output:
- Filtered dataset split into chunks matching input file count
- .metadata file with filtering statistics
- Preserves all original columns

Usage:
    # Filter for single language
    python langdetect_language_filter.py --input_dir data/ --output_dir filtered/ \
        --languages portuguese
    
    # Filter for multiple languages
    python langdetect_language_filter.py --input_dir data/ --output_dir filtered/ \
        --languages english portuguese spanish --num_proc 16
    
    # Save excluded samples for testing
    python langdetect_language_filter.py --input_dir data/ --output_dir excluded/ \
        --languages english --save_excluded
"""
import argparse
import glob
from langdetect import detect, LangDetectException
from utils import DatasetLoader, save_dataset, is_messages_column, flatten_messages

LANGUAGE_CODES = {
    'english': 'en',
    'portuguese': 'pt',
    'spanish': 'es',
    'french': 'fr',
    'german': 'de',
    'italian': 'it',
    'russian': 'ru',
    'ukrainian': 'uk',
    'arabic': 'ar',
    'greek': 'el',
    'hebrew': 'he',
    'hindi': 'hi',
    'bengali': 'bn',
    'chinese': 'zh-cn',
    'japanese': 'ja',
    'korean': 'ko',
    'thai': 'th',
    'vietnamese': 'vi',
}

def create_language_filter(languages):
    target_codes = set()
    for lang in languages:
        if lang.lower() in LANGUAGE_CODES:
            target_codes.add(LANGUAGE_CODES[lang.lower()])
        else:
            print(f"[WARNING] Unknown language '{lang}', skipping...")
    
    if not target_codes:
        raise ValueError(f"No valid languages specified. Available languages: {list(LANGUAGE_CODES.keys())}")
    
    print(f"[INFO] Target language codes: {sorted(target_codes)}")
    
    def filter_language(text):
        if not text or len(text.strip()) < 10:
            return False
        try:
            detected_lang = detect(text)
            return detected_lang in target_codes
        except LangDetectException:
            return False
    
    return filter_language


def main(args):
    assert args.output_type in ["jsonl", "parquet"], "Output type must be either 'jsonl' or 'parquet'."

    loader = DatasetLoader(path=args.input_dir, cache_dir=args.cache_dir)
    dataset = loader.load()
    data_files = glob.glob(f"{args.input_dir}/*.{args.input_type}")

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
    else:
        print(f"\n[INFO] Filtering samples using langdetect for: {', '.join(args.languages)}")
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

    filtered_tokens = sum(filtered_dataset['token_count']) if 'token_count' in filtered_dataset.column_names else 0
    save_dataset(filtered_dataset, args.output_dir, args.output_type, tokens_per_chunk=1_000_000, token_count=filtered_tokens, n_chunks=len(data_files))
    print("\n[INFO] Language filtering complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--languages", type=str, nargs='+', required=True,
                        help=f"Languages to keep. Available: {', '.join(sorted(LANGUAGE_CODES.keys()))}")
    parser.add_argument("--input_type", choices=["jsonl", "parquet"], default="parquet")
    parser.add_argument("--output_type", choices=["jsonl", "parquet"], default="parquet")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--save_excluded", action="store_true")

    args = parser.parse_args()
    print(f"Filtering dataset for {', '.join(args.languages)} text using langdetect...")
    main(args)
    print("Done! 🎆")
