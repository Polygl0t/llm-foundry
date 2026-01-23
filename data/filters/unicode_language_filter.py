"""
Unicode-based Language Filtering

Filters datasets by character set validation using Unicode ranges for 18+ languages.
Complementary to langdetect_language_filter.py - useful when language detection fails.

Supported languages:
English, Portuguese, Spanish, French, German, Italian (Latin scripts)
Russian, Ukrainian (Cyrillic)
Arabic, Hebrew (RTL scripts)
Greek, Hindi, Bengali, Thai, Vietnamese
Chinese (CJK), Japanese (Hiragana + Katakana + Kanji), Korean (Hangul)

Output:
- Filtered dataset split into chunks matching input file count
- .metadata: Document/token counts and statistics
- Optional excluded samples for debugging

Usage:
    # Filter for Portuguese only
    python unicode_language_filter.py --input_dir data/ --output_dir filtered/ \
        --languages portuguese --text_column text
    
    # Multi-language (Latin scripts)
    python unicode_language_filter.py --input_dir data/ --output_dir filtered/ \
        --languages english portuguese spanish french --num_proc 16
    
    # Save excluded samples for debugging
    python unicode_language_filter.py --input_dir data/ --output_dir excluded/ \
        --languages english --save_excluded
"""
import datasets
import argparse
import glob
import os
import re
import numpy as np


# Language to Unicode range mapping
LANGUAGE_RANGES = {
    'english': r'\u0041-\u005A\u0061-\u007A',  # A-Z, a-z
    'portuguese': r'\u0041-\u005A\u0061-\u007A\u00C0-\u00FF',  # English + Latin Extended
    'spanish': r'\u0041-\u005A\u0061-\u007A\u00C0-\u00FF',  # English + Latin Extended
    'french': r'\u0041-\u005A\u0061-\u007A\u00C0-\u00FF',  # English + Latin Extended
    'german': r'\u0041-\u005A\u0061-\u007A\u00C0-\u00FF',  # English + Latin Extended
    'italian': r'\u0041-\u005A\u0061-\u007A\u00C0-\u00FF',  # English + Latin Extended
    'russian': r'\u0400-\u04FF',  # Cyrillic
    'ukrainian': r'\u0400-\u04FF',  # Cyrillic
    'arabic': r'\u0600-\u06FF',  # Arabic
    'greek': r'\u0370-\u03FF',  # Greek
    'hebrew': r'\u0590-\u05FF',  # Hebrew
    'hindi': r'\u0900-\u097F',  # Devanagari
    'bengali': r'\u0980-\u09FF',  # Bengali
    'chinese': r'\u4E00-\u9FFF',  # CJK Unified Ideographs
    'japanese': r'\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF',  # Hiragana + Katakana + Kanji
    'korean': r'\uAC00-\uD7AF',  # Hangul
    'thai': r'\u0E00-\u0E7F',  # Thai
    'vietnamese': r'\u0041-\u005A\u0061-\u007A\u00C0-\u00FF\u0100-\u017F',  # Latin + Extensions
}

def create_language_filter(languages):
    """
    Create a filter function to detect specified language characters.
    Numbers, punctuation, whitespace, emojis, and symbols are always allowed.
    
    Args:
        languages: List of language names from LANGUAGE_RANGES
    """
    # Always allow:
    # - Numbers (0-9), basic punctuation, spaces, and common symbols
    # - Emojis and various symbol blocks
    base_pattern = (
        r'\u0020-\u0040'  # Space to @
        r'\u005B-\u0060'  # [ to `
        r'\u007B-\u007E'  # { to ~
        r'\u0009-\u000D'  # Tab, newline, etc.
        r'\u0020'  # Space (redundant but explicit)
        r'\u2000-\u206F'  # General Punctuation
        r'\u2070-\u209F'  # Superscripts and Subscripts
        r'\u20A0-\u20CF'  # Currency Symbols
        r'\u2100-\u214F'  # Letterlike Symbols
        r'\u2150-\u218F'  # Number Forms
        r'\u2190-\u21FF'  # Arrows
        r'\u2200-\u22FF'  # Mathematical Operators
        r'\u2300-\u23FF'  # Miscellaneous Technical
        r'\u2460-\u24FF'  # Enclosed Alphanumerics
        r'\u2500-\u257F'  # Box Drawing
        r'\u2580-\u259F'  # Block Elements
        r'\u25A0-\u25FF'  # Geometric Shapes
        r'\u2600-\u26FF'  # Miscellaneous Symbols
        r'\u2700-\u27BF'  # Dingbats
        r'\u2B00-\u2BFF'  # Miscellaneous Symbols and Arrows
        r'\u1F300-\u1F5FF'  # Miscellaneous Symbols and Pictographs
        r'\u1F600-\u1F64F'  # Emoticons
        r'\u1F680-\u1F6FF'  # Transport and Map Symbols
        r'\u1F700-\u1F77F'  # Alchemical Symbols
        r'\u1F780-\u1F7FF'  # Geometric Shapes Extended
        r'\u1F800-\u1F8FF'  # Supplemental Arrows-C
        r'\u1F900-\u1F9FF'  # Supplemental Symbols and Pictographs
        r'\u1FA00-\u1FA6F'  # Chess Symbols
        r'\u1FA70-\u1FAFF'  # Symbols and Pictographs Extended-A
    )
    
    # Build character ranges from selected languages
    language_ranges = []
    for lang in languages:
        if lang.lower() in LANGUAGE_RANGES:
            language_ranges.append(LANGUAGE_RANGES[lang.lower()])
        else:
            print(f"[WARNING] Unknown language '{lang}', skipping...")
    
    if not language_ranges:
        raise ValueError(f"No valid languages specified. Available languages: {list(LANGUAGE_RANGES.keys())}")
    
    # Combine base pattern with language ranges
    combined_pattern = base_pattern + ''.join(language_ranges)
    allowed_pattern = re.compile(f'^[{combined_pattern}]+$')
    
    def filter_language(text):
        """Check if text contains only allowed characters."""
        if not text:
            return False
        return bool(allowed_pattern.match(text))
    
    return filter_language


def is_messages_column(dataset, column_name):
    """
    Check if a column contains messages in the expected format.
    
    Args:
        dataset: The dataset to check
        column_name: Name of the column to check
    
    Returns:
        bool: True if the column appears to be a messages column
    """
    if column_name not in dataset.column_names:
        return False
    
    # Check the first non-null entry
    for example in dataset:
        value = example.get(column_name)
        if value is None:
            continue
        
        # Check if it's a list of dicts with 'content' key
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], dict) and 'content' in value[0]:
                return True
        break
    
    return False


def flatten_messages(messages):
    """
    Flatten a messages list into a single text string.
    
    Args:
        messages: List of message dictionaries with 'content' field
    
    Returns:
        str: Flattened text with messages separated by newlines
    """
    if not messages:
        return ""
    
    contents = []
    for msg in messages:
        if isinstance(msg, dict) and 'content' in msg:
            content = msg['content']
            if content:  # Only add non-empty content
                contents.append(str(content))
    
    return '\n'.join(contents)


def main(args):
    # Validate arguments
    assert args.input_type in ["jsonl", "parquet"], "Dataset type must be either 'jsonl' or 'parquet'."
    assert args.output_type in ["jsonl", "parquet"], "Output type must be either 'jsonl' or 'parquet'."
    
    # Load dataset
    data_files = glob.glob(f"{args.input_dir}/*.{args.input_type}")
    if not data_files:
        raise ValueError(f"No {args.input_type.upper()} files found in '{args.input_dir}'.")
    
    dataset = datasets.load_dataset(
        "json" if args.input_type == "jsonl" else "parquet",
        data_files=data_files,
        split="train",
        cache_dir=args.cache_dir,
        num_proc=len(data_files),
    )
    print(f"[INFO] Loaded dataset with {len(dataset):,} examples from {args.input_type.upper()} files.")
    print(f"[INFO] Columns: {dataset.column_names}")
    
    # Verify text column exists
    if args.text_column not in dataset.column_names:
        raise ValueError(f"Column '{args.text_column}' not found in dataset. Available columns: {dataset.column_names}")
    
    # Check if the column is a messages column
    is_messages = is_messages_column(dataset, args.text_column)
    if is_messages:
        print(f"[INFO] Detected messages format in column '{args.text_column}'")
        print("[INFO] Messages will be flattened before filtering")
    
    # Calculate initial token count if column exists
    original_count = len(dataset)
    if 'token_count' in dataset.column_names:
        original_tokens = sum(dataset['token_count'])
        print(f"[INFO] Original tokens: {original_tokens:,}")
    
    # Filter out non-matching language samples
    if args.save_excluded:
        print(f"\n[INFO] Saving EXCLUDED samples (those NOT matching: {', '.join(args.languages)})")
        print("[INFO] (Numbers, punctuation, spaces, emojis, and symbols are always allowed)")
    else:
        print(f"\n[INFO] Filtering samples to keep only: {', '.join(args.languages)}")
        print("[INFO] (Numbers, punctuation, spaces, emojis, and symbols are always allowed)")
    language_filter = create_language_filter(args.languages)
    
    # Create a modified filter that works with the specified text column
    def filter_with_column(example):
        value = example[args.text_column]
        
        # Handle messages column
        if is_messages:
            text = flatten_messages(value)
        else:
            text = value
        
        result = language_filter(text)
        # Invert the filter if we want to save excluded samples
        return not result if args.save_excluded else result
    
    filtered_dataset = dataset.filter(
        filter_with_column,
        num_proc=args.num_proc or 1,
        desc="Filtering by language",
    )
    
    # Calculate statistics
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine number of chunks (equal to number of input files)
    n_chunks = len(data_files)
    print(f"\n[INFO] Splitting dataset into {n_chunks} chunks (matching input file count)")
    
    # Split dataset into chunks
    indices = np.array_split(np.arange(filtered_count), n_chunks)
    chunks = [filtered_dataset.select(idx.tolist()) for idx in indices if len(idx) > 0]
    
    # Save chunks
    extension = args.output_type if args.output_type == "parquet" else "jsonl"
    save_fn = lambda chunk, path: (
        chunk.to_parquet(path) if args.output_type == "parquet" else chunk.to_json(path)
    )
    
    for i, chunk in enumerate(chunks):
        filename = f"{args.output_dir}/train-{i:05d}-of-{n_chunks:05d}.{extension}"
        save_fn(chunk, filename)
        print(f"[INFO] Saved chunk {i+1}/{n_chunks} with {len(chunk):,} examples to {filename}")
    
    # Save metadata
    with open(f"{args.output_dir}/.metadata", "w") as meta_file:
        meta_file.write(f"Number of samples: {filtered_count}\n")
        if 'token_count' in filtered_dataset.column_names:
            meta_file.write(f"Number of tokens: {filtered_tokens}\n")
        meta_file.write(f"Chunks: {n_chunks}\n")
        meta_file.write(f"Columns: {filtered_dataset.column_names}\n")
    
    print("\n[INFO] Metadata saved to .metadata file")
    print("\n[INFO] Language filtering complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter a dataset to keep only specified language text samples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available languages:
{', '.join(sorted(LANGUAGE_RANGES.keys()))}

Examples:
  # Filter for English and Portuguese
  python unicode_language_filter.py --input_dir data/ --output_dir filtered/ --languages english portuguese
  
  # Filter for Chinese only
  python unicode_language_filter.py --input_dir data/ --output_dir filtered/ --languages chinese
  
  # Filter for multiple languages
  python unicode_language_filter.py --input_dir data/ --output_dir filtered/ --languages english spanish french
"""
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the input dataset files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save filtered dataset")
    parser.add_argument("--languages", type=str, nargs='+', required=True, 
                        help=f"Languages to keep (space-separated). Available: {', '.join(sorted(LANGUAGE_RANGES.keys()))}")
    parser.add_argument("--input_type", choices=["jsonl", "parquet"], default="parquet", help="Type of the input files")
    parser.add_argument("--output_type", choices=["jsonl", "parquet"], default="parquet", help="Type of the output files")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column in the dataset")
    parser.add_argument("--cache_dir", type=str, help="Cache directory for dataset loading")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes to use")
    parser.add_argument("--save_excluded", action="store_true", help="Save excluded samples instead of included ones (for testing)")
    
    args = parser.parse_args()
    
    print(f"Filtering dataset for {', '.join(args.languages)} text only...")
    main(args)
    print("Done! 🎆")
