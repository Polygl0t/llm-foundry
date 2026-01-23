"""
CommonCrawl Language Extraction Pipeline (Lightweight)

Streamlined single-stage pipeline for language identification and text extraction from
CommonCrawl WARC archives. Focuses on speed and language separation without quality filtering.

Pipeline stages:
1. WARC Reading: Parse CommonCrawl WARC.gz files
2. URL Filtering: Apply optional blocklists for spam/adult content
3. Text Extraction: Clean HTML using Trafilatura (precision mode)
4. Token Counting: Count tokens using specified tokenizer
5. Language Detection: FT176 (176 langs) or GlotLID (1665 langs)
6. Output: Write to language-separated JSONL files

Usage:
    # Extract all languages from CC-MAIN-2025-30
    python process_cc_dump_all_languages.py \
        --warc_files_folder /data/cc/CC-MAIN-2025-30/ \
        --output_folder all_languages/ \
        --dump CC-MAIN-2025-30 \
        --tasks 32 --workers 32
    
    # Extract specific languages with GlotLID backend
    python process_cc_dump_all_languages.py \
        --warc_files_folder /data/cc/warc/ \
        --output_folder extracted/ \
        --dump CC-MAIN-2025-30 \
        --languages pt bn hi ar \
        --language_filter_backend glotlid \
        --language_threshold 0.7
    
    # Incremental run (appends to existing)
    python process_cc_dump_all_languages.py \
        --warc_files_folder /data/cc/CC-MAIN-2025-31/ \
        --output_folder all_languages/ \
        --dump CC-MAIN-2025-31 \
        --expand_metadata
"""
import argparse
import shutil
import os
import json
import uuid
import glob

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    LanguageFilter,
    URLFilter,
)

from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.tokens import TokensCounter


def read_metadata(metadata_file):
    """Read metadata from file in YAML-like format."""
    if not os.path.exists(metadata_file):
        return None
    
    metadata = {}
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                # Convert numeric values
                try:
                    if '.' in value:
                        metadata[key] = float(value)
                    else:
                        metadata[key] = int(value)
                except ValueError:
                    metadata[key] = value
    return metadata


def write_metadata(metadata_file, metadata):
    """Write metadata to file in YAML-like format."""
    with open(metadata_file, 'w', encoding='utf-8') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")


def initialize_or_load_metadata(lang_output_path):
    """Initialize metadata by scanning existing files or load from .metadata file."""
    metadata_file = os.path.join(lang_output_path, '.metadata')
    
    # Try to load existing metadata
    metadata = read_metadata(metadata_file)
    if metadata is not None:
        return metadata
    
    # No metadata file exists, load all jsonl files in the language folder
    all_jsonl_files = glob.glob(os.path.join(lang_output_path, '*.jsonl'))
    
    if not all_jsonl_files:
        return {'lines': 0, 'tokens': 0}
    
    # Scan existing consolidated file to build metadata
    total_lines = 0
    total_tokens = 0

    for jsonl_file in all_jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    total_lines += 1
                    total_tokens += data.get('token_count', 0)
                except json.JSONDecodeError:
                    continue
    
    metadata = {
        'lines': total_lines,
        'tokens': total_tokens
    }
    
    write_metadata(metadata_file, metadata)
    return metadata


def main(args):

    TASKS = args.tasks
    WORKERS = args.workers
    DUMP = args.dump

    # WARCS should be downloaded from https://commoncrawl.org/the-data/get-started/
    WARC_FILES_FOLDER = args.warc_files_folder
    LOGS_FOLDER = args.logs_folder
    TEMP_OUTPUT_FOLDER = args.temp_output_folder  # Temporary folder for this iteration
    OUTPUT_FOLDER = args.output_folder  # Final output folder (append mode)
    TOKENIZER_NAME_OR_PATH = args.tokenizer_name_or_path # Default: Qwen3 tokenizer (a good general-purpose multilingual tokenizer)
    
    # Create cache folder for problematic files
    ERROR_CACHE_FOLDER = os.path.join(OUTPUT_FOLDER, ".error_cache")
    os.makedirs(ERROR_CACHE_FOLDER, exist_ok=True)

    # Language filtering and extraction pipeline
    pipeline = LocalPipelineExecutor(
        pipeline=[
            # [readers: HuggingFaceDatasetReader, JsonlReader, ParquetReader](https://github.com/huggingface/datatrove/tree/main/src/datatrove/pipeline/readers)
            # [WarcReader](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/readers/warc.py)
            # CommonCrawl data is available in two main formats: WARC and WET. 
            # - WARC ([Web ARChive format](https://en.wikipedia.org/wiki/WARC_(file_format))) files contain the raw data from the crawl
            # - WET (WARC Encapsulated Text) files provide a text only version of those websites.
            WarcReader(
                data_folder=WARC_FILES_FOLDER,
                glob_pattern="*.warc.gz",
                default_metadata={"source": DUMP},
                #limit=1_000_000,  # Uncomment to limit the number of WARC files processed (useful for debugging)
            ),

            # [URLFilter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/url_filter.py)
            # Example of blocklists: https://github.com/maravento/blackweb/tree/master 
            # We can also specify banned_words, banned_subwords, soft_banned_words
            URLFilter(exclusion_writer=None),

            # [Trafilatura](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/extractors/trafilatura.py)
            # [Original documentation](https://trafilatura.readthedocs.io/en/latest/usage-python.html)
            # Trafilatura provides a better extraction of text content from HTML pages then the default HTML parser CommonCrawl uses the WET format.
            # Ablation results available in https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html#starting_point:_text_extraction
            Trafilatura(favour_precision=True),

            # [TokensCounter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/tokens/counter.py#L7)
            TokensCounter(tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH),

            # [LanguageFilter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/language_filter.py)
            # Default option is [FT176](https://fasttext.cc/docs/en/language-identification.html)
            # FT176 gives support to ~176 languages.
            # GlotLID gives supports 1665 languages (2102 labels).
            LanguageFilter(
                languages=args.languages if args.languages else None,  # None keeps all languages
                backend=args.language_filter_backend,
                language_threshold=args.language_threshold,
            ),

            # [writers: JsonlWriter, ParquetWriter, HuggingFaceDatasetWriter](https://github.com/huggingface/datatrove/tree/main/src/datatrove/pipeline/writers)
            # [JsonlWriter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/writers/jsonl.py)
            # Write documents that passed the language filter
            JsonlWriter(
                TEMP_OUTPUT_FOLDER,
                output_filename="${language}/${language}.jsonl",
                compression=None,
                expand_metadata=args.expand_metadata,
            ),
        ],
        tasks=TASKS,
        workers=WORKERS,
        logging_dir=f"{LOGS_FOLDER}/language_filter",
    )

    # Run the pipeline
    pipeline.run()
    
    # ==============================================================================
    # POST-PROCESSING: Consolidate extracted data into a final output folder/files
    # ==============================================================================
    
    if not os.path.exists(TEMP_OUTPUT_FOLDER):
        print("[ERROR] No temporary output folder found.")
        return
    
    language_stats = {}
    
    for lang in os.listdir(TEMP_OUTPUT_FOLDER):
        lang_temp_path = os.path.join(TEMP_OUTPUT_FOLDER, lang)
        if not os.path.isdir(lang_temp_path):
            continue
        
        lang_output_path = os.path.join(OUTPUT_FOLDER, lang)
        os.makedirs(lang_output_path, exist_ok=True)
        
        # Load existing metadata (if any)
        previous_metadata = initialize_or_load_metadata(lang_output_path)
        
        # Consolidated file path
        consolidated_file = os.path.join(lang_output_path, f"{lang}.jsonl")
        
        # Track new data added in this iteration
        new_lines = 0
        new_tokens = 0
        invalid_lines = 0
        
        # Append new data to consolidated file
        with open(consolidated_file, 'a', encoding='utf-8') as outfile:
            for jsonl_file in os.listdir(lang_temp_path):
                if not jsonl_file.endswith('.jsonl'):
                    continue
                
                temp_file_path = os.path.join(lang_temp_path, jsonl_file)
                
                try:
                    with open(temp_file_path, 'r', encoding='utf-8', errors='replace') as infile:
                        for line in infile:
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                data = json.loads(line)
                                outfile.write(line + '\n')
                                new_lines += 1
                                new_tokens += data.get('token_count', 0)
                            except (json.JSONDecodeError, ValueError):
                                invalid_lines += 1
                except Exception as e:
                    # Cache problematic files for debugging
                    cache_path = os.path.join(ERROR_CACHE_FOLDER, f"{lang}_{uuid.uuid4().hex[:8]}_{jsonl_file}")
                    shutil.copy2(temp_file_path, cache_path)
                    print(f"[WARNING] Could not process {jsonl_file}: {e}. Cached for inspection.")
        
        if new_lines == 0:
            print(f"[WARNING] No valid data found for {lang}")
            continue
        
        if invalid_lines > 0:
            print(f"[WARNING] {lang}: Skipped {invalid_lines:,} invalid lines")
        
        # Update metadata
        updated_metadata = {
            'lines': previous_metadata.get('lines', 0) + new_lines,
            'tokens': previous_metadata.get('tokens', 0) + new_tokens
        }
        
        write_metadata(os.path.join(lang_output_path, '.metadata'), updated_metadata)
        
        # Store stats for summary
        language_stats[lang] = {
            'new_lines': new_lines,
            'new_tokens': new_tokens,
            'old_lines': previous_metadata.get('lines', 0),
            'old_tokens': previous_metadata.get('tokens', 0),
            'total_lines': updated_metadata['lines'],
            'total_tokens': updated_metadata['tokens']
        }
    
    # ==============================================================================
    # SUMMARY
    # ==============================================================================
    
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    
    if not language_stats:
        print("\n[WARNING] No languages were successfully processed.")
        print("="*80)
        return
    
    # Calculate totals for current iteration
    new_total_lines = sum(stats['new_lines'] for stats in language_stats.values())
    new_total_tokens = sum(stats['new_tokens'] for stats in language_stats.values())

    # Get previous cumulative totals
    previous_cumulative_lines = sum(stats['old_lines'] for stats in language_stats.values())
    previous_cumulative_tokens = sum(stats['old_tokens'] for stats in language_stats.values())
    
    # Calculate cumulative totals
    cumulative_total_lines = sum(stats['total_lines'] for stats in language_stats.values())
    cumulative_total_tokens = sum(stats['total_tokens'] for stats in language_stats.values())
    
    print(f"\nProcessed {len(language_stats)} language(s) in this iteration\n")
    
    # Detailed stats table with old, new, and total counts
    print("DETAILED STATISTICS:")
    print(f"{'Language':<15} {'Old Lines':<15} {'New Lines':<15} {'Total Lines':<15} {'Old Tokens':<18} {'New Tokens':<18} {'Total Tokens':<18}")
    print("=" * 129)
    
    for lang in sorted(language_stats.keys()):
        stats = language_stats[lang]
        print(f"{lang:<15} {stats['old_lines']:<15,} {stats['new_lines']:<15,} {stats['total_lines']:<15,} {stats['old_tokens']:<18,} {stats['new_tokens']:<18,} {stats['total_tokens']:<18,}")
    
    print("=" * 129)
    print(f"{'TOTAL':<15} {previous_cumulative_lines:<15,} {new_total_lines:<15,} {cumulative_total_lines:<15,} {previous_cumulative_tokens:<18,} {new_total_tokens:<18,} {cumulative_total_tokens:<18,}")
    print("=" * 129)
    
    print(f"\n📊 Summary:")
    print(f"   • Added this iteration: {new_total_lines:,} lines | {new_total_tokens:,} tokens")
    print(f"   • Grand Total: {cumulative_total_lines:,} lines | {cumulative_total_tokens:,} tokens")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CommonCrawl dump and separate by language"
    )

    parser.add_argument(
        "--warc_files_folder",
        type=str,
        required=True,
        help="Folder containing WARC files",
    )
    parser.add_argument(
        "--temp_output_folder",
        type=str,
        default="./language_filter_output",
        help="Temporary folder to store intermediate output (cleared each run)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./all_languages",
        help="Final output folder (results are appended)",
    )
    parser.add_argument(
        "--logs_folder",
        type=str,
        default="./logs",
        help="Folder to store logs",
    )
    parser.add_argument(
        "--dump",
        type=str,
        required=True,
        help="CommonCrawl dump name (e.g., CC-MAIN-2025-30)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        type=str,
        default=None,
        help="List of languages to filter (e.g., --languages bn pt hi). If not specified, all languages are kept.",
    )
    parser.add_argument(
        "--language_filter_backend",
        type=str,
        default="ft176",
        choices=["ft176", "glotlid"],
        help="Backend for language filtering: 'ft176' (default) or 'glotlid'",
    )
    parser.add_argument(
        "--language_threshold",
        type=float,
        default=0.65,
        help="Threshold for language filtering (default: 0.65)",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=10,
        help="Number of tasks per worker",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of processing workers",
    )
    parser.add_argument(
        "--expand_metadata",
        action="store_true",
        help="Whether to expand metadata in the output JSONL files",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="Qwen/Qwen3-0.6B-Base",
        help="Tokenizer name or path for token counting (default: Qwen3)",
    )

    args = parser.parse_args()
    main(args)