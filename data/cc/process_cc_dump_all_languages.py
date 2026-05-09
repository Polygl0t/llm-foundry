"""
CommonCrawl Language Extraction Pipeline

Streamlined single-stage pipeline for language identification and text extraction from
CommonCrawl WARC archives.

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
    
Notes:
- Ensure WARC files are downloaded and accessible in the specified folder.
- Adjust language filter backend and threshold based on desired precision/recall.
- This script can be re-submitted if the job fails or if you want to process additional WARC files.
"""
import argparse
import shutil
import os
import json
import uuid
import glob

from utils import get_logger, write_metadata, initialize_or_load_metadata

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    LanguageFilter,
    URLFilter,
)

from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.tokens import TokensCounter

from langcodes import GLOTLID_LANGUAGE_CODES, FT176_LANGUAGE_CODES

logger = get_logger("CC-Processing-Pipeline")


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

    # Assert that the language code is valid
    if args.languages:
        assert all(lang in GLOTLID_LANGUAGE_CODES for lang in args.languages) \
            or all(lang in FT176_LANGUAGE_CODES for lang in args.languages), \
            "Invalid language code provided. Check the supported languages for the chosen backend."

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
                limit=args.limit,
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
    
    # POST-PROCESSING: Consolidate extracted data into a final output folder/files
    if not os.path.exists(TEMP_OUTPUT_FOLDER):
        logger.error("No temporary output folder found.")
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
                    logger.warning("Could not process %s: %s. Cached for inspection.", jsonl_file, e)
        
        if new_lines == 0:
            logger.warning("No valid data found for %s", lang)
            continue
        
        if invalid_lines > 0:
            logger.warning("%s: Skipped %d invalid lines", lang, invalid_lines)
        
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
    
    # SUMMARY    
    logger.info("=" * 80)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 80)

    if not language_stats:
        logger.warning("No languages were successfully processed.")
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

    logger.info("Processed %d language(s) in this iteration", len(language_stats))

    # Detailed stats table with old, new, and total counts
    header = f"{'Language':<15} {'Old Lines':<15} {'New Lines':<15} {'Total Lines':<15} {'Old Tokens':<18} {'New Tokens':<18} {'Total Tokens':<18}"
    separator = "=" * 129
    logger.info("DETAILED STATISTICS:")
    logger.info(header)
    logger.info(separator)

    for lang in sorted(language_stats.keys()):
        stats = language_stats[lang]
        logger.info(
            "%s %s %s %s %s %s %s",
            f"{lang:<15}",
            f"{stats['old_lines']:<15,}",
            f"{stats['new_lines']:<15,}",
            f"{stats['total_lines']:<15,}",
            f"{stats['old_tokens']:<18,}",
            f"{stats['new_tokens']:<18,}",
            f"{stats['total_tokens']:<18,}",
        )

    logger.info(separator)
    logger.info(
        "%s %s %s %s %s %s %s",
        f"{'TOTAL':<15}",
        f"{previous_cumulative_lines:<15,}",
        f"{new_total_lines:<15,}",
        f"{cumulative_total_lines:<15,}",
        f"{previous_cumulative_tokens:<18,}",
        f"{new_total_tokens:<18,}",
        f"{cumulative_total_tokens:<18,}",
    )
    logger.info(separator)
    logger.info("Added this iteration: %d lines | %d tokens", new_total_lines, new_total_tokens)
    logger.info("Grand Total: %d lines | %d tokens", cumulative_total_lines, cumulative_total_tokens)
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CommonCrawl Language Extraction Pipeline",
    )

    parser.add_argument(
        "--warc_files_folder",
        type=str,
        required=True,
        help="Folder containing WARC files",
    )
    parser.add_argument(
            "--limit", type=int, default=-1, help="Limit the number of WARC files to process (useful for debugging)"
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
        help=(
            "List of languages to filter (e.g., --languages bn pt hi). If not specified, all languages are kept. "
            "Available languages depend on the language filter backend. For FT176, supported languages are: \n"
            + ", ".join(FT176_LANGUAGE_CODES)
            + ". For GlotLID, supported languages are: \n"
            + ", ".join(GLOTLID_LANGUAGE_CODES)
        )
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