"""
CommonCrawl Multi-stage Quality Filtering Pipeline

Processes CommonCrawl WARC archives through comprehensive filtering to extract
multilingual text data. Implements a 2-stage pipeline with 
language-specific configurations.

Pipeline stages:

Stage 1 - WARC Extraction:
1. Read WARC files from CommonCrawl dumps
2. Filter URLs (optional blocklists for spam/adult content)
3. Extract clean text using Trafilatura (precision-focused)
4. Perform initial language detection (FT176 - 176 languages)
5. Write language-separated intermediate files

Stage 2 - Quality Filtering (per-language):
1. Secondary language detection (GlotLID - 1665 languages, 2102 labels)
2. Language score thresholding (custom per language)
3. Gopher Repetition Filter (line/n-gram deduplication)
4. FineWeb Quality Filter (punctuation, newlines, char duplicates)
5. Gopher Quality Filter (word length, stop words, alpha ratio)
6. Formatting cleanup (FTFY encoding fixes, PII removal, symbol lines)
7. Token counting with a selected tokenizer (default: Qwen/Qwen3-0.6B)
8. Output to final JSONL files

Output structure:
- warc_extraction/LANG/: Intermediate language-separated files
- quality_filter/LANG/: Filtered files before final processing
- output/LANG/output.jsonl: Final consolidated dataset per language
- output/LANG/.metadata: Statistics (lines, tokens)

Notes: 
- Output fields are customizable via KEEP_KEYS.
- To make things simple, we have a hardcoded list of languages 
  and their corresponding config files. You can easily add more 
  languages by adding their config files and updating the 
  `LANG_CONFIG_MAPPING` and `LANG_FILTER_MAPPING`.

Usage:
    # Process CC-MAIN-2025-30 for Portuguese and Bengali
    python process_cc_dump_with_quality_filters.py \
        --warc_files_folder /data/cc/CC-MAIN-2025-30/ \
        --config_folder .configs/ \
        --final_output_folder output/ \
        --dump CC-MAIN-2025-30 \
        --languages pt bn \
        --tasks 32 --workers 32
"""
import argparse
import yaml
import os
from functools import partial
import glob
import json
from datetime import datetime

from utils import get_logger, write_metadata, initialize_or_load_metadata

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    URLFilter,
    LambdaFilter,
)

from datatrove.pipeline.formatters import PIIFormatter, FTFYFormatter, SymbolLinesFormatter
from datatrove.pipeline.readers import WarcReader, JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.tokens import TokensCounter

from langcodes import FT176_LANGUAGE_CODES

logger = get_logger("CC-Processing-Pipeline")


def main(args):

    TASKS = args.tasks
    WORKERS = args.workers
    DUMP = args.dump
    TOKENIZER_NAME_OR_PATH = args.tokenizer_name_or_path

    # WARCS should be downloaded from https://commoncrawl.org/the-data/get-started/
    WARC_FILES_FOLDER = args.warc_files_folder
    CONFIG_FOLDER = args.config_folder
    LOGS_FOLDER = args.logs_folder
    WARC_EXTRACTION_OUTPUT = args.warc_extraction_output
    QUALITY_FILTER_OUTPUT = args.quality_filter_output
    FINAL_OUTPUT_FOLDER = args.final_output_folder
    OUTPUT_FILE = args.output_file

    # Assert that the language code is valid
    if args.languages:
        assert all(lang in FT176_LANGUAGE_CODES for lang in args.languages), \
            "Invalid language code provided. Check the supported languages for the FT176 backend."

    # All available language configuration files can be found here: data/.configs
    # Eg.,
    # - Portuguese
    # - Bengali
    # - Hindi
    # - German
    # - etc ...
    LANG_CONFIG_MAPPING = {
        "pt": f"{CONFIG_FOLDER}/por_Latn.yml", # portuguese
        "bn": f"{CONFIG_FOLDER}/ben_Beng.yml", # bengali
        "hi": f"{CONFIG_FOLDER}/hin_Deva.yml", # hindi
        "de": f"{CONFIG_FOLDER}/deu_Latn.yml", # german
        # Add more languages and their corresponding config files here as needed
    }

    # All languages supported: https://raw.githubusercontent.com/huggingface/datatrove/refs/heads/main/src/datatrove/utils/typeshelper.py
    # We need to set this for the quality filters. If None, it will use english as the default language ("eng").
    LANG_FILTER_MAPPING = {
        "pt": "por_Latn",
        "bn": "ben",
        "hi": "hin",
        "de": "deu"
    }

    # List of columns to keep when writing the final output file. 
    # You can customize this list based on your needs.
    KEEP_KEYS = [
        "text",
        "id",
        "source",
        "url",
        "date",
        "file_path",
        "language",
        "language_score",
        "token_count",
    ]

    # Helper function for JSON serialization
    def json_serializer(obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    # Stage 1: Download and extract content from CommonCrawl WARC files
    warc_extract = LocalPipelineExecutor(
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
            # We can also  specify banned_words, banned_subwords, soft_banned_words
            URLFilter(exclusion_writer=None),

            # [Trafilatura](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/extractors/trafilatura.py)
            # [Original documentation](https://trafilatura.readthedocs.io/en/latest/usage-python.html)
            # Trafilatura provides a better extraction of text content from HTML pages then the default HTML parser CommonCrawl uses the WET format.
            # Ablation results available in https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html#starting_point:_text_extraction
            Trafilatura(favour_precision=True),

            # [LanguageFilter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/language_filter.py)
            # Default option is [FT176](https://fasttext.cc/docs/en/language-identification.html)
            # FT176 gives support to ~176 languages.
            LanguageFilter(
                languages=args.languages if args.languages else None,
                exclusion_writer=None,
            ),

            # [writers: JsonlWriter, ParquetWriter, HuggingFaceDatasetWriter](https://github.com/huggingface/datatrove/tree/main/src/datatrove/pipeline/writers)
            # [JsonlWriter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/writers/jsonl.py)
            JsonlWriter(
                WARC_EXTRACTION_OUTPUT,
                output_filename="${language}/${language}.jsonl", 
                compression=None, expand_metadata=args.expand_metadata),
        ],
        tasks=TASKS,
        workers=WORKERS,
        logging_dir=f"{LOGS_FOLDER}/warc_extraction",
    )

    # Run the WARC extraction pipeline
    logger.info("Starting WARC extraction pipeline...")
    warc_extract.run()
    logger.info("WARC extraction pipeline completed.")

    # Stage 2.1: Apply quality filters to the extracted content
    # Language specific processing
    logger.info("Starting language-specific processing...")
    
    # Dictionary to store statistics for each language
    language_statistics = {}
    
    for lang in os.listdir(WARC_EXTRACTION_OUTPUT):

        # Define the current working folder
        lang_folder = os.path.join(WARC_EXTRACTION_OUTPUT, lang)

        if not os.path.isdir(lang_folder):
            logger.info("Could not find language folder: '%s'", lang_folder)
            continue  # Skip if not a directory
        
        # Define the current output folder
        lang_output_folder = os.path.join(QUALITY_FILTER_OUTPUT, lang)
        os.makedirs(lang_output_folder, exist_ok=True)

        logger.info("Processing language: '%s'", lang)
        # Load the specific thresholds, stopwords and other configurations for the language
        with open(LANG_CONFIG_MAPPING[lang], "r") as f:
            filter_config = yaml.safe_load(f)
        
        def above_lang_threshold(doc, threshold):
            """
            Check if the document's language score is above the specified threshold.
            """
            return doc.metadata["language_score"] >= threshold
        
        filtering_pipeline = LocalPipelineExecutor(
            pipeline=[
                # [readers: HuggingFaceDatasetReader, JsonlReader, ParquetReader](https://github.com/huggingface/datatrove/tree/main/src/datatrove/pipeline/readers)
                # [JsonlWriter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/writers/jsonl.py)
                JsonlReader(lang_folder),

                # [LanguageFilter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/language_filter.py)
                # Using [GlotLID](https://github.com/cisnlp/GlotLID)
                # GlotLID gives supports 1665 languages (2102 labels).
                # Paper: https://aclanthology.org/2023.findings-emnlp.410/
                # What is happening? ft176 must be above `threshold`, and the alternative labels (from GlotLID) must also be above `threshold` for a document to be kept.
                LanguageFilter(
                    backend="glotlid", 
                    label_only=True, # if True, only the language label is added to the metadata and no documents are removed
                    keep_top_pairs_threshold=0.01, # keep a list of all language pairs with at least this score. -1 to disable
                ),

                # [LambdaFilter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/lambda_filter.py#L8)
                LambdaFilter(
                    # Finaly, we only keep the documents that have a language score a language specific threshold
                    filter_function=partial(above_lang_threshold, threshold=filter_config["language_score"]),
                    exclusion_writer=None
                ),

                # [GopherRepetitionFilter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/gopher_repetition_filter.py#L73)
                GopherRepetitionFilter(
                    language=LANG_FILTER_MAPPING[lang],  # We need this to know which word tokenizer to use to split into words and ngrams.
                    dup_para_frac=0,
                    dup_line_char_frac=0,
                    dup_para_char_frac=0,
                    dup_line_frac=filter_config['dup_line_frac'],
                    top_n_grams=filter_config["top_n_grams"],
                    dup_n_grams=filter_config["dup_n_grams"],
                    exclusion_writer=None,
                ),

                # [FineWebQualityFilter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/fineweb_quality_filter.py)
                FineWebQualityFilter(
                    language=LANG_FILTER_MAPPING[lang],
                    short_line_thr=999,
                    char_duplicates_ratio=0.1,
                    line_punct_thr=filter_config["line_punct_thr"],
                    new_line_ratio=filter_config['new_line_ratio'],
                    exclusion_writer=None,
                ),

                # [GopherQualityFilter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/gopher_quality_filter.py#L13)
                GopherQualityFilter(
                    language=LANG_FILTER_MAPPING[lang],
                    max_avg_word_length=filter_config['max_avg_word_length'],
                    min_avg_word_length=filter_config['min_avg_word_length'],
                    stop_words=filter_config['stopwords'],
                    max_non_alpha_words_ratio=filter_config['max_non_alpha_words_ratio'],
                    min_stop_words=2,
                    exclusion_writer=None,
                ),

                #[formatters: FTFYFormatter, PIIFormatter, SymbolLinesFormatter](https://github.com/huggingface/datatrove/tree/main/src/datatrove/pipeline/formatters)
                # [FTFYFormatter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/formatters/ftfy.py)
                FTFYFormatter(),  # Fix encoding issues. Important in a multilingual setting!

                # [PIIFormatter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/formatters/pii.py#L42)
                # This will remove PII from the dataset, but it will not remove the samples that contain PII.
                PIIFormatter(),

                # [SymbolLinesFormatter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/formatters/symbol_lines_remover.py)
                # Removes lines that consist exclusively of symbols. Keeps lines that only have whitespace characters.
                SymbolLinesFormatter(symbols_to_remove=["|"], replace_char="\n"),

                # [TokensCounter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/tokens/counter.py#L7)
                TokensCounter(tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH),

                # [writers: JsonlWriter, ParquetWriter, HuggingFaceDatasetWriter](https://github.com/huggingface/datatrove/tree/main/src/datatrove/pipeline/writers)
                # [JsonlWriter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/writers/jsonl.py)
                JsonlWriter(
                    lang_output_folder,
                    output_filename="${language}.jsonl", 
                    compression=None, 
                    expand_metadata=args.expand_metadata,
                ),
            ],
            tasks=TASKS,
            workers=WORKERS,
            logging_dir=f"{LOGS_FOLDER}/quality_filtering/{lang}",
            depends=warc_extract
        )

        logger.info("Starting language-specific processing for '%s'...", lang)
        filtering_pipeline.run()
        logger.info("Language-specific processing for '%s' completed.", lang)

        # Stage 2.2: Post-processing
        logger.info("POST-PROCESSING: %s", lang.upper())

        # Get all JSONL files in the `lang_output_folder`
        all_files = glob.glob(f"{lang_output_folder}/*.jsonl")
        
        if not all_files:
            logger.warning("No JSONL files found in %s", lang_output_folder)
            continue  # Skip if no files found

        logger.info("Found %d JSONL files", len(all_files))

        # List to store language-specific data
        language_data = []
        token_count = 0
        total_documents_filtered = 0
        
        for file_path in all_files:
            # Read them as a list of JSON objects
            with open(file_path, "r") as f:
                for line in f:
                    try:
                        json_object = json.loads(line)
                        filtered_object = {k: json_object[k] for k in KEEP_KEYS if k in json_object}
                        token_count += filtered_object.get("token_count", 0)
                        language_data.append(filtered_object)
                        total_documents_filtered += 1
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue # Skip lines that are not valid JSON or do not have the expected keys

        # Create an output folder for the language
        final_language_output_folder = os.path.join(FINAL_OUTPUT_FOLDER, lang)
        os.makedirs(final_language_output_folder, exist_ok=True)

        # Load existing metadata (efficient tracking)
        previous_metadata = initialize_or_load_metadata(final_language_output_folder)
        existing_documents = previous_metadata.get('lines', 0)
        existing_tokens = previous_metadata.get('tokens', 0)

        # Create the output file path if it doesn't exist
        output_file_path = os.path.join(final_language_output_folder, OUTPUT_FILE)
        
        if not os.path.exists(output_file_path):
            with open(output_file_path, "w") as f:
                pass
        else:
            # If the file already exists, we can append to it
            logger.info("Appending to existing file: %s", output_file_path.split('/')[-1])
            logger.info("Existing: %d documents, %d tokens", existing_documents, existing_tokens)
        
        # Write the filtered data to the output file
        # First we try with `ensure_ascii=False`
        try:
            with open(output_file_path, "a", encoding="utf-8") as f:
                for item in language_data:
                    f.write(json.dumps(item, default=json_serializer, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Error writing to %s with ensure_ascii=False: %s", output_file_path.split('/')[-1], e)
            # If it fails, we do it with `ensure_ascii=True`
            with open(output_file_path, "a", encoding="utf-8") as f:
                for item in language_data:
                    f.write(json.dumps(item, default=json_serializer, ensure_ascii=True) + "\n")

        # Calculate statistics
        new_documents = total_documents_filtered
        new_tokens = token_count
        total_documents = existing_documents + new_documents
        total_tokens = existing_tokens + new_tokens
        
        # Update metadata
        updated_metadata = {
            'lines': total_documents,
            'tokens': total_tokens
        }
        write_metadata(os.path.join(final_language_output_folder, '.metadata'), updated_metadata)
        
        # Store statistics
        language_statistics[lang] = {
            "existing_documents": existing_documents,
            "existing_tokens": existing_tokens,
            "new_documents": new_documents,
            "new_tokens": new_tokens,
            "total_documents": total_documents,
            "total_tokens": total_tokens,
        }
        
        # Print formatted statistics
        logger.info("-" * 80)
        logger.info("STATISTICS FOR '%s'", lang.upper())
        logger.info("-" * 80)
        logger.info("  New Documents Added    : %15d", new_documents)
        logger.info("  New Tokens Added       : %15d", new_tokens)
        logger.info("  Total Documents        : %15d", total_documents)
        logger.info("  Total Tokens           : %15d", total_tokens)
        if total_documents > 0:
            avg_tokens_per_doc = total_tokens / total_documents
            logger.info("  Avg Tokens/Document    : %19.2f", avg_tokens_per_doc)
        logger.info("-" * 80)
        logger.info("Post-processing for '%s' completed.", lang)

    # Print overall summary
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY - ALL LANGUAGES")
    logger.info("=" * 80)

    if language_statistics:
        # Calculate totals
        total_all_documents = sum(stats["total_documents"] for stats in language_statistics.values())
        total_all_tokens = sum(stats["total_tokens"] for stats in language_statistics.values())
        total_new_documents = sum(stats["new_documents"] for stats in language_statistics.values())
        total_new_tokens = sum(stats["new_tokens"] for stats in language_statistics.values())

        # Print per-language summary table
        header = f"{'Language':<12} {'Documents':>15} {'Tokens':>18} {'Avg Tokens/Doc':>18}"
        separator = "-" * 63
        logger.info(header)
        logger.info(separator)

        for lang, stats in sorted(language_statistics.items()):
            avg_tokens = stats["total_tokens"] / stats["total_documents"] if stats["total_documents"] > 0 else 0
            logger.info("%s %15d %18d %18.2f", f"{lang:<12}", stats["total_documents"], stats["total_tokens"], avg_tokens)

        logger.info(separator)
        avg_all = total_all_tokens / total_all_documents if total_all_documents > 0 else 0
        logger.info("%s %15d %18d %18.2f", f"{'TOTAL':<12}", total_all_documents, total_all_tokens, avg_all)

        logger.info("-" * 80)
        logger.info("NEW DATA ADDED IN THIS RUN")
        logger.info("-" * 80)
        logger.info("  Total New Documents    : %15d", total_new_documents)
        logger.info("  Total New Tokens       : %15d", total_new_tokens)
        logger.info("=" * 80)
    else:
        logger.warning("No language data was processed.")

    logger.info("All language-specific processing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CommonCrawl Processing Pipeline with Quality Filters")

    parser.add_argument(
        "--config_folder", type=str, required=True, help="The folder containing language configuration files"
    )
    parser.add_argument(
        "--warc_files_folder", type=str, required=True, help="Folder containing WARC files"
    )
    parser.add_argument(
            "--limit", type=int, default=-1, help="Limit the number of WARC files to process (useful for debugging)"
    )
    parser.add_argument(
        "--logs_folder", type=str, default="./logs", help="Folder to store logs"
    )
    parser.add_argument(
        "--warc_extraction_output", type=str, default="./warc_extraction", help="Folder to store WARC extraction output"
    )
    parser.add_argument(
        "--quality_filter_output", type=str, default="./quality_filter", help="Folder to store quality filter output"
    )
    parser.add_argument(
        "--final_output_folder", type=str, default="./output", help="Folder to store final output"
    )
    parser.add_argument(
        "--output_file", type=str, default="./output.jsonl", help="Path to the output JSONL file"
    )
    parser.add_argument(
        "--dump", type=str, required=True, help="CommonCrawl dump name (e.g., CC-MAIN-2023-23)"
    )
    parser.add_argument(
        "--tokenizer_name_or_path", type=str, default="Qwen/Qwen3-0.6B", help="Tokenizer name or path (default: Qwen/Qwen3-0.6B)"
    )
    parser.add_argument(
        "--languages", nargs='+', type=str, default=None, help="List of languages to filter (e.g., --languages bn pt hi)"
    )
    parser.add_argument("--tasks", type=int, default=10, help="Number of tasks")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument(
        "--expand_metadata", action="store_true", help="Whether to expand metadata in the output JSONL files"
    )

    args = parser.parse_args()
    main(args)
