"""
CommonCrawl Multi-stage Quality Filtering Pipeline

Processes CommonCrawl WARC archives through comprehensive filtering to extract high-quality
multilingual text data. Implements a 2-stage pipeline with language-specific configurations.

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

Supported languages (configurable):
- Portuguese (por_Latn), Bengali (ben_Beng), Hindi (hin_Deva)
- Extensible to any language supported by DataTrove

Output structure:
- warc_extraction/LANG/: Intermediate language-separated files
- quality_filter/LANG/: Filtered files before final processing
- output/LANG/output.jsonl: Final consolidated dataset per language
- output/LANG/.metadata: Statistics (lines, tokens)

Output fields (customizable via KEEP_KEYS):
- text, id, source, url, date, file_path
- language, language_score, token_count

Usage:
    # Process CC-MAIN-2025-30 for Portuguese and Bengali
    python process_cc_dump.py \
        --warc_files_folder /data/cc/CC-MAIN-2025-30/ \
        --config_folder .configs/ \
        --final_output_folder output/ \
        --dump CC-MAIN-2025-30 \
        --languages pt bn \
        --tasks 32 --workers 32
"""
import argparse
import shutil
import yaml
import os
from functools import partial
import glob
import json
from datetime import datetime

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
    TOKENIZER_NAME_OR_PATH = args.tokenizer_name_or_path if args.tokenizer_name_or_path else "Qwen/Qwen3-0.6B"

    # WARCS should be downloaded from https://commoncrawl.org/the-data/get-started/
    WARC_FILES_FOLDER = args.warc_files_folder
    CONFIG_FOLDER = args.config_folder
    LOGS_FOLDER = args.logs_folder
    WARC_EXTRACTION_OUTPUT = args.warc_extraction_output
    QUALITY_FILTER_OUTPUT = args.quality_filter_output
    FINAL_OUTPUT_FOLDER = args.final_output_folder
    OUTPUT_FILE = args.output_file

    # All available language configuration files can be found here: data/.configs
    # Languages we are currently interested in:
    # - Portuguese
    # - Bengali
    # - Hindi
    lang_script_dict = {
        "pt": f"{CONFIG_FOLDER}/por_Latn.yml", # portuguese
        "bn": f"{CONFIG_FOLDER}/ben_Beng.yml", # bengali
        "hi": f"{CONFIG_FOLDER}/hin_Deva.yml", # hindi
    }

    # All languages supported: https://raw.githubusercontent.com/huggingface/datatrove/refs/heads/main/src/datatrove/utils/typeshelper.py
    # We need to set this for the quality filters. If None, it will use english as the default language ("eng").
    lang_id_dict = {
        "pt": "por_Latn",
        "bn": "ben",
        "hi": "hin"
    }

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
                #limit=1_000_000,  # Uncomment to limit the number of WARC files processed (useful for debugging)
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
    print("[INFO] Starting WARC extraction pipeline...")
    warc_extract.run()
    print("[INFO] WARC extraction pipeline completed.")

    # Stage 2.1: Apply quality filters to the extracted content
    # Language specific processing
    print("[INFO] Starting language-specific processing...")
    
    # Dictionary to store statistics for each language
    language_statistics = {}
    
    for lang in os.listdir(WARC_EXTRACTION_OUTPUT):

        # Define the current working folder
        lang_folder = os.path.join(WARC_EXTRACTION_OUTPUT, lang)

        if not os.path.isdir(lang_folder):
            print(f"[INFO] Could not find language folder: '{lang_folder}'")
            continue  # Skip if not a directory
        
        # Define the current output folder
        lang_output_folder = os.path.join(QUALITY_FILTER_OUTPUT, lang)
        os.makedirs(lang_output_folder, exist_ok=True)

        print(f"[INFO] Processing language: '{lang}'")
        # Load the specific thresholds, stopwords and other configurations for the language
        with open(lang_script_dict[lang], "r") as f:
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
                    language=lang_id_dict[lang],  # We need this to know which word tokenizer to use to split into words and ngrams.
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
                    language=lang_id_dict[lang],
                    short_line_thr=999,
                    char_duplicates_ratio=0.1,
                    line_punct_thr=filter_config["line_punct_thr"],
                    new_line_ratio=filter_config['new_line_ratio'],
                    exclusion_writer=None,
                ),

                # [GopherQualityFilter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/gopher_quality_filter.py#L13)
                GopherQualityFilter(
                    language=lang_id_dict[lang],
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

        print(f"[INFO] Starting language-specific processing for '{lang}'...")
        filtering_pipeline.run()
        print(f"[INFO] Language-specific processing for '{lang}' completed.")

        # Stage 2.2: Post-processing
        print(f"\n{'='*80}")
        print(f"POST-PROCESSING: {lang.upper()}")
        print(f"{'='*80}")

        # Get all JSONL files in the `lang_output_folder`
        all_files = glob.glob(f"{lang_output_folder}/*.jsonl")
        
        if not all_files:
            print(f"⚠️  No JSONL files found in {lang_output_folder}")
            continue  # Skip if no files found

        print(f"📂 Found {len(all_files)} JSONL files")

        # List of columns to keep
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
            print(f"📝 Appending to existing file: {output_file_path.split('/')[-1]}")
            print(f"   └─ Existing: {existing_documents:,} documents, {existing_tokens:,} tokens")
        
        # Write the filtered data to the output file
        # First we try with `ensure_ascii=False`
        try:
            with open(output_file_path, "a", encoding="utf-8") as f:
                for item in language_data:
                    f.write(json.dumps(item, default=json_serializer, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"⚠️  Error writing to {output_file_path.split('/')[-1]} with ensure_ascii=False:\nError: {e}")
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
        print(f"\n{'─'*80}")
        print(f"📊 STATISTICS FOR '{lang.upper()}'")
        print(f"{'─'*80}")
        print(f"  New Documents Added    : {new_documents:>15,}")
        print(f"  New Tokens Added       : {new_tokens:>15,}")
        print(f"  {'─'*78}")
        print(f"  Total Documents        : {total_documents:>15,}")
        print(f"  Total Tokens           : {total_tokens:>15,}")
        if total_documents > 0:
            avg_tokens_per_doc = total_tokens / total_documents
            print(f"  Avg Tokens/Document    : {avg_tokens_per_doc:>15,.2f}")
        print(f"{'─'*80}")
        print(f"✅ Post-processing for '{lang}' completed.\n")

    # Print overall summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY - ALL LANGUAGES")
    print(f"{'='*80}\n")
    
    if language_statistics:
        # Calculate totals
        total_all_documents = sum(stats["total_documents"] for stats in language_statistics.values())
        total_all_tokens = sum(stats["total_tokens"] for stats in language_statistics.values())
        total_new_documents = sum(stats["new_documents"] for stats in language_statistics.values())
        total_new_tokens = sum(stats["new_tokens"] for stats in language_statistics.values())
        
        # Print per-language summary table
        print(f"{'Language':<12} {'Documents':>15} {'Tokens':>18} {'Avg Tokens/Doc':>18}")
        print(f"{'─'*12} {'─'*15} {'─'*18} {'─'*18}")
        
        for lang, stats in sorted(language_statistics.items()):
            avg_tokens = stats["total_tokens"] / stats["total_documents"] if stats["total_documents"] > 0 else 0
            print(f"{lang:<12} {stats['total_documents']:>15,} {stats['total_tokens']:>18,} {avg_tokens:>18,.2f}")
        
        print(f"{'─'*12} {'─'*15} {'─'*18} {'─'*18}")
        avg_all = total_all_tokens / total_all_documents if total_all_documents > 0 else 0
        print(f"{'TOTAL':<12} {total_all_documents:>15,} {total_all_tokens:>18,} {avg_all:>18,.2f}")
        
        print(f"\n{'─'*80}")
        print(f"📈 NEW DATA ADDED IN THIS RUN")
        print(f"{'─'*80}")
        print(f"  Total New Documents    : {total_new_documents:>15,}")
        print(f"  Total New Tokens       : {total_new_tokens:>15,}")
        print(f"{'='*80}\n")
    else:
        print("⚠️  No language data was processed.\n")

    print("✅ All language-specific processing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CommonCrawl dump using DataTrove")

    parser.add_argument(
        "--config_folder", type=str, required=True, help="The folder containing language configuration files"
    )
    parser.add_argument(
        "--warc_files_folder", type=str, required=True, help="Folder containing WARC files"
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
        "--tokenizer_name_or_path", type=str, default=None, help="Tokenizer name or path (default: Qwen/Qwen3-0.6B)"
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
