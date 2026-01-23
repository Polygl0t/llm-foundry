"""
Multi-stage Quality Filtering Pipeline using DataTrove

Applies comprehensive quality filters to text datasets with language-specific configurations.
Designed for large-scale web corpus cleaning and dataset curation.

Filtering stages:
1. FastText Language ID (FT176): Initial language detection across 176 languages
2. GlotLID Language ID: Secondary validation using 1665 languages (2102 labels)
3. Language Score Threshold: Keeps only documents above language-specific confidence
4. Gopher Repetition Filter: Removes documents with excessive line/n-gram repetition
5. FineWeb Quality Filter: Checks line punctuation, newline ratios, character duplicates
6. Gopher Quality Filter: Validates word length, stop word presence, alpha ratio
7. Formatting: Fixes encoding (FTFY), removes PII, cleans symbol-only lines
8. Token Counting: Adds token counts using specified tokenizer

Language configurations (in .configs/):
- Portuguese (por_Latn), Bengali (ben_Beng), Hindi (hin_Deva)
- Custom thresholds for: language_score, dup_line_frac, top_n_grams, dup_n_grams,
  line_punct_thr, new_line_ratio, max/min_avg_word_length, stopwords, etc.

Output:
- Filtered JSONL files with token_count field
- .metadata: Document/token counts and averages
- Logs in logs/quality_filters/

Usage:
    # Filter Portuguese dataset
    python quality_filters.py --data_folder raw/ --final_output_folder filtered/ \
        --language pt --config_folder .configs/ \
        --tokenizer_name_or_path Qwen/Qwen3-0.6B \
        --tasks 32 --workers 32
    
    # Filter with metadata expansion
    python quality_filters.py --data_folder data/ --final_output_folder clean/ \
        --language bn --expand_metadata --cache_dir .cache/
"""
import os
import yaml
import glob
import json
import datasets
import argparse
from functools import partial

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.filters import (
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    LambdaFilter
)

from datatrove.pipeline.formatters import PIIFormatter, FTFYFormatter, SymbolLinesFormatter
from datatrove.pipeline.readers import JsonlReader
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

def main(args):

    TASKS = args.tasks
    WORKERS = args.workers
    CONFIG_FOLDER = args.config_folder
    DATA_FOLDER = args.data_folder
    LOGS_FOLDER = args.logs_folder
    FINAL_OUTPUT_FOLDER = args.final_output_folder
    TOKENIZER_NAME_OR_PATH = args.tokenizer_name_or_path
    LANGUAGE = args.language

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
    
    # Load the specific thresholds, stopwords and other configurations for the language
    with open(lang_script_dict[LANGUAGE], "r") as f:
        filter_config = yaml.safe_load(f)

    # Define a lambda function to filter by language score
    def above_lang_threshold(doc, threshold):
        """
        Check if the document's language score is above the specified threshold.
        """
        return doc.metadata["language_score"] >= threshold

    # Run Quality Filters 
    quality_filters = LocalPipelineExecutor(
        pipeline=[
            # [readers: HuggingFaceDatasetReader, JsonlReader, ParquetReader](https://github.com/huggingface/datatrove/tree/main/src/datatrove/pipeline/readers)
            # [JsonlWriter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/writers/jsonl.py)
            JsonlReader(data_folder=DATA_FOLDER),

            # [LanguageFilter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/language_filter.py)
            # Default option is [FT176](https://fasttext.cc/docs/en/language-identification.html)
            # FT176 gives support to ~176 languages.
            LanguageFilter(
                languages=args.languages if args.languages else None,
                exclusion_writer=None,
            ),

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
                language=lang_id_dict[LANGUAGE],  # We need this to know which word tokenizer to use to split into words and ngrams.
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
                language=lang_id_dict[LANGUAGE],
                short_line_thr=999,
                char_duplicates_ratio=0.1,
                line_punct_thr=filter_config["line_punct_thr"],
                new_line_ratio=filter_config['new_line_ratio'],
                exclusion_writer=None,
            ),

            # [GopherQualityFilter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/gopher_quality_filter.py#L13)
            GopherQualityFilter(
                language=lang_id_dict[LANGUAGE],
                max_avg_word_length=filter_config['max_avg_word_length'],
                min_avg_word_length=filter_config['min_avg_word_length'],
                stop_words=filter_config['stopwords'],
                max_non_alpha_words_ratio=filter_config['max_non_alpha_words_ratio'],
                min_stop_words=2,
                exclusion_writer=None,
            ),

            # [TokensCounter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/tokens/counter.py#L7)
            TokensCounter(tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH),

            ##[formatters: FTFYFormatter, PIIFormatter, SymbolLinesFormatter](https://github.com/huggingface/datatrove/tree/main/src/datatrove/pipeline/formatters)
            # [FTFYFormatter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/formatters/ftfy.py)
            FTFYFormatter(),  # Fix encoding issues. Important in a multilingual setting!

            # [PIIFormatter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/formatters/pii.py#L42)
            # This will remove PII from the dataset, but it will not remove the samples that contain PII.
            PIIFormatter(),

            # [SymbolLinesFormatter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/formatters/symbol_lines_remover.py)
            # Removes lines that consist exclusively of symbols. Keeps lines that only have whitespace characters.
            SymbolLinesFormatter(symbols_to_remove=["|"], replace_char="\n"),

            # [writers: JsonlWriter, ParquetWriter, HuggingFaceDatasetWriter](https://github.com/huggingface/datatrove/tree/main/src/datatrove/pipeline/writers)
            # [JsonlWriter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/writers/jsonl.py)
            JsonlWriter(FINAL_OUTPUT_FOLDER, compression=None, expand_metadata=args.expand_metadata),
        ],
        tasks=TASKS,
        workers=WORKERS,
        logging_dir=LOGS_FOLDER + "/quality_filters",
    )

    print(f"[INFO] Running quality filters for: '{LANGUAGE}'")
    quality_filters.run()
    print(f"[INFO] Quality filters for '{LANGUAGE}' completed successfully. Output saved to: {FINAL_OUTPUT_FOLDER}")

    # Post-processing: Calculate and save metadata
    print(f"\n{'='*80}")
    print(f"[POST-PROCESSING] {LANGUAGE.upper()}")
    print(f"{'='*80}")

    # Get all JSONL files in the output folder
    all_files = glob.glob(f"{FINAL_OUTPUT_FOLDER}/*.jsonl")
    
    if not all_files:
        print(f"⚠️  No JSONL files found in {FINAL_OUTPUT_FOLDER}")
        return

    print(f"📂 Found {len(all_files)} JSONL files")

    # Calculate statistics from output files
    total_documents = 0
    total_tokens = 0
    

    try:
        # Try to load with datasets library for efficiency
        data = datasets.load_dataset(
            "json",
            data_files=all_files, 
            split="train",
            cache_dir=args.cache_dir,
            num_proc=len(all_files),
        )

        total_documents = len(data)
        if 'token_count' in data.column_names:
            total_tokens = sum(data['token_count'])
    except Exception as e:
        # Fallback to file-by-file, line-by-line parsing
        print(f"⚠️  Using fallback parsing for statistics due to error: {e}")
        for file_path in all_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json_object = json.loads(line)
                        total_documents += 1
                        total_tokens += json_object.get('token_count', 0)
                    except Exception:
                        continue

    # Save metadata
    metadata = {
        'lines': total_documents,
        'tokens': total_tokens
    }
    metadata_file = os.path.join(FINAL_OUTPUT_FOLDER, '.metadata')
    write_metadata(metadata_file, metadata)
    
    # Print formatted statistics
    print(f"\n{'─'*80}")
    print(f"📊 STATISTICS FOR '{LANGUAGE.upper()}'")
    print(f"{'─'*80}")
    print(f"  Total Documents        : {total_documents:>15,}")
    print(f"  Total Tokens           : {total_tokens:>15,}")
    if total_documents > 0:
        avg_tokens_per_doc = total_tokens / total_documents
        print(f"  Avg Tokens/Document    : {avg_tokens_per_doc:>15,.2f}")
    print(f"{'─'*80}")
    print(f"✅ Metadata saved to: {metadata_file}")
    print(f"✅ Post-processing for '{LANGUAGE}' completed.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quality Filters for text data")

    parser.add_argument("--tasks", type=int, default=32, help="Number of tasks")
    parser.add_argument("--workers", type=int, default=32, help="Number of workers")
    parser.add_argument("--cache_dir", type=str, default="./.cache", help="Cache directory for datasets library")
    parser.add_argument("--config_folder", type=str, default="./.configs", help="Folder containing language configuration files")
    parser.add_argument("--data_folder", type=str, default="./dataset", help="Path to the data folder")
    parser.add_argument("--logs_folder", type=str, default="./logs", help="Path to the logs folder")
    parser.add_argument("--expand_metadata", action="store_true", help="Expand metadata")
    parser.add_argument("--final_output_folder", type=str, default="./final_output", help="Final output folder")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="Qwen/Qwen3-0.6B", help="Tokenizer name or path")
    parser.add_argument("--language", type=str, required=True, help=f"Language code. Currently supported: pt, bn, hi")

    args = parser.parse_args()
    main(args)
