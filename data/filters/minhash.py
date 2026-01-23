"""
MinHash-based Dataset Deduplication Pipeline

Implements scalable fuzzy deduplication using MinHash signatures via DataTrove library.
Detects and removes near-duplicate documents while preserving one representative per cluster.

Methodology (4-stage pipeline):
1. Signature Generation: Computes MinHash signatures for each document using n-gram hashing
2. Bucket Clustering: Groups similar signatures into buckets using LSH (Locality-Sensitive Hashing)
3. Cluster Formation: Creates duplicate clusters across all buckets
4. Filtering: Removes duplicates, keeping one sample per cluster, and counts tokens

MinHash Configuration (FineWeb settings):
- num_buckets: 14, hashes_per_bucket: 8, n_grams: 5
- Hash function: xxhash with 64-bit precision
- Probability formula: P(x,y) = [1 - (1 - sim(x,y)^n_hashes)^n_buckets]
- Language-specific word tokenization for n-gram generation

Output structure:
- minhash_signatures/: MinHash signature files
- minhash_bucket/: Bucket clustering results
- removed_ids/: IDs of removed duplicates
- duplicated_samples/: Excluded duplicate documents
- deduplication_final/: Clean deduplicated dataset with token counts
- .metadata: Dataset statistics

Usage:
    # Deduplicate Portuguese dataset
    python minhash.py --data_folder raw_data/ --language pt \
        --output_deduplication_final deduplicated/ \
        --tokenizer_name_or_path Qwen/Qwen3-0.6B \
        --tasks 32 --workers 32
    
    # Deduplicate with metadata expansion
    python minhash.py --data_folder data/ --language bn \
        --output_deduplication_final clean_data/ \
        --expand_metadata --cache_dir .cache/
"""
import os
import glob
import json
import datasets
import argparse
from functools import partial

from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.utils.hashing import HashConfig
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.executor import LocalPipelineExecutor
from datatrove.utils.typeshelper import Languages
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
    DATA_FOLDER = args.data_folder
    LOGS_FOLDER = args.logs_folder

    OUTPUT_FOLDER_MINHASH_SIGNATURES = args.output_minhash_signatures
    OUTPUT_FOLDER_MINHASH_BUCKET = args.output_minhash_bucket
    OUTPUT_FOLDER_REMOVED_IDS = args.output_removed_ids
    OUTPUT_FOLDER_DUPLICATED_SAMPLES = args.output_duplicated_samples
    OUTPUT_DEDUPLICATION_FINAL = args.output_deduplication_final

    TOKENIZER_NAME_OR_PATH = args.tokenizer_name_or_path
    LANGUAGE = args.language

    # All languages supported: https://raw.githubusercontent.com/huggingface/datatrove/refs/heads/main/src/datatrove/utils/typeshelper.py
    # We need to set this for the quality filters. If None, it will use english as the default language ("eng").
    lang_id_dict = {
        "pt": "por_Latn",
        "bn": "ben",
        "hi": "hin"
    }

    # [MinhashConfig](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/dedup/minhash.py)
    minhash_config = MinhashConfig(
        # [HashConfig](https://github.com/huggingface/datatrove/blob/main/src/datatrove/utils/hashing.py)
        hash_config=HashConfig(
            hash_fc="xxhash", 
            precision=64, # better precision -> fewer false positives (collisions)
        ),
        num_buckets=14,
        hashes_per_bucket=8,
        n_grams=5,
        # This is the same configuration used to create the FineWeb dataset
        #
        # What is the probability that two documents are similar given their MinHash signatures? The probability of similarity can be estimated using the formula:
        #
        # $$\text{Probability}(x,y) = [1 - \left(1 - \text{similarity}(x,y)^{n_{\text{hashes}}}\right)^{n_{\text{buckets}}}]$$
        #
        # Where:
        #
        # - $\text{similarity}(x,y)$ is the estimated similarity between the documents $x$ and $y$.
        # - $n_{\text{hashes}}$ is the number of hash functions used in the MinHash process.
        # - $n_{\text{buckets}}$ is the number of buckets used to cluster the MinHash signatures.
    )

    # Stage 1: Computes minhash signatures
    stage1 = LocalPipelineExecutor(
        [
        # [readers: HuggingFaceDatasetReader, JsonlReader, ParquetReader](https://github.com/huggingface/datatrove/tree/main/src/datatrove/pipeline/readers)
        # [JsonlWriter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/writers/jsonl.py)
        JsonlReader(
            data_folder=DATA_FOLDER
        ),

        # [MinhashDedupSignature](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/dedup/minhash.py)
        MinhashDedupSignature(
            output_folder=OUTPUT_FOLDER_MINHASH_SIGNATURES, 
            config=minhash_config, 
            language=lang_id_dict[LANGUAGE],  # We need this to know which word tokenizer to use to split into words and ngrams.
        ),
        ],
        tasks=TASKS,
        workers=WORKERS,
        logging_dir=LOGS_FOLDER + "/minhash_signatures",
    )

    # Stage 2: Computes matches between signatures
    stage2 = LocalPipelineExecutor(
        pipeline=[
            # [MinhashDedupBuckets](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/dedup/minhash.py)
            MinhashDedupBuckets(
                input_folder=OUTPUT_FOLDER_MINHASH_SIGNATURES,
                output_folder=OUTPUT_FOLDER_MINHASH_BUCKET,
                config=MinhashConfig(hash_config=minhash_config.hash_config),
            ),
        ],
        tasks=minhash_config.num_buckets, # the code supports parallelizing each bucket (num_buckets * n, where n is the number of workers per bucket)
        workers=WORKERS,
        logging_dir=LOGS_FOLDER + "/minhash_buckets",
        depends=stage1,
    )

    # Stage 3: Creates clusters of duplicates using the results from all buckets
    stage3 = LocalPipelineExecutor(
        pipeline=[
            # [MinhashDedupCluster](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/dedup/minhash.py)
            MinhashDedupCluster(
                input_folder=OUTPUT_FOLDER_MINHASH_BUCKET,
                output_folder=OUTPUT_FOLDER_REMOVED_IDS,
                config=minhash_config,
            ),
        ],
        tasks=1,
        workers=WORKERS,
        logging_dir=LOGS_FOLDER + "/minhash_clusters",
        depends=stage2,
    )

    # Stage 4: Reads the original input data and removes all but 1 sample per duplicate cluster.
    # The data must match exactly stage 1, so number of tasks and the input source must be the same
    stage4 = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
            data_folder=DATA_FOLDER
            ),
            # [MinhashDedupFilter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/dedup/minhash.py)
            MinhashDedupFilter(
                input_folder=OUTPUT_FOLDER_REMOVED_IDS,
                exclusion_writer=JsonlWriter(OUTPUT_FOLDER_DUPLICATED_SAMPLES, compression=None, expand_metadata=args.expand_metadata),
            ),

            # [TokensCounter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/tokens/counter.py#L7)
            TokensCounter(tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH),

            # [writers: JsonlWriter, ParquetWriter, HuggingFaceDatasetWriter](https://github.com/huggingface/datatrove/tree/main/src/datatrove/pipeline/writers)
            # [JsonlWriter](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/writers/jsonl.py)
            JsonlWriter(output_folder=OUTPUT_DEDUPLICATION_FINAL, compression=None, expand_metadata=args.expand_metadata),
        ],
        tasks=TASKS,
        workers=WORKERS,
        logging_dir=LOGS_FOLDER + "/minhash_filtering",
        depends=stage3,
    )

    print(f"[INFO] Running deduplication pipeline for: '{LANGUAGE}'")
    stage4.run()
    print(f"[INFO] Deduplication completed successfully. Output saved to: {OUTPUT_DEDUPLICATION_FINAL}")

    # Post-processing: Calculate and save metadata
    print(f"\n{'='*80}")
    print(f"[POST-PROCESSING] {LANGUAGE.upper()}")
    print(f"{'='*80}")

    # Get all JSONL files in the output folder
    all_files = glob.glob(f"{OUTPUT_DEDUPLICATION_FINAL}/*.jsonl")
    
    if not all_files:
        print(f"⚠️  No JSONL files found in {OUTPUT_DEDUPLICATION_FINAL}")
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
    metadata_file = os.path.join(OUTPUT_DEDUPLICATION_FINAL, '.metadata')
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
    parser = argparse.ArgumentParser(description="Data processing pipeline using DataTrove")

    parser.add_argument("--tasks", type=int, default=32, help="Number of tasks")
    parser.add_argument("--workers", type=int, default=32, help="Number of workers")
    parser.add_argument("--cache_dir", type=str, default="./.cache", help="Cache directory for datasets library")
    parser.add_argument("--data_folder", type=str, default="./dataset", help="Path to the data folder")
    parser.add_argument("--logs_folder", type=str, default="./logs", help="Path to the logs folder")
    parser.add_argument("--expand_metadata", action="store_true", help="Expand metadata")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="Qwen/Qwen3-0.6B", help="Tokenizer name or path")
    parser.add_argument("--output_minhash_signatures", type=str, default="./minhash_signatures", help="Output folder for Minhash Signatures")
    parser.add_argument("--output_minhash_bucket", type=str, default="./minhash_bucket", help="Output folder for Minhash Buckets")
    parser.add_argument("--output_removed_ids", type=str, default="./removed_ids", help="Output folder for removed IDs")
    parser.add_argument("--output_duplicated_samples", type=str, default="./duplicated_samples", help="Output folder for duplicated samples")
    parser.add_argument("--output_deduplication_final", type=str, default="./deduplication_final", help="Output folder for deduplication final")
    parser.add_argument("--language", type=str, required=True, help=f"Language code. Currently supported: pt, bn, hi")

    args = parser.parse_args()
    main(args)
