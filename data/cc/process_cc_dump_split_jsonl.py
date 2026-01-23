"""
JSONL File Splitter by Token Count

Splits large JSONL files into manageable chunks based on token count thresholds.
Designed for post-processing CommonCrawl extractions or other large datasets.

Workflow:
- Scans directory for JSONL files exceeding size threshold (default: 1 GB)
- Splits files into chunks with max token count per chunk (default: 100M tokens)
- Reads 'token_count' field from each JSON object to track accumulation
- Generates unique chunk filenames with hash prefix to avoid collisions
- Automatically removes original file after successful chunking
- Skips files already ending with '-chunk-' to prevent re-splitting

Output naming:
- Original: language.jsonl (e.g., pt.jsonl - 5GB)
- Chunks: {hash}-chunk-{N}.jsonl (e.g., a3f8b2c1-chunk-0.jsonl)

Usage:
    # Split files over 1GB into 100M token chunks
    python process_cc_dump_split_jsonl.py --directory output/pt/
    
"""
import argparse
import os
import json
import glob
import uuid


def get_file_size_gb(file_path: str) -> float:
    """Get file size in GB."""
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 ** 3)


def split_jsonl_file(file_path: str, max_tokens_per_chunk: int = 100_000_000, size_threshold_gb: float = 1.0):
    """Split a large JSONL file into chunks based on token count."""
    
    file_size_gb = get_file_size_gb(file_path)
    
    if file_size_gb <= size_threshold_gb:
        return False
    
    directory = os.path.dirname(file_path)
    unique_hash = uuid.uuid4().hex[:8]
    
    chunk_idx = 0
    current_chunk_tokens = 0
    chunks_created = []
    
    chunk_filename = f"{unique_hash}-chunk-{chunk_idx}.jsonl"
    chunk_path = os.path.join(directory, chunk_filename)
    chunk_file = open(chunk_path, 'w', encoding='utf-8')
    chunks_created.append(chunk_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    token_count = data.get('token_count', 0)
                    
                    if current_chunk_tokens > 0 and current_chunk_tokens + token_count > max_tokens_per_chunk:
                        chunk_file.close()
                        
                        chunk_idx += 1
                        chunk_filename = f"{unique_hash}-chunk-{chunk_idx}.jsonl"
                        chunk_path = os.path.join(directory, chunk_filename)
                        chunk_file = open(chunk_path, 'w', encoding='utf-8')
                        chunks_created.append(chunk_path)
                        
                        current_chunk_tokens = 0
                    
                    chunk_file.write(line + '\n')
                    current_chunk_tokens += token_count
                
                except json.JSONDecodeError:
                    continue
        
        chunk_file.close()
        
    except Exception:
        chunk_file.close()
        for chunk_path in chunks_created:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
        return False
    
    return True


def main(args):
    
    if not os.path.isdir(args.directory):
        return
    
    pattern = os.path.join(args.directory, "*.jsonl")
    jsonl_files = glob.glob(pattern)
    
    if not jsonl_files:
        return
    
    for file_path in sorted(jsonl_files):
        if '-chunk-' in os.path.basename(file_path):
            continue
        
        was_split = split_jsonl_file(
            file_path,
            max_tokens_per_chunk=args.max_tokens_per_chunk,
            size_threshold_gb=args.size_threshold_gb
        )
        
        if was_split:
            try:
                os.remove(file_path)
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split large JSONL files into smaller chunks based on token count"
    )
    
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Directory containing JSONL files to process",
    )
    parser.add_argument(
        "--max_tokens_per_chunk",
        type=int,
        default=100_000_000,
        help="Maximum number of tokens per chunk (default: 100,000,000)",
    )
    parser.add_argument(
        "--size_threshold_gb",
        type=float,
        default=1.0,
        help="Only split files larger than this size in GB (default: 1.0)",
    )
    
    args = parser.parse_args()
    main(args)
