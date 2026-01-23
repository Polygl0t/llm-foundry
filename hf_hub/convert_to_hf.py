"""
Convert JSONL/Parquet Files to Hugging Face Dataset Format

Usage:
    python convert_to_hf.py \
        --directory_path ./raw_data \
        --dataset_type jsonl \
        --output_path ./processed_data \
        --default_dataset_name default \
        --num_workers 16
"""
import time
import os
import argparse
import glob

import numpy as np
import datasets
from huggingface_hub import HfApi
from tqdm import tqdm


def convert_single_file(input_file, output_dir, file_start_idx, dataset_type, cache_dir, 
                        max_chunk_size_gb=5.0, compression_ratio=0.25):
    """
    Convert a single JSONL/Parquet file to Parquet format using datasets library.
    
    Chunking is determined by max_chunk_size_gb to limit output file sizes.
    
    Args:
        input_file: Path to input file
        output_dir: Directory to write output files
        file_start_idx: Starting index for output file naming
        dataset_type: 'jsonl' or 'parquet'
        cache_dir: Cache directory for datasets
        max_chunk_size_gb: Max size per output file in GB (default 5GB)
        compression_ratio: Expected compression from JSONL to Parquet (default 0.25)
    
    Returns a list of result dicts (one per output chunk).
    """
    
    try:
        # Use the datasets library - same as original, but one file at a time
        loader_type = "json" if dataset_type == "jsonl" else "parquet"
        
        ds = datasets.load_dataset(
            loader_type,
            data_files=input_file,
            split="train",
            cache_dir=cache_dir,
        )
        
        # Calculate chunks needed based on file size
        input_size_bytes = os.path.getsize(input_file)
        if dataset_type == "jsonl":
            estimated_output_size = input_size_bytes * compression_ratio
        else:
            # Parquet to parquet - size stays roughly the same
            estimated_output_size = input_size_bytes
        
        max_chunk_bytes = max_chunk_size_gb * 1024**3
        n_chunks = max(1, int(np.ceil(estimated_output_size / max_chunk_bytes)))
        
        # Get schema/features info (same for all chunks)
        features_info = {}
        for name, feature in ds.features.items():
            if isinstance(feature, datasets.features.Sequence):
                features_info[name] = {"type": "sequence", "feature": str(feature.feature)}
            else:
                features_info[name] = {"type": "value", "dtype": str(feature.dtype)}
        
        results = []
        
        if n_chunks == 1:
            # No chunking needed - write single file
            output_file = os.path.join(output_dir, f"train-{file_start_idx:05d}.parquet")
            ds.to_parquet(output_file)
            
            token_count = 0
            if "token_count" in ds.column_names:
                token_count = sum(ds["token_count"])
            
            results.append({
                'success': True,
                'input_file': input_file,
                'output_file': output_file,
                'num_rows': len(ds),
                'size_bytes': os.path.getsize(output_file),
                'token_count': token_count,
                'features': features_info,
            })
        else:
            # Split into chunks
            indices = np.array_split(np.arange(len(ds)), n_chunks)
            
            for chunk_idx, chunk_indices in enumerate(indices):
                if len(chunk_indices) == 0:
                    continue
                    
                chunk_ds = ds.select(chunk_indices.tolist())
                output_file = os.path.join(output_dir, f"train-{file_start_idx + chunk_idx:05d}.parquet")
                chunk_ds.to_parquet(output_file)
                
                token_count = 0
                if "token_count" in chunk_ds.column_names:
                    token_count = sum(chunk_ds["token_count"])
                
                results.append({
                    'success': True,
                    'input_file': input_file,
                    'output_file': output_file,
                    'num_rows': len(chunk_ds),
                    'size_bytes': os.path.getsize(output_file),
                    'token_count': token_count,
                    'features': features_info,
                })
                
                del chunk_ds
        
        # Clean up cache for this file to free memory
        ds.cleanup_cache_files()
        del ds
        
        return results
        
    except Exception as e:
        return [{
            'success': False,
            'input_file': input_file,
            'error': str(e),
        }]


def process_folder(folder_name, folder_path, output_path, dataset_type, cache_dir, 
                   max_chunk_size_gb=5.0, compression_ratio=0.25):
    """Process a single folder containing data files, one file at a time."""
    print(f"\n{'='*60}")
    print(f"Processing folder: {folder_name}")
    print(f"{'='*60}")
    
    # Find all data files
    pattern = f"*.{dataset_type}"
    data_files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    
    if not data_files:
        print(f"No {dataset_type} files found in {folder_path}")
        return None
    
    print(f"Found {len(data_files)} {dataset_type.upper()} files")
    print(f"Max output file size: {max_chunk_size_gb:.1f} GB")
    
    # Create output directory
    output_dir = os.path.join(output_path, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process files sequentially (one at a time to control memory)
    total_rows = 0
    total_size = 0
    total_tokens = 0
    total_output_files = 0
    features_info = None
    failed_files = []
    
    # Track the next output file index
    next_file_idx = 0
    
    for file_num, input_file in enumerate(data_files, 1):
        input_basename = os.path.basename(input_file)
        input_size_gb = os.path.getsize(input_file) / 1024**3
        tqdm.write(f"\n[{file_num}/{len(data_files)}] Processing: {input_basename} ({input_size_gb:.2f} GB)")
        
        results = convert_single_file(
            input_file, 
            output_dir, 
            next_file_idx, 
            dataset_type, 
            cache_dir,
            max_chunk_size_gb,
            compression_ratio,
        )
        
        for result in results:
            if result['success']:
                total_rows += result['num_rows']
                total_size += result['size_bytes']
                total_tokens += result.get('token_count', 0)
                total_output_files += 1
                next_file_idx += 1
                if features_info is None:
                    features_info = result['features']
            else:
                failed_files.append(result)
                tqdm.write(f"  ❌ Failed: {result['error']}")
        
        if results and results[0]['success']:
            chunks_created = len(results)
            output_size_gb = sum(r['size_bytes'] for r in results) / 1024**3
            tqdm.write(f"  ✓ Done: {chunks_created} chunk(s), {output_size_gb:.2f} GB output, {len(data_files) - file_num} files remaining")
    
    # Rename files to include total count (train-00000-of-00XXX.parquet)
    if total_output_files > 0:
        print(f"\nRenaming {total_output_files} output files...")
        for i in range(total_output_files):
            old_name = os.path.join(output_dir, f"train-{i:05d}.parquet")
            new_name = os.path.join(output_dir, f"train-{i:05d}-of-{total_output_files:05d}.parquet")
            if os.path.exists(old_name):
                os.rename(old_name, new_name)
    
    if failed_files:
        print(f"\n⚠️  {len(failed_files)} files failed to convert")
        for f in failed_files:
            print(f"  - {f['input_file']}: {f['error']}")
    
    print(f"\n✅ Folder {folder_name} completed:")
    print(f"   Input files: {len(data_files)}")
    print(f"   Output files: {total_output_files}")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Total size: {total_size / 1024**3:.2f} GB")
    if total_tokens > 0:
        print(f"   Total tokens: {total_tokens:,}")
    
    return {
        'folder_name': folder_name,
        'num_files': total_output_files,
        'num_rows': total_rows,
        'size_bytes': total_size,
        'token_count': total_tokens,
        'features': features_info,
        'failed_files': len(failed_files),
    }


def generate_yaml_config(folder_stats, output_path, default_dataset_name):
    """Generate the README.md with YAML config from collected stats."""
    
    yaml_config = "---\ndataset_info:\n"
    
    for stats in folder_stats:
        if stats is None:
            continue
            
        folder_name = stats['folder_name']
        yaml_config += f"- config_name: {folder_name}\n  features:\n"
        
        # Add features from schema
        if stats['features']:
            for feature_name, info in stats['features'].items():
                if info['type'] == 'sequence':
                    yaml_config += f"  - name: {feature_name}\n    sequence: {info['feature']}\n"
                else:
                    yaml_config += f"  - name: {feature_name}\n    dtype: {info['dtype']}\n"
        
        yaml_config += f"  splits:\n  - name: train\n    num_bytes: {stats['size_bytes']}\n    num_examples: {stats['num_rows']}\n"
        yaml_config += f"  download_size: {stats['size_bytes']}\n  dataset_size: {stats['size_bytes']}\n"
    
    # Add configs section
    yaml_config += "configs:\n"
    for stats in folder_stats:
        if stats is None:
            continue
        folder_name = stats['folder_name']
        yaml_config += f"- config_name: {folder_name}\n"
        if folder_name == default_dataset_name:
            yaml_config += f"  default: true\n"
        yaml_config += f"  data_files:\n  - split: train\n    path: {folder_name}/train-*\n"
    
    yaml_config += "---\n\n# Dataset Card\n\n"
    
    # Add token count information
    token_counts = {s['folder_name']: s['token_count'] for s in folder_stats if s and s.get('token_count', 0) > 0}
    if token_counts:
        yaml_config += "\n## Token Counts per Subset\n\n"
        total_tokens = 0
        for folder_name, token_count in token_counts.items():
            yaml_config += f"- **{folder_name}**: {token_count:,} tokens\n"
            total_tokens += token_count
        yaml_config += f"\n**Total tokens across all subsets**: {total_tokens:,} tokens\n"
    
    # Add statistics
    yaml_config += "\n## Dataset Statistics\n\n"
    yaml_config += "| Subset | Files | Rows | Size |\n"
    yaml_config += "|--------|-------|------|------|\n"
    total_files = 0
    total_rows = 0
    total_size_bytes = 0
    for stats in folder_stats:
        if stats is None:
            continue
        size_gb = stats['size_bytes'] / 1024**3
        yaml_config += f"| {stats['folder_name']} | {stats['num_files']} | {stats['num_rows']:,} | {size_gb:.2f} GB |\n"
        total_files += stats['num_files']
        total_rows += stats['num_rows']
        total_size_bytes += stats['size_bytes']
    # Add total row
    total_size_gb = total_size_bytes / 1024**3
    yaml_config += f"| **Total** | **{total_files}** | **{total_rows:,}** | **{total_size_gb:.2f} GB** |\n"
    
    # Write README.md
    readme_path = os.path.join(output_path, "README.md")
    with open(readme_path, "w") as f:
        f.write(yaml_config)
    
    print(f"\n📝 Generated README.md at {readme_path}")


def main(args):
    start_time = time.time()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Find all folders to process
    folder_names = sorted(os.listdir(args.directory_path))
    folder_paths = [os.path.join(args.directory_path, name) for name in folder_names]
    folder_paths = [(name, path) for name, path in zip(folder_names, folder_paths) if os.path.isdir(path)]
    
    print(f"Found {len(folder_paths)} folders to process")
    for name, path in folder_paths:
        print(f"  - {name}")
    
    # Process each folder
    all_stats = []
    for folder_name, folder_path in folder_paths:
        stats = process_folder(
            folder_name=folder_name,
            folder_path=folder_path,
            output_path=args.output_path,
            dataset_type=args.dataset_type,
            cache_dir=args.cache_dir,
            max_chunk_size_gb=args.max_chunk_size_gb,
            compression_ratio=args.compression_ratio,
        )
        all_stats.append(stats)
    
    # Generate YAML config from collected stats (no second pass needed!)
    generate_yaml_config(all_stats, args.output_path, args.default_dataset_name)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Conversion completed in {elapsed/3600:.2f} hours")
    print(f"{'='*60}")
    
    # Upload if requested
    if args.new_repo_id and args.token:
        print(f"\n📤 Uploading to Hugging Face: {args.new_repo_id}")
        
        api = HfApi(token=args.token)
        
        api.create_repo(
            repo_id=args.new_repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        
        upload_success = False
        max_retries = 5
        for attempt in range(max_retries):
            try:
                api.upload_large_folder(
                    folder_path=args.output_path,
                    repo_id=args.new_repo_id,
                    repo_type="dataset",
                    num_workers=args.num_workers,
                )
                upload_success = True
                break
            except Exception as e:
                print(f"Upload attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print("Retrying in 60 seconds...")
                    time.sleep(60)
        
        if upload_success:
            print(f"✅ Uploaded dataset to: {args.new_repo_id}")
        else:
            print(f"❌ Failed to upload after {max_retries} attempts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converter for large JSONL/Parquet datasets to HuggingFace format"
    )
    parser.add_argument("--directory_path", type=str, required=True, help="Path to the directory containing data folders")
    parser.add_argument("--dataset_type", type=str, choices=['jsonl', 'parquet'], default='jsonl', help="Type of source files (jsonl or parquet)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the converted dataset")
    parser.add_argument("--cache_dir", type=str, default="./.cache", help="Path to cache directory for datasets")
    parser.add_argument("--max_chunk_size_gb", type=float, default=5.0, help="Maximum output file size in GB. Default: 5.0")
    parser.add_argument("--compression_ratio", type=float, default=0.25, help="Expected JSONL to Parquet compression ratio. Default: 0.25")
    parser.add_argument("--default_dataset_name", type=str, default="default", help="Default config name for the dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers for upload")
    parser.add_argument("--new_repo_id", type=str, default=None, help="HuggingFace repository ID for upload")
    parser.add_argument("--private", action='store_true', help="Create private repository")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token")
    
    args = parser.parse_args()
    
    print("🚀 Starting conversion process!")
    print(f"   Source: {args.directory_path}")
    print(f"   Output: {args.output_path}")
    print(f"   Type: {args.dataset_type}")
    print(f"   Cache: {args.cache_dir}")
    print(f"   Max chunk size: {args.max_chunk_size_gb:.1f} GB")
    print(f"   Compression ratio: {args.compression_ratio}")
    
    main(args)
    
    print("\n🎉 Done!")
