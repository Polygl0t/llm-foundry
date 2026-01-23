"""
Download Repository from Hugging Face Hub via Snapshot Download

This script downloads complete repositories (datasets, models, or spaces) from the
Hugging Face Hub to a local directory using the efficient snapshot_download method.

Usage:
    python download_repository.py \
        --repo_name username/repo-name \
        --output_dir /path/to/output \
        --token YOUR_HF_TOKEN \
        --repo_type dataset \
        --cache_dir ./.cache
"""
from huggingface_hub import snapshot_download
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download a repository from Hugging Face Hub")
    parser.add_argument("--repo_name", help="Name of the repository to download (e.g., 'Polygl0t/bengali-edu-qwen-annotations')")
    parser.add_argument("--output_dir", help="Directory where the repository will be downloaded")
    parser.add_argument("--cache_dir", default="./.cache", 
                       help="Cache directory for Hugging Face Hub")
    parser.add_argument("--token", required=True, help="Hugging Face token for authentication")
    parser.add_argument("--repo_type", default="dataset", choices=["dataset", "model", "space"],
                       help="Type of repository to download")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    snapshot_download(
        repo_id=args.repo_name,
        repo_type=args.repo_type,
        cache_dir=args.cache_dir,
        token=args.token,
        local_dir_use_symlinks=False,
        local_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()


