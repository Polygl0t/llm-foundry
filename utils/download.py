"""
Download Repository from Hugging Face Hub via Snapshot Download

This script downloads complete repositories (datasets, models, or spaces) from the
Hugging Face Hub to a local directory using the efficient snapshot_download method.

Usage:
    python download.py \\
        --repo_name username/repo-name \\
        --output_dir /path/to/output \\
        --token YOUR_HF_TOKEN \\
        --repo_type dataset \\
        --cache_dir ./.cache \\
        --allow_patterns "*"
"""
import os
import argparse
from huggingface_hub import snapshot_download


def main(args):
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download the repository using snapshot_download
    # See https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download#huggingface_hub.snapshot_download
    snapshot_download(
        repo_id=args.repo_name,
        repo_type=args.repo_type,
        cache_dir=args.cache_dir,
        token=args.token,
        local_dir=args.output_dir,
        allow_patterns=args.allow_patterns
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--repo_name", help="Name of the repository to download (e.g., 'Polygl0t/bengali-edu-qwen-annotations')")
    parser.add_argument("--output_dir", help="Directory where the repository will be downloaded")
    parser.add_argument("--cache_dir", default="./.cache", 
                       help="Cache directory for Hugging Face Hub")
    parser.add_argument("--token", required=True, help="Hugging Face token for authentication")
    parser.add_argument("--repo_type", default="dataset", choices=["dataset", "model", "space"],
                       help="Type of repository to download")
    parser.add_argument("--allow_patterns", nargs="+", default=["*"],
                       help="Optional glob patterns to filter files to download (e.g., 'de/*' '*.md')")
    
    args = parser.parse_args()

    main(args)


