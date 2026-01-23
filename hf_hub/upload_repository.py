"""
Upload Repository to Hugging Face Hub with Automatic Retry Logic

This script uploads a local directory to the Hugging Face Hub as either a dataset
or model repository, with automatic retry on failure.

Usage:
    python upload_repository.py \
        --main_dir /path/to/local/folder \
        --new_repo_id username/repo-name \
        --token YOUR_HF_TOKEN \
        --repo_type dataset \
        --num_workers 8 \
        --private
"""
from huggingface_hub import HfApi
import argparse
import time

def main(args):

    api = HfApi(token=args.token)

    api.create_repo(
        repo_id=args.new_repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )

    upload_flag = False
    while not upload_flag:
        try:
            api.upload_large_folder(
                folder_path=args.main_dir,
                repo_id=args.new_repo_id,
                repo_type=args.repo_type,
                num_workers=args.num_workers,
            )
            upload_flag = True
        except Exception as e:
            print(f"Error uploading dataset: {e}")
            print("Retrying in 60 seconds...")
            time.sleep(60)

    print(f"Repository uploaded to the hub at {args.new_repo_id}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a repository to the Hugging Face Hub")
    parser.add_argument("--main_dir", type=str, default=None, help="Directory to upload")
    parser.add_argument("--new_repo_id", type=str, default=None, help="New repository ID on the Hugging Face Hub")
    parser.add_argument("--private", action='store_true', help="Make the repository private")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for uploading")
    parser.add_argument("--repo_type", type=str, default="dataset", help="Type of repository (dataset/model)")

    args = parser.parse_args()
    main(args)
