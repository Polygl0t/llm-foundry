"""
Upload files or folders to a Hugging Face repository across branches.

Usage:
  # Upload individual files
  python upload_quick.py --repo Polygl0t/MyModel --files a.json b.yaml

  # Upload a folder
  python upload_quick.py --repo Polygl0t/MyModel --folder ./plots

  # Upload folder to a specific path inside the repo
  python upload_quick.py --repo Polygl0t/MyModel --folder ./plots --repo-folder plots

  # Push to main branch only
  python upload_quick.py --repo Polygl0t/MyModel --folder ./plots --main-only

  # Use a dataset repo
  python upload_quick.py --repo Polygl0t/MyDataset --folder ./data --repo-type dataset

Note: Token can also be set via the HF_TOKEN environment variable
"""

import argparse
import os

from huggingface_hub import HfApi, login


def _get_branches(api, repo_id, repo_type, main_only):
    branches = api.list_repo_refs(repo_id, repo_type=repo_type).branches
    names = [b.name for b in branches]
    if main_only:
        names = [n for n in names if n == "main"]
    print(f"Target branches: {names}")
    return names


def push_files(api, repo_id, files, repo_type, main_only=False):
    branches = _get_branches(api, repo_id, repo_type, main_only)
    for branch in branches:
        print(f"\nUploading to branch: {branch}")
        for file_path in files:
            if not os.path.exists(file_path):
                print(f"  ❌ File not found: {file_path}")
                continue
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=os.path.basename(file_path),
                    repo_id=repo_id,
                    revision=branch,
                    repo_type=repo_type,
                )
                print(f"  ✅ {file_path}")
            except Exception as e:
                print(f"  ❌ {file_path}: {e}")


def push_folder(api, repo_id, folder_path, repo_type, repo_folder=None, main_only=False):
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")

    branches = _get_branches(api, repo_id, repo_type, main_only)
    target = repo_folder or os.path.basename(os.path.normpath(folder_path))

    for branch in branches:
        print(f"\nUploading folder to branch: {branch}")
        try:
            api.upload_folder(
                folder_path=folder_path,
                path_in_repo=target,
                repo_id=repo_id,
                revision=branch,
                repo_type=repo_type,
            )
            print(f"  ✅ {folder_path} -> {target}")
        except Exception as e:
            print(f"  ❌ {folder_path} -> {target}: {e}")


def main(args):
    login(token=args.token)
    api = HfApi(token=args.token)

    if args.files:
        push_files(api, args.repo, args.files, args.repo_type, main_only=args.main_only)
    else:
        push_folder(
            api,
            args.repo,
            args.folder,
            args.repo_type,
            repo_folder=args.repo_folder,
            main_only=args.main_only,
        )

    print("\nDone 🎉🎉🎉")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--repo", required=True, help="Repository ID, e.g. Polygl0t/MyModel")
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Repository type (default: model)",
    )
    parser.add_argument(
        "--token", default=None, help="Hugging Face token. Falls back to HF_TOKEN env var."
    )
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="Upload to the main branch only instead of all branches",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--files", nargs="+", metavar="FILE", help="One or more file paths to upload"
    )
    source.add_argument("--folder", metavar="DIR", help="Local folder to upload")

    parser.add_argument(
        "--repo-folder",
        default=None,
        help="Destination path inside the repo when using --folder "
        "(defaults to the local folder name)",
    )

    args = parser.parse_args()

    if args.token is None:
        args.token = os.getenv("HF_TOKEN", None)
        if args.token is None:
            parser.error("Provide --token or set the HF_TOKEN environment variable.")

    main(args)
