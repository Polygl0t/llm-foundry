"""
Launch a Trackio dashboard for a given project, either locally or as a
Hugging Face Space.

Usage:
  # Launch the dashboard locally
  python launch_trackio.py --project  MyProjectName

  # Sync to a public Hugging Face Space (Gradio SDK, live server)
  python launch_trackio.py --project  MyProjectName --space-id myname/my-space --token hf_...

  # Sync to a private Hugging Face Space
  python launch_trackio.py --project  MyProjectName --db-dir ./MyProjectName --space-id myname/my-space --private --token hf_...

  # Sync to a static (read-only, browser-only) Space (data must be public)
  python launch_trackio.py --project  MyProjectName --sdk static --space-id myname/my-static-space --token hf_...

  # Specify a custom project database folder instead of the default location
  python launch_trackio.py --project  MyProjectName --db-dir ./MyProjectName
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def _resolve_hf_token(token: str | None) -> str | None:
    """
    Resolve the Hugging Face token, trying (in order):
    1. Explicit --token argument
    2. HF_TOKEN environment variable
    3. huggingface-cli login cache

    Sets HF_TOKEN in the environment so trackio picks it up automatically.
    Returns the resolved token, or None if not authenticated.
    """
    if token:
        os.environ["HF_TOKEN"] = token
        return token

    # Let huggingface_hub resolve it (env var or login cache)
    try:
        import huggingface_hub

        resolved = huggingface_hub.utils.get_token()
        if resolved:
            os.environ["HF_TOKEN"] = resolved
        return resolved
    except ImportError:
        return os.environ.get("HF_TOKEN")


# Default directory where trackio stores its SQLite databases.
DEFAULT_CACHE = Path.home() / ".cache" / "huggingface" / "trackio"


def _ensure_project_in_cache(project: str, db_dir: str | None = None) -> None:
    """
    If *db_dir* is provided, symlink (or copy) its .db file into the default
    trackio cache so that ``trackio.show(project=...)`` can find it.
    """
    if db_dir is None:
        return  # assume the db is already in the default cache

    src = Path(db_dir)
    db_file = src / f"{project}.db"
    lock_file = src / f"{project}.lock"

    if not db_file.exists():
        print(f"⚠️  Database file not found: {db_file}")
        sys.exit(1)

    DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
    dst_db = DEFAULT_CACHE / f"{project}.db"
    dst_lock = DEFAULT_CACHE / f"{project}.lock"

    if not dst_db.exists() or dst_db.resolve() != db_file.resolve():
        try:
            dst_db.symlink_to(db_file.resolve())
            print(f"🔗 Linked {db_file} -> {dst_db}")
        except OSError:
            shutil.copy2(db_file, dst_db)
            print(f"📋 Copied {db_file} -> {dst_db}")

    if lock_file.exists() and not dst_lock.exists():
        try:
            dst_lock.symlink_to(lock_file.resolve())
        except OSError:
            shutil.copy2(lock_file, dst_lock)


def launch_local(project: str) -> None:
    """Start the Trackio dashboard on the local machine."""
    import trackio

    print(f"🚀 Launching Trackio dashboard locally for project '{project}' ...")
    trackio.show(project=project)


def launch_space(
    project: str,
    space_id: str | None,
    private: bool,
    sdk: str,
    force: bool,
    token: str | None = None,
) -> None:
    """Sync the project database to a Hugging Face Space and print the URL."""
    import trackio

    # Resolve token before proceeding
    hf_token = _resolve_hf_token(token)
    if hf_token is None:
        print(
            "❌ Not authenticated with Hugging Face Hub.\n"
            "   Run 'huggingface-cli login' or pass --token <your-token>.\n"
            "   Get a token at: https://huggingface.co/settings/tokens"
        )
        sys.exit(1)

    print(f"🔑 Authenticated (token: {hf_token[:6]}...{hf_token[-4:]})")

    if space_id is None:
        print(
            "⚠️ No --space-id provided. A random Space name will be generated.\n"
            "   If you want to reuse an existing Space, pass --space-id <user/name>."
        )

    print(f"☁️  Syncing project '{project}' to Hugging Face Space …")
    if private:
        print("   🔒 Space will be private.")
    else:
        print("   🌐 Space will be public.")

    result_id = trackio.sync(
        project=project,
        space_id=space_id,
        private=private if private else None,
        force=force,
        sdk=sdk,
    )

    print("\n✅ Dashboard deployed!")
    print(f"   https://huggingface.co/spaces/{result_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a Trackio dashboard locally or on Hugging Face Spaces.",
    )

    parser.add_argument(
        "--project",
        required=True,
        help="Trackio project name (see the name of your .db file, e.g. 'MyProjectName' for MyProjectName.db).",
    )
    parser.add_argument(
        "--db-dir",
        default=None,
        help=(
            "Directory containing the .db / .lock files."
            " If omitted, the default trackio cache is used."
        ),
    )
    parser.add_argument(
        "--space-id",
        default=None,
        help="HF Space ID, e.g. 'username/my-space'. When provided, syncs to a Space instead of launching locally.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Make the Space private (Gradio SDK only).",
    )
    parser.add_argument(
        "--sdk",
        choices=["gradio", "static"],
        default="gradio",
        help=(
            "Space type: 'gradio' for a live server (supports private), "
            "'static' for a read-only browser-only Space (must be public). "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite an existing database in the Space without prompting.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Hugging Face token for authentication. "
            "If omitted, uses HF_TOKEN env var or huggingface-cli login cache."
        ),
    )

    args = parser.parse_args()

    _ensure_project_in_cache(args.project, args.db_dir)

    if args.space_id is not None:
        launch_space(
            project=args.project,
            space_id=args.space_id,
            private=args.private,
            sdk=args.sdk,
            force=args.force,
            token=args.token,
        )
    else:
        launch_local(args.project)


if __name__ == "__main__":
    main()
