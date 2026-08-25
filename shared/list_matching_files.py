import glob
import os


def list_matching_files(directory: str, *patterns: str) -> list[str]:
    """Return a sorted list of files in `directory` matching any glob pattern."""
    matches: set[str] = set()
    for pattern in patterns:
        matches.update(glob.glob(os.path.join(directory, pattern)))
    return sorted(matches)
