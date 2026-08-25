import os


def read_metadata(metadata_file: str) -> dict | None:
    """
    Read metadata from a file in YAML-like key: value format.

    Args:
        metadata_file: Path to the metadata file.

    Returns:
        Dictionary with metadata values, or None if the file does not exist.
    """
    if not os.path.exists(metadata_file):
        return None

    metadata = {}
    with open(metadata_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                try:
                    if "." in value:
                        metadata[key] = float(value)
                    else:
                        metadata[key] = int(value)
                except ValueError:
                    metadata[key] = value
    return metadata
