def write_metadata(metadata_file: str, metadata: dict) -> None:
    """
    Write metadata to a file in YAML-like key: value format.

    Args:
        metadata_file: Path to the metadata file to write.
        metadata: Dictionary of metadata values to persist.
    """
    with open(metadata_file, "w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
