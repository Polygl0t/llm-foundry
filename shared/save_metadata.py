import os


def save_metadata(output_dir, **kwargs):
    """Write key-value metadata to `<output_dir>/.metadata`."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, ".metadata"), "w") as f:
        for key, value in kwargs.items():
            f.write(f"{key}: {value}\n")
