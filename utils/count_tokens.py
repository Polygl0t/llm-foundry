"""
Token Statistics Aggregation and Reporting Tool

This script scans directories containing tokenized datasets, aggregates statistics from
.metadata files, and generates comprehensive hierarchical reports with token counts,
sample counts, and other metrics across the entire dataset structure.

Usage:
    python count_tokens.py \\
        --main-dir data/tokenized \\
        --output-file dataset_report.txt
"""

import argparse
import os


def main(main_dir, output_file):
    report_lines = []
    report_lines.append("=" * 140)

    metadata_fields = [
        "Samples",
        "Tokens",
        "Tokens per chunk",
        "Block size",
        "Chunks",
        "Strategy",
        "Packed columns",
        "Tokenizer",
    ]

    key_aliases = {
        "samples": "Samples",
        "tokens": "Tokens",
        "tokens per chunk": "Tokens per chunk",
        "tokens_per_chunk": "Tokens per chunk",
        "block size": "Block size",
        "block_size": "Block size",
        "chunks": "Chunks",
        "strategy": "Strategy",
        "packed_columns": "Packed columns",
        "tokenizer": "Tokenizer",
    }

    folder_metadata = {}

    for root, _dirs, files in os.walk(main_dir):
        for file in files:
            if file.endswith(".metadata"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, main_dir)
                parts = [p for p in rel_path.split(os.sep) if p and p != "."]
                if not parts:
                    subfolder = os.path.basename(os.path.abspath(main_dir))
                    subsubfolder = None
                elif len(parts) == 1:
                    subfolder = parts[0]
                    subsubfolder = None
                else:
                    subfolder = parts[0]
                    subsubfolder = parts[1]

                metadata = {}
                with open(file_path, encoding="utf-8") as f:
                    for line in f:
                        if ":" not in line:
                            continue
                        raw_key, raw_value = line.split(":", 1)
                        key = raw_key.strip().lower()
                        field = key_aliases.get(key)
                        if field is None:
                            continue
                        value = raw_value.strip()
                        try:
                            metadata[field] = int(value)
                        except ValueError:
                            metadata[field] = value

                if subfolder not in folder_metadata:
                    folder_metadata[subfolder] = {}
                key = subsubfolder if subsubfolder else subfolder

                if key not in folder_metadata[subfolder]:
                    folder_metadata[subfolder][key] = {}

                for field, value in metadata.items():
                    if isinstance(value, int):
                        folder_metadata[subfolder][key][field] = (
                            folder_metadata[subfolder][key].get(field, 0) + value
                        )
                    else:
                        folder_metadata[subfolder][key][field] = value

    def format_value(value):
        if isinstance(value, int):
            return f"{value:,}"
        return value

    total_tokens = 0
    total_samples = 0
    total_chunks = 0

    # Define column widths
    col_widths = {
        "Subfolder": 35,
        "Samples": 15,
        "Tokens": 20,
        "Tokens per chunk": 20,
        "Block size": 12,
        "Chunks": 10,
        "Strategy": 14,
        "Packed columns": 30,
        "Tokenizer": 50,
    }

    header = f"{'Subfolder':<{col_widths['Subfolder']}}"
    for field in metadata_fields:
        header += f" | {field:^{col_widths[field]}}"
    report_lines.append(header)
    report_lines.append("-" * 140)

    for subfolder in sorted(folder_metadata):
        report_lines.append(f"\n{'=' * 30} {subfolder.upper()} {'=' * 30}")

        header = f"{'Subfolder':<{col_widths['Subfolder']}}"
        for field in metadata_fields:
            header += f" | {field:^{col_widths[field]}}"
        report_lines.append(header)
        report_lines.append("-" * 140)

        subfolder_total_tokens = 0
        subfolder_total_samples = 0
        subfolder_total_chunks = 0

        for subsubfolder in sorted(folder_metadata[subfolder]):
            metadata = folder_metadata[subfolder][subsubfolder]

            row = f"{subsubfolder:<{col_widths['Subfolder']}}"
            for field in metadata_fields:
                value = metadata.get(field, "N/A")
                formatted_value = format_value(value)
                # Truncate long string fields with ellipsis.
                if (
                    field in ("Tokenizer", "Packed columns")
                    and len(formatted_value) > col_widths[field]
                ):
                    formatted_value = "..." + formatted_value[-(col_widths[field] - 3) :]
                elif field == "Strategy" and len(formatted_value) > col_widths[field]:
                    formatted_value = formatted_value[: col_widths[field]]
                row += f" | {formatted_value:>{col_widths[field]}}"
            report_lines.append(row)

            subfolder_total_tokens += metadata.get("Tokens", 0)
            subfolder_total_samples += metadata.get("Samples", 0)
            subfolder_total_chunks += metadata.get("Chunks", 0)

        report_lines.append("-" * 140)
        total_row = f"{'TOTAL':<{col_widths['Subfolder']}}"
        for field in metadata_fields:
            if field == "Samples":
                total_row += f" | {format_value(subfolder_total_samples):>{col_widths[field]}}"
            elif field == "Tokens":
                total_row += f" | {format_value(subfolder_total_tokens):>{col_widths[field]}}"
            elif field == "Chunks":
                total_row += f" | {format_value(subfolder_total_chunks):>{col_widths[field]}}"
            else:
                total_row += f" | {'':>{col_widths[field]}}"
        report_lines.append(total_row)
        report_lines.append("=" * 140)

        total_tokens += subfolder_total_tokens
        total_samples += subfolder_total_samples
        total_chunks += subfolder_total_chunks

    report_lines.append(f"\n{'GRAND TOTALS':<{col_widths['Subfolder']}}")
    grand_totals_row = f"{'':>{col_widths['Subfolder']}}"
    for field in metadata_fields:
        if field == "Samples":
            grand_totals_row += f" | {format_value(total_samples):>{col_widths[field]}}"
        elif field == "Tokens":
            grand_totals_row += f" | {format_value(total_tokens):>{col_widths[field]}}"
        elif field == "Chunks":
            grand_totals_row += f" | {format_value(total_chunks):>{col_widths[field]}}"
        else:
            grand_totals_row += f" | {'':>{col_widths[field]}}"
    report_lines.append(grand_totals_row)
    report_lines.append("=" * 140)

    output_file_path = os.path.join(main_dir, output_file)
    with open(output_file_path, "w", encoding="utf-8") as report_file:
        for line in report_lines:
            report_file.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--main-dir", type=str, help="Path to the main directory containing tokenized datasets."
    )
    parser.add_argument(
        "--output-file", type=str, default="report.txt", help="Output report file name."
    )
    args = parser.parse_args()

    main(args.main_dir, output_file=args.output_file)
