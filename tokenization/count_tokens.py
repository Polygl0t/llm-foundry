"""
Token Statistics Aggregation and Reporting Tool

This script scans directories containing tokenized datasets, aggregates statistics from
.metadata files, and generates comprehensive hierarchical reports with token counts,
sample counts, and other metrics across the entire dataset structure.

Example usage:
    python count_tokens.py \
        --main-dir data/tokenized \
        --output-file dataset_report.txt
"""
import os
import argparse

def main(main_dir, output_file):
    report_lines = []
    report_lines.append("=" * 140)

    metadata_fields = [
        "Samples", "Tokens", "Tokens per chunk", "Block size", "Chunks", "Tokenizer"
    ]

    folder_metadata = {}

    for root, dirs, files in os.walk(main_dir):
        for file in files:
            if file.endswith('.metadata'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, main_dir)
                parts = rel_path.split(os.sep)
                if len(parts) == 1:
                    subfolder = parts[0]
                    subsubfolder = None
                else:
                    subfolder = parts[0]
                    subsubfolder = parts[1]
                
                metadata = {}
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        for field in metadata_fields:
                            if line.startswith(f"{field}:"):
                                value = line.split(":", 1)[1].strip()

                                try:
                                    metadata[field] = int(value)
                                except ValueError:
                                    metadata[field] = value
                
                if subfolder not in folder_metadata:
                    folder_metadata[subfolder] = {}
                key = subsubfolder if subsubfolder else subfolder
                
                if key not in folder_metadata[subfolder]:
                    folder_metadata[subfolder][key] = {}
                    for field in metadata_fields:
                        folder_metadata[subfolder][key][field] = 0
                
                for field, value in metadata.items():
                    if isinstance(value, int):
                        folder_metadata[subfolder][key][field] += value
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
        "Tokenizer": 50
    }

    header = f"{'Subfolder':<{col_widths['Subfolder']}}"
    for field in metadata_fields:
        header += f" | {field:^{col_widths[field]}}"
    report_lines.append(header)
    report_lines.append("-" * 140)

    for subfolder in sorted(folder_metadata):
        report_lines.append(f"\n{'='*30} {subfolder.upper()} {'='*30}")
        
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
                if field == "Tokenizer" and len(formatted_value) > col_widths[field]:

                    formatted_value = "..." + formatted_value[-(col_widths[field]-3):]
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
    parser = argparse.ArgumentParser(description="Count tokens in tokenized datasets and generate a report.")
    parser.add_argument("--main-dir", type=str, help="Path to the main directory containing tokenized datasets.")
    parser.add_argument("--output-file", type=str, default="report.txt", help="Output report file name.")
    args = parser.parse_args()
    
    print("🚀")
    main(args.main_dir, output_file=args.output_file)
    print("🎉")

