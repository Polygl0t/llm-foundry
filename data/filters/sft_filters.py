"""
SFT Dataset Filter

A comprehensive filtering and conversion tool for instruction-tuning (SFT) datasets.
Designed to clean, validate, and transform instruction-following datasets by removing
malformed code blocks, corrupted code content, undecoded Unicode sequences, and other
crap that passed trough the quality filter.

Expected Dataset Format:
The script expects datasets with a 'messages' column containing conversation data:
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "token_count": 123  # Optional, enables token-based features
    }

Usage Examples:
    python sft_filter.py \
        --input_dir ./raw_data \
        --output_dir ./filtered_data \
        --input_type parquet \
        --output_type parquet \
        --filter_incomplete_sentences \
        --filter_malformed_code_blocks \
        --filter_corrupted_code \
        --filter_undecoded_sequences \
        --filter_invalid_markers \
        --remove_system_messages \
        --max_tokens_per_chunk 100000000
"""
import datasets
import numpy as np
import argparse
import glob
import os
import re
import matplotlib.pyplot as plt


# =============================================================================
# CONSTANTS & PATTERNS
# =============================================================================

# Valid programming language tags for code block validation.
# Used to detect malformed code blocks with invalid or corrupted language identifiers.
# Includes common programming languages, markup languages, shell scripting, and config formats.
VALID_LANGUAGE_TAGS = {
    # Common languages
    'python', 'py', 'python3', 'py3',
    'javascript', 'js', 'node', 'nodejs',
    'typescript', 'ts',
    'java',
    'c', 'cpp', 'c++', 'cxx', 'cc',
    'csharp', 'cs', 'c#',
    'ruby', 'rb',
    'go', 'golang',
    'rust', 'rs',
    'php',
    'swift',
    'kotlin', 'kt',
    'scala',
    'r',
    'perl', 'pl',
    'lua',
    'julia', 'jl',
    'haskell', 'hs',
    'clojure', 'clj',
    'elixir', 'ex', 'exs',
    'erlang', 'erl',
    'fsharp', 'fs', 'f#',
    'ocaml', 'ml',
    'lisp', 'elisp', 'commonlisp', 'cl',
    'scheme', 'scm', 'racket', 'rkt',
    'prolog',
    'cobol', 'cob',
    'fortran', 'f90', 'f95', 'f03', 'f08',
    'pascal', 'pas', 'delphi',
    'ada',
    'assembly', 'asm', 'nasm', 'masm',
    'vb', 'vbnet', 'visualbasic', 'vba',
    'powershell', 'ps1', 'pwsh',
    'matlab', 'm',
    'objectivec', 'objc', 'objective-c',
    'd', 'dlang',
    'nim',
    'crystal', 'cr',
    'zig',
    'v', 'vlang',
    'dart',
    'groovy',
    # Web/Markup
    'html', 'htm', 'xhtml',
    'css', 'scss', 'sass', 'less', 'stylus',
    'xml', 'xsl', 'xslt',
    'svg',
    'json', 'jsonc', 'json5',
    'yaml', 'yml',
    'toml',
    'ini', 'cfg', 'conf',
    'markdown', 'md', 'mdown',
    'restructuredtext', 'rst',
    'latex', 'tex',
    'graphql', 'gql',
    # Shell/Scripting
    'bash', 'sh', 'shell', 'zsh', 'fish', 'ksh', 'csh', 'tcsh',
    'bat', 'batch', 'cmd',
    'awk', 'sed',
    # Database
    'sql', 'mysql', 'postgresql', 'postgres', 'sqlite', 'plsql', 'tsql',
    'mongodb', 'cql', 'cassandra',
    # DevOps/Config
    'dockerfile', 'docker',
    'makefile', 'make', 'cmake',
    'terraform', 'tf', 'hcl',
    'ansible',
    'puppet',
    'chef',
    'nginx',
    'apache',
    # Other
    'diff', 'patch',
    'plaintext', 'text', 'txt',
    'console', 'terminal', 'output',
    'log',
    'csv', 'tsv',
    'protobuf', 'proto',
    'thrift',
    'avro',
    'wasm', 'wat', 'webassembly',
    'solidity', 'sol',
    'vyper',
    'move',
    'cairo',
    'regex', 'regexp',
    'ebnf', 'bnf',
    'mermaid',
    'plantuml',
    'dot', 'graphviz',
    # Empty/generic (allowed)
    '', 'code', 'source', 'snippet',
}

# Pattern to detect corrupted/split language tags.
# Matches cases where a language name is split across lines (e.g., "```c\nsharp" instead of "```csharp").
# This typically occurs due to text processing errors or malformed generation.
CORRUPTED_LANG_TAG_PATTERN = re.compile(
    r'```\s*([a-zA-Z]+)\s*[\r\n]+\s*([a-zA-Z]+)\s*[\r\n]',
    re.MULTILINE
)

# Pattern to extract code blocks with their language tags
CODE_BLOCK_PATTERN = re.compile(
    r'```([^\n`]*)\n(.*?)```',
    re.DOTALL
)

# Pattern to detect undecoded Unicode escape sequences
UNICODE_ESCAPE_PATTERN = re.compile(
    r'\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}'
)

# Pattern to detect invalid structural markers (###, ####, or ##### followed by number)
INVALID_MARKER_PATTERN = re.compile(
    r'^#{3,5}\s*\d+',
    re.MULTILINE
)

# Pattern to detect mistranslated programming keywords inside code blocks.
# When code is incorrectly translated to Portuguese (or other languages), keywords like
# "class", "function", "return" become "classe", "função", "retornar", etc.
# This indicates a translation error that corrupted the source code.
MISTRANSLATED_CODE_INDICATORS = re.compile(
    r'\b(classe\s+pública|função|método|variável|retornar|enquanto|senão|verdadeiro|falso|'
    r'público|privado|protegido|estático|abstrato|interface|herança|exceção|importar|'
    r'definição|declaração|parâmetro|argumento|atributo|propriedade|instância|objeto|'
    r'construtor|destrutor|sobrescrever|implementar|estender|módulo|pacote|biblioteca)\b',
    re.IGNORECASE
)


def get_all_content(example):
    """
    Extract and concatenate all text content from a sample's messages.
    
    Args:
        example: A dataset sample containing a 'messages' field with conversation data.
    
    Returns:
        str: All message contents joined by newlines, or empty string if no content.
    """
    try:
        messages = example.get("messages", [])
        if not messages:
            return ""
        return "\n".join(msg.get("content", "") for msg in messages if msg.get("content"))
    except (KeyError, TypeError, AttributeError):
        return ""


def filter_malformed_code_blocks(example):
    """
    Filter out samples containing malformed code blocks.
    
    Detects two types of malformations:
    1. Corrupted/split language tags (e.g., "```c\nsharp" instead of "```csharp")
    2. Invalid/unrecognized language identifiers not in VALID_LANGUAGE_TAGS
    
    Args:
        example: A dataset sample to validate.
    
    Returns:
        bool: True if sample is valid (should be kept), False if malformed.
    """
    content = get_all_content(example)
    if not content:
        return True  # Keep empty samples (will be filtered by other rules)
    
    # Check for corrupted/split language tags
    if CORRUPTED_LANG_TAG_PATTERN.search(content):
        return False
    
    # Extract all code blocks and validate language tags
    code_blocks = CODE_BLOCK_PATTERN.findall(content)
    for lang_tag, _ in code_blocks:
        # Clean and normalize the language tag
        lang_tag = lang_tag.strip().lower()
        
        # Skip empty tags (generic code blocks are allowed)
        if not lang_tag:
            continue
        
        # Check for newlines or invalid characters in the tag
        if '\n' in lang_tag or '\r' in lang_tag:
            return False
        
        # Extract just the language part (ignore additional metadata)
        lang_parts = lang_tag.split()
        if lang_parts:
            primary_lang = lang_parts[0].strip()
            # Remove common suffixes/prefixes
            primary_lang = re.sub(r'[^a-z0-9#+]', '', primary_lang)
            
            if primary_lang and primary_lang not in VALID_LANGUAGE_TAGS:
                return False
    
    return True


def filter_corrupted_code_content(example):
    """
    Filter out samples with mistranslated code content.
    
    Detects code blocks containing Portuguese (or other language) translations of
    programming keywords, which indicates the source code was incorrectly translated.
    Examples: "função" instead of "function", "retornar" instead of "return".
    
    Note: Markdown and plaintext blocks are excluded from this check.
    
    Args:
        example: A dataset sample to validate.
    
    Returns:
        bool: True if sample is valid (should be kept), False if corrupted.
    """
    content = get_all_content(example)
    if not content:
        return True
    
    # Extract code blocks
    code_blocks = CODE_BLOCK_PATTERN.findall(content)
    for lang_tag, code_content in code_blocks:
        lang_tag = lang_tag.strip().lower()
        
        # Skip markdown and text blocks
        if lang_tag in {'markdown', 'md', 'text', 'txt', 'plaintext', ''}:
            continue
        
        # Check for mistranslated programming keywords in code
        if MISTRANSLATED_CODE_INDICATORS.search(code_content):
            return False
    
    return True


def filter_undecoded_sequences(example):
    """
    Filter out samples containing undecoded Unicode escape sequences.
    
    Detects raw escape sequences that should have been decoded, such as:
    - Unicode escapes: \\u00e3, \\u00f3 (should be 'ã', 'ó')
    - Hex escapes: \\x00, \\xff
    
    These typically indicate encoding/decoding errors during data processing.
    
    Args:
        example: A dataset sample to validate.
    
    Returns:
        bool: True if sample is valid (should be kept), False if contains escapes.
    """
    content = get_all_content(example)
    if not content:
        return True
    
    # Check for undecoded escape sequences
    if UNICODE_ESCAPE_PATTERN.search(content):
        return False
    
    return True


def filter_invalid_structural_markers(example):
    """
    Filter out samples containing invalid structural markers.
    
    Detects lines starting with 3-5 hash symbols followed by a number (e.g., "### 1",
    "#### 42"). These patterns typically indicate:
    - Leaked internal markup from document processing
    - Malformed headers that should have been cleaned
    - Numbering artifacts from automated generation
    
    Args:
        example: A dataset sample to validate.
    
    Returns:
        bool: True if sample is valid (should be kept), False if contains markers.
    """
    content = get_all_content(example)
    if not content:
        return True
    
    # Check for invalid structural markers
    if INVALID_MARKER_PATTERN.search(content):
        return False
    
    return True


def filter_minimum_tokens(example, min_tokens):
    """
    Filter out samples below the minimum token count threshold.
    
    Args:
        example: A dataset sample with a 'token_count' field.
        min_tokens: Minimum number of tokens required.
    
    Returns:
        bool: True if sample meets the threshold, False otherwise.
    """
    try:
        token_count = example.get("token_count", 0)
        return token_count >= min_tokens
    except (KeyError, TypeError):
        return False

def remove_system_messages(example):
    """
    Remove all system messages from the sample's conversation.
    
    Filters out messages where role='system', keeping only user and assistant
    messages. Useful for when system messages are not relevant for training, or
    they are corrupted.
    
    Args:
        example: A dataset sample containing a 'messages' field.
    
    Returns:
        dict: A copy of the example with system messages removed.
    """
    try:
        messages = example.get("messages", [])
        if not messages:
            return example
        
        # Filter out system messages
        filtered_messages = [
            msg for msg in messages
            if msg.get("role", "").lower() != "system"
        ]
        
        # Create a copy of the example with filtered messages
        new_example = dict(example)
        new_example["messages"] = filtered_messages
        return new_example
    except (KeyError, TypeError, AttributeError):
        return example


def strip_message_content(example):
    """
    Strip leading and trailing whitespace from all message content.
    
    Applied by default to normalize text before any filtering operations.
    This ensures consistent handling of messages regardless of source formatting.
    
    Args:
        example: A dataset sample containing a 'messages' field.
    
    Returns:
        dict: A copy of the example with whitespace-stripped message content.
    """
    try:
        messages = example.get("messages", [])
        if not messages:
            return example
        
        stripped_messages = []
        for msg in messages:
            content = msg.get("content", "")
            # Create a copy of the message with stripped content
            stripped_msg = dict(msg)
            stripped_msg["content"] = content.strip() if content else ""
            stripped_messages.append(stripped_msg)
        
        # Create a copy of the example with stripped messages
        new_example = dict(example)
        new_example["messages"] = stripped_messages
        return new_example
    except (KeyError, TypeError, AttributeError):
        return example


def filter_incomplete_sentences(example):
    """
    Filter out samples where the final message doesn't end with proper punctuation.
    
    Checks if the last message in the conversation ends with sentence-ending
    punctuation (. ! ? … $) optionally followed by closing quotes.
    The $ is included to support LaTeX math expressions.
    
    Args:
        example: A dataset sample containing a 'messages' field.
    
    Returns:
        bool: True if the final message ends properly, False otherwise.
    """
    try:
        messages = example["messages"]
        if not messages or len(messages) == 0:
            return False
        
        # Get the last message
        last_message = messages[-1]
        content = last_message.get("content", "").strip()
        
        if not content:
            return False
        
        # Check if it ends with sentence-ending punctuation for Portuguese
        # Covers: period, exclamation, question mark, ellipsis, or LaTeX closing ($)
        # Optionally followed by closing quotes
        sentence_ending_pattern = r'[.!?…$]["\'\'»]*$'
        return bool(re.search(sentence_ending_pattern, content))
    except (KeyError, IndexError, TypeError):
        return False


def filter_token_count(example, max_tokens):
    """
    Filter out samples exceeding the maximum token count threshold.
    
    Args:
        example: A dataset sample with a 'token_count' field.
        max_tokens: Maximum number of tokens allowed.
    
    Returns:
        bool: True if sample is within the limit, False otherwise.
    """
    try:
        token_count = example.get("token_count", 0)
        return token_count <= max_tokens
    except (KeyError, TypeError):
        return False


def load_dataset(input_dir, input_type="jsonl", cache_dir=None):
    """
    Load dataset from JSONL or Parquet files in the specified directory.
    
    Args:
        input_dir: Path to directory containing dataset files.
        input_type: File format, either 'jsonl' or 'parquet'.
        cache_dir: Optional cache directory for HuggingFace datasets.
    
    Returns:
        tuple: (dataset, num_files) - The loaded dataset and count of input files.
    
    Raises:
        ValueError: If no files of the specified type are found.
    """
    data_files = glob.glob(f"{input_dir}/*.{input_type}")
    if not data_files:
        raise ValueError(f"No {input_type.upper()} files found in '{input_dir}'.")
    
    print(f"[INFO] Found {len(data_files)} {input_type} file(s)")
    
    dataset = datasets.load_dataset(
        "json" if input_type == "jsonl" else "parquet",
        data_files=data_files,
        split="train",
        cache_dir=cache_dir,
        num_proc=len(data_files),
    )
    
    return dataset, len(data_files)


def plot_token_distribution(dataset, output_dir):
    """
    Generate and save a histogram of the token count distribution.
    
    Creates a histogram visualization with statistics (mean, median, min, max)
    and saves it as a PNG file in the output directory.
    
    Args:
        dataset: The filtered dataset with a 'token_count' column.
        output_dir: Directory where the histogram will be saved.
    
    Note:
        Skips if 'token_count' column is not present in the dataset.
    """
    if "token_count" not in dataset.column_names:
        print("[WARNING] 'token_count' column not found. Skipping histogram.")
        return
    
    token_counts = dataset["token_count"]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    plt.hist(token_counts, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Token Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Token Counts', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_tokens = np.mean(token_counts)
    median_tokens = np.median(token_counts)
    max_tokens = np.max(token_counts)
    min_tokens = np.min(token_counts)
    
    stats_text = f'Mean: {mean_tokens:.0f}\nMedian: {median_tokens:.0f}\nMin: {min_tokens}\nMax: {max_tokens}'
    plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save the plot
    plot_path = f"{output_dir}/token_distribution.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Token distribution histogram saved to {plot_path}")
    print(f"[INFO] Token statistics: Mean={mean_tokens:.0f}, Median={median_tokens:.0f}, Min={min_tokens}, Max={max_tokens}")


def save_dataset(dataset, output_dir, output_type="jsonl", num_chunks=1, total_tokens=None):
    """
    Save dataset to disk, optionally splitting into multiple chunks.
    
    Saves the dataset in the specified format with consistent naming:
    - Single chunk: train.{ext}
    - Multiple chunks: train-00000-of-NNNNN.{ext}, train-00001-of-NNNNN.{ext}, ...
    
    Also creates a .metadata file with dataset statistics.
    
    Args:
        dataset: The dataset to save.
        output_dir: Directory where files will be written.
        output_type: Output format, either 'jsonl' or 'parquet'.
        num_chunks: Number of chunks to split the dataset into.
        total_tokens: Optional total token count for metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sample_count = len(dataset)
    print(f"[INFO] Saving {sample_count:,} samples in {num_chunks} chunk(s)")
    
    extension = output_type if output_type == "parquet" else "jsonl"
    
    if num_chunks > 1:
        # Split into chunks
        indices = np.array_split(np.arange(sample_count), num_chunks)
        chunks = [dataset.select(idx) for idx in indices]
        
        # Save each chunk
        for i, chunk in enumerate(chunks):
            filename = f"{output_dir}/train-{i:05d}-of-{num_chunks:05d}.{extension}"
            print(f"[INFO] Saving chunk {i+1}/{num_chunks}: {filename} ({len(chunk):,} samples)")
            
            if output_type == "parquet":
                chunk.to_parquet(filename)
            else:
                chunk.to_json(filename)
    else:
        # Save as single file
        filename = f"{output_dir}/train.{extension}"
        print(f"[INFO] Saving to {filename}")
        
        if output_type == "parquet":
            dataset.to_parquet(filename)
        else:
            dataset.to_json(filename)
    
    # Save metadata
    metadata_path = f"{output_dir}/.metadata"
    with open(metadata_path, "w") as meta_file:
        meta_file.write(f"Samples: {sample_count}\n")
        meta_file.write(f"Chunks: {num_chunks}\n")
        meta_file.write(f"Output type: {output_type}\n")
        if total_tokens is not None:
            meta_file.write(f"Total tokens: {total_tokens:,}\n")
        meta_file.write(f"Columns: {dataset.column_names}\n")
    
    print(f"[INFO] Metadata saved to {metadata_path}")


def main(args):
    # Validate arguments
    assert args.input_type in ["jsonl", "parquet"], "Input type must be either 'jsonl' or 'parquet'."
    assert args.output_type in ["jsonl", "parquet"], "Output type must be either 'jsonl' or 'parquet'."
    
    # Load dataset
    print(f"[INFO] Loading dataset from {args.input_dir}")
    dataset, num_input_files = load_dataset(args.input_dir, args.input_type, args.cache_dir)
    print(f"[INFO] Loaded dataset with {len(dataset):,} examples")
    print(f"[INFO] Columns: {dataset.column_names}\n")
    
    initial_count = len(dataset)
    
    # Default preprocessing: Strip whitespace from all message content
    if "messages" in dataset.column_names:
        print(f"[INFO] Stripping whitespace from message content...")
        dataset = dataset.map(strip_message_content, num_proc=args.num_proc)
        print(f"[INFO] Whitespace stripped from {len(dataset):,} samples\n")
    
    # Preprocessing: Remove system messages if enabled
    if args.remove_system_messages and "messages" in dataset.column_names:
        print(f"[INFO] Removing system messages from all samples...")
        dataset = dataset.map(remove_system_messages, num_proc=args.num_proc)
        print(f"[INFO] System messages removed from {len(dataset):,} samples\n")
    
    # Apply filters
    if args.filter_incomplete_sentences and "messages" in dataset.column_names:
        print(f"[INFO] Filtering samples with incomplete sentences...")
        dataset = dataset.filter(filter_incomplete_sentences, num_proc=args.num_proc)
        filtered_incomplete = initial_count - len(dataset)
        print(f"[INFO] Removed {filtered_incomplete:,} samples with incomplete sentences")
        print(f"[INFO] Remaining samples: {len(dataset):,}\n")
    
    if args.filter_malformed_code_blocks and "messages" in dataset.column_names:
        print(f"[INFO] Filtering samples with malformed code blocks...")
        before_filter = len(dataset)
        dataset = dataset.filter(filter_malformed_code_blocks, num_proc=args.num_proc)
        filtered_count = before_filter - len(dataset)
        print(f"[INFO] Removed {filtered_count:,} samples with malformed code blocks")
        print(f"[INFO] Remaining samples: {len(dataset):,}\n")
    
    if args.filter_corrupted_code and "messages" in dataset.column_names:
        print(f"[INFO] Filtering samples with corrupted/mistranslated code content...")
        before_filter = len(dataset)
        dataset = dataset.filter(filter_corrupted_code_content, num_proc=args.num_proc)
        filtered_count = before_filter - len(dataset)
        print(f"[INFO] Removed {filtered_count:,} samples with corrupted code content")
        print(f"[INFO] Remaining samples: {len(dataset):,}\n")
    
    if args.filter_undecoded_sequences:
        print(f"[INFO] Filtering samples with undecoded Unicode escape sequences...")
        before_filter = len(dataset)
        dataset = dataset.filter(filter_undecoded_sequences, num_proc=args.num_proc)
        filtered_count = before_filter - len(dataset)
        print(f"[INFO] Removed {filtered_count:,} samples with undecoded sequences")
        print(f"[INFO] Remaining samples: {len(dataset):,}\n")
    
    if args.filter_invalid_markers:
        print(f"[INFO] Filtering samples with invalid structural markers (#### followed by number)...")
        before_filter = len(dataset)
        dataset = dataset.filter(filter_invalid_structural_markers, num_proc=args.num_proc)
        filtered_count = before_filter - len(dataset)
        print(f"[INFO] Removed {filtered_count:,} samples with invalid structural markers")
        print(f"[INFO] Remaining samples: {len(dataset):,}\n")
    
    if args.min_token_count is not None and "token_count" in dataset.column_names:
        print(f"[INFO] Filtering samples with token_count < {args.min_token_count:,}...")
        before_filter = len(dataset)
        dataset = dataset.filter(lambda x: filter_minimum_tokens(x, args.min_token_count), num_proc=args.num_proc)
        filtered_count = before_filter - len(dataset)
        print(f"[INFO] Removed {filtered_count:,} samples below minimum token threshold")
        print(f"[INFO] Remaining samples: {len(dataset):,}\n")
    
    if args.max_token_count and "token_count" in dataset.column_names:
        print(f"[INFO] Filtering samples with token_count > {args.max_token_count:,}...")
        before_token_filter = len(dataset)
        dataset = dataset.filter(lambda x: filter_token_count(x, args.max_token_count), num_proc=args.num_proc)
        filtered_tokens = before_token_filter - len(dataset)
        print(f"[INFO] Removed {filtered_tokens:,} samples exceeding token limit")
        print(f"[INFO] Remaining samples: {len(dataset):,}\n")
    
    # Check if any samples remain
    if len(dataset) == 0:
        print("[ERROR] No samples remaining after filtering. Exiting.")
        return
    
    # Calculate total tokens if available
    total_tokens = None
    if "token_count" in dataset.column_names:
        total_tokens = sum(dataset["token_count"])
        print(f"[INFO] Total tokens in filtered dataset: {total_tokens:,}")
    
    # Determine number of chunks based on total tokens after filtering
    if total_tokens is not None and args.max_tokens_per_chunk is not None:
        num_chunks = max(1, (total_tokens + args.max_tokens_per_chunk - 1) // args.max_tokens_per_chunk)
        print(f"[INFO] Calculating chunks based on token count (~{args.max_tokens_per_chunk:,} tokens per chunk, {num_chunks} chunk(s))")
    else:
        # Fallback to number of input files if token count not available
        num_chunks = num_input_files
        print(f"[INFO] Token count not available, using {num_chunks} chunk(s) based on input file count")
    
    # Save dataset
    save_dataset(
        dataset,
        args.output_dir,
        args.output_type,
        num_chunks,
        total_tokens
    )
    
    # Plot token distribution
    print(f"\n[INFO] Generating token distribution histogram...")
    plot_token_distribution(dataset, args.output_dir)
    
    print(f"\n[SUCCESS] Dataset saved to {args.output_dir}")
    print(f"[SUCCESS] Total samples: {len(dataset):,}")
    print(f"[SUCCESS] Format: {args.input_type} → {args.output_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments (I/O paths and types)
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing the dataset files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save the converted dataset")
    parser.add_argument("--input_type", choices=["jsonl", "parquet"], required=True, help="Type of the input files")
    parser.add_argument("--output_type", choices=["jsonl", "parquet"], required=True, help="Type of the output files")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for datasets")
    parser.add_argument("--max_tokens_per_chunk", type=int, default=300_000_000, 
                        help="Maximum tokens per output chunk. The filtered dataset will be split into "
                             "chunks of approximately this size. Requires 'token_count' column. (default: 300M)")
    
    # Filtering options
    parser.add_argument("--filter_incomplete_sentences", action="store_true", help="Filter out samples where the final message doesn't end with punctuation")
    parser.add_argument("--max_token_count", type=int, default=None, help="Maximum token count threshold; samples exceeding this will be filtered out")
    parser.add_argument("--min_token_count", type=int, default=None, help="Minimum token count threshold; samples below this will be filtered out")
    
    parser.add_argument("--filter_malformed_code_blocks", action="store_true", 
                        help="Filter out samples with malformed code blocks (corrupted/split language tags or invalid language identifiers)")
    parser.add_argument("--filter_corrupted_code", action="store_true",
                        help="Filter out samples with corrupted/mistranslated code content (non-ASCII characters or diacritics in source code)")
    parser.add_argument("--filter_undecoded_sequences", action="store_true",
                        help="Filter out samples containing undecoded Unicode escape sequences (e.g., \\u00e3, \\x00)")
    parser.add_argument("--filter_invalid_markers", action="store_true",
                        help="Filter out samples containing invalid structural markers (lines starting with #### followed by a number)")
    parser.add_argument("--remove_system_messages", action="store_true",
                        help="Remove all system messages from samples before processing (disabled by default)")
    
    # Number of processes to use for parallel filtering
    parser.add_argument("--num_proc", type=int, default=4, help="Number of processes to use for filtering (default: 4)")

    args = parser.parse_args()
    main(args)
