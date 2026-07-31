"""
SFT Dataset Filter (Portuguese Version)

This is a filtering and conversion tool for instruction-tuning (SFT) datasets.
Designed to clean, validate, and transform instruction-following datasets by removing
malformed code blocks, corrupted code content, undecoded Unicode sequences, word
repetition loops (degenerate model outputs), and other issues that passed through
the quality filter.

Expected Dataset Format:
The script expects datasets with a messages column (configurable) containing conversation data:
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "token_count": 123  # Optional, enables token-based features
    }

Usage:
    python sft_filter.py \\
        --input_dir ./raw_data \\
        --output_dir ./filtered_data \\
        --input_type parquet \\
        --output_type parquet \\
        --messages_column messages \\
        --token_count_column token_count \\
        --filter_incomplete_sentences \\
        --filter_malformed_code_blocks \\
        --filter_corrupted_code \\
        --filter_undecoded_sequences \\
        --filter_invalid_markers \\
        --filter_repetition_loops \\
        --remove_system_messages \\
        --quality_score_column instruct_score \\
        --min_quality_score 4.5

TODO: Currently focused on Portuguese SFT datasets, but we should add support for
other languages in the future. Many of the filters are language-agnostic, but
some (like repetition loops) need language-specific adjustments.
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from utils import DatasetLoader, flatten_messages, get_logger, save_dataset, write_metadata

logger = get_logger("SFTFilter")

# CONSTANTS & PATTERNS
# Valid programming language tags for code block validation.
# Used to detect malformed code blocks with invalid or corrupted language identifiers.
# Includes common programming languages, markup languages, shell scripting, and config formats.
VALID_LANGUAGE_TAGS = {
    # Common languages
    "python",
    "py",
    "python3",
    "py3",
    "javascript",
    "js",
    "node",
    "nodejs",
    "typescript",
    "ts",
    "java",
    "c",
    "cpp",
    "c++",
    "cxx",
    "cc",
    "csharp",
    "cs",
    "c#",
    "ruby",
    "rb",
    "go",
    "golang",
    "rust",
    "rs",
    "php",
    "swift",
    "kotlin",
    "kt",
    "scala",
    "r",
    "perl",
    "pl",
    "lua",
    "julia",
    "jl",
    "haskell",
    "hs",
    "clojure",
    "clj",
    "elixir",
    "ex",
    "exs",
    "erlang",
    "erl",
    "fsharp",
    "fs",
    "f#",
    "ocaml",
    "ml",
    "lisp",
    "elisp",
    "commonlisp",
    "cl",
    "scheme",
    "scm",
    "racket",
    "rkt",
    "prolog",
    "cobol",
    "cob",
    "fortran",
    "f90",
    "f95",
    "f03",
    "f08",
    "pascal",
    "pas",
    "delphi",
    "ada",
    "assembly",
    "asm",
    "nasm",
    "masm",
    "vb",
    "vbnet",
    "visualbasic",
    "vba",
    "powershell",
    "ps1",
    "pwsh",
    "matlab",
    "m",
    "objectivec",
    "objc",
    "objective-c",
    "d",
    "dlang",
    "nim",
    "crystal",
    "cr",
    "zig",
    "v",
    "vlang",
    "dart",
    "groovy",
    # Web/Markup
    "html",
    "htm",
    "xhtml",
    "css",
    "scss",
    "sass",
    "less",
    "stylus",
    "xml",
    "xsl",
    "xslt",
    "svg",
    "json",
    "jsonc",
    "json5",
    "yaml",
    "yml",
    "toml",
    "ini",
    "cfg",
    "conf",
    "markdown",
    "md",
    "mdown",
    "restructuredtext",
    "rst",
    "latex",
    "tex",
    "graphql",
    "gql",
    # Shell/Scripting
    "bash",
    "sh",
    "shell",
    "zsh",
    "fish",
    "ksh",
    "csh",
    "tcsh",
    "bat",
    "batch",
    "cmd",
    "awk",
    "sed",
    # Database
    "sql",
    "mysql",
    "postgresql",
    "postgres",
    "sqlite",
    "plsql",
    "tsql",
    "mongodb",
    "cql",
    "cassandra",
    # DevOps/Config
    "dockerfile",
    "docker",
    "makefile",
    "make",
    "cmake",
    "terraform",
    "tf",
    "hcl",
    "ansible",
    "puppet",
    "chef",
    "nginx",
    "apache",
    # Other
    "diff",
    "patch",
    "plaintext",
    "text",
    "txt",
    "console",
    "terminal",
    "output",
    "log",
    "csv",
    "tsv",
    "protobuf",
    "proto",
    "thrift",
    "avro",
    "wasm",
    "wat",
    "webassembly",
    "solidity",
    "sol",
    "vyper",
    "move",
    "cairo",
    "regex",
    "regexp",
    "ebnf",
    "bnf",
    "mermaid",
    "plantuml",
    "dot",
    "graphviz",
    # Empty/generic (allowed)
    "",
    "code",
    "source",
    "snippet",
}

# Pattern to detect corrupted/split language tags.
# Matches cases where a language name is split across lines (e.g., "```c\nsharp" instead of "```csharp").
# This typically occurs due to text processing errors or malformed generation.
CORRUPTED_LANG_TAG_PATTERN = re.compile(
    r"```\s*([a-zA-Z]+)\s*[\r\n]+\s*([a-zA-Z]+)\s*[\r\n]", re.MULTILINE
)

# Pattern to extract code blocks with their language tags
CODE_BLOCK_PATTERN = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)

# Pattern to detect undecoded Unicode escape sequences
UNICODE_ESCAPE_PATTERN = re.compile(r"\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}")

# Pattern to detect invalid structural markers (###, ####, or ##### followed by number)
INVALID_MARKER_PATTERN = re.compile(r"^#{3,5}\s*\d+", re.MULTILINE)

# Pattern to detect mistranslated programming keywords inside code blocks.
# When code is incorrectly translated to Portuguese (or other languages), keywords like
# "class", "function", "return" become "classe", "função", "retornar", etc.
# This indicates a translation error that corrupted the source code.
# TODO: Expand to other languages in the future, and consider adding more keywords based on
# common mistranslations. We could, for example, set a parameter so that the user can specify
# the language and we load a corresponding list of keywords (from a file or from a small API).
MISTRANSLATED_CODE_INDICATORS = re.compile(
    r"\b(classe\s+pública|função|método|variável|retornar|enquanto|senão|verdadeiro|falso|"
    r"público|privado|protegido|estático|abstrato|interface|herança|exceção|importar|"
    r"definição|declaração|parâmetro|argumento|atributo|propriedade|instância|objeto|"
    r"construtor|destrutor|sobrescrever|implementar|estender|módulo|pacote|biblioteca)\b",
    re.IGNORECASE,
)

# Stopwords to exclude from uniqueness calculations
# These common words naturally repeat in normal text and shouldn't trigger the filter.
# TODO: Expand this list based on analysis of false positives, and consider making it configurable.
PORTUGUESE_STOPWORDS = {
    "a",
    "o",
    "e",
    "de",
    "da",
    "do",
    "das",
    "dos",
    "em",
    "na",
    "no",
    "nas",
    "nos",
    "um",
    "uma",
    "uns",
    "umas",
    "para",
    "por",
    "com",
    "como",
    "que",
    "se",
    "ou",
    "mais",
    "mas",
    "ao",
    "aos",
    "à",
    "às",
    "ser",
    "ter",
    "estar",
    "foi",
    "são",
    "é",
    "esse",
    "essa",
    "este",
    "esta",
    "isso",
    "isto",
    "aquele",
    "aquela",
    "seu",
    "sua",
    "seus",
    "suas",
    "ele",
    "ela",
    "eles",
    "elas",
    "nós",
    "você",
    "entre",
    "sobre",
    "até",
    "já",
    "também",
    "muito",
    "bem",
    "só",
    "ainda",
    "quando",
    "onde",
    "qual",
    "quais",
    "quem",
    "porque",
    "pode",
    "podem",
    "deve",
    "devem",
    "há",
    "sem",
    "pela",
    "pelo",
    "pelas",
    "pelos",
    "após",
}


def filter_malformed_code_blocks(example, messages_column="messages"):
    """
    Filter out samples containing malformed code blocks.

    Detects two types of malformations:
    1. Corrupted/split language tags (e.g., "```c\nsharp" instead of "```csharp")
    2. Invalid/unrecognized language identifiers not in VALID_LANGUAGE_TAGS

    Args:
        example: A dataset sample to validate.
        messages_column: The name of the column containing messages.

    Returns:
        bool: True if sample is valid (should be kept), False if malformed.
    """
    content = flatten_messages(example.get(messages_column, []))
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
        if "\n" in lang_tag or "\r" in lang_tag:
            return False

        # Extract just the language part (ignore additional metadata)
        lang_parts = lang_tag.split()
        if lang_parts:
            primary_lang = lang_parts[0].strip()
            # Remove common suffixes/prefixes
            primary_lang = re.sub(r"[^a-z0-9#+]", "", primary_lang)

            if primary_lang and primary_lang not in VALID_LANGUAGE_TAGS:
                return False

    return True


def filter_corrupted_code_content(example, messages_column="messages"):
    """
    Filter out samples with mistranslated code content.

    Detects code blocks containing Portuguese (or other language) translations of
    programming keywords, which indicates the source code was incorrectly translated.
    Examples: "função" instead of "function", "retornar" instead of "return".

    Note: Markdown and plaintext blocks are excluded from this check.

    Args:
        example: A dataset sample to validate.
        messages_column: The name of the column containing messages.

    Returns:
        bool: True if sample is valid (should be kept), False if corrupted.
    """
    content = flatten_messages(example.get(messages_column, []))
    if not content:
        return True

    # Extract code blocks
    code_blocks = CODE_BLOCK_PATTERN.findall(content)
    for lang_tag, code_content in code_blocks:
        lang_tag = lang_tag.strip().lower()

        # Skip markdown and text blocks
        if lang_tag in {"markdown", "md", "text", "txt", "plaintext", ""}:
            continue

        # Check for mistranslated programming keywords in code
        if MISTRANSLATED_CODE_INDICATORS.search(code_content):
            return False

    return True


def filter_undecoded_sequences(example, messages_column="messages"):
    """
    Filter out samples containing undecoded Unicode escape sequences.

    Detects raw escape sequences that should have been decoded, such as:
    - Unicode escapes: \\u00e3, \\u00f3 (should be 'ã', 'ó')
    - Hex escapes: \\x00, \\xff

    These typically indicate encoding/decoding errors during data processing.

    Args:
        example: A dataset sample to validate.
        messages_column: The name of the column containing messages.

    Returns:
        bool: True if sample is valid (should be kept), False if contains escapes.
    """
    content = flatten_messages(example.get(messages_column, []))
    if not content:
        return True

    # Check for undecoded escape sequences
    return not UNICODE_ESCAPE_PATTERN.search(content)


def filter_invalid_structural_markers(example, messages_column="messages"):
    """
    Filter out samples containing invalid structural markers.

    Detects lines starting with 3-5 hash symbols followed by a number (e.g., "### 1",
    "#### 42"). These patterns typically indicate:
    - Leaked internal markup from document processing
    - Malformed headers that should have been cleaned
    - Numbering artifacts from automated generation

    Args:
        example: A dataset sample to validate.
        messages_column: The name of the column containing messages.

    Returns:
        bool: True if sample is valid (should be kept), False if contains markers.
    """
    content = flatten_messages(example.get(messages_column, []))
    if not content:
        return True

    # Check for invalid structural markers
    return not INVALID_MARKER_PATTERN.search(content)


def filter_repetition_loops(
    example,
    messages_column="messages",
    min_repeated_words=8,
    max_unique_ratio=0.15,
    window_size=30,
    min_window_matches=5,
    min_consecutive_suffix=12,
    min_ngram_repeats=10,
):
    """
    Filter out samples where the model gets stuck in word repetition loops.

    Detects degenerate text patterns where the model repeatedly generates similar
    words in sequence, typically adverbs, participles, or related terms.
    Examples of repetitive patterns:
    - "precisamente calculadamente programaticamente antecipadamente predeterminadamente"
    - "consolidadas solidificadas fortalecidas reforçadas intensificadas ampliadas"

    The detection uses multiple heuristics:
    1. Sliding window analysis: Checks for windows with very low unique content word ratio
    2. Suffix pattern detection: Identifies long sequences of words with same suffix
    3. N-gram repetition: Detects repeated word sequences (scaled by document length)
    4. Single-word repetition: Detects "word word word word..."

    Note: This filter excludes common stopwords from uniqueness calculations to avoid
    false positives on normal prose that naturally repeats function words.

    Args:
        example: A dataset sample to validate.
        messages_column: The name of the column containing messages.
        min_repeated_words: Minimum consecutive identical words to flag.
        max_unique_ratio: Maximum ratio of unique content words in a window to be flagged.
        window_size: Size of sliding window for analysis.
        min_window_matches: Minimum windows that must match to filter the sample.
        min_consecutive_suffix: Minimum consecutive words with same suffix to flag.
        min_ngram_repeats: Minimum n-gram repetitions to flag.

    Returns:
        bool: True if sample is valid (should be kept), False if contains repetition loops.
    """
    content = flatten_messages(example.get(messages_column, []))
    if not content:
        return True

    # Tokenize into words (simple whitespace + punctuation split)
    words = re.findall(r"\b[a-záàâãéèêíìîóòôõúùûüçñ]+\b", content.lower())

    if len(words) < window_size:
        return True

    # Filter out stopwords for content analysis (but keep original for suffix/ngram checks)
    content_words = [w for w in words if w not in PORTUGUESE_STOPWORDS and len(w) > 2]

    # Heuristic 1: Sliding window with low unique content word ratio
    # This catches sequences where the same or similar content words repeat
    # Only applies to content words (excluding stopwords)
    if len(content_words) >= window_size:
        repetitive_windows = 0
        for i in range(len(content_words) - window_size + 1):
            window = content_words[i : i + window_size]
            unique_ratio = len(set(window)) / len(window)
            if unique_ratio < max_unique_ratio:
                repetitive_windows += 1
                if repetitive_windows >= min_window_matches:
                    return False

    # Heuristic 2: Suffix pattern detection (stricter)
    # Catches sequences like "precisamente calculadamente programaticamente"
    # Only check for the most problematic suffixes that indicate degenerate loops
    degenerate_suffixes = ["mente", "ando", "endo", "indo"]  # Most indicative of loops

    for suffix in degenerate_suffixes:
        consecutive_suffix_count = 0
        max_consecutive = 0
        for word in words:
            # Word must be substantial (not just the suffix)
            if word.endswith(suffix) and len(word) > len(suffix) + 3:
                consecutive_suffix_count += 1
                max_consecutive = max(max_consecutive, consecutive_suffix_count)
            else:
                consecutive_suffix_count = 0

        # High threshold: 12+ consecutive words with the same suffix is truly degenerate
        if max_consecutive >= min_consecutive_suffix:
            return False

    # Heuristic 3: N-gram repetition detection (scaled by document length)
    # Catches exact phrase repetitions, but with thresholds proportional to doc length
    if len(words) >= 50:
        # Scale threshold by document length - longer docs naturally have more repetition
        # For a 500-word doc, threshold is ~10; for 1000-word doc, ~14
        length_factor = max(1, len(words) / 500)
        dynamic_threshold = int(min_ngram_repeats * (0.7 + 0.3 * length_factor))

        for n in [4, 5, 6]:  # Check 4-grams, 5-grams, and 6-grams (not trigrams - too common)
            # Exclude n-grams that are mostly stopwords
            ngrams = []
            for i in range(len(words) - n + 1):
                ng = tuple(words[i : i + n])
                # Only count if at least half of the words are content words
                content_count = sum(1 for w in ng if w not in PORTUGUESE_STOPWORDS)
                if content_count >= n // 2:
                    ngrams.append(ng)

            if ngrams:
                ngram_counts = {}
                for ng in ngrams:
                    ngram_counts[ng] = ngram_counts.get(ng, 0) + 1

                max_count = max(ngram_counts.values())
                if max_count >= dynamic_threshold:
                    return False

    # Heuristic 4: Long sequences of single-word repetition
    # Catches "word word word word word..."
    for i in range(len(words) - min_repeated_words):
        if len(set(words[i : i + min_repeated_words])) == 1:
            return False

    return True


def filter_minimum_tokens(example, min_tokens, token_count_column="token_count"):
    """
    Filter out samples below the minimum token count threshold.

    Args:
        example: A dataset sample with a token count field.
        min_tokens: Minimum number of tokens required.
        token_count_column: Name of the column containing token counts.

    Returns:
        bool: True if sample meets the threshold, False otherwise.
    """
    try:
        token_count = example.get(token_count_column, 0)
        return token_count >= min_tokens
    except (KeyError, TypeError):
        return False


def filter_quality_score(example, score_column, min_score):
    """
    Filter out samples below the minimum quality score threshold.

    Used to filter based on quality annotation columns like 'instruct_score',
    'educational_score', or similar metrics from quality classifiers.

    Args:
        example: A dataset sample with a quality score field.
        score_column: Name of the column containing the quality score.
        min_score: Minimum score required to keep the sample.

    Returns:
        bool: True if sample meets the threshold, False otherwise.
    """
    try:
        score = example.get(score_column, None)
        if score is None:
            return False
        return float(score) >= min_score
    except (KeyError, TypeError, ValueError):
        return False


def remove_system_messages(example, messages_column="messages"):
    """
    Remove all system messages from the sample's conversation.

    Filters out messages where role='system', keeping only user and assistant
    messages. Useful for when system messages are not relevant for training, or
    they are corrupted.

    Args:
        example: A dataset sample containing a messages field.
        messages_column: The name of the column containing messages.

    Returns:
        dict: A copy of the example with system messages removed.
    """
    try:
        messages = example.get(messages_column, [])
        if not messages:
            return example

        # Filter out system messages
        filtered_messages = [msg for msg in messages if msg.get("role", "").lower() != "system"]

        # Create a copy of the example with filtered messages
        new_example = dict(example)
        new_example[messages_column] = filtered_messages
        return new_example
    except (KeyError, TypeError, AttributeError):
        return example


def strip_message_content(example, messages_column="messages"):
    """
    Strip leading and trailing whitespace from all message content.

    Applied by default to normalize text before any filtering operations.
    This ensures consistent handling of messages regardless of source formatting.

    Args:
        example: A dataset sample containing a messages field.
        messages_column: The name of the column containing messages.

    Returns:
        dict: A copy of the example with whitespace-stripped message content.
    """
    try:
        messages = example.get(messages_column, [])
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
        new_example[messages_column] = stripped_messages
        return new_example
    except (KeyError, TypeError, AttributeError):
        return example


def filter_incomplete_sentences(example, messages_column="messages"):
    """
    Filter out samples where the final message doesn't end with proper punctuation.

    Checks if the last message in the conversation ends with sentence-ending
    punctuation (. ! ? … $) optionally followed by closing quotes.
    The $ is included to support LaTeX math expressions.

    Args:
        example: A dataset sample containing a messages field.
        messages_column: The name of the column containing messages.

    Returns:
        bool: True if the final message ends properly, False otherwise.
    """
    try:
        messages = example[messages_column]
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


def filter_token_count(example, max_tokens, token_count_column="token_count"):
    """
    Filter out samples exceeding the maximum token count threshold.

    Args:
        example: A dataset sample with a token count field.
        max_tokens: Maximum number of tokens allowed.
        token_count_column: Name of the column containing token counts.

    Returns:
        bool: True if sample is within the limit, False otherwise.
    """
    try:
        token_count = example.get(token_count_column, 0)
        return token_count <= max_tokens
    except (KeyError, TypeError):
        return False


def plot_token_distribution(dataset, output_dir, token_count_column="token_count"):
    """
    Generate and save a histogram of the token count distribution.

    Creates a histogram visualization with statistics (mean, median, min, max)
    and saves it as a PNG file in the output directory.

    Args:
        dataset: The filtered dataset with a token count column.
        output_dir: Directory where the histogram will be saved.
        token_count_column: Name of the column containing token counts.

    Note:
        Skips if token count column is not present in the dataset.
    """
    if token_count_column not in dataset.column_names:
        logger.warning("'%s' column not found. Skipping histogram.", token_count_column)
        return

    token_counts = dataset[token_count_column]

    # Create figure
    plt.figure(figsize=(12, 6))
    plt.hist(token_counts, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Token Count", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Token Counts", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)

    # Add statistics
    mean_tokens = np.mean(token_counts)
    median_tokens = np.median(token_counts)
    max_tokens = np.max(token_counts)
    min_tokens = np.min(token_counts)

    stats_text = f"Mean: {mean_tokens:.0f}\nMedian: {median_tokens:.0f}\nMin: {min_tokens}\nMax: {max_tokens}"
    plt.text(
        0.98,
        0.97,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    # Save the plot
    plot_path = f"{output_dir}/token_distribution.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Token distribution histogram saved to %s", plot_path)
    logger.info(
        "Token statistics: Mean=%.0f, Median=%.0f, Min=%s, Max=%s",
        mean_tokens,
        median_tokens,
        min_tokens,
        max_tokens,
    )


def main(args):
    # Get the column names
    messages_column = args.messages_column
    token_count_column = args.token_count_column

    # Load dataset
    logger.info("Loading dataset from %s", args.input_dir)
    dataset = DatasetLoader(args.input_dir, cache_dir=args.cache_dir).load()
    logger.info("Loaded dataset with %d examples", len(dataset))
    logger.info("Columns: %s", dataset.column_names)
    logger.info("Messages column: %s", messages_column)

    initial_count = len(dataset)

    # Default preprocessing: Strip whitespace from all message content
    if messages_column in dataset.column_names:
        logger.info("Stripping whitespace from message content...")
        dataset = dataset.map(
            lambda x: strip_message_content(x, messages_column), num_proc=args.num_proc
        )
        logger.info("Whitespace stripped from %d samples", len(dataset))

    # Preprocessing: Remove system messages if enabled
    if args.remove_system_messages and messages_column in dataset.column_names:
        logger.info("Removing system messages from all samples...")
        dataset = dataset.map(
            lambda x: remove_system_messages(x, messages_column), num_proc=args.num_proc
        )
        logger.info("System messages removed from %d samples", len(dataset))

    # Apply filters
    if args.filter_incomplete_sentences and messages_column in dataset.column_names:
        logger.info("Filtering samples with incomplete sentences...")
        dataset = dataset.filter(
            lambda x: filter_incomplete_sentences(x, messages_column), num_proc=args.num_proc
        )
        filtered_incomplete = initial_count - len(dataset)
        logger.info("Removed %d samples with incomplete sentences", filtered_incomplete)
        logger.info("Remaining samples: %d", len(dataset))

    if args.filter_malformed_code_blocks and messages_column in dataset.column_names:
        logger.info("Filtering samples with malformed code blocks...")
        before_filter = len(dataset)
        dataset = dataset.filter(
            lambda x: filter_malformed_code_blocks(x, messages_column), num_proc=args.num_proc
        )
        filtered_count = before_filter - len(dataset)
        logger.info("Removed %d samples with malformed code blocks", filtered_count)
        logger.info("Remaining samples: %d", len(dataset))

    if args.filter_corrupted_code and messages_column in dataset.column_names:
        logger.info("Filtering samples with corrupted/mistranslated code content...")
        before_filter = len(dataset)
        dataset = dataset.filter(
            lambda x: filter_corrupted_code_content(x, messages_column), num_proc=args.num_proc
        )
        filtered_count = before_filter - len(dataset)
        logger.info("Removed %d samples with corrupted code content", filtered_count)
        logger.info("Remaining samples: %d", len(dataset))

    if args.filter_undecoded_sequences and messages_column in dataset.column_names:
        logger.info("Filtering samples with undecoded Unicode escape sequences...")
        before_filter = len(dataset)
        dataset = dataset.filter(
            lambda x: filter_undecoded_sequences(x, messages_column), num_proc=args.num_proc
        )
        filtered_count = before_filter - len(dataset)
        logger.info("Removed %d samples with undecoded sequences", filtered_count)
        logger.info("Remaining samples: %d", len(dataset))

    if args.filter_invalid_markers and messages_column in dataset.column_names:
        logger.info(
            "Filtering samples with invalid structural markers (#### followed by number)..."
        )
        before_filter = len(dataset)
        dataset = dataset.filter(
            lambda x: filter_invalid_structural_markers(x, messages_column), num_proc=args.num_proc
        )
        filtered_count = before_filter - len(dataset)
        logger.info("Removed %d samples with invalid structural markers", filtered_count)
        logger.info("Remaining samples: %d", len(dataset))

    if args.filter_repetition_loops and messages_column in dataset.column_names:
        logger.info("Filtering samples with word repetition loops...")
        before_filter = len(dataset)
        dataset = dataset.filter(
            lambda x: filter_repetition_loops(x, messages_column), num_proc=args.num_proc
        )
        filtered_count = before_filter - len(dataset)
        logger.info("Removed %d samples with repetition loops", filtered_count)
        logger.info("Remaining samples: %d", len(dataset))

    if args.min_quality_score is not None and args.quality_score_column:
        if args.quality_score_column in dataset.column_names:
            logger.info(
                "Filtering samples with %s < %s...",
                args.quality_score_column,
                args.min_quality_score,
            )
            before_filter = len(dataset)
            dataset = dataset.filter(
                lambda x: filter_quality_score(
                    x, args.quality_score_column, args.min_quality_score
                ),
                num_proc=args.num_proc,
            )
            filtered_count = before_filter - len(dataset)
            logger.info("Removed %d samples below quality score threshold", filtered_count)
            logger.info("Remaining samples: %d", len(dataset))
        else:
            logger.warning(
                "Quality score column '%s' not found in dataset. Skipping filter.",
                args.quality_score_column,
            )

    if args.min_token_count is not None and token_count_column in dataset.column_names:
        logger.info("Filtering samples with %s < %d...", token_count_column, args.min_token_count)
        before_filter = len(dataset)
        dataset = dataset.filter(
            lambda x: filter_minimum_tokens(x, args.min_token_count, token_count_column),
            num_proc=args.num_proc,
        )
        filtered_count = before_filter - len(dataset)
        logger.info("Removed %d samples below minimum token threshold", filtered_count)
        logger.info("Remaining samples: %d", len(dataset))

    if args.max_token_count and token_count_column in dataset.column_names:
        logger.info("Filtering samples with %s > %d...", token_count_column, args.max_token_count)
        before_token_filter = len(dataset)
        dataset = dataset.filter(
            lambda x: filter_token_count(x, args.max_token_count, token_count_column),
            num_proc=args.num_proc,
        )
        filtered_tokens = before_token_filter - len(dataset)
        logger.info("Removed %d samples exceeding token limit", filtered_tokens)
        logger.info("Remaining samples: %d", len(dataset))

    # Check if any samples remain
    if len(dataset) == 0:
        logger.error("No samples remaining after filtering. Exiting.")
        return

    # Calculate total tokens if available
    total_tokens = None
    if token_count_column in dataset.column_names:
        total_tokens = sum(dataset[token_count_column])
        logger.info("Total tokens in filtered dataset: %d", total_tokens)

    # Determine number of chunks based on total tokens after filtering
    if total_tokens is not None:
        num_chunks = max(
            1, (total_tokens + args.max_tokens_per_chunk - 1) // args.max_tokens_per_chunk
        )
        logger.info(
            "Calculating chunks based on token count (~%d tokens per chunk, %d chunk(s))",
            args.max_tokens_per_chunk,
            num_chunks,
        )
    else:
        # Fallback to number of input files if token count not available
        num_chunks = max(1, len(glob.glob(f"{args.input_dir}/*.{args.input_type}")))
        logger.info(
            "Token count not available, using %d chunk(s) based on input file count", num_chunks
        )

    # Save dataset
    n_chunks_written = save_dataset(
        dataset,
        args.output_dir,
        args.output_type,
        args.max_tokens_per_chunk,
        token_count=total_tokens or 0,
        n_chunks=num_chunks,
    )
    metadata = {
        "Samples": len(dataset),
        "Chunks": n_chunks_written,
        "Output_type": args.output_type,
        "Columns": dataset.column_names,
    }
    if total_tokens is not None:
        metadata["Total_tokens"] = f"{total_tokens:,}"
    write_metadata(os.path.join(args.output_dir, ".metadata"), metadata)

    # Plot token distribution
    logger.info("Generating token distribution histogram...")
    plot_token_distribution(dataset, args.output_dir, token_count_column)

    logger.info("Dataset saved to %s", args.output_dir)
    logger.info("Total samples: %d", len(dataset))
    logger.info("Format: %s → %s", args.input_type, args.output_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments (I/O paths and types)
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input directory containing the dataset files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save the converted dataset",
    )
    parser.add_argument(
        "--input_type", choices=["jsonl", "parquet"], required=True, help="Type of the input files"
    )
    parser.add_argument(
        "--output_type",
        choices=["jsonl", "parquet"],
        required=True,
        help="Type of the output files",
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for datasets")
    parser.add_argument(
        "--max_tokens_per_chunk",
        type=int,
        default=300_000_000,
        help="Maximum tokens per output chunk. The filtered dataset will be split into "
        "chunks of approximately this size. Requires 'token_count' column. (default: 300M)",
    )

    # Filtering options
    parser.add_argument(
        "--filter_incomplete_sentences",
        action="store_true",
        help="Filter out samples where the final message doesn't end with punctuation",
    )
    parser.add_argument(
        "--max_token_count",
        type=int,
        default=None,
        help="Maximum token count threshold; samples exceeding this will be filtered out",
    )
    parser.add_argument(
        "--min_token_count",
        type=int,
        default=None,
        help="Minimum token count threshold; samples below this will be filtered out",
    )

    parser.add_argument(
        "--filter_malformed_code_blocks",
        action="store_true",
        help="Filter out samples with malformed code blocks (corrupted/split language tags or invalid language identifiers)",
    )
    parser.add_argument(
        "--filter_corrupted_code",
        action="store_true",
        help="Filter out samples with corrupted/mistranslated code content (non-ASCII characters or diacritics in source code)",
    )
    parser.add_argument(
        "--filter_undecoded_sequences",
        action="store_true",
        help="Filter out samples containing undecoded Unicode escape sequences (e.g., \\u00e3, \\x00)",
    )
    parser.add_argument(
        "--filter_invalid_markers",
        action="store_true",
        help="Filter out samples containing invalid structural markers (lines starting with #### followed by a number)",
    )
    parser.add_argument(
        "--remove_system_messages",
        action="store_true",
        help="Remove all system messages from samples before processing (disabled by default)",
    )
    parser.add_argument(
        "--filter_repetition_loops",
        action="store_true",
        help="Filter out samples where the model gets stuck in word repetition loops",
    )

    # Quality score filtering
    parser.add_argument(
        "--quality_score_column",
        type=str,
        default=None,
        help="Name of the column containing quality scores (e.g., 'instruct_score', 'educational_score')",
    )
    parser.add_argument(
        "--min_quality_score",
        type=float,
        default=None,
        help="Minimum quality score threshold; samples below this will be filtered out",
    )

    # Messages column configuration
    parser.add_argument(
        "--messages_column",
        type=str,
        default="messages",
        help="Name of the column containing the messages/conversation data (default: 'messages')",
    )

    # Token count column configuration
    parser.add_argument(
        "--token_count_column",
        type=str,
        default="token_count",
        help="Name of the column containing token counts (default: 'token_count')",
    )

    # Number of processes to use for parallel filtering
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of processes to use for filtering (default: 4)",
    )

    args = parser.parse_args()
    main(args)
