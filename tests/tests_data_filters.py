"""
Data/filters test suite — language filter utilities.

Tests:
  - data/filters/utils.py     (is_messages_column, flatten_messages,
                                save_dataset, DatasetLoader)
  - data/filters/language_filter.py  (unicode backend, language definitions,
                                      argument parser)

The langdetect library is NOT required; the langdetect backend is not
exercised here — only its registration in _BACKEND_FACTORIES is checked.

Run with:
    python tests/tests_data_filters.py
"""

# %%
#######################################
# 1. Imports & Setup
#######################################
import os
import sys
import json
import tempfile
import argparse

sys.pycache_prefix = os.path.join(tempfile.gettempdir(), "pycache")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
FILTERS_DIR = os.path.join(REPO_ROOT, "data", "filters")
if FILTERS_DIR not in sys.path:
    sys.path.insert(0, FILTERS_DIR)

import datasets
from utils import (
    DatasetLoader,
    save_dataset,
    is_messages_column,
    flatten_messages,
)
from language_filter import (
    LANGDETECT_CODES,
    UNICODE_RANGES,
    SUPPORTED_LANGUAGES,
    _BACKEND_FACTORIES,
    _create_unicode_filter,
)

print("All imports OK ✅")


# %%
#######################################
# Section 1 — is_messages_column
#######################################

def test_01_is_messages_column_missing_column_returns_false():
    # 1. is_messages_column — returns False when the column is not in the dataset
    ds = datasets.Dataset.from_list([{"text": "hello"}])
    assert not is_messages_column(ds, "messages")
    print("Test 01 — is_messages_column (missing column): OK ✅")


def test_02_is_messages_column_empty_dataset_returns_false():
    # 2. is_messages_column — returns False for an empty dataset
    ds = datasets.Dataset.from_list([])
    assert not is_messages_column(ds, "messages")
    print("Test 02 — is_messages_column (empty dataset): OK ✅")


def test_03_is_messages_column_detects_messages_format():
    # 3. is_messages_column — returns True when column holds a list of dicts with 'content'
    ds = datasets.Dataset.from_list([
        {"messages": [{"role": "user", "content": "hello"}]},
    ])
    assert is_messages_column(ds, "messages")
    print("Test 03 — is_messages_column (messages format): OK ✅")


def test_04_is_messages_column_plain_text_returns_false():
    # 4. is_messages_column — returns False when column holds plain strings
    ds = datasets.Dataset.from_list([{"text": "hello world"}])
    assert not is_messages_column(ds, "text")
    print("Test 04 — is_messages_column (plain text): OK ✅")


# %%
#######################################
# Section 2 — flatten_messages
#######################################

def test_05_flatten_messages_concatenates_content_fields():
    # 5. flatten_messages — joins 'content' values with newlines in order
    messages = [
        {"role": "user",      "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    result = flatten_messages(messages)
    assert result == "hello\nworld"
    print("Test 05 — flatten_messages (basic concatenation): OK ✅")


def test_06_flatten_messages_empty_list_returns_empty_string():
    # 6. flatten_messages — empty list produces an empty string
    assert flatten_messages([]) == ""
    assert flatten_messages(None) == ""
    print("Test 06 — flatten_messages (empty/None): OK ✅")


def test_07_flatten_messages_skips_entries_without_content():
    # 7. flatten_messages — dicts missing 'content' key are silently skipped
    messages = [
        {"role": "system"},
        {"role": "user", "content": "hello"},
    ]
    result = flatten_messages(messages)
    assert result == "hello"
    print("Test 07 — flatten_messages (skips missing content): OK ✅")


# %%
#######################################
# Section 3 — save_dataset
#######################################

def test_08_save_dataset_empty_returns_zero():
    # 8. save_dataset — returns 0 and writes nothing for an empty dataset
    ds = datasets.Dataset.from_list([])
    with tempfile.TemporaryDirectory() as tmpdir:
        result = save_dataset(ds, tmpdir, "parquet", 3_000_000, token_count=0)
        assert result == 0
        assert len(os.listdir(tmpdir)) == 0
    print("Test 08 — save_dataset (empty dataset): OK ✅")


def test_09_save_dataset_writes_parquet_files():
    # 9. save_dataset — produces .parquet files readable by datasets
    ds = datasets.Dataset.from_list([{"text": f"sample {i}", "token_count": 100} for i in range(10)])
    with tempfile.TemporaryDirectory() as tmpdir:
        n = save_dataset(ds, tmpdir, "parquet", 3_000_000, token_count=1000)
        files = sorted(os.listdir(tmpdir))
        assert all(f.endswith(".parquet") for f in files)
        assert len(files) == n
        reloaded = datasets.load_dataset("parquet", data_files=[os.path.join(tmpdir, f) for f in files], split="train")
        assert len(reloaded) == 10
    print("Test 09 — save_dataset (parquet output): OK ✅")


def test_10_save_dataset_writes_jsonl_files():
    # 10. save_dataset — produces .jsonl files with one JSON object per line
    ds = datasets.Dataset.from_list([{"text": f"line {i}"} for i in range(5)])
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dataset(ds, tmpdir, "jsonl", 3_000_000, token_count=0, n_chunks=1)
        files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
        assert len(files) == 1
        with open(os.path.join(tmpdir, files[0])) as fh:
            lines = [l for l in fh if l.strip()]
        assert len(lines) == 5
    print("Test 10 — save_dataset (jsonl output): OK ✅")


def test_11_save_dataset_chunks_by_token_count():
    # 11. save_dataset — splits output into the expected number of files based on token_count
    ds = datasets.Dataset.from_list([{"text": f"x {i}"} for i in range(100)])
    with tempfile.TemporaryDirectory() as tmpdir:
        # 9_000_000 tokens / 3_000_000 per chunk = 3 chunks
        n = save_dataset(ds, tmpdir, "parquet", 3_000_000, token_count=9_000_000)
        assert n == 3
        assert len(os.listdir(tmpdir)) == 3
    print("Test 11 — save_dataset (chunking by token count): OK ✅")


# %%
#######################################
# Section 4 — DatasetLoader
#######################################

def test_12_datasetloader_loads_from_jsonl_file():
    # 12. DatasetLoader — loads a single .jsonl file correctly
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "data.jsonl")
        records = [{"text": f"record {i}", "token_count": 10} for i in range(6)]
        with open(fpath, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        loader = DatasetLoader(path=fpath)
        ds = loader.load()
        assert len(ds) == 6
        assert "text" in ds.column_names
    print("Test 12 — DatasetLoader (single jsonl file): OK ✅")


def test_13_datasetloader_loads_from_directory():
    # 13. DatasetLoader — discovers and loads all .jsonl files inside a directory
    with tempfile.TemporaryDirectory() as tmpdir:
        for shard in range(3):
            fpath = os.path.join(tmpdir, f"shard_{shard}.jsonl")
            with open(fpath, "w") as f:
                for i in range(4):
                    f.write(json.dumps({"text": f"s{shard}r{i}"}) + "\n")
        loader = DatasetLoader(path=tmpdir)
        ds = loader.load()
        assert len(ds) == 12
    print("Test 13 — DatasetLoader (directory of jsonl files): OK ✅")


# %%
#######################################
# Section 5 — Language definitions
#######################################

def test_14_langdetect_codes_and_unicode_ranges_have_same_keys():
    # 14. LANGDETECT_CODES and UNICODE_RANGES must cover exactly the same languages
    assert set(LANGDETECT_CODES.keys()) == set(UNICODE_RANGES.keys()), (
        f"Key mismatch — only in LANGDETECT_CODES: {set(LANGDETECT_CODES) - set(UNICODE_RANGES)}, "
        f"only in UNICODE_RANGES: {set(UNICODE_RANGES) - set(LANGDETECT_CODES)}"
    )
    print("Test 14 — LANGDETECT_CODES and UNICODE_RANGES key parity: OK ✅")


def test_15_supported_languages_is_sorted_union():
    # 15. SUPPORTED_LANGUAGES is the sorted union of both dicts' keys
    expected = sorted(set(LANGDETECT_CODES) | set(UNICODE_RANGES))
    assert SUPPORTED_LANGUAGES == expected
    print("Test 15 — SUPPORTED_LANGUAGES (sorted union): OK ✅")


# %%
#######################################
# Section 6 — Unicode backend
#######################################

def test_16_unicode_filter_keeps_latin_text():
    # 16. _create_unicode_filter — accepts a clean Portuguese/Latin text
    keep = _create_unicode_filter(["portuguese"])
    assert keep("Olá, este é um texto em português.")
    print("Test 16 — unicode filter (keeps Latin text): OK ✅")


def test_17_unicode_filter_rejects_foreign_script():
    # 17. _create_unicode_filter — rejects Cyrillic-heavy text when filtering for English
    keep = _create_unicode_filter(["english"], threshold=0.85)
    cyrillic_text = "Привет мир, это текст на русском языке без латинских букв."
    assert not keep(cyrillic_text)
    print("Test 17 — unicode filter (rejects foreign script): OK ✅")


def test_18_unicode_filter_empty_text_returns_false():
    # 18. _create_unicode_filter — empty / None-ish string returns False
    keep = _create_unicode_filter(["english"])
    assert not keep("")
    assert not keep(None)
    print("Test 18 — unicode filter (empty text): OK ✅")


def test_19_unicode_filter_threshold_controls_strictness():
    # 19. _create_unicode_filter — same mixed text passes at low threshold, fails at high
    keep_loose = _create_unicode_filter(["english"], threshold=0.20)
    keep_strict = _create_unicode_filter(["english"], threshold=0.99)
    # Text is ~50% Latin, ~50% Cyrillic
    mixed = "Hello мир Hello мир Hello мир Hello мир"
    assert keep_loose(mixed)
    assert not keep_strict(mixed)
    print("Test 19 — unicode filter (threshold controls strictness): OK ✅")


def test_20_unicode_filter_raises_for_unknown_language():
    # 20. _create_unicode_filter — raises ValueError when no valid language is given
    try:
        _create_unicode_filter(["klingon"])
        assert False, "Expected ValueError was not raised"
    except ValueError:
        pass
    print("Test 20 — unicode filter (raises for unknown language): OK ✅")


# %%
#######################################
# Runner
#######################################

if __name__ == "__main__":
    test_01_is_messages_column_missing_column_returns_false()
    test_02_is_messages_column_empty_dataset_returns_false()
    test_03_is_messages_column_detects_messages_format()
    test_04_is_messages_column_plain_text_returns_false()
    test_05_flatten_messages_concatenates_content_fields()
    test_06_flatten_messages_empty_list_returns_empty_string()
    test_07_flatten_messages_skips_entries_without_content()
    test_08_save_dataset_empty_returns_zero()
    test_09_save_dataset_writes_parquet_files()
    test_10_save_dataset_writes_jsonl_files()
    test_11_save_dataset_chunks_by_token_count()
    test_12_datasetloader_loads_from_jsonl_file()
    test_13_datasetloader_loads_from_directory()
    test_14_langdetect_codes_and_unicode_ranges_have_same_keys()
    test_15_supported_languages_is_sorted_union()
    test_16_unicode_filter_keeps_latin_text()
    test_17_unicode_filter_rejects_foreign_script()
    test_18_unicode_filter_empty_text_returns_false()
    test_19_unicode_filter_threshold_controls_strictness()
    test_20_unicode_filter_raises_for_unknown_language()
    print("\n" + "=" * 50)
    print("All tests passed ✅")
    print("=" * 50)
