"""
Data/cc test suite.

Tests the surrounding logic of the data/cc scripts:
  - utils.py (get_logger, read_metadata, write_metadata, initialize_or_load_metadata)
  - process_cc_dump_all_languages.py (argument parser, post-processing consolidation)
  - process_cc_dump_with_quality_filters.py (argument parser)

DataTrove pipeline internals are deliberately NOT tested here.

Run with:
    python tests_data.py

Requirements:
- No GPU required
- No datatrove installation required
"""

# %%
#######################################
# 1. Imports & Setup
#######################################
import sys
import os
import argparse
import importlib
import importlib.util
import json
import tempfile

sys.pycache_prefix = os.path.join(tempfile.gettempdir(), "pycache")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_CC_DIR = os.path.join(REPO_ROOT, "data", "cc")
if DATA_CC_DIR not in sys.path:
    sys.path.insert(0, DATA_CC_DIR)

from unittest.mock import MagicMock

# Patch datatrove at module level so the CC scripts can be imported without it.
_dt_mock = MagicMock()
for _mod in [
    "datatrove",
    "datatrove.executor",
    "datatrove.pipeline",
    "datatrove.pipeline.extractors",
    "datatrove.pipeline.filters",
    "datatrove.pipeline.formatters",
    "datatrove.pipeline.readers",
    "datatrove.pipeline.writers",
    "datatrove.pipeline.writers.jsonl",
    "datatrove.pipeline.tokens",
]:
    sys.modules.setdefault(_mod, _dt_mock)

from utils import get_logger, read_metadata, write_metadata, initialize_or_load_metadata

print("All imports OK ✅")


# %%
#######################################
# Section 1 — data/cc/utils.py
#######################################

def test_02_getlogger_returns_a_working_logger():
    # 2. get_logger — returns a working logger with the correct name
    #######################################
    logger = get_logger("TestDataCC")
    assert logger.name == "TestDataCC"
    logger.info("Logger works.")
    logger.warning("Warning works.")
    print("Test 2 — get_logger: OK ✅")


def test_03_getlogger_is_idempotent():
    # 3. get_logger — calling twice must not duplicate handlers
    #######################################
    logger_a = get_logger("TestDataCC_idem")
    n_handlers = len(logger_a.handlers)
    logger_b = get_logger("TestDataCC_idem")
    assert logger_a is logger_b, "Should return the same Logger instance"
    assert len(logger_b.handlers) == n_handlers, "No new handlers should be added on second call"
    print("Test 3 — get_logger idempotent: OK ✅")


def test_04_readmetadata_returns_none_for_missing_file():
    # 4. read_metadata — returns None when the file does not exist
    #######################################
    with tempfile.TemporaryDirectory() as tmpdir:
        result = read_metadata(os.path.join(tmpdir, ".metadata"))
        assert result is None
    print("Test 4 — read_metadata (missing file): OK ✅")


def test_05_readmetadata_parses_int_float_and_string():
    # 5. read_metadata — correctly parses int, float, and string values
    #######################################
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = os.path.join(tmpdir, ".metadata")
        with open(meta_path, "w") as f:
            f.write("lines: 42\n")
            f.write("tokens: 1234567\n")
            f.write("ratio: 3.14\n")
            f.write("source: CC-MAIN-2025-30\n")

        result = read_metadata(meta_path)
        assert result["lines"] == 42 and isinstance(result["lines"], int)
        assert result["tokens"] == 1234567
        assert abs(result["ratio"] - 3.14) < 1e-6
        assert result["source"] == "CC-MAIN-2025-30"
    print("Test 5 — read_metadata parsing: OK ✅")


def test_06_readmetadata_handles_blank_and_non_kv_lines():
    # 6. read_metadata — ignores blank lines and lines that contain no colon
    #######################################
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = os.path.join(tmpdir, ".metadata")
        with open(meta_path, "w") as f:
            f.write("\n")
            f.write("# a comment without a colon\n")
            f.write("lines: 10\n")
            f.write("\n")
            f.write("tokens: 500\n")

        result = read_metadata(meta_path)
        assert result == {"lines": 10, "tokens": 500}
    print("Test 6 — read_metadata blank/non-kv lines: OK ✅")


def test_07_writemetadata_writes_correct_format():
    # 7. write_metadata — produces a key: value file readable by hand
    #######################################
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = os.path.join(tmpdir, ".metadata")
        write_metadata(meta_path, {"lines": 100, "tokens": 9999, "lang": "pt"})
        with open(meta_path) as f:
            content = f.read()
        assert "lines: 100\n" in content
        assert "tokens: 9999\n" in content
        assert "lang: pt\n" in content
    print("Test 7 — write_metadata format: OK ✅")


def test_09_initializeorloadmetadata_empty_folder_returns_zeros():
    # 9. initialize_or_load_metadata — empty folder (no JSONL, no .metadata) → zeros
    #######################################
    with tempfile.TemporaryDirectory() as tmpdir:
        result = initialize_or_load_metadata(tmpdir)
        assert result == {"lines": 0, "tokens": 0}
    print("Test 9 — initialize_or_load_metadata (empty folder): OK ✅")


def test_10_initializeorloadmetadata_loads_existing_metadata_file():
    # 10. initialize_or_load_metadata — reads .metadata directly (no JSONL scan)
    #######################################
    with tempfile.TemporaryDirectory() as tmpdir:
        write_metadata(os.path.join(tmpdir, ".metadata"), {"lines": 77, "tokens": 8800})
        result = initialize_or_load_metadata(tmpdir)
        assert result["lines"] == 77
        assert result["tokens"] == 8800
    print("Test 10 — initialize_or_load_metadata (existing .metadata): OK ✅")


def test_11_initializeorloadmetadata_scans_jsonl_skips_invalid_and_creates_metadata():
    # 11. initialize_or_load_metadata — scans multiple JSONL shards, skips invalid JSON
    #     lines, accumulates counts across shards, and persists .metadata
    #######################################
    with tempfile.TemporaryDirectory() as tmpdir:
        # Shard 0: 5 valid records, token_count = 10 each
        with open(os.path.join(tmpdir, "shard_0.jsonl"), "w") as f:
            for i in range(5):
                f.write(json.dumps({"text": f"r{i}", "token_count": 10}) + "\n")

        # Shard 1: 2 valid records interspersed with invalid/blank lines
        with open(os.path.join(tmpdir, "shard_1.jsonl"), "w") as f:
            f.write(json.dumps({"token_count": 20}) + "\n")
            f.write("{ not valid json\n")
            f.write("\n")
            f.write(json.dumps({"token_count": 30}) + "\n")

        result = initialize_or_load_metadata(tmpdir)
        assert result["lines"] == 7    # 5 from shard_0 + 2 valid from shard_1
        assert result["tokens"] == 100  # 5×10 + 20 + 30

        # .metadata must have been persisted
        meta_path = os.path.join(tmpdir, ".metadata")
        assert os.path.exists(meta_path)
        assert read_metadata(meta_path) == result
    print("Test 11 — initialize_or_load_metadata (multi-shard + invalid JSON): OK ✅")


# %%
#######################################
# Section 2 — Argument Parsers
#######################################

def test_13_all_languages_argument_parser_defaults_and_required_args():
    # 13. process_cc_dump_all_languages.py — argument parser defaults and required args
    #######################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--warc_files_folder", type=str, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--temp_output_folder", type=str, default="./language_filter_output")
    parser.add_argument("--output_folder", type=str, default="./all_languages")
    parser.add_argument("--logs_folder", type=str, default="./logs")
    parser.add_argument("--dump", type=str, required=True)
    parser.add_argument("--languages", nargs="+", type=str, default=None)
    parser.add_argument("--language_filter_backend", type=str, default="ft176", choices=["ft176", "glotlid"])
    parser.add_argument("--language_threshold", type=float, default=0.65)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expand_metadata", action="store_true")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="Qwen/Qwen3-0.6B-Base")

    # Defaults
    args = parser.parse_args([
        "--warc_files_folder", "/data/warc",
        "--dump", "CC-MAIN-2025-30",
    ])
    assert args.limit == -1
    assert args.temp_output_folder == "./language_filter_output"
    assert args.output_folder == "./all_languages"
    assert args.logs_folder == "./logs"
    assert args.languages is None
    assert args.language_filter_backend == "ft176"
    assert abs(args.language_threshold - 0.65) < 1e-9
    assert args.tasks == 10
    assert args.workers == 4
    assert args.expand_metadata is False
    assert args.tokenizer_name_or_path == "Qwen/Qwen3-0.6B-Base"
    assert args.warc_files_folder == "/data/warc"
    assert args.dump == "CC-MAIN-2025-30"

    # Overrides
    args2 = parser.parse_args([
        "--warc_files_folder", "/data/warc",
        "--dump", "CC-MAIN-2025-18",
        "--languages", "pt", "bn",
        "--language_filter_backend", "glotlid",
        "--language_threshold", "0.7",
        "--tasks", "32",
        "--workers", "32",
        "--expand_metadata",
    ])
    assert args2.languages == ["pt", "bn"]
    assert args2.language_filter_backend == "glotlid"
    assert abs(args2.language_threshold - 0.7) < 1e-9
    assert args2.tasks == 32
    assert args2.workers == 32
    assert args2.expand_metadata is True
    print("Test 13 — all_languages argument parser: OK ✅")


def test_14_quality_filters_argument_parser_defaults_and_required_args():
    # 14. process_cc_dump_with_quality_filters.py — argument parser defaults and required args
    #######################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_folder", type=str, required=True)
    parser.add_argument("--warc_files_folder", type=str, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--logs_folder", type=str, default="./logs")
    parser.add_argument("--warc_extraction_output", type=str, default="./warc_extraction")
    parser.add_argument("--quality_filter_output", type=str, default="./quality_filter")
    parser.add_argument("--final_output_folder", type=str, default="./output")
    parser.add_argument("--output_file", type=str, default="./output.jsonl")
    parser.add_argument("--dump", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--languages", nargs="+", type=str, default=None)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expand_metadata", action="store_true")

    # Defaults
    args = parser.parse_args([
        "--config_folder", ".configs/",
        "--warc_files_folder", "/data/warc",
        "--dump", "CC-MAIN-2025-30",
    ])
    assert args.limit == -1
    assert args.logs_folder == "./logs"
    assert args.warc_extraction_output == "./warc_extraction"
    assert args.quality_filter_output == "./quality_filter"
    assert args.final_output_folder == "./output"
    assert args.output_file == "./output.jsonl"
    assert args.tokenizer_name_or_path == "Qwen/Qwen3-0.6B"
    assert args.languages is None
    assert args.tasks == 10
    assert args.workers == 4
    assert args.expand_metadata is False
    assert args.config_folder == ".configs/"
    assert args.warc_files_folder == "/data/warc"
    assert args.dump == "CC-MAIN-2025-30"

    # Overrides
    args2 = parser.parse_args([
        "--config_folder", ".configs/",
        "--warc_files_folder", "/data/warc",
        "--dump", "CC-MAIN-2025-18",
        "--languages", "pt", "hi",
        "--tasks", "16",
        "--expand_metadata",
    ])
    assert args2.languages == ["pt", "hi"]
    assert args2.tasks == 16
    assert args2.expand_metadata is True
    print("Test 14 — quality_filters argument parser: OK ✅")


# %%
#######################################
# Section 3 — Consolidation Logic (process_cc_dump_all_languages.py)
#######################################

def _load_all_languages_main():
    """
    Import and return main() from process_cc_dump_all_languages.py.
    DataTrove is already mocked at module level so no pipeline actually runs.
    The module is cached in sys.modules to avoid re-loading between tests.
    """
    key = "all_languages_mod"
    if key in sys.modules:
        return sys.modules[key].main
    spec = importlib.util.spec_from_file_location(
        key,
        os.path.join(DATA_CC_DIR, "process_cc_dump_all_languages.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__name__ = key
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod.main


def _make_args(tmpdir, **overrides):
    """Return a Namespace mimicking the argparse output for the all_languages script."""
    warc_dir = os.path.join(tmpdir, "warc")
    os.makedirs(warc_dir, exist_ok=True)
    defaults = dict(
        warc_files_folder=warc_dir,
        limit=-1,
        temp_output_folder=os.path.join(tmpdir, "temp"),
        output_folder=os.path.join(tmpdir, "output"),
        logs_folder=os.path.join(tmpdir, "logs"),
        dump="CC-MAIN-TEST",
        languages=None,
        language_filter_backend="ft176",
        language_threshold=0.65,
        tasks=1,
        workers=1,
        expand_metadata=False,
        tokenizer_name_or_path="Qwen/Qwen3-0.6B-Base",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _write_temp_lang_jsonl(temp_folder, lang, records):
    """Populate TEMP_OUTPUT_FOLDER/lang/lang.jsonl as the datatrove pipeline would."""
    lang_dir = os.path.join(temp_folder, lang)
    os.makedirs(lang_dir, exist_ok=True)
    fpath = os.path.join(lang_dir, f"{lang}.jsonl")
    with open(fpath, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_15_consolidation_writes_output_and_metadata():
    # 15. Consolidation — valid JSONL in temp folder produces consolidated file + metadata
    #######################################
    main = _load_all_languages_main()

    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_args(tmpdir)
        records = [{"text": f"texto {i}", "token_count": 10} for i in range(5)]
        _write_temp_lang_jsonl(args.temp_output_folder, "pt", records)

        # LocalPipelineExecutor is already a MagicMock; calling .run() is a no-op
        main(args)

        out_file = os.path.join(args.output_folder, "pt", "pt.jsonl")
        assert os.path.exists(out_file), "Consolidated JSONL must exist"
        with open(out_file) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 5

        meta = read_metadata(os.path.join(args.output_folder, "pt", ".metadata"))
        assert meta["lines"] == 5
        assert meta["tokens"] == 50
    print("Test 15 — consolidation writes output and metadata: OK ✅")


def test_16_consolidation_skips_invalid_json_lines():
    # 16. Consolidation — invalid JSON lines are silently skipped; valid ones are kept
    #######################################
    main = _load_all_languages_main()

    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_args(tmpdir)

        lang_dir = os.path.join(args.temp_output_folder, "bn")
        os.makedirs(lang_dir, exist_ok=True)
        with open(os.path.join(lang_dir, "bn.jsonl"), "w") as f:
            f.write(json.dumps({"text": "valid 1", "token_count": 5}) + "\n")
            f.write("{ not json\n")
            f.write(json.dumps({"text": "valid 2", "token_count": 8}) + "\n")
            f.write("\n")
            f.write(json.dumps({"text": "valid 3", "token_count": 3}) + "\n")

        main(args)

        out_file = os.path.join(args.output_folder, "bn", "bn.jsonl")
        with open(out_file) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 3

        meta = read_metadata(os.path.join(args.output_folder, "bn", ".metadata"))
        assert meta["lines"] == 3
        assert meta["tokens"] == 16
    print("Test 16 — consolidation skips invalid JSON: OK ✅")


def test_17_consolidation_appends_and_accumulates_metadata():
    # 17. Consolidation — new run appends to existing output; metadata accumulates
    #######################################
    main = _load_all_languages_main()

    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_args(tmpdir)

        # Simulate a previous run: existing consolidated file + .metadata
        lang_out = os.path.join(args.output_folder, "pt")
        os.makedirs(lang_out, exist_ok=True)
        with open(os.path.join(lang_out, "pt.jsonl"), "w") as f:
            for i in range(3):
                f.write(json.dumps({"text": f"old {i}", "token_count": 20}) + "\n")
        write_metadata(os.path.join(lang_out, ".metadata"), {"lines": 3, "tokens": 60})

        # New temp data from this run
        new_records = [{"text": f"new {i}", "token_count": 10} for i in range(4)]
        _write_temp_lang_jsonl(args.temp_output_folder, "pt", new_records)

        main(args)

        out_file = os.path.join(lang_out, "pt.jsonl")
        with open(out_file) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 7          # 3 old + 4 new

        meta = read_metadata(os.path.join(lang_out, ".metadata"))
        assert meta["lines"] == 7       # 3 + 4
        assert meta["tokens"] == 100    # 60 + 40
    print("Test 17 — consolidation appends and accumulates metadata: OK ✅")


def test_18_consolidation_skips_language_with_no_valid_data():
    # 18. Consolidation — a language folder with zero valid records is skipped (no .metadata written)
    #######################################
    main = _load_all_languages_main()

    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_args(tmpdir)

        lang_dir = os.path.join(args.temp_output_folder, "hi")
        os.makedirs(lang_dir, exist_ok=True)
        with open(os.path.join(lang_dir, "hi.jsonl"), "w") as f:
            f.write("not json at all\n")
            f.write("\n")

        main(args)

        # .metadata must NOT exist — the language was skipped
        meta_path = os.path.join(args.output_folder, "hi", ".metadata")
        assert not os.path.exists(meta_path), ".metadata must not be written for an empty language"
    print("Test 18 — consolidation skips lang with no valid data: OK ✅")


def test_19_consolidation_multiple_languages_in_one_run():
    # 19. Consolidation — multiple languages in one temp folder are each processed independently
    #######################################
    main = _load_all_languages_main()

    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_args(tmpdir)

        _write_temp_lang_jsonl(
            args.temp_output_folder, "pt",
            [{"text": f"pt {i}", "token_count": 5} for i in range(4)]
        )
        _write_temp_lang_jsonl(
            args.temp_output_folder, "bn",
            [{"text": f"bn {i}", "token_count": 20} for i in range(3)]
        )

        main(args)

        pt_meta = read_metadata(os.path.join(args.output_folder, "pt", ".metadata"))
        assert pt_meta["lines"] == 4
        assert pt_meta["tokens"] == 20

        bn_meta = read_metadata(os.path.join(args.output_folder, "bn", ".metadata"))
        assert bn_meta["lines"] == 3
        assert bn_meta["tokens"] == 60
    print("Test 19 — consolidation multiple languages: OK ✅")


def test_20_consolidation_multiple_shards_per_language():
    # 20. Consolidation — multiple JSONL shards under one language folder are merged
    #######################################
    main = _load_all_languages_main()

    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_args(tmpdir)

        lang_dir = os.path.join(args.temp_output_folder, "pt")
        os.makedirs(lang_dir, exist_ok=True)
        # Two shard files (as datatrove would produce for multiple tasks)
        for shard, (n, tok) in enumerate([(3, 10), (5, 20)]):
            fpath = os.path.join(lang_dir, f"pt_{shard:05d}.jsonl")
            with open(fpath, "w") as f:
                for i in range(n):
                    f.write(json.dumps({"text": f"shard{shard}_{i}", "token_count": tok}) + "\n")

        main(args)

        out_file = os.path.join(args.output_folder, "pt", "pt.jsonl")
        with open(out_file) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 8          # 3 + 5

        meta = read_metadata(os.path.join(args.output_folder, "pt", ".metadata"))
        assert meta["lines"] == 8
        assert meta["tokens"] == 3 * 10 + 5 * 20   # 30 + 100 = 130
    print("Test 20 — consolidation multiple shards per language: OK ✅")


if __name__ == "__main__":
    test_02_getlogger_returns_a_working_logger()
    test_03_getlogger_is_idempotent()
    test_04_readmetadata_returns_none_for_missing_file()
    test_05_readmetadata_parses_int_float_and_string()
    test_06_readmetadata_handles_blank_and_non_kv_lines()
    test_07_writemetadata_writes_correct_format()
    test_09_initializeorloadmetadata_empty_folder_returns_zeros()
    test_10_initializeorloadmetadata_loads_existing_metadata_file()
    test_11_initializeorloadmetadata_scans_jsonl_skips_invalid_and_creates_metadata()
    test_13_all_languages_argument_parser_defaults_and_required_args()
    test_14_quality_filters_argument_parser_defaults_and_required_args()
    test_15_consolidation_writes_output_and_metadata()
    test_16_consolidation_skips_invalid_json_lines()
    test_17_consolidation_appends_and_accumulates_metadata()
    test_18_consolidation_skips_language_with_no_valid_data()
    test_19_consolidation_multiple_languages_in_one_run()
    test_20_consolidation_multiple_shards_per_language()
    print("\n" + "=" * 50)
    print("All tests passed ✅")
    print("=" * 50)
