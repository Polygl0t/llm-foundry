"""
Data/preprocess test suite.

Tests the surrounding logic of the data/preprocess.py script:
  - built-in parser and external parser resolution
  - RoutingWriter JSONL/parquet output behavior
  - main() orchestration and metadata generation

DataTrove pipeline internals are deliberately NOT tested here.

Run with:
    python tests/tests_data.py

Requirements:
- pyarrow
- No datatrove installation required
"""

import argparse
import json
import os
import sys
import tempfile
import types
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

sys.pycache_prefix = os.path.join(tempfile.gettempdir(), "pycache")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)


class StubPipelineStep:
    def __init__(self):
        self._stat_totals = {}

    def stat_update(self, name, value=1):
        self._stat_totals[name] = self._stat_totals.get(name, 0) + value


class StubParquetReader:
    def __init__(self, data_folder, glob_pattern):
        self.data_folder = data_folder
        self.glob_pattern = glob_pattern


class StubLocalPipelineExecutor:
    def __init__(self, pipeline, tasks, workers, logging_dir):
        self.pipeline = pipeline
        self.tasks = tasks
        self.workers = workers
        self.logging_dir = logging_dir

    def run(self):
        return SimpleNamespace(stats=[])


def _install_datatrove_stubs():
    datatrove_mod = types.ModuleType("datatrove")
    executor_mod = types.ModuleType("datatrove.executor")
    pipeline_mod = types.ModuleType("datatrove.pipeline")
    base_mod = types.ModuleType("datatrove.pipeline.base")
    readers_mod = types.ModuleType("datatrove.pipeline.readers")

    executor_mod.LocalPipelineExecutor = StubLocalPipelineExecutor
    base_mod.PipelineStep = StubPipelineStep
    readers_mod.ParquetReader = StubParquetReader

    datatrove_mod.executor = executor_mod
    datatrove_mod.pipeline = pipeline_mod
    pipeline_mod.base = base_mod
    pipeline_mod.readers = readers_mod

    sys.modules.setdefault("datatrove", datatrove_mod)
    sys.modules.setdefault("datatrove.executor", executor_mod)
    sys.modules.setdefault("datatrove.pipeline", pipeline_mod)
    sys.modules.setdefault("datatrove.pipeline.base", base_mod)
    sys.modules.setdefault("datatrove.pipeline.readers", readers_mod)


_install_datatrove_stubs()

import preprocess
from utils import ParseResult, document_to_row

print("All imports OK ✅")


class FakeDocument:
    def __init__(self, text, doc_id, metadata=None):
        self.text = text
        self.id = doc_id
        self.metadata = metadata or {}


def _make_args(tmpdir, **overrides):
    datasets_dir = os.path.join(tmpdir, "datasets")
    output_dir = os.path.join(tmpdir, "output")
    logs_folder = os.path.join(tmpdir, "logs")
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    defaults = dict(
        datasets_dir=datasets_dir,
        output_dir=output_dir,
        output_type="jsonl",
        token_count_column="token_count",
        parser_path=None,
        parser_config=None,
        default_subset_name="default",
        stratify_by_column=None,
        write_batch_size=2,
        tasks=1,
        workers=1,
        logs_folder=logs_folder,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _write_parquet_rows(directory, rows, file_name="00000.parquet"):
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, file_name)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, file_path)
    return file_path


def _read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_metadata(file_path):
    metadata = {}
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            if value.isdigit():
                metadata[key] = int(value)
            else:
                metadata[key] = value
    return metadata


def _pipeline_stats_from_writer(writer):
    routing_stats = {
        name: SimpleNamespace(total=total)
        for name, total in writer._stat_totals.items()
    }
    routing = SimpleNamespace(name="RoutingWriter", stats=routing_stats)
    return SimpleNamespace(stats=[routing])


#######################################
# Section 1 — Parser Resolution
#######################################

def test_01_apply_stratification_overrides_subset_when_enabled():
    args = argparse.Namespace(stratify_by_column="language", default_subset_name="fallback")
    doc = FakeDocument("texto", "doc-1", {"language": "pt"})
    parsed = ParseResult("ignored", {"text": "texto", "id": "doc-1"})

    result = preprocess.apply_stratification(parsed, doc, args)

    assert result.subset == "pt"
    assert result.row == parsed.row
    print("Test 1 — apply_stratification override: OK ✅")


def test_02_build_builtin_parser_without_stratification_uses_default_subset():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_args(tmpdir, default_subset_name="all")
        parser_fn, expected_subsets = preprocess.build_builtin_parser(args)
        doc = FakeDocument("texto", "doc-2", {"language": "bn", "token_count": 7})

        result = parser_fn(doc)

        assert result == ParseResult("all", document_to_row(doc))
        assert expected_subsets == ["all"]
    print("Test 2 — built-in parser default subset: OK ✅")


def test_03_build_builtin_parser_with_stratification_discovers_parquet_values():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_args(tmpdir, stratify_by_column="language", default_subset_name="other")
        _write_parquet_rows(
            args.datasets_dir,
            [
                {"id": "1", "text": "ola", "language": "pt", "token_count": 3},
                {"id": "2", "text": "namaskar", "language": "bn", "token_count": 5},
                {"id": "3", "text": "hello", "language": "pt", "token_count": 7},
            ],
        )
        parser_fn, expected_subsets = preprocess.build_builtin_parser(args)
        doc = FakeDocument("ola", "doc-3", {"language": "pt", "token_count": 3})

        result = parser_fn(doc)

        assert expected_subsets == ["bn", "pt"]
        assert result == ParseResult("pt", document_to_row(doc))
    print("Test 3 — built-in parser stratification: OK ✅")


def test_04_build_active_parser_loads_add_uuid_parser():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_args(
            tmpdir,
            parser_path=os.path.join(DATA_DIR, "parsers", "add_uuid_parser.py"),
            parser_config={"uuid_column": "sample_uuid"},
        )
        parser_fn, parser_name, expected_subsets = preprocess.build_active_parser(args)
        doc = FakeDocument("texto", "doc-4", {"language": "pt", "token_count": 11})

        result = parser_fn(doc)

        assert parser_name == "add-uuid"
        assert expected_subsets == ["default"]
        assert result.subset == "default"
        assert result.row["sample_uuid"]
        assert result.row["text"] == "texto"
        assert result.row["id"] == "doc-4"
    print("Test 4 — active parser add_uuid: OK ✅")


def test_05_build_active_parser_combines_filtering_and_stratification():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_args(
            tmpdir,
            parser_path=os.path.join(DATA_DIR, "parsers", "score_threshold_filter_parser.py"),
            parser_config={"score_column": "edu_int_score", "minimum_score": 3},
            stratify_by_column="language",
        )
        _write_parquet_rows(
            args.datasets_dir,
            [
                {"id": "1", "text": "ola", "language": "pt", "edu_int_score": 4},
                {"id": "2", "text": "namaskar", "language": "bn", "edu_int_score": 5},
            ],
        )
        parser_fn, parser_name, expected_subsets = preprocess.build_active_parser(args)
        keep_doc = FakeDocument(
            "namaskar",
            "doc-5",
            {"language": "bn", "edu_int_score": 5, "token_count": 9},
        )
        drop_doc = FakeDocument(
            "short",
            "doc-6",
            {"language": "pt", "edu_int_score": 1, "token_count": 2},
        )

        keep_result = parser_fn(keep_doc)
        drop_result = parser_fn(drop_doc)

        assert parser_name == "score-threshold-filter"
        assert expected_subsets == ["bn", "pt"]
        assert keep_result == ParseResult("bn", document_to_row(keep_doc))
        assert drop_result is None
    print("Test 5 — active parser filter + stratification: OK ✅")


#######################################
# Section 2 — RoutingWriter
#######################################

def test_06_routing_writer_writes_jsonl_and_tracks_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "jsonl_output")

        def parse_document(doc):
            if doc.metadata.get("drop"):
                return None
            return ParseResult(doc.metadata["subset"], {"id": doc.id, "text": doc.text})

        writer = preprocess.RoutingWriter(
            parse_document=parse_document,
            output_dir=output_dir,
            output_type="jsonl",
            token_count_column="token_count",
            write_batch_size=2,
        )
        docs = [
            FakeDocument("ola", "doc-1", {"subset": "pt", "token_count": 5}),
            FakeDocument("namaskar", "doc-2", {"subset": "pt", "token_count": 7}),
            FakeDocument("drop", "doc-3", {"subset": "pt", "token_count": 3, "drop": True}),
        ]

        list(writer.run(iter(docs), rank=3, world_size=1))

        written = _read_jsonl(os.path.join(output_dir, "pt", "00003.jsonl"))
        assert written == [
            {"id": "doc-1", "text": "ola"},
            {"id": "doc-2", "text": "namaskar"},
        ]
        assert writer._stat_totals["pt_documents"] == 2
        assert writer._stat_totals["pt_tokens"] == 12
        assert writer._stat_totals["filtered_documents"] == 1
    print("Test 6 — RoutingWriter JSONL: OK ✅")


def test_07_routing_writer_writes_parquet_rows():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "parquet_output")

        def parse_document(doc):
            return ParseResult(
                doc.metadata["subset"],
                {"id": doc.id, "text": doc.text, "token_count": doc.metadata["token_count"]},
            )

        writer = preprocess.RoutingWriter(
            parse_document=parse_document,
            output_dir=output_dir,
            output_type="parquet",
            token_count_column="token_count",
            write_batch_size=1,
        )
        docs = [
            FakeDocument("ola", "doc-1", {"subset": "pt", "token_count": 5}),
            FakeDocument("namaskar", "doc-2", {"subset": "bn", "token_count": 7}),
        ]

        list(writer.run(iter(docs), rank=4, world_size=1))

        pt_rows = pq.read_table(os.path.join(output_dir, "pt", "00004.parquet")).to_pylist()
        bn_rows = pq.read_table(os.path.join(output_dir, "bn", "00004.parquet")).to_pylist()
        assert pt_rows == [{"id": "doc-1", "text": "ola", "token_count": 5}]
        assert bn_rows == [{"id": "doc-2", "text": "namaskar", "token_count": 7}]
    print("Test 7 — RoutingWriter parquet: OK ✅")


#######################################
# Section 3 — main()
#######################################

def test_08_main_runs_writer_and_emits_subset_metadata():
    class FakeExecutor:
        docs = []
        last_instance = None

        def __init__(self, pipeline, tasks, workers, logging_dir):
            self.pipeline = pipeline
            self.tasks = tasks
            self.workers = workers
            self.logging_dir = logging_dir
            FakeExecutor.last_instance = self

        def run(self):
            reader, writer = self.pipeline
            assert isinstance(reader, StubParquetReader)
            list(writer.run(iter(self.docs), rank=0, world_size=1))
            return _pipeline_stats_from_writer(writer)

    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_args(
            tmpdir,
            parser_path=os.path.join(DATA_DIR, "parsers", "score_threshold_filter_parser.py"),
            parser_config=json.dumps({"score_column": "edu_int_score", "minimum_score": 3}),
            stratify_by_column="language",
            output_type="jsonl",
            tasks=3,
            workers=2,
        )
        _write_parquet_rows(
            args.datasets_dir,
            [
                {"id": "1", "text": "ola", "language": "pt", "edu_int_score": 5},
                {"id": "2", "text": "namaskar", "language": "bn", "edu_int_score": 4},
            ],
        )
        FakeExecutor.docs = [
            FakeDocument("ola", "doc-1", {"language": "pt", "edu_int_score": 5, "token_count": 5}),
            FakeDocument("skip", "doc-2", {"language": "bn", "edu_int_score": 1, "token_count": 8}),
            FakeDocument("namaskar", "doc-3", {"language": "bn", "edu_int_score": 4, "token_count": 7}),
        ]

        with patch.object(preprocess, "LocalPipelineExecutor", FakeExecutor):
            preprocess.main(args)

        assert FakeExecutor.last_instance is not None
        assert FakeExecutor.last_instance.tasks == 3
        assert FakeExecutor.last_instance.workers == 2
        assert FakeExecutor.last_instance.logging_dir == args.logs_folder

        pt_rows = _read_jsonl(os.path.join(args.output_dir, "pt", "00000.jsonl"))
        bn_rows = _read_jsonl(os.path.join(args.output_dir, "bn", "00000.jsonl"))
        assert pt_rows == [{"text": "ola", "id": "doc-1", "language": "pt", "edu_int_score": 5, "token_count": 5}]
        assert bn_rows == [{"text": "namaskar", "id": "doc-3", "language": "bn", "edu_int_score": 4, "token_count": 7}]

        pt_meta = _read_metadata(os.path.join(args.output_dir, "pt", ".metadata"))
        bn_meta = _read_metadata(os.path.join(args.output_dir, "bn", ".metadata"))
        assert pt_meta["samples"] == 1
        assert pt_meta["tokens"] == 5
        assert pt_meta["chunks"] == 1
        assert pt_meta["subset"] == "pt"
        assert bn_meta["samples"] == 1
        assert bn_meta["tokens"] == 7
        assert bn_meta["chunks"] == 1
        assert bn_meta["subset"] == "bn"
    print("Test 8 — main orchestration + metadata: OK ✅")


if __name__ == "__main__":
    test_01_apply_stratification_overrides_subset_when_enabled()
    test_02_build_builtin_parser_without_stratification_uses_default_subset()
    test_03_build_builtin_parser_with_stratification_discovers_parquet_values()
    test_04_build_active_parser_loads_add_uuid_parser()
    test_05_build_active_parser_combines_filtering_and_stratification()
    test_06_routing_writer_writes_jsonl_and_tracks_stats()
    test_07_routing_writer_writes_parquet_rows()
    test_08_main_runs_writer_and_emits_subset_metadata()
    print("\n" + "=" * 50)
    print("All tests passed ✅")
    print("=" * 50)
