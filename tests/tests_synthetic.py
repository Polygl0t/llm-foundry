"""
Synthetic generation test suite.

Run with:
    python tests_synthetic.py

Requirements:
- torch
- transformers
- datasets
"""

import json
import importlib
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.pycache_prefix = os.path.join(tempfile.gettempdir(), "pycache")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SYNTHETIC_DIR = os.path.join(REPO_ROOT, "synthetic")
if SYNTHETIC_DIR not in sys.path:
    sys.path.insert(0, SYNTHETIC_DIR)

# Patch heavyweight optional imports so utility tests can run on CPU-only machines.
sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("datatrove", MagicMock())
sys.modules.setdefault("datatrove.utils", MagicMock())
sys.modules.setdefault("datatrove.utils.logging", MagicMock(logger=MagicMock()))

from utils import (  # noqa: E402
    DatasetLoader,
    chunk_text,
    constitutional_generation,
    detect_failure_reason,
    get_nvidia_smi_vram,
    get_starting_row,
    run_cai_rollouts,
    run_rollouts,
    save_cai_sample,
    save_samples,
    setup_triton_cache,
)

print("All imports OK ✅")


def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _mock_word_tokenizer():
    tokenizer = MagicMock()

    def tokenize(text, **kwargs):
        result = MagicMock()
        result.input_ids = list(range(len(text.split())))
        return result

    tokenizer.side_effect = tokenize
    tokenizer.__call__ = tokenize
    tokenizer.decode = lambda ids, **kwargs: " ".join(f"w{i}" for i in ids)
    return tokenizer


def test_detect_failure_reason_handles_known_failures_and_empty_inputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        cases = {
            "oom.log": ("torch.OutOfMemoryError: CUDA out of memory\n", "OOM"),
            "timeout.log": ("Running...\nCANCELLED DUE TO TIME LIMIT\n", "timeout"),
            "server.log": ("Failed to start VLLMServer server\n", "server_fail"),
            "clean.log": ("Generation complete\n", None),
            "empty.log": ("", None),
        }
        for filename, (content, expected) in cases.items():
            path = Path(tmpdir) / filename
            path.write_text(content)
            assert detect_failure_reason(path) == expected

        large_log = Path(tmpdir) / "large.log"
        large_log.write_text("x" * 200_000 + "\nOutOfMemoryError\n")
        assert detect_failure_reason(large_log) == "OOM"
        assert detect_failure_reason(Path(tmpdir) / "missing.log") is None
        assert detect_failure_reason(None) is None
    print("Test 1 — detect_failure_reason: OK ✅")


def test_get_starting_row_resumes_from_valid_rows_and_ignores_bad_lines():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert get_starting_row("/nonexistent/file.jsonl", row_start=10) == 10
        assert get_starting_row("/nonexistent/file.jsonl", row_start=0) == 0
        assert get_starting_row(os.path.join(tmpdir, "missing.jsonl"), row_start=None) == 0

        progress_path = os.path.join(tmpdir, "progress.jsonl")
        with open(progress_path, "w", encoding="utf-8") as handle:
            handle.write("not json at all\n")
            for row in [0, 5, 3, 10, 7]:
                handle.write(json.dumps({"row": row}) + "\n")
            handle.write("{bad json\n")

        assert get_starting_row(progress_path, row_start=None) == 11

        empty_path = os.path.join(tmpdir, "empty.jsonl")
        open(empty_path, "w", encoding="utf-8").close()
        assert get_starting_row(empty_path, row_start=None) == 0
    print("Test 2 — get_starting_row: OK ✅")


def test_save_samples_appends_jsonl_with_chunk_and_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "output.jsonl")

        save_samples(output_path, row=0, seed_text="Hello", rollouts=["One"], metadata={})
        save_samples(
            output_path,
            row=1,
            seed_text="Chunked",
            rollouts=["Two", "Three"],
            metadata={"source": "test"},
            chunk=3,
        )

        records = _read_jsonl(output_path)
        assert records == [
            {"row": 0, "seed_text": "Hello", "rollouts": ["One"], "chunk": None, "metadata": {}},
            {
                "row": 1,
                "seed_text": "Chunked",
                "rollouts": ["Two", "Three"],
                "chunk": 3,
                "metadata": {"source": "test"},
            },
        ]
        assert get_starting_row(output_path, row_start=None) == 2
    print("Test 3 — save_samples: OK ✅")


def test_save_cai_sample_preserves_nested_results_metadata_and_unicode():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "cai_output.jsonl")
        result = {
            "initial_responses": ["Ola"],
            "final_responses": ["Olá, tudo ótimo!"],
            "critiques": [["Add accents"]],
            "revisions": [["Olá, tudo ótimo!"]],
        }

        save_cai_sample(output_path, row=4, instruction="Cumprimente.", cai_result=result, metadata={"lang": "pt"})

        record = _read_jsonl(output_path)[0]
        assert record == {
            "row": 4,
            "instruction": "Cumprimente.",
            "initial_responses": ["Ola"],
            "final_responses": ["Olá, tudo ótimo!"],
            "critiques": [["Add accents"]],
            "revisions": [["Olá, tudo ótimo!"]],
            "metadata": {"lang": "pt"},
        }
    print("Test 4 — save_cai_sample: OK ✅")


def test_chunk_text_splits_on_token_boundaries_and_can_keep_first_chunk_only():
    tokenizer = _mock_word_tokenizer()

    assert chunk_text("one two three", tokenizer, max_chunk_size=10, chunk_once=False) == ["w0 w1 w2"]
    assert chunk_text(" ".join(f"word{i}" for i in range(11)), tokenizer, 10, False) == [
        "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9",
        "w10",
    ]
    assert chunk_text(" ".join(f"word{i}" for i in range(20)), tokenizer, 5, True) == ["w0 w1 w2 w3 w4"]
    print("Test 5 — chunk_text: OK ✅")


def test_setup_triton_cache_sets_rank_directory_and_removes_stale_files():
    original_env = {key: os.environ.get(key) for key in ["TRITON_CACHE_DIR", "SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES"]}
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = os.path.join(tmpdir, "triton")
            os.environ["TRITON_CACHE_DIR"] = cache_root
            os.environ["SLURM_JOB_ID"] = "99"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

            rank_dir = os.path.join(cache_root, "99", "rank_0-1")
            os.makedirs(rank_dir, exist_ok=True)
            stale_file = os.path.join(rank_dir, "stale_kernel.so")
            fresh_file = os.path.join(rank_dir, "fresh_kernel.so")
            Path(stale_file).write_text("old data")
            Path(fresh_file).write_text("new data")
            old_time = time.time() - 7200
            os.utime(stale_file, (old_time, old_time))

            setup_triton_cache()

            assert os.environ["TRITON_CACHE_DIR"] == rank_dir
            assert os.path.isdir(rank_dir)
            assert not os.path.exists(stale_file)
            assert os.path.exists(fresh_file)
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("Test 6 — setup_triton_cache: OK ✅")


def test_dataset_loader_reads_local_jsonl_files_and_directories():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "data.jsonl")
        _write_jsonl(file_path, [{"text": f"Sample {index}", "label": index} for index in range(3)])

        file_dataset = DatasetLoader(path=file_path, cache_dir=tmpdir).load()
        assert len(file_dataset) == 3
        assert file_dataset[0]["text"] == "Sample 0"
        assert "label" in file_dataset.column_names

        data_dir = os.path.join(tmpdir, "dataset_dir")
        os.makedirs(data_dir)
        for shard in range(2):
            _write_jsonl(os.path.join(data_dir, f"shard_{shard}.jsonl"), [{"text": f"s{shard}_{i}"} for i in range(2)])

        directory_dataset = DatasetLoader(path=data_dir, cache_dir=tmpdir).load()
        assert len(directory_dataset) == 4
    print("Test 7 — DatasetLoader local jsonl/directory: OK ✅")


def test_dataset_loader_shuffle_is_seeded_and_optional():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "shuffle_data.jsonl")
        _write_jsonl(file_path, [{"text": f"Item {index}", "idx": index} for index in range(20)])

        seeded_a = DatasetLoader(path=file_path, seed=42, cache_dir=tmpdir).load()
        seeded_b = DatasetLoader(path=file_path, seed=42, cache_dir=tmpdir).load()
        different_seed = DatasetLoader(path=file_path, seed=123, cache_dir=tmpdir).load()
        unshuffled = DatasetLoader(path=file_path, cache_dir=tmpdir).load()

        assert list(seeded_a["idx"]) == list(seeded_b["idx"])
        assert list(seeded_a["idx"]) != list(different_seed["idx"])
        assert list(unshuffled["idx"]) == list(range(20))
    print("Test 8 — DatasetLoader shuffle: OK ✅")


def test_dataset_loader_rejects_unsupported_files_and_empty_directories():
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = os.path.join(tmpdir, "data.csv")
        Path(bad_path).write_text("col1,col2\na,b\n")

        try:
            DatasetLoader(path=bad_path, cache_dir=tmpdir).load()
            assert False, "Unsupported file format should raise ValueError"
        except ValueError as error:
            assert "Unsupported file format" in str(error)

        empty_dir = os.path.join(tmpdir, "empty_dir")
        os.makedirs(empty_dir)
        try:
            DatasetLoader(path=empty_dir, cache_dir=tmpdir).load()
            assert False, "Empty directory should raise ValueError"
        except ValueError as error:
            assert "No .jsonl or .parquet" in str(error)
    print("Test 9 — DatasetLoader invalid inputs: OK ✅")


def test_dataset_loader_reads_parquet_when_pyarrow_is_available():
    try:
        pa = importlib.import_module("pyarrow")
        pq = importlib.import_module("pyarrow.parquet")
    except ImportError:
        print("DatasetLoader parquet test skipped: pyarrow not installed")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, "data.parquet")
        table = pa.table({"text": [f"row_{index}" for index in range(8)], "id": list(range(8))})
        pq.write_table(table, parquet_path)

        dataset = DatasetLoader(path=parquet_path, cache_dir=tmpdir).load()
        assert len(dataset) == 8
        assert "text" in dataset.column_names
    print("Test 10 — DatasetLoader parquet: OK ✅")


def test_get_nvidia_smi_vram_parses_output_and_falls_back_on_errors():
    with patch("subprocess.check_output", side_effect=FileNotFoundError("nvidia-smi not found")):
        assert get_nvidia_smi_vram() == [0.0]

    with patch("subprocess.check_output", return_value=b"4096\n8192\n"):
        assert get_nvidia_smi_vram() == [4.0, 8.0]
    print("Test 11 — get_nvidia_smi_vram: OK ✅")


def test_constitutional_generation_returns_initial_responses_when_critique_is_disabled():
    with patch("utils.generate_rollouts", return_value=["Response A", "Response B"]):
        result = constitutional_generation(
            model=MagicMock(),
            tokenizer=MagicMock(),
            user_prompt="Test prompt",
            system="Be helpful.",
            sampling_params=MagicMock(),
            enable_critique=False,
        )

    assert result == {
        "initial_responses": ["Response A", "Response B"],
        "final_responses": ["Response A", "Response B"],
        "critiques": [],
        "revisions": [],
    }
    print("Test 12 — constitutional_generation no critique: OK ✅")


def test_constitutional_generation_runs_requested_critique_revision_iterations():
    with patch("utils.generate_rollouts", return_value=["Initial resp"]), \
         patch("utils.critique_response", return_value=["Critique text"]) as critique_mock, \
         patch("utils.revise_response", return_value=["Revised resp"]) as revise_mock:
        result = constitutional_generation(
            model=MagicMock(),
            tokenizer=MagicMock(),
            user_prompt="Write a poem",
            system="Be creative.",
            sampling_params=MagicMock(),
            enable_critique=True,
            max_revisions=2,
        )

    assert result["initial_responses"] == ["Initial resp"]
    assert result["final_responses"] == ["Revised resp"]
    assert result["critiques"] == [["Critique text"], ["Critique text"]]
    assert result["revisions"] == [["Revised resp"], ["Revised resp"]]
    assert critique_mock.call_count == 2
    assert revise_mock.call_count == 2
    print("Test 13 — constitutional_generation critique/revision: OK ✅")


def test_run_rollouts_generates_for_each_chunk_and_saves_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "rollouts.jsonl")
        tokenizer = _mock_word_tokenizer()

        with patch("utils.generate_rollouts", return_value=["Generated output"]) as generate_mock:
            run_rollouts(
                sample={"text": "one two three four five six", "source": "test_suite", "ignored": True},
                counter=7,
                text_column="text",
                metadata_columns=["source", "missing"],
                model=MagicMock(),
                tokenizer=tokenizer,
                sampling_params=MagicMock(),
                file_path=output_path,
                system="Summarize.",
                prompt_prefix="PREFIX: ",
                prompt_suffix=" :SUFFIX",
                max_chunk_size=3,
                chunk_once=False,
            )

        records = _read_jsonl(output_path)
        assert len(records) == 2
        assert [record["chunk"] for record in records] == [0, 1]
        assert all(record["row"] == 7 for record in records)
        assert all(record["metadata"] == {"source": "test_suite"} for record in records)
        assert generate_mock.call_args_list[0].kwargs["input_string"] == "PREFIX: w0 w1 w2 :SUFFIX"
        assert generate_mock.call_args_list[1].kwargs["input_string"] == "PREFIX: w3 w4 w5 :SUFFIX"
    print("Test 14 — run_rollouts: OK ✅")


def test_run_cai_rollouts_saves_result_and_skips_oversized_prompts():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "cai_rollouts.jsonl")
        tokenizer = MagicMock(return_value=MagicMock(input_ids=[0] * 10))
        cai_result = {
            "initial_responses": ["Gravity is..."],
            "final_responses": ["Gravity is a force..."],
            "critiques": [["Could be clearer"]],
            "revisions": [["Gravity is a force..."]],
        }

        with patch("utils.constitutional_generation", return_value=cai_result) as generation_mock:
            run_cai_rollouts(
                sample={"instruction": "Explain gravity", "topic": "physics"},
                counter=5,
                prompt_column="instruction",
                metadata_columns=["topic", "missing"],
                model=MagicMock(),
                tokenizer=tokenizer,
                sampling_params=MagicMock(),
                file_path=output_path,
                system="Be helpful.",
                prompt_prefix="Q: ",
                prompt_suffix="?",
                max_chunk_size=8192,
                enable_critique=True,
                max_revisions=1,
            )

        record = _read_jsonl(output_path)[0]
        assert record["row"] == 5
        assert record["instruction"] == "Explain gravity"
        assert record["final_responses"] == ["Gravity is a force..."]
        assert record["metadata"] == {"topic": "physics"}
        assert generation_mock.call_args.kwargs["user_prompt"] == "Q: Explain gravity?"

        tokenizer.return_value = MagicMock(input_ids=[0] * 10_000)
        with patch("utils.constitutional_generation") as oversized_generation_mock:
            run_cai_rollouts(
                sample={"instruction": "Very long prompt"},
                counter=6,
                prompt_column="instruction",
                metadata_columns=[],
                model=MagicMock(),
                tokenizer=tokenizer,
                sampling_params=MagicMock(),
                file_path=output_path,
                system="Be helpful.",
                prompt_prefix="",
                prompt_suffix="",
                max_chunk_size=100,
            )

        oversized_generation_mock.assert_not_called()
        assert len(_read_jsonl(output_path)) == 1
    print("Test 15 — run_cai_rollouts: OK ✅")


if __name__ == "__main__":
    tests = [
        test_detect_failure_reason_handles_known_failures_and_empty_inputs,
        test_get_starting_row_resumes_from_valid_rows_and_ignores_bad_lines,
        test_save_samples_appends_jsonl_with_chunk_and_metadata,
        test_save_cai_sample_preserves_nested_results_metadata_and_unicode,
        test_chunk_text_splits_on_token_boundaries_and_can_keep_first_chunk_only,
        test_setup_triton_cache_sets_rank_directory_and_removes_stale_files,
        test_dataset_loader_reads_local_jsonl_files_and_directories,
        test_dataset_loader_shuffle_is_seeded_and_optional,
        test_dataset_loader_rejects_unsupported_files_and_empty_directories,
        test_dataset_loader_reads_parquet_when_pyarrow_is_available,
        test_get_nvidia_smi_vram_parses_output_and_falls_back_on_errors,
        test_constitutional_generation_returns_initial_responses_when_critique_is_disabled,
        test_constitutional_generation_runs_requested_critique_revision_iterations,
        test_run_rollouts_generates_for_each_chunk_and_saves_metadata,
        test_run_cai_rollouts_saves_result_and_skips_oversized_prompts,
    ]
    for test in tests:
        test()
    print("\n" + "=" * 50)
    print("All tests passed ✅")
    print("=" * 50)
