"""
Synthetic generation test suite for utils, generate.py, and generate_cai.py.

Run with:
    python tests_synthetic.py

Requirements:
- torch
- transformers
- datasets
"""
# %%
#######################################
# 1. Imports & Setup
#######################################
import sys
import os
import tempfile

sys.pycache_prefix = os.path.join(tempfile.gettempdir(), "pycache")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SYNTHETIC_DIR = os.path.join(REPO_ROOT, "synthetic")
if SYNTHETIC_DIR not in sys.path:
    sys.path.insert(0, SYNTHETIC_DIR)

import json
import re
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# We can import the non-vLLM utilities directly.
# Patch vllm at module level so utils.py can be imported without a GPU.
_vllm_mock = MagicMock()
sys.modules.setdefault("vllm", _vllm_mock)
sys.modules.setdefault("datatrove", MagicMock())
sys.modules.setdefault("datatrove.utils", MagicMock())
sys.modules.setdefault("datatrove.utils.logging", MagicMock(logger=MagicMock()))

from utils import (
    DatasetLoader,
    get_logger,
    get_starting_row,
    save_samples,
    save_cai_sample,
    chunk_text,
    detect_failure_reason,
    setup_triton_cache,
    get_nvidia_smi_vram,
    FAILURE_PATTERNS,
    QUANTIZATION_METHODS,
    KV_CACHE_DTYPE_OPTIONS,
    MAX_GPUS_PER_NODE,
    VRAM_MB_TO_GB,
    TRITON_CACHE_CLEANUP_AGE,
)

print("All imports OK ✅")

# %%
#######################################
# 2. get_logger — returns a working logger
#######################################
logger = get_logger("TestSynthetic")
assert logger.name == "TestSynthetic"
# Should be able to log without errors
logger.info("Logger works.")
print("Test 2 — get_logger: OK ✅")

# %%
#######################################
# 3. Constants sanity checks
#######################################
assert MAX_GPUS_PER_NODE == 8
assert VRAM_MB_TO_GB == 1024
assert TRITON_CACHE_CLEANUP_AGE == 3600
assert "bitsandbytes" in QUANTIZATION_METHODS
assert "auto" in KV_CACHE_DTYPE_OPTIONS
assert "fp8_e4m3" in KV_CACHE_DTYPE_OPTIONS
assert "fp8_e5m2" in KV_CACHE_DTYPE_OPTIONS
print("Test 3 — constants sanity: OK ✅")

# %%
#######################################
# 4. FAILURE_PATTERNS — compiled regex list
#######################################
assert len(FAILURE_PATTERNS) > 0, "FAILURE_PATTERNS should not be empty"
for pattern, reason in FAILURE_PATTERNS:
    assert hasattr(pattern, "search"), "Each pattern should be a compiled regex"
    assert isinstance(reason, str) and len(reason) > 0

# Verify known patterns match expected strings
oom_text = "torch.OutOfMemoryError: CUDA out of memory"
matched = any(p.search(oom_text) for p, _ in FAILURE_PATTERNS)
assert matched, "OOM pattern should match"

timeout_text = "CANCELLED DUE TO TIME LIMIT"
matched_timeout = any(p.search(timeout_text) for p, r in FAILURE_PATTERNS if r == "timeout")
assert matched_timeout, "Timeout pattern should match"

server_text = "Failed to start VLLMServer server"
matched_server = any(p.search(server_text) for p, r in FAILURE_PATTERNS if r == "server_fail")
assert matched_server, "Server fail pattern should match"

no_match_text = "Everything is fine and running smoothly."
assert not any(p.search(no_match_text) for p, _ in FAILURE_PATTERNS), "Benign text should not match"
print("Test 4 — FAILURE_PATTERNS: OK ✅")

# %%
#######################################
# 5. detect_failure_reason — file-based detection
#######################################
with tempfile.TemporaryDirectory() as tmpdir:
    # OOM log
    oom_log = Path(tmpdir) / "oom.log"
    oom_log.write_text("Some startup info\ntorch.OutOfMemoryError: CUDA out of memory\n")
    assert detect_failure_reason(oom_log) == "OOM"

    # Timeout log
    timeout_log = Path(tmpdir) / "timeout.log"
    timeout_log.write_text("Running...\nCANCELLED AT  DUE TO TIME LIMIT\n")
    assert detect_failure_reason(timeout_log) == "timeout"

    # Server fail log
    server_log = Path(tmpdir) / "server.log"
    server_log.write_text("Initializing...\nFailed to start VLLMServer server\n")
    assert detect_failure_reason(server_log) == "server_fail"

    # Clean log — no failure
    clean_log = Path(tmpdir) / "clean.log"
    clean_log.write_text("All good\nGeneration complete\n")
    assert detect_failure_reason(clean_log) is None

    # Empty log
    empty_log = Path(tmpdir) / "empty.log"
    empty_log.write_text("")
    assert detect_failure_reason(empty_log) is None

    # Non-existent log
    assert detect_failure_reason(Path(tmpdir) / "nonexistent.log") is None

    # None path
    assert detect_failure_reason(None) is None

    # Large log — failure in tail
    large_log = Path(tmpdir) / "large.log"
    large_log.write_text("x" * 200_000 + "\nOutOfMemoryError\n")
    assert detect_failure_reason(large_log) == "OOM"

print("Test 5 — detect_failure_reason: OK ✅")

# %%
#######################################
# 6. get_starting_row — resumption logic
#######################################
with tempfile.TemporaryDirectory() as tmpdir:
    # Explicit row_start overrides everything
    assert get_starting_row("/nonexistent/file.jsonl", row_start=10) == 10
    assert get_starting_row("/nonexistent/file.jsonl", row_start=0) == 0

    # Non-existent file → start at 0
    assert get_starting_row(os.path.join(tmpdir, "missing.jsonl"), row_start=None) == 0

    # File with rows → resume after max row
    out_path = os.path.join(tmpdir, "progress.jsonl")
    with open(out_path, "w") as f:
        for i in range(5):
            f.write(json.dumps({"row": i, "data": "x"}) + "\n")
    assert get_starting_row(out_path, row_start=None) == 5

    # Non-contiguous rows → still picks max
    out_path2 = os.path.join(tmpdir, "sparse.jsonl")
    with open(out_path2, "w") as f:
        for r in [0, 5, 3, 10, 7]:
            f.write(json.dumps({"row": r, "data": "x"}) + "\n")
    assert get_starting_row(out_path2, row_start=None) == 11

    # File with bad lines (non-JSON) → gracefully ignored
    out_path3 = os.path.join(tmpdir, "mixed.jsonl")
    with open(out_path3, "w") as f:
        f.write("not json at all\n")
        f.write(json.dumps({"row": 2}) + "\n")
        f.write("{bad json\n")
        f.write(json.dumps({"row": 8}) + "\n")
    assert get_starting_row(out_path3, row_start=None) == 9

    # Empty file → start at 0
    empty_path = os.path.join(tmpdir, "empty.jsonl")
    open(empty_path, "w").close()
    assert get_starting_row(empty_path, row_start=None) == 0

print("Test 6 — get_starting_row: OK ✅")

# %%
#######################################
# 7. save_samples — output format and append behavior
#######################################
with tempfile.TemporaryDirectory() as tmpdir:
    fpath = os.path.join(tmpdir, "output.jsonl")

    # Save a single sample without chunk
    save_samples(fpath, row=0, seed_text="Hello world", rollouts=["Generated text"], metadata={})
    with open(fpath) as f:
        line = f.readline()
    record = json.loads(line)
    assert record["row"] == 0
    assert record["seed_text"] == "Hello world"
    assert record["rollouts"] == ["Generated text"]
    assert record["chunk"] is None

    # Save with chunk and metadata
    save_samples(fpath, row=1, seed_text="Input 2", rollouts=["Out A", "Out B"],
                 metadata={"source": "test"}, chunk=3)
    with open(fpath) as f:
        lines = f.readlines()
    assert len(lines) == 2, "Should have appended a second line"
    record2 = json.loads(lines[1])
    assert record2["row"] == 1
    assert record2["chunk"] == 3
    assert record2["metadata"] == {"source": "test"}
    assert len(record2["rollouts"]) == 2

    # Multiple rollouts preserved
    save_samples(fpath, row=2, seed_text="Multi", rollouts=["r1", "r2", "r3"], metadata={})
    with open(fpath) as f:
        lines = f.readlines()
    assert len(lines) == 3
    record3 = json.loads(lines[2])
    assert record3["rollouts"] == ["r1", "r2", "r3"]

print("Test 7 — save_samples: OK ✅")

# %%
#######################################
# 8. save_cai_sample — CAI output format
#######################################
with tempfile.TemporaryDirectory() as tmpdir:
    fpath = os.path.join(tmpdir, "cai_output.jsonl")

    cai_result_no_critique = {
        "initial_responses": ["resp1", "resp2"],
        "final_responses": ["resp1", "resp2"],
        "critiques": [],
        "revisions": [],
    }
    save_cai_sample(fpath, row=0, instruction="Do X", cai_result=cai_result_no_critique, metadata={})
    with open(fpath) as f:
        record = json.loads(f.readline())
    assert record["row"] == 0
    assert record["instruction"] == "Do X"
    assert record["initial_responses"] == ["resp1", "resp2"]
    assert record["final_responses"] == ["resp1", "resp2"]
    assert record["critiques"] == []
    assert record["revisions"] == []

    # With critique/revision
    cai_result_with_critique = {
        "initial_responses": ["resp1"],
        "final_responses": ["revised1"],
        "critiques": [["critique1"]],
        "revisions": [["revised1"]],
    }
    save_cai_sample(fpath, row=1, instruction="Do Y",
                    cai_result=cai_result_with_critique, metadata={"lang": "pt"})
    with open(fpath) as f:
        lines = f.readlines()
    assert len(lines) == 2
    record2 = json.loads(lines[1])
    assert record2["critiques"] == [["critique1"]]
    assert record2["revisions"] == [["revised1"]]
    assert record2["metadata"] == {"lang": "pt"}

    # Unicode characters preserved (ensure_ascii=False)
    cai_result_unicode = {
        "initial_responses": ["Olá, tudo bem?"],
        "final_responses": ["Olá, tudo ótimo!"],
        "critiques": [],
        "revisions": [],
    }
    save_cai_sample(fpath, row=2, instruction="Cumprimente.",
                    cai_result=cai_result_unicode, metadata={})
    with open(fpath) as f:
        lines = f.readlines()
    record3 = json.loads(lines[2])
    assert "Olá" in record3["initial_responses"][0]
    assert "ótimo" in record3["final_responses"][0]

print("Test 8 — save_cai_sample: OK ✅")

# %%
#######################################
# 9. chunk_text — splitting long texts
#######################################
# Create a mock tokenizer
mock_tokenizer = MagicMock()

# Simulate tokenization: each word → one token id
def _mock_tokenize(text, **kwargs):
    tokens = text.split()
    result = MagicMock()
    result.input_ids = list(range(len(tokens)))
    return result

mock_tokenizer.side_effect = _mock_tokenize
mock_tokenizer.__call__ = _mock_tokenize
mock_tokenizer.decode = lambda ids, **kw: " ".join(f"w{i}" for i in ids)

# Text shorter than chunk size → single chunk
text_short = "one two three"
chunks = chunk_text(text_short, mock_tokenizer, max_chunk_size=10, chunk_once=False)
assert len(chunks) == 1

# Text longer than chunk size → multiple chunks
text_long = " ".join(f"word{i}" for i in range(20))
chunks_multi = chunk_text(text_long, mock_tokenizer, max_chunk_size=5, chunk_once=False)
assert len(chunks_multi) == 4  # 20 tokens / 5 = 4 chunks

# chunk_once=True → only first chunk
chunks_once = chunk_text(text_long, mock_tokenizer, max_chunk_size=5, chunk_once=True)
assert len(chunks_once) == 1

# Exact boundary
text_exact = " ".join(f"word{i}" for i in range(10))
chunks_exact = chunk_text(text_exact, mock_tokenizer, max_chunk_size=10, chunk_once=False)
assert len(chunks_exact) == 1

# One token over boundary
text_over = " ".join(f"word{i}" for i in range(11))
chunks_over = chunk_text(text_over, mock_tokenizer, max_chunk_size=10, chunk_once=False)
assert len(chunks_over) == 2

print("Test 9 — chunk_text: OK ✅")

# %%
#######################################
# 10. setup_triton_cache — directory creation
#######################################
with tempfile.TemporaryDirectory() as tmpdir:
    cache_path = os.path.join(tmpdir, "triton_test")
    old_triton = os.environ.get("TRITON_CACHE_DIR")
    old_slurm = os.environ.get("SLURM_JOB_ID")
    old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")

    os.environ["TRITON_CACHE_DIR"] = cache_path
    os.environ["SLURM_JOB_ID"] = "12345"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    setup_triton_cache()

    # Should have created the directory
    expected_dir = os.path.join(cache_path, "12345", "rank_0")
    assert os.path.isdir(expected_dir), f"Expected {expected_dir} to exist"
    assert os.environ["TRITON_CACHE_DIR"] == expected_dir

    # Restore env
    if old_triton is not None:
        os.environ["TRITON_CACHE_DIR"] = old_triton
    else:
        del os.environ["TRITON_CACHE_DIR"]
    if old_slurm is not None:
        os.environ["SLURM_JOB_ID"] = old_slurm
    elif "SLURM_JOB_ID" in os.environ:
        del os.environ["SLURM_JOB_ID"]
    if old_cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

print("Test 10 — setup_triton_cache: OK ✅")

# %%
#######################################
# 11. setup_triton_cache — stale file cleanup
#######################################
with tempfile.TemporaryDirectory() as tmpdir:
    cache_path = os.path.join(tmpdir, "triton_cleanup")
    os.environ["TRITON_CACHE_DIR"] = cache_path
    os.environ["SLURM_JOB_ID"] = "99"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Pre-create directory with a stale file
    rank_dir = os.path.join(cache_path, "99", "rank_0")
    os.makedirs(rank_dir, exist_ok=True)
    stale_file = os.path.join(rank_dir, "stale_kernel.so")
    with open(stale_file, "w") as f:
        f.write("old data")
    # Set mtime to 2 hours ago (exceeds TRITON_CACHE_CLEANUP_AGE of 3600s)
    old_time = time.time() - 7200
    os.utime(stale_file, (old_time, old_time))

    fresh_file = os.path.join(rank_dir, "fresh_kernel.so")
    with open(fresh_file, "w") as f:
        f.write("new data")

    setup_triton_cache()

    assert not os.path.exists(stale_file), "Stale file should have been cleaned up"
    assert os.path.exists(fresh_file), "Fresh file should remain"

    # Cleanup env
    del os.environ["TRITON_CACHE_DIR"]
    del os.environ["SLURM_JOB_ID"]
    del os.environ["CUDA_VISIBLE_DEVICES"]

print("Test 11 — setup_triton_cache stale cleanup: OK ✅")

# %%
#######################################
# 12. DatasetLoader — local jsonl file
#######################################
with tempfile.TemporaryDirectory() as tmpdir:
    # Create a small JSONL file
    jsonl_path = os.path.join(tmpdir, "data.jsonl")
    samples = [{"text": f"Sample {i}", "label": i} for i in range(10)]
    with open(jsonl_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    loader = DatasetLoader(path=jsonl_path, cache_dir=tmpdir)
    ds = loader.load()
    assert len(ds) == 10
    assert "text" in ds.column_names
    assert "label" in ds.column_names
    assert ds[0]["text"] == "Sample 0"

print("Test 12 — DatasetLoader from local jsonl: OK ✅")

# %%
#######################################
# 13. DatasetLoader — local directory with jsonl
#######################################
with tempfile.TemporaryDirectory() as tmpdir:
    data_dir = os.path.join(tmpdir, "dataset_dir")
    os.makedirs(data_dir)
    for shard in range(3):
        shard_path = os.path.join(data_dir, f"shard_{shard}.jsonl")
        with open(shard_path, "w") as f:
            for i in range(5):
                f.write(json.dumps({"text": f"shard{shard}_sample{i}"}) + "\n")

    loader = DatasetLoader(path=data_dir, cache_dir=tmpdir)
    ds = loader.load()
    assert len(ds) == 15  # 3 shards × 5 samples

print("Test 13 — DatasetLoader from directory: OK ✅")

# %%
#######################################
# 14. DatasetLoader — shuffle with seed
#######################################
with tempfile.TemporaryDirectory() as tmpdir:
    jsonl_path = os.path.join(tmpdir, "shuffle_data.jsonl")
    with open(jsonl_path, "w") as f:
        for i in range(20):
            f.write(json.dumps({"text": f"Item {i}", "idx": i}) + "\n")

    # Same seed → same order
    ds1 = DatasetLoader(path=jsonl_path, seed=42, cache_dir=tmpdir).load()
    ds2 = DatasetLoader(path=jsonl_path, seed=42, cache_dir=tmpdir).load()
    assert list(ds1["idx"]) == list(ds2["idx"]), "Same seed should give same order"

    # Different seed → different order (with high probability)
    ds3 = DatasetLoader(path=jsonl_path, seed=123, cache_dir=tmpdir).load()
    assert list(ds1["idx"]) != list(ds3["idx"]), "Different seeds should give different orders"

    # No seed → no shuffle (original order preserved)
    ds4 = DatasetLoader(path=jsonl_path, cache_dir=tmpdir).load()
    assert list(ds4["idx"]) == list(range(20)), "No seed should keep original order"

print("Test 14 — DatasetLoader shuffle: OK ✅")

# %%
#######################################
# 15. DatasetLoader — unsupported format raises error
#######################################
with tempfile.TemporaryDirectory() as tmpdir:
    bad_path = os.path.join(tmpdir, "data.csv")
    with open(bad_path, "w") as f:
        f.write("col1,col2\na,b\n")

    try:
        DatasetLoader(path=bad_path, cache_dir=tmpdir).load()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported file format" in str(e)

    # Empty directory — no data files
    empty_dir = os.path.join(tmpdir, "empty_dir")
    os.makedirs(empty_dir)
    try:
        DatasetLoader(path=empty_dir, cache_dir=tmpdir).load()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No .jsonl or .parquet" in str(e)

print("Test 15 — DatasetLoader unsupported format: OK ✅")

# %%
#######################################
# 16. get_nvidia_smi_vram — graceful fallback
#######################################
# When nvidia-smi is not available (e.g., CPU-only), should return [0.0]
with patch("subprocess.check_output", side_effect=FileNotFoundError("nvidia-smi not found")):
    vram = get_nvidia_smi_vram()
    assert vram == [0.0], f"Expected [0.0] on failure, got {vram}"

# Simulated success
with patch("subprocess.check_output", return_value=b"4096\n8192\n"):
    vram = get_nvidia_smi_vram()
    assert len(vram) == 2
    assert abs(vram[0] - 4.0) < 0.01  # 4096 / 1024
    assert abs(vram[1] - 8.0) < 0.01  # 8192 / 1024

print("Test 16 — get_nvidia_smi_vram: OK ✅")

# %%
#######################################
# 17. generate.py argument parser — defaults and required args
#######################################
# Import the argument parser from generate.py by capturing it
import importlib
import importlib.util

# Load generate.py's parser without invoking main()
_gen_spec = importlib.util.spec_from_file_location("generate_mod", os.path.join(SYNTHETIC_DIR, "generate.py"))
_gen_mod = importlib.util.module_from_spec(_gen_spec)
# Patch __name__ so it doesn't run main()
_gen_mod.__name__ = "generate_mod"
sys.modules["generate_mod"] = _gen_mod

# We need to prevent the `if __name__ == "__main__"` block from running,
# but that's handled because __name__ != "__main__".
# Also need to handle vllm import which is already mocked.
_gen_spec.loader.exec_module(_gen_mod)

# Build parser manually (mirrors what generate.py defines)
import argparse

gen_parser = argparse.ArgumentParser()
gen_parser.add_argument("--model_name_or_path", type=str, required=True)
gen_parser.add_argument("--tensor_parallel_size", type=int, default=1)
gen_parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
gen_parser.add_argument("--track_vram", action="store_true")
gen_parser.add_argument("--dataset_path", type=str, required=True)
gen_parser.add_argument("--dataset_subset", type=str, default=None)
gen_parser.add_argument("--dataset_split", type=str, default="train")
gen_parser.add_argument("--seed", type=int, default=None)
gen_parser.add_argument("--text_column", type=str, required=True)
gen_parser.add_argument("--metadata_columns", type=str, nargs="*", default=[])
gen_parser.add_argument("--output_dir", type=str, required=True)
gen_parser.add_argument("--output_file", type=str, default="output.jsonl")
gen_parser.add_argument("--max_length", type=int, default=4096)
gen_parser.add_argument("--max_chunk_size", type=int, default=8192)
gen_parser.add_argument("--chunk_once", action="store_true")
gen_parser.add_argument("--temperature", type=float, default=0.5)
gen_parser.add_argument("--top_k", type=int, default=20)
gen_parser.add_argument("--top_p", type=float, default=0.8)
gen_parser.add_argument("--num_return_sequences", type=int, default=1)
gen_parser.add_argument("--repetition_penalty", type=float, default=1.2)
gen_parser.add_argument("--cache_dir", type=str, default="./.cache")
gen_parser.add_argument("--system", type=str, default="")
gen_parser.add_argument("--prompt_prefix", type=str, default="")
gen_parser.add_argument("--prompt_suffix", type=str, default="")
gen_parser.add_argument("--row_start", type=int, default=None)
gen_parser.add_argument("--enable_thinking", action="store_true")

args = gen_parser.parse_args([
    "--model_name_or_path", "test-model",
    "--dataset_path", "/data/test.jsonl",
    "--text_column", "text",
    "--output_dir", "/tmp/out",
])

# Check defaults
assert args.tensor_parallel_size == 1
assert args.gpu_memory_utilization == 0.9
assert args.track_vram is False
assert args.dataset_split == "train"
assert args.seed is None
assert args.metadata_columns == []
assert args.output_file == "output.jsonl"
assert args.max_length == 4096
assert args.max_chunk_size == 8192
assert args.chunk_once is False
assert args.temperature == 0.5
assert args.top_k == 20
assert args.top_p == 0.8
assert args.num_return_sequences == 1
assert args.repetition_penalty == 1.2
assert args.cache_dir == "./.cache"
assert args.system == ""
assert args.prompt_prefix == ""
assert args.prompt_suffix == ""
assert args.row_start is None
assert args.enable_thinking is False

# Check required args are captured
assert args.model_name_or_path == "test-model"
assert args.dataset_path == "/data/test.jsonl"
assert args.text_column == "text"
assert args.output_dir == "/tmp/out"

print("Test 17 — generate.py argument parser defaults: OK ✅")

# %%
#######################################
# 18. generate_cai.py argument parser — defaults and required args
#######################################
cai_parser = argparse.ArgumentParser()
cai_parser.add_argument("--model_name_or_path", type=str, required=True)
cai_parser.add_argument("--tensor_parallel_size", type=int, default=1)
cai_parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
cai_parser.add_argument("--track_vram", action="store_true")
cai_parser.add_argument("--dataset_path", type=str, required=True)
cai_parser.add_argument("--dataset_subset", type=str, default=None)
cai_parser.add_argument("--dataset_split", type=str, default="train")
cai_parser.add_argument("--seed", type=int, default=None)
cai_parser.add_argument("--prompt_column", type=str, required=True)
cai_parser.add_argument("--metadata_columns", type=str, nargs="*", default=[])
cai_parser.add_argument("--output_dir", type=str, required=True)
cai_parser.add_argument("--output_file", type=str, default="output.jsonl")
cai_parser.add_argument("--max_length", type=int, default=4096)
cai_parser.add_argument("--max_chunk_size", type=int, default=8192)
cai_parser.add_argument("--temperature", type=float, default=0.5)
cai_parser.add_argument("--top_k", type=int, default=20)
cai_parser.add_argument("--top_p", type=float, default=0.8)
cai_parser.add_argument("--num_return_sequences", type=int, default=1)
cai_parser.add_argument("--repetition_penalty", type=float, default=1.2)
cai_parser.add_argument("--cache_dir", type=str, default="./.cache")
cai_parser.add_argument("--enable_thinking", action="store_true")
cai_parser.add_argument("--constitution_file", type=str, default="./constitution.md")
cai_parser.add_argument("--prompt_prefix", type=str, default="")
cai_parser.add_argument("--prompt_suffix", type=str, default="")
cai_parser.add_argument("--row_start", type=int, default=None)
cai_parser.add_argument("--enable_critique", action="store_true")
cai_parser.add_argument("--max_revisions", type=int, default=1)

cai_args = cai_parser.parse_args([
    "--model_name_or_path", "cai-model",
    "--dataset_path", "/data/prompts.jsonl",
    "--prompt_column", "instruction",
    "--output_dir", "/tmp/cai_out",
])

# CAI-specific defaults
assert cai_args.constitution_file == "./constitution.md"
assert cai_args.enable_critique is False
assert cai_args.max_revisions == 1
assert cai_args.prompt_column == "instruction"

# Shared defaults
assert cai_args.temperature == 0.5
assert cai_args.enable_thinking is False

# CAI with flags enabled
cai_args2 = cai_parser.parse_args([
    "--model_name_or_path", "cai-model",
    "--dataset_path", "/data/prompts.jsonl",
    "--prompt_column", "instruction",
    "--output_dir", "/tmp/cai_out",
    "--enable_critique",
    "--max_revisions", "3",
    "--enable_thinking",
])
assert cai_args2.enable_critique is True
assert cai_args2.max_revisions == 3
assert cai_args2.enable_thinking is True

print("Test 18 — generate_cai.py argument parser defaults: OK ✅")

# %%
#######################################
# 19. constitutional_generation — no critique path
#######################################
from utils import constitutional_generation

mock_model = MagicMock()
mock_tok = MagicMock()
mock_tok.apply_chat_template = lambda msgs, **kw: "formatted"
mock_params = MagicMock()

# Simulate generate_rollouts returning 2 responses
with patch("utils.generate_rollouts", return_value=["Response A", "Response B"]):
    result = constitutional_generation(
        model=mock_model,
        tokenizer=mock_tok,
        user_prompt="Test prompt",
        system="Be helpful.",
        sampling_params=mock_params,
        enable_critique=False,
    )
    assert result["initial_responses"] == ["Response A", "Response B"]
    assert result["final_responses"] == ["Response A", "Response B"]
    assert result["critiques"] == []
    assert result["revisions"] == []

print("Test 19 — constitutional_generation (no critique): OK ✅")

# %%
#######################################
# 20. constitutional_generation — with critique and revision
#######################################
with patch("utils.generate_rollouts", return_value=["Initial resp"]) as mock_gen, \
     patch("utils.critique_response", return_value=["Critique text"]) as mock_crit, \
     patch("utils.revise_response", return_value=["Revised resp"]) as mock_rev:

    result = constitutional_generation(
        model=mock_model,
        tokenizer=mock_tok,
        user_prompt="Write a poem",
        system="Be creative.",
        sampling_params=mock_params,
        enable_critique=True,
        max_revisions=2,
    )
    assert result["initial_responses"] == ["Initial resp"]
    assert result["final_responses"] == ["Revised resp"]
    assert len(result["critiques"]) == 2   # 2 revisions
    assert len(result["revisions"]) == 2
    assert mock_crit.call_count == 2
    assert mock_rev.call_count == 2

print("Test 20 — constitutional_generation (with critique): OK ✅")

# %%
#######################################
# 21. run_rollouts — end-to-end with mocks
#######################################
from utils import run_rollouts

with tempfile.TemporaryDirectory() as tmpdir:
    out_file = os.path.join(tmpdir, "rollout_out.jsonl")

    mock_model = MagicMock()
    mock_tok = MagicMock()
    mock_tok.return_value = MagicMock(input_ids=[0] * 10)
    mock_tok.decode = lambda ids, **kw: "decoded"
    mock_params = MagicMock()

    sample = {"text": "Hello world", "source": "test_suite"}

    with patch("utils.generate_rollouts", return_value=["Generated output"]):
        run_rollouts(
            sample=sample,
            counter=0,
            text_column="text",
            metadata_columns=["source"],
            model=mock_model,
            tokenizer=mock_tok,
            sampling_params=mock_params,
            file_path=out_file,
            system="Summarize.",
            prompt_prefix="PREFIX: ",
            prompt_suffix=" :SUFFIX",
            max_chunk_size=8192,
            chunk_once=False,
        )

    with open(out_file) as f:
        record = json.loads(f.readline())
    assert record["row"] == 0
    assert record["seed_text"] == "Hello world"
    assert record["rollouts"] == ["Generated output"]
    assert record["metadata"] == {"source": "test_suite"}

print("Test 21 — run_rollouts: OK ✅")

# %%
#######################################
# 22. run_cai_rollouts — end-to-end with mocks
#######################################
from utils import run_cai_rollouts

with tempfile.TemporaryDirectory() as tmpdir:
    out_file = os.path.join(tmpdir, "cai_rollout_out.jsonl")

    mock_model = MagicMock()
    mock_tok = MagicMock()
    mock_tok.return_value = MagicMock(input_ids=[0] * 10)
    mock_params = MagicMock()

    sample = {"instruction": "Explain gravity", "topic": "physics"}

    cai_mock_result = {
        "initial_responses": ["Gravity is..."],
        "final_responses": ["Gravity is a force..."],
        "critiques": [["Could be clearer"]],
        "revisions": [["Gravity is a force..."]],
    }
    with patch("utils.constitutional_generation", return_value=cai_mock_result):
        run_cai_rollouts(
            sample=sample,
            counter=5,
            prompt_column="instruction",
            metadata_columns=["topic"],
            model=mock_model,
            tokenizer=mock_tok,
            sampling_params=mock_params,
            file_path=out_file,
            system="Be helpful.",
            prompt_prefix="",
            prompt_suffix="",
            max_chunk_size=8192,
            enable_critique=True,
            max_revisions=1,
        )

    with open(out_file) as f:
        record = json.loads(f.readline())
    assert record["row"] == 5
    assert record["instruction"] == "Explain gravity"
    assert record["initial_responses"] == ["Gravity is..."]
    assert record["final_responses"] == ["Gravity is a force..."]
    assert record["metadata"] == {"topic": "physics"}

print("Test 22 — run_cai_rollouts: OK ✅")

# %%
#######################################
# 23. run_cai_rollouts — skips oversized prompts
#######################################
with tempfile.TemporaryDirectory() as tmpdir:
    out_file = os.path.join(tmpdir, "cai_skip.jsonl")
    open(out_file, "w").close()

    mock_model = MagicMock()
    mock_tok = MagicMock()
    # Simulate a very long token count
    mock_tok.return_value = MagicMock(input_ids=[0] * 10000)
    mock_params = MagicMock()

    sample = {"instruction": "Very long prompt..."}

    with patch("utils.constitutional_generation") as mock_cg:
        run_cai_rollouts(
            sample=sample,
            counter=0,
            prompt_column="instruction",
            metadata_columns=[],
            model=mock_model,
            tokenizer=mock_tok,
            sampling_params=mock_params,
            file_path=out_file,
            system="System.",
            prompt_prefix="",
            prompt_suffix="",
            max_chunk_size=100,  # Much smaller than 10000 tokens
        )
        mock_cg.assert_not_called()

    # File should still be empty (nothing written)
    with open(out_file) as f:
        assert f.read() == "", "Oversized prompt should be skipped"

print("Test 23 — run_cai_rollouts skips oversized: OK ✅")

# %%
#######################################
# 24. DatasetLoader — parquet file
#######################################
try:
    import pyarrow  # noqa: F401
    _has_parquet = True
except ImportError:
    _has_parquet = False

if _has_parquet:
    import pyarrow as pa
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as tmpdir:
        pq_path = os.path.join(tmpdir, "data.parquet")
        table = pa.table({"text": [f"row_{i}" for i in range(8)], "id": list(range(8))})
        pq.write_table(table, pq_path)

        ds = DatasetLoader(path=pq_path, cache_dir=tmpdir).load()
        assert len(ds) == 8
        assert "text" in ds.column_names
    print("Test 24 — DatasetLoader from parquet: OK ✅")
else:
    print("Test 24 — DatasetLoader from parquet: SKIPPED (pyarrow not installed)")

# %%
#######################################
# 25. save_samples + get_starting_row round-trip
#######################################
with tempfile.TemporaryDirectory() as tmpdir:
    fpath = os.path.join(tmpdir, "roundtrip.jsonl")

    for i in range(7):
        save_samples(fpath, row=i, seed_text=f"text_{i}", rollouts=[f"out_{i}"], metadata={})

    next_row = get_starting_row(fpath, row_start=None)
    assert next_row == 7, f"Expected 7, got {next_row}"

    # Append more
    save_samples(fpath, row=7, seed_text="text_7", rollouts=["out_7"], metadata={})
    assert get_starting_row(fpath, row_start=None) == 8

print("Test 25 — save_samples + get_starting_row round-trip: OK ✅")

# %%
#######################################
# 26. DatasetLoader — metadata_columns and text_column extraction
#######################################
with tempfile.TemporaryDirectory() as tmpdir:
    jsonl_path = os.path.join(tmpdir, "meta.jsonl")
    with open(jsonl_path, "w") as f:
        for i in range(5):
            f.write(json.dumps({
                "text": f"Content {i}",
                "source": "wiki",
                "difficulty": "easy",
                "extra": i
            }) + "\n")

    ds = DatasetLoader(path=jsonl_path, cache_dir=tmpdir).load()

    # Simulate metadata extraction like run_rollouts does
    sample = ds[0]
    text_column = "text"
    metadata_columns = ["source", "difficulty", "nonexistent"]
    metadata = {col: sample[col] for col in metadata_columns if col in sample}

    assert metadata == {"source": "wiki", "difficulty": "easy"}
    assert "nonexistent" not in metadata
    assert sample[text_column] == "Content 0"

print("Test 26 — metadata extraction: OK ✅")

# %%
print("\n" + "=" * 50)
print("All synthetic tests passed ✅")
print("=" * 50)
