"""
Synthetic agents test suite (peripheral functions only. No inference).

Run with:
    python tests_synthetic_agents.py

Requirements:
- torch
- transformers
- datasets
- pyyaml
"""

import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.pycache_prefix = os.path.join(tempfile.gettempdir(), "pycache")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
AGENTS_DIR = os.path.join(REPO_ROOT, "synthetic", "agents")
if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Mock smolagents *before* importing utils, since utils.py does module-level
# imports from smolagents.  We use real stub classes for the memory types
# because _extract_step_type() relies on isinstance() checks against them.
# ---------------------------------------------------------------------------


class _StubTaskStep:
    def __init__(self, task: str = ""):
        self.task = task


class _StubPlanningStep:
    def __init__(self, plan: str = "", facts: str = ""):
        self.plan = plan
        self.facts = facts


class _StubActionStep:
    pass


class _StubFinalAnswerStep:
    pass


class _StubSystemPromptStep:
    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt


_smolagents = MagicMock()
_smolagents.CodeAgent = MagicMock()
_smolagents_agents = MagicMock()
_smolagents_agents.EMPTY_PROMPT_TEMPLATES = {}
_smolagents_agents.PromptTemplates = MagicMock()
_smolagents_agents.populate_template = MagicMock()
_smolagents_default_tools = MagicMock()
_smolagents_lpe = MagicMock()
_smolagents_lpe.BASE_BUILTIN_MODULES = []
_smolagents_memory = MagicMock()
_smolagents_memory.TaskStep = _StubTaskStep
_smolagents_memory.PlanningStep = _StubPlanningStep
_smolagents_memory.ActionStep = _StubActionStep
_smolagents_memory.FinalAnswerStep = _StubFinalAnswerStep
_smolagents_memory.SystemPromptStep = _StubSystemPromptStep
_smolagents_models = MagicMock()
_smolagents_monitoring = MagicMock()
_smolagents_tools = MagicMock()
_smolagents_tools.Tool = MagicMock()
sys.modules["smolagents"] = _smolagents
sys.modules["smolagents.agents"] = _smolagents_agents
sys.modules["smolagents.default_tools"] = _smolagents_default_tools
sys.modules["smolagents.local_python_executor"] = _smolagents_lpe
sys.modules["smolagents.memory"] = _smolagents_memory
sys.modules["smolagents.models"] = _smolagents_models
sys.modules["smolagents.monitoring"] = _smolagents_monitoring
sys.modules["smolagents.tools"] = _smolagents_tools
_tools_mock = MagicMock()
sys.modules.setdefault("tools", _tools_mock)

from utils import (  # noqa: E402
    LANGUAGE_CONFIGS,
    SUPPORTED_LANGUAGES,
    OutputManager,
    TraceRecord,
    _conversation_has_unclosed_think,
    _extract_code_action,
    _extract_error,
    _extract_model_output_text,
    _extract_observations,
    _extract_step_type,
    _make_tool_call_xml,
    _make_tool_response_xml,
    append_metadata_entry,
    compare_answer,
    format_trace_as_conversation,
    load_dataset,
    load_metadata_entries,
    load_processed_ids,
    load_system_prompt,
    normalize_answer,
    setup_triton_cache,
    summarize_trace_metadata,
)

# Re-export the stub classes under the canonical names so the test for
# _extract_step_type reads naturally.
ActionStep = _StubActionStep
FinalAnswerStep = _StubFinalAnswerStep
PlanningStep = _StubPlanningStep
SystemPromptStep = _StubSystemPromptStep
TaskStep = _StubTaskStep

print("All imports OK ✅")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _make_mock_step(
    step_type: str,
    *,
    task: str = "",
    plan: str = "",
    model_output_message: str | dict | None = None,
    code_action: str = "",
    observations: str = "",
    error: str | dict | None = None,
    output: str | None = None,
    system_prompt: str = "",
) -> dict:
    """Build a minimal step dict as it would appear in TraceRecord.steps."""
    step: dict = {"_step_type": step_type}
    if step_type == "TaskStep":
        step["task"] = task
    elif step_type == "PlanningStep":
        step["plan"] = plan
    elif step_type == "ActionStep":
        if model_output_message is not None:
            step["model_output_message"] = model_output_message
        step["code_action"] = code_action
        step["observations"] = observations
        if error is not None:
            step["error"] = error
    elif step_type == "FinalAnswerStep":
        if output is not None:
            step["output"] = output
    elif step_type == "SystemPromptStep":
        step["system_prompt"] = system_prompt
    return step


# ---------------------------------------------------------------------------
# Test 1 — TraceRecord defaults
# ---------------------------------------------------------------------------


def test_trace_record_defaults_and_field_assignment():
    tr = TraceRecord(trace_id="abc-123", prompt="What is 2+2?")
    assert tr.trace_id == "abc-123"
    assert tr.prompt == "What is 2+2?"
    assert tr.ground_truth is None
    assert tr.status == "fail"
    assert tr.final_answer is None
    assert tr.steps == []
    assert tr.system_prompt == ""
    assert tr.started_at == ""
    assert tr.ended_at == ""
    assert tr.duration_seconds == 0.0
    assert tr.num_steps == 0
    assert tr.error is None
    assert tr.metadata == {}

    # Full assignment
    tr2 = TraceRecord(
        trace_id="xyz",
        prompt="Hello",
        ground_truth="Hi",
        status="success",
        final_answer="Hi",
        steps=[{"_step_type": "ActionStep", "code_action": "final_answer('Hi')"}],
        system_prompt="You are helpful.",
        started_at="2025-01-01T00:00:00Z",
        ended_at="2025-01-01T00:00:05Z",
        duration_seconds=5.0,
        num_steps=3,
        error=None,
        metadata={"source": "test"},
    )
    assert tr2.status == "success"
    assert tr2.num_steps == 3
    assert tr2.metadata == {"source": "test"}
    print("Test 1 — TraceRecord defaults: OK ✅")


# ---------------------------------------------------------------------------
# Test 2 — OutputManager: basic append + JSON array format
# ---------------------------------------------------------------------------


def test_output_manager_appends_valid_json_array_and_rotates():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "traces"
        mgr = OutputManager(output_dir, max_entries_per_file=3)

        for i in range(5):
            tr = TraceRecord(
                trace_id=f"trace_{i:03d}",
                prompt=f"Prompt {i}",
                status="success" if i % 2 == 0 else "fail",
                num_steps=i + 1,
                duration_seconds=i * 0.5,
            )
            mgr.save_raw_trace(tr)

        # Should have two files (3 + 2 entries)
        raw_dir = output_dir / "raw_traces"
        files = sorted(raw_dir.glob("raw_traces_*.json"))
        assert len(files) == 2

        with open(files[0], encoding="utf-8") as fh:
            batch1 = json.load(fh)
        with open(files[1], encoding="utf-8") as fh:
            batch2 = json.load(fh)

        assert isinstance(batch1, list)
        assert len(batch1) == 3
        assert isinstance(batch2, list)
        assert len(batch2) == 2

        # Verify content integrity
        all_ids = [r["trace_id"] for r in batch1] + [r["trace_id"] for r in batch2]
        assert all_ids == [f"trace_{i:03d}" for i in range(5)]
    print("Test 2 — OutputManager append + rotation: OK ✅")


# ---------------------------------------------------------------------------
# Test 3 — OutputManager: resume scanning
# ---------------------------------------------------------------------------


def test_output_manager_resume_scans_existing_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "traces"
        # First run: write 2 traces
        mgr1 = OutputManager(output_dir, max_entries_per_file=5)
        for i in range(2):
            mgr1.save_raw_trace(TraceRecord(trace_id=f"run1_{i}", prompt=f"P{i}", status="success"))

        # Second run: should pick up where we left off
        mgr2 = OutputManager(output_dir, max_entries_per_file=5)
        assert mgr2._raw_count == 2  # scanned existing entries
        for i in range(3):
            mgr2.save_raw_trace(TraceRecord(trace_id=f"run2_{i}", prompt=f"Q{i}", status="success"))

        raw_dir = output_dir / "raw_traces"
        files = sorted(raw_dir.glob("raw_traces_*.json"))
        assert len(files) == 1  # all 5 in one file
        with open(files[0], encoding="utf-8") as fh:
            all_records = json.load(fh)
        ids = [r["trace_id"] for r in all_records]
        assert ids == ["run1_0", "run1_1", "run2_0", "run2_1", "run2_2"]
    print("Test 3 — OutputManager resume scanning: OK ✅")


# ---------------------------------------------------------------------------
# Test 4 — OutputManager: save_formatted_trace skips failures & think guard
# ---------------------------------------------------------------------------


def test_output_manager_formatted_trace_skips_failures_and_think_guard():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "traces"
        mgr = OutputManager(output_dir)

        # Failure trace → should NOT save formatted
        fail_tr = TraceRecord(trace_id="fail_1", prompt="P1", status="fail")
        path = mgr.save_formatted_trace(fail_tr)
        assert path is None

        # Success trace with valid steps → should save
        success_tr = TraceRecord(
            trace_id="ok_1",
            prompt="What is the capital of France?",
            status="success",
            system_prompt="You are a helpful assistant.",
            steps=[
                _make_mock_step(
                    "ActionStep",
                    model_output_message="Let me think about this.",
                    code_action="final_answer('Paris')",
                    observations="Final answer submitted.",
                ),
                _make_mock_step("FinalAnswerStep", output="Paris"),
            ],
        )
        path = mgr.save_formatted_trace(success_tr)
        assert path is not None

        fmt_dir = output_dir / "formatted_traces"
        files = sorted(fmt_dir.glob("formatted_traces_*.json"))
        assert len(files) == 1
        with open(files[0], encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, list)
        assert len(data) == 1
        assert "messages" in data[0]
        assert data[0]["trace_id"] == "ok_1"
    print("Test 4 — OutputManager formatted traces: OK ✅")


# ---------------------------------------------------------------------------
# Test 5 — load_dataset: local .jsonl
# ---------------------------------------------------------------------------


def test_load_dataset_reads_jsonl_with_auto_trace_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "data.jsonl")
        _write_jsonl(file_path, [{"prompt": "Hello"}, {"prompt": "World"}])

        rows = load_dataset(path=file_path, prompt_column="prompt", cache_dir=tmpdir)
        assert len(rows) == 2
        assert rows[0]["prompt"] == "Hello"
        assert rows[1]["prompt"] == "World"
        # Auto-generated trace IDs (SHA-256 hash of prompt)
        assert "_trace_id" in rows[0]
        assert rows[0]["_trace_id"] != rows[1]["_trace_id"]
    print("Test 5 — load_dataset jsonl: OK ✅")


# ---------------------------------------------------------------------------
# Test 6 — load_dataset: id column preserved
# ---------------------------------------------------------------------------


def test_load_dataset_uses_id_column_when_present():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "data.jsonl")
        _write_jsonl(
            file_path,
            [
                {"prompt": "Hello", "id": "abc-001"},
                {"prompt": "World", "id": "xyz-002"},
            ],
        )

        rows = load_dataset(
            path=file_path,
            prompt_column="prompt",
            id_column="id",
            cache_dir=tmpdir,
        )
        assert rows[0]["_trace_id"] == "abc-001"
        assert rows[1]["_trace_id"] == "xyz-002"
    print("Test 6 — load_dataset id column: OK ✅")


# ---------------------------------------------------------------------------
# Test 7 — load_dataset: ground truth column
# ---------------------------------------------------------------------------


def test_load_dataset_adds_ground_truth_column():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "data.jsonl")
        _write_jsonl(
            file_path,
            [{"prompt": "Q1", "answer": "Paris"}, {"prompt": "Q2", "answer": "Rome"}],
        )

        rows = load_dataset(
            path=file_path,
            prompt_column="prompt",
            ground_truth_column="answer",
            cache_dir=tmpdir,
        )
        assert rows[0]["_ground_truth"] == "Paris"
        assert rows[1]["_ground_truth"] == "Rome"
    print("Test 7 — load_dataset ground truth column: OK ✅")


# ---------------------------------------------------------------------------
# Test 8 — load_dataset: directory of .jsonl shards
# ---------------------------------------------------------------------------


def test_load_dataset_reads_directory_of_jsonl_shards():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "shards")
        os.makedirs(data_dir)
        _write_jsonl(
            os.path.join(data_dir, "shard_0.jsonl"), [{"prompt": f"s0_{i}"} for i in range(3)]
        )
        _write_jsonl(
            os.path.join(data_dir, "shard_1.jsonl"), [{"prompt": f"s1_{i}"} for i in range(2)]
        )

        rows = load_dataset(path=data_dir, prompt_column="prompt", cache_dir=tmpdir)
        assert len(rows) == 5
        prompts = {r["prompt"] for r in rows}
        assert prompts == {"s0_0", "s0_1", "s0_2", "s1_0", "s1_1"}
    print("Test 8 — load_dataset directory: OK ✅")


# ---------------------------------------------------------------------------
# Test 9 — load_dataset: rejects unsupported format
# ---------------------------------------------------------------------------


def test_load_dataset_rejects_unsupported_file_format():
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = os.path.join(tmpdir, "data.csv")
        Path(bad_path).write_text("col1,col2\na,b\n")
        try:
            load_dataset(path=bad_path, prompt_column="prompt", cache_dir=tmpdir)
            raise AssertionError("Unsupported format should raise ValueError")
        except ValueError as error:
            assert "Unsupported file format" in str(error)
    print("Test 9 — load_dataset unsupported format: OK ✅")


# ---------------------------------------------------------------------------
# Test 10 — load_dataset: missing prompt column
# ---------------------------------------------------------------------------


def test_load_dataset_raises_on_missing_prompt_column():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "data.jsonl")
        _write_jsonl(file_path, [{"text": "Hello", "label": 0}])

        try:
            load_dataset(path=file_path, prompt_column="prompt", cache_dir=tmpdir)
            raise AssertionError("Missing column should raise ValueError")
        except ValueError as error:
            assert "prompt" in str(error)
    print("Test 10 — load_dataset missing column: OK ✅")


# ---------------------------------------------------------------------------
# Test 11 — load_dataset: empty directory and empty file
# ---------------------------------------------------------------------------


def test_load_dataset_raises_on_empty_inputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty directory
        empty_dir = os.path.join(tmpdir, "empty_dir")
        os.makedirs(empty_dir)
        try:
            load_dataset(path=empty_dir, prompt_column="prompt", cache_dir=tmpdir)
            raise AssertionError("Empty directory should raise ValueError")
        except ValueError as error:
            assert "No .jsonl or .parquet" in str(error)

        # Empty .jsonl file — datasets may raise StopIteration or ValueError
        # depending on version, but should never silently succeed.
        empty_file = os.path.join(tmpdir, "empty.jsonl")
        open(empty_file, "w", encoding="utf-8").close()
        try:
            load_dataset(path=empty_file, prompt_column="prompt", cache_dir=tmpdir)
            raise AssertionError("Empty file should raise an exception")
        except (ValueError, StopIteration):
            pass  # expected — empty input is rejected
    print("Test 11 — load_dataset empty inputs: OK ✅")


# ---------------------------------------------------------------------------
# Test 12 — load_dataset: parquet
# ---------------------------------------------------------------------------


def test_load_dataset_reads_parquet_when_pyarrow_available():
    try:
        pa = importlib.import_module("pyarrow")
        pq = importlib.import_module("pyarrow.parquet")
    except ImportError:
        print("load_dataset parquet test skipped: pyarrow not installed")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, "data.parquet")
        table = pa.table({"prompt": [f"row_{i}" for i in range(4)], "id": list(range(4))})
        pq.write_table(table, parquet_path)

        rows = load_dataset(
            path=parquet_path, prompt_column="prompt", id_column="id", cache_dir=tmpdir
        )
        assert len(rows) == 4
        assert rows[2]["prompt"] == "row_2"
        assert rows[2]["_trace_id"] == "2"
    print("Test 12 — load_dataset parquet: OK ✅")


# ---------------------------------------------------------------------------
# Test 13 — load_processed_ids (resume)
# ---------------------------------------------------------------------------


def test_load_processed_ids_parses_metadata_jsonl_and_handles_malformed():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        # No file yet → empty set
        assert load_processed_ids(out_dir) == set()

        metadata_path = out_dir / "metadata.jsonl"
        with open(metadata_path, "w", encoding="utf-8") as fh:
            fh.write("garbage line\n")
            fh.write('{"trace_id": "abc", "status": "success"}\n')
            fh.write('{"trace_id": "xyz", "status": "fail"}\n')
            fh.write("{bad json\n")
            fh.write('{"trace_id": "def"}\n')

        ids = load_processed_ids(out_dir)
        assert ids == {"abc", "xyz", "def"}

        # Empty metadata.jsonl
        metadata_path.write_text("")
        assert load_processed_ids(out_dir) == set()
    print("Test 13 — load_processed_ids: OK ✅")


# ---------------------------------------------------------------------------
# Test 14 — append_metadata_entry
# ---------------------------------------------------------------------------


def test_append_metadata_entry_writes_one_line_per_trace():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        tr1 = TraceRecord(
            trace_id="t1",
            prompt="Q1",
            status="success",
            num_steps=3,
            duration_seconds=2.5,
            started_at="2025-06-01T10:00:00Z",
            ground_truth="42",
        )
        tr2 = TraceRecord(
            trace_id="t2",
            prompt="Q2",
            status="fail",
            num_steps=0,
            duration_seconds=0.0,
            error="Timeout",
        )

        append_metadata_entry(tr1, out_dir)
        append_metadata_entry(tr2, out_dir)

        records = _read_jsonl(out_dir / "metadata.jsonl")
        assert len(records) == 2
        assert records[0]["trace_id"] == "t1"
        assert records[0]["status"] == "success"
        assert records[0]["has_ground_truth"] is True
        assert records[1]["trace_id"] == "t2"
        assert records[1]["error"] == "Timeout"
        assert records[1]["has_ground_truth"] is False
    print("Test 14 — append_metadata_entry: OK ✅")


# ---------------------------------------------------------------------------
# Test 15 — normalize_answer
# ---------------------------------------------------------------------------


def test_normalize_answer_various_cases():
    # Basic
    assert normalize_answer("  Hello World!  ") == "hello world"
    assert normalize_answer("YES.") == "yes"
    assert normalize_answer("No...") == "no"
    assert normalize_answer("¿Qué?") == "¿qué"

    # Numeric normalization
    assert normalize_answer("42") == "42"
    assert normalize_answer("42.0") == "42"
    assert normalize_answer("42.00") == "42"
    assert normalize_answer("42.50") == "42.5"
    assert normalize_answer("-3.140") == "-3.14"
    assert normalize_answer("+100") == "100"

    # Comma normalization: thousands separator
    assert normalize_answer("40,000") == "40000"
    # Comma as decimal (trailing 1-2 digits)
    assert normalize_answer("42,5") == "42.5"
    # Mixed US style
    assert normalize_answer("1,234.56") == "1234.56"
    # Mixed EU style
    assert normalize_answer("1.234,56") == "1234.56"

    # Non-string input
    assert normalize_answer(42) == "42"
    assert normalize_answer(None) == "none"

    # Whitespace collapse
    assert normalize_answer("first    second\tthird") == "first second third"

    # Empty / whitespace only
    assert normalize_answer("") == ""
    assert normalize_answer("   ") == ""

    print("Test 15 — normalize_answer: OK ✅")


# ---------------------------------------------------------------------------
# Test 16 — compare_answer
# ---------------------------------------------------------------------------


def test_compare_answer_exact_and_substring_and_edge_cases():
    # Exact match
    assert compare_answer("Paris", "Paris") is True
    assert compare_answer("paris", "Paris") is True
    assert compare_answer("  Paris!  ", "Paris") is True

    # Substring containment
    assert compare_answer("The capital is Paris.", "Paris") is True
    assert compare_answer("Paris", "The capital is Paris.") is True

    # Numeric match
    assert compare_answer("42.0", "42") is True
    assert compare_answer("The answer is 42.", "42") is True

    # Mismatch
    assert compare_answer("Paris", "London") is False

    # Edge cases
    assert compare_answer(None, "something") is False
    assert compare_answer("something", "") is False
    assert compare_answer("", "") is False
    assert compare_answer("", "something") is False

    print("Test 16 — compare_answer: OK ✅")


# ---------------------------------------------------------------------------
# Test 17 — _conversation_has_unclosed_think
# ---------------------------------------------------------------------------


def test_conversation_has_unclosed_think_detection():
    # Balanced tags → OK
    assert (
        _conversation_has_unclosed_think(
            [{"role": "assistant", "content": "<think>reasoning</think> done"}]
        )
        is False
    )

    # Unclosed opening tag
    assert (
        _conversation_has_unclosed_think([{"role": "assistant", "content": "<think>reasoning..."}])
        is True
    )

    # Nested (balanced)
    assert (
        _conversation_has_unclosed_think(
            [{"role": "assistant", "content": "<think>outer<think>inner</think></think>"}]
        )
        is False
    )

    # Nested (unbalanced — one extra open)
    assert (
        _conversation_has_unclosed_think(
            [{"role": "assistant", "content": "<think>outer<think>inner</think>"}]
        )
        is True
    )

    # No tags at all
    assert (
        _conversation_has_unclosed_think([{"role": "assistant", "content": "just plain text"}])
        is False
    )

    # Empty content
    assert _conversation_has_unclosed_think([{"role": "assistant", "content": ""}]) is False

    # Multiple messages — only one has the problem
    assert (
        _conversation_has_unclosed_think(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "<think>oops</think>"},
                {"role": "assistant", "content": "<think>unclosed"},
            ]
        )
        is True
    )

    print("Test 17 — _conversation_has_unclosed_think: OK ✅")


# ---------------------------------------------------------------------------
# Test 18 — _extract_model_output_text
# ---------------------------------------------------------------------------


def test_extract_model_output_text_variants():
    # String format
    assert _extract_model_output_text({"model_output_message": "hello"}) == "hello"

    # ChatMessage dict format
    assert (
        _extract_model_output_text(
            {"model_output_message": {"role": "assistant", "content": "I think..."}}
        )
        == "I think..."
    )

    # Content as list of text blocks
    assert (
        _extract_model_output_text(
            {"model_output_message": {"content": [{"text": "part1"}, {"text": "part2"}]}}
        )
        == "part1\npart2"
    )

    # Missing field
    assert _extract_model_output_text({}) == ""

    # None model_output_message
    assert _extract_model_output_text({"model_output_message": None}) == ""

    print("Test 18 — _extract_model_output_text: OK ✅")


# ---------------------------------------------------------------------------
# Test 19 — _extract_code_action, _extract_observations, _extract_error
# ---------------------------------------------------------------------------


def test_extract_step_field_helpers():
    # code_action
    assert _extract_code_action({"code_action": "print(1)"}) == "print(1)"
    assert _extract_code_action({}) == ""

    # observations
    assert _extract_observations({"observations": "output"}) == "output"
    assert _extract_observations({}) == ""

    # error: string
    assert _extract_error({"error": "something broke"}) == "something broke"
    # error: dict
    assert _extract_error({"error": {"message": "type error"}}) == "type error"
    assert _extract_error({"error": {}}) == ""
    # error: None / missing
    assert _extract_error({}) == ""
    assert _extract_error({"error": None}) == ""

    print("Test 19 — step field helpers: OK ✅")


# ---------------------------------------------------------------------------
# Test 20 — _extract_step_type
# ---------------------------------------------------------------------------


def test_extract_step_type_labels():
    assert _extract_step_type(TaskStep(task="do X")) == "TaskStep"
    assert _extract_step_type(PlanningStep(plan="step 1", facts="f")) == "PlanningStep"
    assert _extract_step_type(ActionStep()) == "ActionStep"
    assert _extract_step_type(FinalAnswerStep()) == "FinalAnswerStep"
    assert _extract_step_type(SystemPromptStep(system_prompt="sys")) == "SystemPromptStep"

    # Unknown / plain object
    class CustomStep:
        pass

    assert _extract_step_type(CustomStep()) == "CustomStep"

    print("Test 20 — _extract_step_type: OK ✅")


# ---------------------------------------------------------------------------
# Test 21 — _make_tool_call_xml / _make_tool_response_xml
# ---------------------------------------------------------------------------


def test_xml_wrapping_helpers():
    assert _make_tool_call_xml("print(1)", "<code>", "</code>") == "<code>\nprint(1)\n</code>"
    assert _make_tool_call_xml("x = y", "```python", "```") == "```python\nx = y\n```"
    assert (
        _make_tool_response_xml("Execution logs:\noutput")
        == "<tool_response>\nExecution logs:\noutput\n</tool_response>"
    )
    print("Test 21 — XML wrapping helpers: OK ✅")


# ---------------------------------------------------------------------------
# Test 22 — format_trace_as_conversation
# ---------------------------------------------------------------------------


def test_format_trace_as_conversation_basic_structure():
    trace = TraceRecord(
        trace_id="conv_001",
        prompt="What is the capital of France?",
        status="success",
        system_prompt="You are a helpful assistant.",
        steps=[
            _make_mock_step(
                "ActionStep",
                model_output_message="Let me search for this.",
                code_action='web_search("capital of France")',
                observations="The capital of France is Paris.",
            ),
            _make_mock_step(
                "ActionStep",
                model_output_message="Now I know.",
                code_action='final_answer("Paris")',
                observations="Final answer submitted.",
            ),
            _make_mock_step("FinalAnswerStep", output="Paris"),
        ],
    )

    conv = format_trace_as_conversation(trace)

    # Roles are in order
    roles = [m["role"] for m in conv]
    assert "system" in roles
    assert "user" in roles
    assert "assistant" in roles

    # System prompt is first
    assert conv[0]["role"] == "system"
    assert "helpful assistant" in conv[0]["content"]

    # User prompt is second
    assert conv[1]["role"] == "user"
    assert "capital of France" in conv[1]["content"]

    # Tool call and tool response appear
    all_content = "\n".join(m["content"] for m in conv)
    assert "<code>" in all_content
    assert "web_search" in all_content
    assert "<tool_response>" in all_content
    assert "capital of France is Paris" in all_content

    print("Test 22 — format_trace_as_conversation: OK ✅")


# ---------------------------------------------------------------------------
# Test 23 — format_trace_as_conversation: PlanningStep
# ---------------------------------------------------------------------------


def test_format_trace_as_conversation_includes_planning_step():
    trace = TraceRecord(
        trace_id="plan_001",
        prompt="Solve a complex problem",
        status="success",
        steps=[
            _make_mock_step("PlanningStep", plan="Step 1: search. Step 2: compute."),
            _make_mock_step(
                "ActionStep",
                code_action='final_answer("42")',
                observations="Done.",
            ),
        ],
    )

    conv = format_trace_as_conversation(trace)
    plan_contents = [m["content"] for m in conv if m["role"] == "assistant"]
    assert any("Step 1: search" in c for c in plan_contents)
    print("Test 23 — format_trace_as_conversation planning: OK ✅")


# ---------------------------------------------------------------------------
# Test 24 — format_trace_as_conversation: TaskStep duplicate skip
# ---------------------------------------------------------------------------


def test_format_trace_as_conversation_skips_duplicate_task_step():
    trace = TraceRecord(
        trace_id="task_001",
        prompt="What is 2+2?",
        status="success",
        steps=[
            _make_mock_step("TaskStep", task="What is 2+2?"),  # same as prompt
            _make_mock_step(
                "ActionStep",
                code_action='final_answer("4")',
                observations="Done.",
            ),
        ],
    )

    conv = format_trace_as_conversation(trace)
    # Should NOT have an extra user message duplicating the prompt.
    # Exclude <tool_response> messages (also role="user") from this count.
    user_contents = [
        m["content"] for m in conv if m["role"] == "user" and "<tool_response>" not in m["content"]
    ]
    assert len(user_contents) == 1  # only the original prompt

    # A different task should appear
    trace2 = TraceRecord(
        trace_id="task_002",
        prompt="What is 2+2?",
        status="success",
        steps=[
            _make_mock_step("TaskStep", task="Rephrased: compute 2+2"),
            _make_mock_step(
                "ActionStep",
                code_action='final_answer("4")',
                observations="Done.",
            ),
        ],
    )
    conv2 = format_trace_as_conversation(trace2)
    user_contents2 = [
        m["content"] for m in conv2 if m["role"] == "user" and "<tool_response>" not in m["content"]
    ]
    assert len(user_contents2) == 2  # original prompt + rephrased task
    print("Test 24 — format_trace_as_conversation task dedup: OK ✅")


# ---------------------------------------------------------------------------
# Test 25 — format_trace_as_conversation: error in ActionStep
# ---------------------------------------------------------------------------


def test_format_trace_as_conversation_includes_error_in_tool_response():
    trace = TraceRecord(
        trace_id="err_001",
        prompt="Run invalid code",
        status="fail",
        steps=[
            _make_mock_step(
                "ActionStep",
                model_output_message="Let me try.",
                code_action="undefined_fn()",
                observations="",
                error="NameError: name 'undefined_fn' is not defined",
            ),
        ],
    )

    conv = format_trace_as_conversation(trace)
    tool_responses = [m["content"] for m in conv if "<tool_response>" in str(m.get("content", ""))]
    assert len(tool_responses) >= 1
    assert "NameError" in tool_responses[0]
    print("Test 25 — format_trace_as_conversation error: OK ✅")


# ---------------------------------------------------------------------------
# Test 26 — format_trace_as_conversation: "Thought:" prefix stripping
# ---------------------------------------------------------------------------


def test_format_trace_as_conversation_strips_thought_prefix():
    trace = TraceRecord(
        trace_id="thought_001",
        prompt="Q",
        status="success",
        steps=[
            _make_mock_step(
                "ActionStep",
                model_output_message="Thought: I need to search.\n<code>\nsearch()\n</code>",
                code_action="search()",
                observations="Result.",
            ),
        ],
    )

    conv = format_trace_as_conversation(trace)
    assistant_messages = [m for m in conv if m["role"] == "assistant"]
    # Should not start with "Thought:"
    for msg in assistant_messages:
        assert not msg["content"].startswith("Thought:")
        assert not msg["content"].startswith("Pensamento:")
    print("Test 26 — format_trace_as_conversation thought prefix: OK ✅")


# ---------------------------------------------------------------------------
# Test 27 — format_trace_as_conversation: Portuguese translations
# ---------------------------------------------------------------------------


def test_format_trace_as_conversation_portuguese_translations():
    trace = TraceRecord(
        trace_id="pt_001",
        prompt="Qual é a capital?",
        status="success",
        steps=[
            _make_mock_step(
                "PlanningStep",
                plan="Here are the facts I know and the plan of action that I will follow to solve the task: search then answer.",
            ),
            _make_mock_step(
                "ActionStep",
                model_output_message="Pensamento: vou pesquisar.",
                code_action='web_search("capital of France")',
                observations="Execution logs:\nsearching...\nLast output from code snippet:\nParis",
            ),
        ],
    )

    conv = format_trace_as_conversation(trace, language="pt")
    # Check that English strings were translated
    all_content = "\n".join(m["content"] for m in conv)
    assert "Aqui estão os fatos" in all_content
    assert "Registros de execução" in all_content
    assert "Última saída do trecho de código" in all_content

    # The "Pensamento:" prefix should be stripped
    assistant_messages = [m for m in conv if m["role"] == "assistant"]
    for msg in assistant_messages:
        assert not msg["content"].startswith("Pensamento:")

    print("Test 27 — format_trace_as_conversation Portuguese: OK ✅")


# ---------------------------------------------------------------------------
# Test 28 — load_system_prompt: user file
# ---------------------------------------------------------------------------


def test_load_system_prompt_reads_yaml_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = os.path.join(tmpdir, "prompts.yaml")
        with open(yaml_path, "w", encoding="utf-8") as fh:
            fh.write("system_prompt: |\n  You are a test bot.\n")

        templates = load_system_prompt(yaml_path)
        assert templates["system_prompt"].strip() == "You are a test bot."
    print("Test 28 — load_system_prompt YAML file: OK ✅")


# ---------------------------------------------------------------------------
# Test 29 — load_system_prompt: missing file
# ---------------------------------------------------------------------------


def test_load_system_prompt_raises_on_missing_file():
    try:
        load_system_prompt("/nonexistent/prompts.yaml")
        raise AssertionError("Missing file should raise FileNotFoundError")
    except FileNotFoundError as error:
        assert "not found" in str(error)
    print("Test 29 — load_system_prompt missing file: OK ✅")


# ---------------------------------------------------------------------------
# Test 30 — load_system_prompt: library default fallback
# ---------------------------------------------------------------------------


def test_load_system_prompt_falls_back_to_language_file():
    # No path given → load the bundled prompt file for the language.
    templates = load_system_prompt(None, language="en")
    assert isinstance(templates, dict)
    assert "system_prompt" in templates
    assert len(templates["system_prompt"]) > 0

    templates_pt = load_system_prompt(None, language="pt")
    assert "system_prompt" in templates_pt
    assert len(templates_pt["system_prompt"]) > 0
    # Portuguese prompt should differ from the English one.
    assert templates_pt["system_prompt"] != templates["system_prompt"]
    print("Test 30 — load_system_prompt language fallback: OK ✅")


# ---------------------------------------------------------------------------
# Test 31 — setup_triton_cache (agents variant, same as synthetic)
# ---------------------------------------------------------------------------


def test_setup_triton_cache_creates_rank_dir_and_removes_stale_files():
    original_env = {
        key: os.environ.get(key)
        for key in ["TRITON_CACHE_DIR", "SLURM_JOB_ID", "LOCAL_RANK", "CUDA_VISIBLE_DEVICES"]
    }
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = os.path.join(tmpdir, "triton")
            os.environ["TRITON_CACHE_DIR"] = cache_root
            os.environ["SLURM_JOB_ID"] = "42"
            # LOCAL_RANK takes priority over CUDA_VISIBLE_DEVICES when both are set.
            os.environ["LOCAL_RANK"] = "3"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

            # The rank should come from LOCAL_RANK, not CUDA_VISIBLE_DEVICES.
            rank_dir = os.path.join(cache_root, "42", "rank_3")
            os.makedirs(rank_dir, exist_ok=True)
            stale_file = os.path.join(rank_dir, "old_kernel.so")
            fresh_file = os.path.join(rank_dir, "new_kernel.so")
            Path(stale_file).write_text("stale")
            Path(fresh_file).write_text("fresh")
            old_time = time.time() - 7200  # 2 hours ago
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
    print("Test 31 — setup_triton_cache: OK ✅")


# ---------------------------------------------------------------------------
# Test 32 — format_trace_as_conversation: empty trace
# ---------------------------------------------------------------------------


def test_format_trace_as_conversation_empty_trace():
    trace = TraceRecord(trace_id="empty", prompt="Q", status="fail")
    conv = format_trace_as_conversation(trace)
    # Should have at least the user prompt
    assert len(conv) >= 1
    assert conv[0]["role"] == "user"
    assert conv[0]["content"] == "Q"
    print("Test 32 — format_trace_as_conversation empty: OK ✅")


# ---------------------------------------------------------------------------
# Test 33 — format_trace_as_conversation: empty tool_response discarded
# ---------------------------------------------------------------------------


def test_format_trace_as_conversation_discards_empty_tool_response():
    trace = TraceRecord(
        trace_id="empty_tr",
        prompt="Q",
        status="success",
        steps=[
            _make_mock_step(
                "ActionStep",
                model_output_message="test",
                code_action='final_answer("ok")',
                observations="Execution logs:\nLast output from code snippet:\nNone",
            ),
        ],
    )

    conv = format_trace_as_conversation(trace)
    # The empty tool_response should be discarded
    tool_responses = [m for m in conv if "<tool_response>" in str(m.get("content", ""))]
    assert len(tool_responses) == 0
    print("Test 33 — format_trace_as_conversation empty tool_response: OK ✅")


# ---------------------------------------------------------------------------
# Test 34 — format_trace_as_conversation: SystemPromptStep ignored
# ---------------------------------------------------------------------------


def test_format_trace_as_conversation_ignores_system_prompt_step():
    trace = TraceRecord(
        trace_id="sys_001",
        prompt="Q",
        system_prompt="You are helpful.",
        status="success",
        steps=[
            _make_mock_step("SystemPromptStep", system_prompt="You are helpful."),
            _make_mock_step(
                "ActionStep",
                code_action='final_answer("ok")',
                observations="Done.",
            ),
        ],
    )

    conv = format_trace_as_conversation(trace)
    # System prompt should appear exactly once (from trace.system_prompt, not from step)
    system_count = sum(1 for m in conv if m["role"] == "system")
    assert system_count == 1
    print("Test 34 — format_trace_as_conversation SystemPromptStep: OK ✅")


# ---------------------------------------------------------------------------
# Test 35 — load_metadata_entries
# ---------------------------------------------------------------------------


def test_load_metadata_entries_returns_latest_entry_per_trace_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        # No file yet → empty dict
        assert load_metadata_entries(out_dir) == {}

        metadata_path = out_dir / "metadata.jsonl"
        with open(metadata_path, "w", encoding="utf-8") as fh:
            fh.write('{"trace_id": "a", "status": "fail"}\n')
            fh.write('{"trace_id": "b", "status": "success"}\n')
            fh.write('{"trace_id": "a", "status": "success"}\n')  # reprocessed
            fh.write("garbage line\n")
            fh.write("{bad json\n")
            fh.write('{"trace_id": "c", "status": "fail"}\n')

        entries = load_metadata_entries(out_dir)
        assert set(entries) == {"a", "b", "c"}
        # Latest entry wins for duplicate trace IDs
        assert entries["a"]["status"] == "success"
        assert entries["b"]["status"] == "success"
        assert entries["c"]["status"] == "fail"
    print("Test 35 — load_metadata_entries: OK ✅")


# ---------------------------------------------------------------------------
# Test 36 — summarize_trace_metadata
# ---------------------------------------------------------------------------


def test_summarize_trace_metadata_counts_whole_dataset():
    rows = [
        {"_trace_id": "a", "prompt": "Q1"},
        {"_trace_id": "b", "prompt": "Q2"},
        {"_trace_id": "c", "prompt": "Q3"},
        {"_trace_id": "d", "prompt": "Q4"},
    ]
    entries = {
        "a": {"trace_id": "a", "status": "success"},
        "b": {"trace_id": "b", "status": "fail"},
        "c": {"trace_id": "c", "status": "success"},
        # "d" has no metadata entry → still pending
    }

    stats = summarize_trace_metadata(rows, entries)
    assert stats["total"] == 4
    assert stats["processed"] == 3
    assert stats["success"] == 2
    assert stats["failed"] == 1
    assert stats["remaining"] == 1

    # Empty dataset
    stats = summarize_trace_metadata([], {})
    assert stats["total"] == 0
    assert stats["processed"] == 0
    assert stats["remaining"] == 0
    print("Test 36 — summarize_trace_metadata: OK ✅")


# ---------------------------------------------------------------------------
# Test 37 — load_system_prompt: unsupported language
# ---------------------------------------------------------------------------


def test_load_system_prompt_raises_on_unsupported_language():
    try:
        load_system_prompt(None, language="zz")
        raise AssertionError("Unsupported language should raise ValueError")
    except ValueError as error:
        assert "zz" in str(error)
    print("Test 37 — load_system_prompt unsupported language: OK ✅")


# ---------------------------------------------------------------------------
# Test 38 — load_system_prompt: missing language file
# ---------------------------------------------------------------------------


def test_load_system_prompt_raises_when_language_file_missing():
    with tempfile.TemporaryDirectory() as tmpdir, patch("utils.PROMPTS_DIR", Path(tmpdir)):
        try:
            load_system_prompt(None, language="en")
            raise AssertionError("Missing language file should raise FileNotFoundError")
        except FileNotFoundError as error:
            assert "en" in str(error)
    print("Test 38 — load_system_prompt missing language file: OK ✅")


# ---------------------------------------------------------------------------
# Test 39 — format_trace_as_conversation: English stays English
# ---------------------------------------------------------------------------


def test_format_trace_as_conversation_english_not_translated():
    trace = TraceRecord(
        trace_id="en_001",
        prompt="What is the capital?",
        status="success",
        steps=[
            _make_mock_step(
                "PlanningStep",
                plan="Here are the facts I know and the plan of action that I will follow to solve the task: search then answer.",
            ),
            _make_mock_step(
                "ActionStep",
                model_output_message="Thought: let me search.",
                code_action='web_search("capital of France")',
                observations="Execution logs:\nsearching...\nLast output from code snippet:\nParis",
            ),
        ],
    )
    conv = format_trace_as_conversation(trace, language="en")
    all_content = "\n".join(m["content"] for m in conv)
    assert "Here are the facts I know" in all_content
    assert "Execution logs:" in all_content
    assert "Last output from code snippet:" in all_content
    assert "Aqui estão os fatos" not in all_content
    assert "Registros de execução" not in all_content
    print("Test 39 — format_trace_as_conversation English: OK ✅")


# ---------------------------------------------------------------------------
# Test 40 — LANGUAGE_CONFIGS contract
# ---------------------------------------------------------------------------


def test_language_configs_have_required_keys():
    required = {
        "prompt_file",
        "wikipedia_language",
        "thought_prefixes",
        "planning_prefixes",
        "labels",
    }
    for language, config in LANGUAGE_CONFIGS.items():
        missing = required - set(config)
        assert not missing, f"{language!r} is missing keys: {missing}"
    assert sorted(LANGUAGE_CONFIGS) == SUPPORTED_LANGUAGES
    assert "en" in SUPPORTED_LANGUAGES
    print("Test 40 — LANGUAGE_CONFIGS contract: OK ✅")


# ---------------------------------------------------------------------------
# Test 41 — format_trace_as_conversation: unknown language falls back to English
# ---------------------------------------------------------------------------


def test_format_trace_as_conversation_unknown_language_falls_back_to_english():
    trace = TraceRecord(
        trace_id="zz_001",
        prompt="Q",
        status="success",
        steps=[
            _make_mock_step(
                "ActionStep",
                model_output_message="Thought: hi.",
                code_action='web_search("x")',
                observations="Execution logs:\nsome output\nLast output from code snippet:\n42",
            ),
        ],
    )
    conv = format_trace_as_conversation(trace, language="zz")
    all_content = "\n".join(m["content"] for m in conv)
    assert "Execution logs:" in all_content
    assert "Registros de execução" not in all_content
    assert "Aqui estão os fatos" not in all_content
    print("Test 41 — format_trace_as_conversation unknown language: OK ✅")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_trace_record_defaults_and_field_assignment,
        test_output_manager_appends_valid_json_array_and_rotates,
        test_output_manager_resume_scans_existing_files,
        test_output_manager_formatted_trace_skips_failures_and_think_guard,
        test_load_dataset_reads_jsonl_with_auto_trace_id,
        test_load_dataset_uses_id_column_when_present,
        test_load_dataset_adds_ground_truth_column,
        test_load_dataset_reads_directory_of_jsonl_shards,
        test_load_dataset_rejects_unsupported_file_format,
        test_load_dataset_raises_on_missing_prompt_column,
        test_load_dataset_raises_on_empty_inputs,
        test_load_dataset_reads_parquet_when_pyarrow_available,
        test_load_processed_ids_parses_metadata_jsonl_and_handles_malformed,
        test_append_metadata_entry_writes_one_line_per_trace,
        test_normalize_answer_various_cases,
        test_compare_answer_exact_and_substring_and_edge_cases,
        test_conversation_has_unclosed_think_detection,
        test_extract_model_output_text_variants,
        test_extract_step_field_helpers,
        test_extract_step_type_labels,
        test_xml_wrapping_helpers,
        test_format_trace_as_conversation_basic_structure,
        test_format_trace_as_conversation_includes_planning_step,
        test_format_trace_as_conversation_skips_duplicate_task_step,
        test_format_trace_as_conversation_includes_error_in_tool_response,
        test_format_trace_as_conversation_strips_thought_prefix,
        test_format_trace_as_conversation_portuguese_translations,
        test_load_system_prompt_reads_yaml_file,
        test_load_system_prompt_raises_on_missing_file,
        test_load_system_prompt_falls_back_to_language_file,
        test_setup_triton_cache_creates_rank_dir_and_removes_stale_files,
        test_format_trace_as_conversation_empty_trace,
        test_format_trace_as_conversation_discards_empty_tool_response,
        test_format_trace_as_conversation_ignores_system_prompt_step,
        test_load_metadata_entries_returns_latest_entry_per_trace_id,
        test_summarize_trace_metadata_counts_whole_dataset,
        test_load_system_prompt_raises_on_unsupported_language,
        test_load_system_prompt_raises_when_language_file_missing,
        test_format_trace_as_conversation_english_not_translated,
        test_language_configs_have_required_keys,
        test_format_trace_as_conversation_unknown_language_falls_back_to_english,
    ]
    for test in tests:
        test()
    print("\n" + "=" * 50)
    print(f"All {len(tests)} tests passed ✅")
    print("=" * 50)
