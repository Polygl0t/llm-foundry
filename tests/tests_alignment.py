"""
Alignment utilities test suite.

Tests the helper functions in alignment/utils.py:
  - get_logger
  - setup_distributed_state
  - load_training_dataset
  - split_dataset
  - load_tokenizer
  - resolve_checkpoint_path
  - run_training

Trainer internals, TRL, Transformers model loading, and real datasets loading are
mocked or faked deliberately.

Run with:
    python tests/tests_alignment.py

Requirements:
- No GPU required
- No datasets, transformers, accelerate, or trl installation required
"""

# %%
#######################################
# 1. Imports & Setup
#######################################
import os
import sys
import tempfile
import types

sys.pycache_prefix = os.path.join(tempfile.gettempdir(), "pycache")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
ALIGNMENT_DIR = os.path.join(REPO_ROOT, "alignment")
if ALIGNMENT_DIR not in sys.path:
    sys.path.insert(0, ALIGNMENT_DIR)


class FakeState:
    def __init__(self, process_index=0):
        self.process_index = process_index
        self.wait_calls = 0

    def wait_for_everyone(self):
        self.wait_calls += 1

    def __str__(self):
        return f"FakeState(process_index={self.process_index})"


class FakeAccelerateModule:
    process_index = 0
    last_state = None

    @classmethod
    def PartialState(cls):
        cls.last_state = FakeState(cls.process_index)
        return cls.last_state


class FakeDatasetsModule:
    last_load_dataset_kwargs = None
    return_value = object()

    @classmethod
    def load_dataset(cls, *args, **kwargs):
        cls.last_load_dataset_kwargs = {"args": args, "kwargs": kwargs}
        return cls.return_value


class FakeAutoTokenizer:
    next_tokenizer = None
    last_from_pretrained_kwargs = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.last_from_pretrained_kwargs = {"args": args, "kwargs": kwargs}
        return cls.next_tokenizer


class FakeTransformersModule:
    AutoTokenizer = FakeAutoTokenizer


sys.modules.setdefault("accelerate", FakeAccelerateModule)
sys.modules.setdefault("datasets", FakeDatasetsModule)
sys.modules.setdefault("transformers", FakeTransformersModule)

from utils import (  # noqa: E402
    get_logger,
    load_tokenizer,
    load_training_dataset,
    resolve_checkpoint_path,
    run_training,
    setup_distributed_state,
    split_dataset,
)

print("All imports OK ✅")


# %%
#######################################
# Section 1 - Logging / Distributed State
#######################################

def test_01_getlogger_returns_a_working_logger():
    logger = get_logger("TestAlignment")
    assert logger.name == "TestAlignment"
    logger.info("Logger works.")
    logger.warning("Warning works.")
    print("Test 1 - get_logger: OK ✅")


def test_02_getlogger_is_idempotent():
    logger_a = get_logger("TestAlignment_idem")
    n_handlers = len(logger_a.handlers)
    logger_b = get_logger("TestAlignment_idem")
    assert logger_a is logger_b, "Should return the same Logger instance"
    assert len(logger_b.handlers) == n_handlers, "No new handlers should be added on second call"
    print("Test 2 - get_logger idempotent: OK ✅")


def test_03_setupdistributedstate_identifies_master_process():
    FakeAccelerateModule.process_index = 0
    logger = get_logger("TestAlignment_master")
    state, master_process = setup_distributed_state(logger)
    assert state.process_index == 0
    assert master_process is True
    print("Test 3 - setup_distributed_state master: OK ✅")


def test_04_setupdistributedstate_identifies_worker_process():
    FakeAccelerateModule.process_index = 2
    logger = get_logger("TestAlignment_worker")
    state, master_process = setup_distributed_state(logger)
    assert state.process_index == 2
    assert master_process is False
    FakeAccelerateModule.process_index = 0
    print("Test 4 - setup_distributed_state worker: OK ✅")


# %%
#######################################
# Section 2 - Dataset Helpers
#######################################

def test_05_loadtrainingdataset_collects_sorted_jsonl_files_and_waits():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "data")
        os.makedirs(data_dir)
        explicit_file = os.path.join(tmpdir, "explicit.jsonl")
        shard_b = os.path.join(data_dir, "b.jsonl")
        shard_a = os.path.join(data_dir, "a.jsonl")
        ignored_file = os.path.join(data_dir, "ignored.parquet")

        for path in (explicit_file, shard_b, shard_a, ignored_file):
            with open(path, "w") as f:
                f.write("{}\n")

        state = FakeState()
        FakeDatasetsModule.return_value = "loaded-dataset"
        result = load_training_dataset(
            [data_dir, explicit_file, ignored_file],
            "jsonl",
            num_proc=16,
            cache_dir="/tmp/cache",
            state=state,
        )

        expected_files = sorted([shard_a, shard_b, explicit_file])
        call = FakeDatasetsModule.last_load_dataset_kwargs
        assert result == "loaded-dataset"
        assert state.wait_calls == 1
        assert call["args"] == ("json",)
        assert call["kwargs"]["data_files"] == expected_files
        assert call["kwargs"]["split"] == "train"
        assert call["kwargs"]["num_proc"] == 3
        assert call["kwargs"]["cache_dir"] == "/tmp/cache"
    print("Test 5 - load_training_dataset jsonl collection: OK ✅")


def test_06_loadtrainingdataset_rejects_unknown_dataset_type():
    state = FakeState()
    try:
        load_training_dataset("/tmp/nowhere", "csv", 1, None, state)
    except AssertionError as exc:
        assert "Dataset type must be either" in str(exc)
    else:
        raise AssertionError("Expected AssertionError for unsupported dataset type")
    print("Test 6 - load_training_dataset invalid type: OK ✅")


class FakeTestSplit:
    def __init__(self):
        self.to_json_calls = []

    def to_json(self, path, orient, lines):
        self.to_json_calls.append((path, orient, lines))
        with open(path, "w") as f:
            f.write('{"id": 1}\n')


class FakeSplittableDataset:
    def __init__(self):
        self.train_test_split_calls = []
        self.test = FakeTestSplit()

    def train_test_split(self, test_size, seed):
        self.train_test_split_calls.append((test_size, seed))
        return {"train": "train-split", "test": self.test}



def test_07_splitdataset_returns_original_when_no_test_size():
    dataset = FakeSplittableDataset()
    state = FakeState()
    result = split_dataset(dataset, None, 42, "/tmp/checkpoints", True, True, state)
    assert result is dataset
    assert dataset.train_test_split_calls == []
    assert state.wait_calls == 0
    print("Test 7 - split_dataset no split: OK ✅")


def test_08_splitdataset_saves_test_set_on_master():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = FakeSplittableDataset()
        state = FakeState()
        result = split_dataset(dataset, 2, 123, tmpdir, True, True, state)

        test_file = os.path.join(tmpdir, "test_set.jsonl")
        assert result["train"] == "train-split"
        assert result["test"] is dataset.test
        assert dataset.train_test_split_calls == [(2, 123)]
        assert dataset.test.to_json_calls == [(test_file, "records", True)]
        assert os.path.exists(test_file)
        assert state.wait_calls == 1
    print("Test 8 - split_dataset saves test set: OK ✅")


def test_09_splitdataset_worker_does_not_save_test_set():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = FakeSplittableDataset()
        state = FakeState()
        split_dataset(dataset, 0.1, 99, tmpdir, True, False, state)
        assert dataset.test.to_json_calls == []
        assert not os.path.exists(os.path.join(tmpdir, "test_set.jsonl"))
        assert state.wait_calls == 1
    print("Test 9 - split_dataset worker no save: OK ✅")


# %%
#######################################
# Section 3 - Tokenizer Helper
#######################################

class FakeTokenizer:
    def __init__(self, chat_template="template", pad_token="<pad>", eos_token="<eos>"):
        self.chat_template = chat_template
        self.pad_token = pad_token
        self.eos_token = eos_token



def test_10_loadtokenizer_uses_from_pretrained_options():
    tokenizer = FakeTokenizer()
    FakeAutoTokenizer.next_tokenizer = tokenizer
    result = load_tokenizer("model-name", 4096, "/tmp/cache")

    call = FakeAutoTokenizer.last_from_pretrained_kwargs
    assert result is tokenizer
    assert call["args"] == ("model-name",)
    assert call["kwargs"]["model_max_length"] == 4096
    assert call["kwargs"]["cache_dir"] == "/tmp/cache"
    assert call["kwargs"]["use_fast"] is True
    assert call["kwargs"]["trust_remote_code"] is True
    print("Test 10 - load_tokenizer from_pretrained options: OK ✅")


def test_11_loadtokenizer_reads_chat_template_when_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        template_path = os.path.join(tmpdir, "chat_template.jinja")
        with open(template_path, "w") as f:
            f.write("{{ messages }}")

        tokenizer = FakeTokenizer(chat_template=None)
        FakeAutoTokenizer.next_tokenizer = tokenizer
        result = load_tokenizer("model-name", 2048, None, chat_template_path=template_path)

        assert result.chat_template == "{{ messages }}"
    print("Test 11 - load_tokenizer reads chat template: OK ✅")


def test_12_loadtokenizer_requires_chat_template_when_missing():
    FakeAutoTokenizer.next_tokenizer = FakeTokenizer(chat_template=None)
    try:
        load_tokenizer("model-name", 2048, None)
    except AssertionError as exc:
        assert "Tokenizer does not have a chat template" in str(exc)
    else:
        raise AssertionError("Expected AssertionError for missing chat template")
    print("Test 12 - load_tokenizer missing chat template: OK ✅")


def test_13_loadtokenizer_allows_eos_pad_token_when_requested():
    tokenizer = FakeTokenizer(pad_token=None, eos_token="<eos>")
    FakeAutoTokenizer.next_tokenizer = tokenizer
    result = load_tokenizer("model-name", 2048, None, allow_eos_pad_token=True)
    assert result.pad_token == "<eos>"
    print("Test 13 - load_tokenizer allow eos pad: OK ✅")


def test_14_loadtokenizer_requires_pad_token_by_default():
    FakeAutoTokenizer.next_tokenizer = FakeTokenizer(pad_token=None, eos_token="<eos>")
    try:
        load_tokenizer("model-name", 2048, None)
    except AssertionError as exc:
        assert "does not have a pad token" in str(exc)
    else:
        raise AssertionError("Expected AssertionError for missing pad token")
    print("Test 14 - load_tokenizer missing pad token: OK ✅")


def test_15_loadtokenizer_rejects_pad_equal_to_eos_by_default():
    FakeAutoTokenizer.next_tokenizer = FakeTokenizer(pad_token="<eos>", eos_token="<eos>")
    try:
        load_tokenizer("model-name", 2048, None)
    except AssertionError as exc:
        assert "pad token is the same as the eos token" in str(exc)
    else:
        raise AssertionError("Expected AssertionError for pad_token == eos_token")
    print("Test 15 - load_tokenizer pad equals eos: OK ✅")


# %%
#######################################
# Section 4 - Checkpoint / Training Helpers
#######################################

def test_16_resolvecheckpointpath_returns_latest_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "checkpoint-2"))
        os.makedirs(os.path.join(tmpdir, "checkpoint-10"))
        os.makedirs(os.path.join(tmpdir, "not-a-checkpoint"))

        logger = get_logger("TestAlignment_checkpoint")
        result = resolve_checkpoint_path(tmpdir, True, logger)
        assert result == os.path.join(tmpdir, "checkpoint-10")
    print("Test 16 - resolve_checkpoint_path latest checkpoint: OK ✅")


def test_17_resolvecheckpointpath_keeps_direct_path():
    direct_path = "/tmp/checkpoint-123"
    logger = get_logger("TestAlignment_checkpoint_direct")
    result = resolve_checkpoint_path(direct_path, False, logger)
    assert result == direct_path
    print("Test 17 - resolve_checkpoint_path direct path: OK ✅")


class FakeTrainer:
    def __init__(self, fail=False):
        self.fail = fail
        self.train_calls = []
        self.save_model_calls = []

    def train(self, resume_from_checkpoint=None):
        self.train_calls.append(resume_from_checkpoint)
        if self.fail:
            raise RuntimeError("boom")

    def save_model(self, path):
        self.save_model_calls.append(path)



def test_18_runtraining_trains_and_saves_final_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = FakeTrainer()
        logger = get_logger("TestAlignment_run_training")
        run_training(trainer, "checkpoint-path", tmpdir, True, logger)

        assert trainer.train_calls == ["checkpoint-path"]
        assert trainer.save_model_calls == [os.path.join(tmpdir, "final")]
    print("Test 18 - run_training success: OK ✅")


def test_19_runtraining_saves_last_model_on_failure_and_reraises():
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = FakeTrainer(fail=True)
        logger = get_logger("TestAlignment_run_training_fail")

        try:
            run_training(trainer, None, tmpdir, True, logger)
        except RuntimeError as exc:
            assert str(exc) == "boom"
        else:
            raise AssertionError("Expected RuntimeError from failed trainer")

        assert trainer.train_calls == [None]
        assert trainer.save_model_calls == [os.path.join(tmpdir, "last")]
    print("Test 19 - run_training failure saves last: OK ✅")


if __name__ == "__main__":
    test_01_getlogger_returns_a_working_logger()
    test_02_getlogger_is_idempotent()
    test_03_setupdistributedstate_identifies_master_process()
    test_04_setupdistributedstate_identifies_worker_process()
    test_05_loadtrainingdataset_collects_sorted_jsonl_files_and_waits()
    test_06_loadtrainingdataset_rejects_unknown_dataset_type()
    test_07_splitdataset_returns_original_when_no_test_size()
    test_08_splitdataset_saves_test_set_on_master()
    test_09_splitdataset_worker_does_not_save_test_set()
    test_10_loadtokenizer_uses_from_pretrained_options()
    test_11_loadtokenizer_reads_chat_template_when_missing()
    test_12_loadtokenizer_requires_chat_template_when_missing()
    test_13_loadtokenizer_allows_eos_pad_token_when_requested()
    test_14_loadtokenizer_requires_pad_token_by_default()
    test_15_loadtokenizer_rejects_pad_equal_to_eos_by_default()
    test_16_resolvecheckpointpath_returns_latest_checkpoint()
    test_17_resolvecheckpointpath_keeps_direct_path()
    test_18_runtraining_trains_and_saves_final_model()
    test_19_runtraining_saves_last_model_on_failure_and_reraises()
    print("\n" + "=" * 50)
    print("All tests passed ✅")
    print("=" * 50)
