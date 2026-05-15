"""
Data/tokenization test suite.

Tests the surrounding logic of the data/tokenization scripts:
  - decontaminate.py (argument parser, exact/approximate filtering, output chunking)
  - make_validation_split.py (metadata, file selection, split creation)
  - pack.py (packing functions and main validation paths)
  - tokenize.py (tokenizer loading, tokenization functions, main filtering/metadata paths)

Heavy HuggingFace/Datasets internals are deliberately NOT tested here.

Run with:
    python tests/tests_data_tokenization.py

Requirements:
- No GPU required
- No network required
- No real transformers/datasets installation required
"""

# %%
#######################################
# 1. Imports & Setup
#######################################
import sys
import os
import argparse
import importlib
import json
import tempfile
import types
from unittest.mock import MagicMock, patch

sys.pycache_prefix = os.path.join(tempfile.gettempdir(), "pycache")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TOKENIZATION_DIR = os.path.join(REPO_ROOT, "data", "tokenization")
if TOKENIZATION_DIR not in sys.path:
    sys.path.insert(0, TOKENIZATION_DIR)


class TinyDataset:
    """Small in-memory stand-in for the subset of datasets.Dataset used here."""

    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]
        self.column_names = list(self.rows[0].keys()) if self.rows else []
        self.features = {name: object() for name in self.column_names}
        self.saved_json_path = None
        self.saved_parquet_path = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]

    def __repr__(self):
        return f"TinyDataset(num_rows={len(self)})"

    def select(self, indices):
        return TinyDataset([self.rows[int(i)] for i in list(indices)])

    def filter(self, fn, batched=False, batch_size=None, num_proc=None, desc=None):
        if batched:
            batch = {col: [row[col] for row in self.rows] for col in self.column_names}
            keep = fn(batch)
            return TinyDataset([row for row, should_keep in zip(self.rows, keep) if should_keep])
        return TinyDataset([row for row in self.rows if fn(row)])

    def map(self, fn, batched=False, remove_columns=None, desc=None, num_proc=None, load_from_cache_file=True):
        if batched:
            batch = {col: [row[col] for row in self.rows] for col in self.column_names}
            mapped = fn(batch)
            if not mapped:
                return TinyDataset([])
            keys = list(mapped.keys())
            n_rows = len(mapped[keys[0]]) if keys else 0
            return TinyDataset([{key: mapped[key][i] for key in keys} for i in range(n_rows)])

        mapped_rows = []
        for row in self.rows:
            mapped_rows.append(fn(dict(row)))
        return TinyDataset(mapped_rows)

    def remove_columns(self, columns):
        if isinstance(columns, str):
            columns = [columns]
        return TinyDataset([
            {key: value for key, value in row.items() if key not in columns}
            for row in self.rows
        ])

    def shuffle(self, seed=None):
        return TinyDataset(self.rows)

    def to_json(self, path):
        self.saved_json_path = path
        with open(path, "w") as f:
            for row in self.rows:
                f.write(json.dumps(row) + "\n")

    def to_parquet(self, path):
        self.saved_parquet_path = path
        with open(path, "w") as f:
            f.write(json.dumps(self.rows))


class FakeDatasetsModule(types.ModuleType):
    def __init__(self):
        super().__init__("datasets")
        self.load_calls = []
        self.datasets_by_file = {}
        self.next_datasets = []

    def load_dataset(self, fmt, data_files=None, split="train", cache_dir=None, **kwargs):
        self.load_calls.append({
            "fmt": fmt,
            "data_files": data_files,
            "split": split,
            "cache_dir": cache_dir,
            **kwargs,
        })
        if isinstance(data_files, list):
            rows = []
            for path in data_files:
                if path in self.datasets_by_file:
                    rows.extend(self.datasets_by_file[path].rows)
            if rows:
                return TinyDataset(rows)
        elif data_files in self.datasets_by_file:
            return self.datasets_by_file[data_files]
        if self.next_datasets:
            return self.next_datasets.pop(0)
        return TinyDataset([])

    def concatenate_datasets(self, datasets_list):
        rows = []
        for dataset in datasets_list:
            rows.extend(dataset.rows)
        return TinyDataset(rows)


fake_datasets = FakeDatasetsModule()
fake_transformers = types.ModuleType("transformers")
fake_transformers.AutoTokenizer = MagicMock()
sys.modules.setdefault("datasets", fake_datasets)
sys.modules.setdefault("transformers", fake_transformers)


def _load_module_from_tokenization_dir(module_name, filename):
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(TOKENIZATION_DIR, filename),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


make_validation_split = _load_module_from_tokenization_dir(
    "tokenization_make_validation_split_mod", "make_validation_split.py"
)
decontaminate = _load_module_from_tokenization_dir("tokenization_decontaminate_mod", "decontaminate.py")
pack = _load_module_from_tokenization_dir("tokenization_pack_mod", "pack.py")
tokenize = _load_module_from_tokenization_dir("tokenization_tokenize_mod", "tokenize.py")
from utils import save_metadata

print("All imports OK ✅")


def _reset_fake_datasets():
    fake_datasets.load_calls = []
    fake_datasets.datasets_by_file = {}
    fake_datasets.next_datasets = []


# %%
#######################################
# Section 1 - make_validation_split.py
#######################################

def test_01_read_metadata_returns_empty_dict_for_missing_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = make_validation_split.read_metadata(os.path.join(tmpdir, ".metadata"))
        assert result == {}
    print("Test 1 - read_metadata missing file: OK ✅")


def test_02_read_metadata_parses_key_value_strings():
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = os.path.join(tmpdir, ".metadata")
        with open(meta_path, "w") as f:
            f.write("Samples: 10\n")
            f.write("Tokens: 40\n")
            f.write("ignored line\n")
        assert make_validation_split.read_metadata(meta_path) == {"Samples": "10", "Tokens": "40"}
    print("Test 2 - read_metadata parsing: OK ✅")


def test_03_get_files_from_dirs_returns_sorted_json_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        for name in ["b.jsonl", "a.jsonl", "skip.txt"]:
            open(os.path.join(tmpdir, name), "w").close()
        files = make_validation_split.get_files_from_dirs([tmpdir], "json")
        assert [os.path.basename(path) for path in files] == ["a.jsonl", "b.jsonl"]
    print("Test 3 - get_files_from_dirs json sorted: OK ✅")


def test_04_get_files_from_dirs_can_sample_n_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(5):
            open(os.path.join(tmpdir, f"part_{i}.parquet"), "w").close()
        with patch.object(make_validation_split.random, "sample", return_value=[
            os.path.join(tmpdir, "part_4.parquet"),
            os.path.join(tmpdir, "part_1.parquet"),
        ]):
            files = make_validation_split.get_files_from_dirs([tmpdir], "parquet", n_files=2)
        assert [os.path.basename(path) for path in files] == ["part_1.parquet", "part_4.parquet"]
    print("Test 4 - get_files_from_dirs sampled files: OK ✅")


def test_05_get_files_from_dirs_raises_when_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            make_validation_split.get_files_from_dirs([tmpdir], "json")
            raise AssertionError("Expected FileNotFoundError")
        except FileNotFoundError:
            pass
    print("Test 5 - get_files_from_dirs empty folder: OK ✅")


def test_06_validation_split_main_writes_json_split_and_metadata():
    _reset_fake_datasets()
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(input_dir)
        file_a = os.path.join(input_dir, "a.jsonl")
        file_b = os.path.join(input_dir, "b.jsonl")
        open(file_a, "w").close()
        open(file_b, "w").close()
        save_metadata(input_dir, Tokenizer="toy-tokenizer")
        fake_datasets.datasets_by_file[file_a] = TinyDataset([
            {"input_ids": [1, 2, 3, 4], "id": "a0"},
            {"input_ids": [5, 6, 7, 8], "id": "a1"},
            {"input_ids": [9, 10, 11, 12], "id": "a2"},
        ])
        fake_datasets.datasets_by_file[file_b] = TinyDataset([
            {"input_ids": [13, 14, 15, 16], "id": "b0"},
            {"input_ids": [17, 18, 19, 20], "id": "b1"},
            {"input_ids": [21, 22, 23, 24], "id": "b2"},
        ])

        make_validation_split.main([input_dir], output_dir, "json", "valid", n_samples=4)

        with open(os.path.join(output_dir, "valid.jsonl")) as f:
            valid_rows = [json.loads(line) for line in f]
        assert [row["id"] for row in valid_rows] == ["a0", "a1", "b0", "b1"]
        assert make_validation_split.read_metadata(os.path.join(output_dir, ".metadata")) == {
            "Samples": "4",
            "Tokens": "16",
            "Tokens per chunk": "16",
            "Block size": "4",
            "Chunks": "1",
            "Tokenizer": "toy-tokenizer",
        }
    print("Test 6 - validation split main json output: OK ✅")


def test_07_validation_split_main_accumulates_existing_metadata():
    _reset_fake_datasets()
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        file_path = os.path.join(input_dir, "a.jsonl")
        open(file_path, "w").close()
        save_metadata(output_dir, Samples=2, Tokens=8, Chunks=1)
        fake_datasets.datasets_by_file[file_path] = TinyDataset([
            {"input_ids": [1, 2], "id": "new0"},
            {"input_ids": [3, 4], "id": "new1"},
        ])

        make_validation_split.main([input_dir], output_dir, "json", "valid", n_samples=1)

        meta = make_validation_split.read_metadata(os.path.join(output_dir, ".metadata"))
        assert meta["Samples"] == "3"
        assert meta["Tokens"] == "10"
        assert meta["Chunks"] == "2"
    print("Test 7 - validation split accumulates metadata: OK ✅")


def test_08_validation_split_main_rejects_too_many_samples():
    _reset_fake_datasets()
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(input_dir)
        file_path = os.path.join(input_dir, "a.jsonl")
        open(file_path, "w").close()
        fake_datasets.datasets_by_file[file_path] = TinyDataset([{"input_ids": [1]}])
        try:
            make_validation_split.main([input_dir], output_dir, "json", "valid", n_samples=2)
            raise AssertionError("Expected ValueError")
        except ValueError as exc:
            assert "greater than total" in str(exc)
    print("Test 8 - validation split rejects too many samples: OK ✅")


# %%
#######################################
# Section 2 - pack.py
#######################################

def test_09_concatenate_pack_splits_blocks_and_discards_tail():
    pack_fn = pack.create_concatenate_function(4, ["input_ids", "attention_mask"])
    result = pack_fn({
        "input_ids": [[1, 2], [3, 4, 5], [6, 7, 8, 9]],
        "attention_mask": [[1, 1], [1, 1, 1], [1, 1, 1, 1]],
    })
    assert result["input_ids"] == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert result["attention_mask"] == [[1, 1, 1, 1], [1, 1, 1, 1]]
    assert result["seq_lengths"] == [4, 4]
    print("Test 9 - concatenate packing: OK ✅")


def test_10_concatenate_pack_handles_no_full_blocks():
    pack_fn = pack.create_concatenate_function(10, ["input_ids"])
    result = pack_fn({"input_ids": [[1, 2, 3]]})
    assert result == {"input_ids": [], "seq_lengths": []}
    print("Test 10 - concatenate packing no blocks: OK ✅")


def test_11_bfd_pack_pads_partial_chunks():
    pack_fn = pack.create_bfd_function(5, ["input_ids", "labels", "attention_mask"], {
        "input_ids": 0,
        "labels": -100,
        "attention_mask": 0,
    })
    result = pack_fn({
        "input_ids": [[1, 2, 3], [4]],
        "labels": [[1, 2, 3], [4]],
        "attention_mask": [[1, 1, 1], [1]],
    })
    assert result["input_ids"] == [[1, 2, 3, 4, 0]]
    assert result["labels"] == [[1, 2, 3, 4, -100]]
    assert result["attention_mask"] == [[1, 1, 1, 1, 0]]
    assert result["seq_lengths"] == [5]
    print("Test 11 - bfd pads partial chunks: OK ✅")


def test_12_bfd_pack_discards_empty_and_too_long_sequences():
    pack_fn = pack.create_bfd_function(4, ["input_ids"], {"input_ids": 0})
    result = pack_fn({"input_ids": [[], [1, 2, 3, 4, 5], [6, 7]]})
    assert result["input_ids"] == [[6, 7, 0, 0]]
    assert result["seq_lengths"] == [4]
    print("Test 12 - bfd discards invalid lengths: OK ✅")


def test_13_bfd_pack_uses_seq_lengths_when_present():
    pack_fn = pack.create_bfd_function(4, ["input_ids"], {"input_ids": 0})
    result = pack_fn({"input_ids": [[1, 2, 99], [3, 4]], "seq_lengths": [2, 2]})
    assert result["input_ids"] == [[1, 2, 3, 4]]
    assert result["seq_lengths"] == [4]
    print("Test 13 - bfd uses seq_lengths: OK ✅")


def test_14_pack_main_rejects_dataset_without_input_ids():
    args = argparse.Namespace(
        input_path="ignored",
        cache_dir=None,
        seed=None,
        strategy="concatenate",
        block_size=4,
        pad_token_id=None,
        num_proc=1,
        max_tokens=None,
        output_dir="ignored",
        output_type="jsonl",
        tokens_per_chunk=100,
    )
    with patch.object(pack, "DatasetLoader") as loader_cls:
        loader_cls.return_value.load.return_value = TinyDataset([{"text": "hello"}])
        try:
            pack.main(args)
            raise AssertionError("Expected ValueError")
        except ValueError as exc:
            assert "input_ids" in str(exc)
    print("Test 14 - pack main requires input_ids: OK ✅")


def test_15_pack_main_bfd_requires_pad_token_id():
    args = argparse.Namespace(
        input_path="ignored",
        cache_dir=None,
        seed=None,
        strategy="bfd",
        block_size=4,
        pad_token_id=None,
        num_proc=1,
        max_tokens=None,
        output_dir="ignored",
        output_type="jsonl",
        tokens_per_chunk=100,
    )
    with patch.object(pack, "DatasetLoader") as loader_cls:
        loader_cls.return_value.load.return_value = TinyDataset([{"input_ids": [1, 2]}])
        try:
            pack.main(args)
            raise AssertionError("Expected ValueError")
        except ValueError as exc:
            assert "pad_token_id" in str(exc)
    print("Test 15 - pack main bfd requires pad token: OK ✅")


def test_16_pack_main_saves_packed_dataset_and_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = argparse.Namespace(
            input_path="ignored",
            cache_dir=None,
            seed=None,
            strategy="concatenate",
            block_size=3,
            pad_token_id=None,
            num_proc=1,
            max_tokens=None,
            output_dir=tmpdir,
            output_type="jsonl",
            tokens_per_chunk=6,
        )
        dataset = TinyDataset([{"input_ids": [1, 2]}, {"input_ids": [3, 4]}, {"input_ids": [5, 6]}])
        with patch.object(pack, "DatasetLoader") as loader_cls:
            loader_cls.return_value.load.return_value = dataset
            pack.main(args)
        files = sorted(name for name in os.listdir(tmpdir) if name.endswith(".jsonl"))
        assert files == ["train-00000-of-00001.jsonl"]
        meta = make_validation_split.read_metadata(os.path.join(tmpdir, ".metadata"))
        assert meta["samples"] == "2"
        assert meta["tokens"] == "6"
        assert meta["strategy"] == "concatenate"
        assert meta["packed_columns"] == "input_ids"
    print("Test 16 - pack main saves output: OK ✅")


# %%
#######################################
# Section 3 - tokenize.py
#######################################

class FakeTokenizer:
    def __init__(self):
        self.bos_token_id = 101
        self.eos_token_id = 102
        self.chat_template = "template"

    def __call__(self, texts, return_attention_mask=False, return_token_type_ids=False, add_special_tokens=False):
        return {"input_ids": [[ord(ch) % 50 for ch in text] for text in texts]}

    def apply_chat_template(self, messages_batch, return_assistant_tokens_mask, return_dict, add_generation_prompt):
        input_ids = []
        assistant_masks = []
        for messages in messages_batch:
            seq = []
            mask = []
            for message in messages:
                role_mask = 1 if message["role"] == "assistant" else 0
                tokens = [len(message["content"]), len(message["role"])]
                seq.extend(tokens)
                mask.extend([role_mask] * len(tokens))
            input_ids.append(seq)
            assistant_masks.append(mask)
        return {"input_ids": input_ids, "assistant_masks": assistant_masks}


def _tokenize_args(**overrides):
    defaults = dict(
        tokenizer_name="fake-tokenizer",
        cache_dir=None,
        token=None,
        chat_template_path=None,
        text_column="text",
        apply_chat_template=False,
        add_bos_token=False,
        add_eos_token=False,
        return_seq_lengths=True,
        return_attention_mask=False,
        return_labels=False,
        return_assistant_masks=False,
        input_path="ignored",
        output_dir="ignored",
        output_type="jsonl",
        split="train",
        subset=None,
        seed=None,
        num_proc=1,
        max_length=None,
        max_tokens=None,
        tokens_per_chunk=100,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_17_load_tokenizer_rejects_assistant_masks_without_chat_template():
    args = _tokenize_args(return_assistant_masks=True, apply_chat_template=False)
    with patch.object(tokenize.AutoTokenizer, "from_pretrained", return_value=FakeTokenizer()):
        try:
            tokenize.load_tokenizer(args)
            raise AssertionError("Expected ValueError")
        except ValueError as exc:
            assert "requires --apply_chat_template" in str(exc)
    print("Test 17 - load_tokenizer assistant masks require template: OK ✅")


def test_18_load_tokenizer_loads_chat_template_from_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        template_path = os.path.join(tmpdir, "template.jinja")
        with open(template_path, "w") as f:
            f.write("custom template")
        tokenizer_obj = FakeTokenizer()
        tokenizer_obj.chat_template = None
        args = _tokenize_args(apply_chat_template=True, chat_template_path=template_path)
        with patch.object(tokenize.AutoTokenizer, "from_pretrained", return_value=tokenizer_obj):
            loaded = tokenize.load_tokenizer(args)
        assert loaded.chat_template == "custom template"
    print("Test 18 - load_tokenizer reads chat template file: OK ✅")


def test_19_load_tokenizer_requires_chat_template_when_missing():
    tokenizer_obj = FakeTokenizer()
    tokenizer_obj.chat_template = None
    args = _tokenize_args(apply_chat_template=True)
    with patch.object(tokenize.AutoTokenizer, "from_pretrained", return_value=tokenizer_obj):
        try:
            tokenize.load_tokenizer(args)
            raise AssertionError("Expected ValueError")
        except ValueError as exc:
            assert "no chat template" in str(exc)
    print("Test 19 - load_tokenizer missing chat template: OK ✅")


def test_20_standard_tokenize_adds_special_tokens_and_masks():
    args = _tokenize_args(add_bos_token=True, add_eos_token=True, return_attention_mask=True, return_labels=True)
    tokenize_fn = tokenize.create_tokenize_function(FakeTokenizer(), args)
    result = tokenize_fn({"text": ["ab", "c"]})
    assert result["input_ids"] == [[101, 47, 48, 102], [101, 49, 102]]
    assert result["seq_lengths"] == [4, 3]
    assert result["attention_mask"] == [[1, 1, 1, 1], [1, 1, 1]]
    assert result["labels"] == result["input_ids"]
    print("Test 20 - standard tokenize function: OK ✅")


def test_21_chat_tokenize_returns_assistant_masks_and_masked_labels():
    args = _tokenize_args(
        text_column="messages",
        apply_chat_template=True,
        return_assistant_masks=True,
        return_labels=True,
    )
    tokenize_fn = tokenize.create_tokenize_function(FakeTokenizer(), args)
    result = tokenize_fn({"messages": [[
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "ok"},
    ]]})
    # Chat template is solely responsible for special tokens — no extra BOS/EOS injected.
    assert result["input_ids"] == [[5, 4, 2, 9]]
    assert result["assistant_masks"] == [[0, 0, 1, 1]]
    assert result["labels"] == [[-100, -100, 2, 9]]
    print("Test 21 - chat tokenize masks labels: OK ✅")


def test_21b_load_tokenizer_rejects_chat_template_with_bos_or_eos():
    for kwargs in ({"add_bos_token": True}, {"add_eos_token": True}):
        args = _tokenize_args(apply_chat_template=True, **kwargs)
        with patch.object(tokenize.AutoTokenizer, "from_pretrained", return_value=FakeTokenizer()):
            try:
                tokenize.load_tokenizer(args)
                raise AssertionError("Expected ValueError")
            except ValueError as exc:
                assert "must not be combined with --apply_chat_template" in str(exc)
    print("Test 21b - load_tokenizer rejects chat template + BOS/EOS: OK ✅")


def test_22_tokenize_main_rejects_missing_text_column():
    args = _tokenize_args(text_column="missing")
    with patch.object(tokenize, "DatasetLoader") as loader_cls:
        loader_cls.return_value.load.return_value = TinyDataset([{"text": "hello"}])
        try:
            tokenize.main(args)
            raise AssertionError("Expected ValueError")
        except ValueError as exc:
            assert "Column 'missing'" in str(exc)
    print("Test 22 - tokenize main validates text column: OK ✅")


def test_23_tokenize_main_filters_truncates_saves_and_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _tokenize_args(
            output_dir=tmpdir,
            add_bos_token=True,
            add_eos_token=True,
            return_seq_lengths=False,
            max_length=5,
            max_tokens=4,
            tokens_per_chunk=10,
        )
        dataset = TinyDataset([{"text": "ab"}, {"text": "abcdef"}, {"text": "c"}])
        with patch.object(tokenize, "DatasetLoader") as loader_cls:
            loader_cls.return_value.load.return_value = dataset
            with patch.object(tokenize.AutoTokenizer, "from_pretrained", return_value=FakeTokenizer()):
                tokenize.main(args)
        output_files = [name for name in os.listdir(tmpdir) if name.endswith(".jsonl")]
        assert output_files == ["train-00000-of-00001.jsonl"]
        with open(os.path.join(tmpdir, output_files[0])) as f:
            rows = [json.loads(line) for line in f]
        assert rows == [{"input_ids": [101, 47, 48, 102]}]
        meta = make_validation_split.read_metadata(os.path.join(tmpdir, ".metadata"))
        assert meta["samples"] == "1"
        assert meta["tokens"] == "4"
        assert meta["tokenizer"] == "fake-tokenizer"
        assert meta["add_bos_token"] == "True"
    print("Test 23 - tokenize main saves output and metadata: OK ✅")


# %%
#######################################
# Section 4 - decontaminate.py
#######################################

def _decontaminate_args(tmpdir, **overrides):
    defaults = dict(
        input_dir=os.path.join(tmpdir, "input"),
        reference_path=os.path.join(tmpdir, "ref.jsonl"),
        cache_dir=None,
        num_proc=1,
        batch_size=2,
        output_dir=os.path.join(tmpdir, "output"),
        output_type="jsonl",
        min_k=3,
        max_k=4,
        allow_one_token_mismatch=False,
        approx_max_k=3,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_24_decontaminate_exact_match_filters_contaminated_rows():
    _reset_fake_datasets()
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir)
        input_file = os.path.join(input_dir, "train.jsonl")
        ref_file = os.path.join(tmpdir, "ref.jsonl")
        open(input_file, "w").close()
        open(ref_file, "w").close()
        fake_datasets.next_datasets = [
            TinyDataset([
                {"input_ids": [9, 1, 2, 3, 8], "id": "bad"},
                {"input_ids": [7, 7, 7, 7], "id": "good"},
            ]),
            TinyDataset([{"input_ids": [101, 1, 2, 3, 102]}]),
        ]
        decontaminate.main(_decontaminate_args(tmpdir))
        with open(os.path.join(tmpdir, "output", "train-00000-of-00001.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        assert rows == [{"input_ids": [7, 7, 7, 7], "id": "good"}]
    print("Test 24 - decontaminate exact match: OK ✅")


def test_25_decontaminate_allows_one_token_mismatch_when_enabled():
    _reset_fake_datasets()
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir)
        input_file = os.path.join(input_dir, "train.jsonl")
        ref_file = os.path.join(tmpdir, "ref.jsonl")
        open(input_file, "w").close()
        open(ref_file, "w").close()
        fake_datasets.next_datasets = [
            TinyDataset([
                {"input_ids": [1, 2, 99], "id": "near-match"},
                {"input_ids": [8, 8, 8], "id": "clean"},
            ]),
            TinyDataset([{"input_ids": [101, 1, 2, 3, 102]}]),
        ]
        decontaminate.main(_decontaminate_args(tmpdir, allow_one_token_mismatch=True, approx_max_k=3))
        with open(os.path.join(tmpdir, "output", "train-00000-of-00001.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        assert rows == [{"input_ids": [8, 8, 8], "id": "clean"}]
    print("Test 25 - decontaminate one-token mismatch: OK ✅")


def test_26_decontaminate_keeps_all_rows_when_references_too_short():
    _reset_fake_datasets()
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir)
        input_file = os.path.join(input_dir, "train.jsonl")
        ref_file = os.path.join(tmpdir, "ref.jsonl")
        open(input_file, "w").close()
        open(ref_file, "w").close()
        fake_datasets.next_datasets = [
            TinyDataset([{"input_ids": [1, 2, 3], "id": "kept"}]),
            TinyDataset([{"input_ids": [101, 1, 102]}]),
        ]
        decontaminate.main(_decontaminate_args(tmpdir, min_k=3))
        with open(os.path.join(tmpdir, "output", "train-00000-of-00001.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        assert rows == [{"input_ids": [1, 2, 3], "id": "kept"}]
    print("Test 26 - decontaminate short references keep rows: OK ✅")


def test_27_decontaminate_chunks_output_by_input_file_count():
    _reset_fake_datasets()
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir)
        for name in ["a.jsonl", "b.jsonl"]:
            open(os.path.join(input_dir, name), "w").close()
        ref_file = os.path.join(tmpdir, "ref.jsonl")
        open(ref_file, "w").close()
        fake_datasets.next_datasets = [
            TinyDataset([{"input_ids": [i, i, i], "id": i} for i in range(4)]),
            TinyDataset([{"input_ids": [1, 2, 3]}]),
        ]
        decontaminate.main(_decontaminate_args(tmpdir, min_k=3))
        files = sorted(name for name in os.listdir(os.path.join(tmpdir, "output")) if name.endswith(".jsonl"))
        assert files == ["train-00000-of-00002.jsonl", "train-00001-of-00002.jsonl"]
    print("Test 27 - decontaminate output chunk count: OK ✅")


def test_28_decontaminate_argument_parser_defaults_and_required_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--reference_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_type", type=str, default="jsonl", choices=["jsonl", "parquet"])
    parser.add_argument("--min_k", type=int, default=8)
    parser.add_argument("--max_k", type=int, default=32)
    parser.add_argument("--allow_one_token_mismatch", action="store_true")
    parser.add_argument("--approx_max_k", type=int, default=10)

    args = parser.parse_args([
        "--input_dir", "data/tokenized_train",
        "--reference_path", "eval_set/",
        "--output_dir", "cleaned",
    ])
    assert args.cache_dir is None
    assert args.num_proc == 8
    assert args.batch_size == 10000
    assert args.output_type == "jsonl"
    assert args.min_k == 8
    assert args.max_k == 32
    assert args.allow_one_token_mismatch is False
    assert args.approx_max_k == 10
    assert args.reference_path == "eval_set/"
    print("Test 28 - decontaminate argument parser: OK ✅")


if __name__ == "__main__":
    test_01_read_metadata_returns_empty_dict_for_missing_file()
    test_02_read_metadata_parses_key_value_strings()
    test_03_get_files_from_dirs_returns_sorted_json_files()
    test_04_get_files_from_dirs_can_sample_n_files()
    test_05_get_files_from_dirs_raises_when_empty()
    test_06_validation_split_main_writes_json_split_and_metadata()
    test_07_validation_split_main_accumulates_existing_metadata()
    test_08_validation_split_main_rejects_too_many_samples()
    test_09_concatenate_pack_splits_blocks_and_discards_tail()
    test_10_concatenate_pack_handles_no_full_blocks()
    test_11_bfd_pack_pads_partial_chunks()
    test_12_bfd_pack_discards_empty_and_too_long_sequences()
    test_13_bfd_pack_uses_seq_lengths_when_present()
    test_14_pack_main_rejects_dataset_without_input_ids()
    test_15_pack_main_bfd_requires_pad_token_id()
    test_16_pack_main_saves_packed_dataset_and_metadata()
    test_17_load_tokenizer_rejects_assistant_masks_without_chat_template()
    test_18_load_tokenizer_loads_chat_template_from_file()
    test_19_load_tokenizer_requires_chat_template_when_missing()
    test_20_standard_tokenize_adds_special_tokens_and_masks()
    test_21_chat_tokenize_returns_assistant_masks_and_masked_labels()
    test_21b_load_tokenizer_rejects_chat_template_with_bos_or_eos()
    test_22_tokenize_main_rejects_missing_text_column()
    test_23_tokenize_main_filters_truncates_saves_and_metadata()
    test_24_decontaminate_exact_match_filters_contaminated_rows()
    test_25_decontaminate_allows_one_token_mismatch_when_enabled()
    test_26_decontaminate_keeps_all_rows_when_references_too_short()
    test_27_decontaminate_chunks_output_by_input_file_count()
    test_28_decontaminate_argument_parser_defaults_and_required_args()
    print("\n" + "=" * 50)
    print("All tests passed ✅")
    print("=" * 50)
