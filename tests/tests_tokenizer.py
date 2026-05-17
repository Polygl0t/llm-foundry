"""
Tokenizer training and evaluation test suite.

Run with:
    python tests_tokenizer.py

Requirements:
- transformers
- datasets
- tokenizers
- sentencepiece
- pandas
- tabulate
"""

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

sys.pycache_prefix = os.path.join(tempfile.gettempdir(), "pycache")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TOKENIZER_DIR = os.path.join(REPO_ROOT, "tokenizer")
if TOKENIZER_DIR not in sys.path:
    sys.path.insert(0, TOKENIZER_DIR)

from utils import EXTRA_TOKENS, load_text_dataset  # noqa: E402

SHAKESPEARE_URL = (
    "https://gist.githubusercontent.com/blakesanie/dde3a2b7e698f52f389532b4b52bc254/raw/"
    "76fe1b5e9efcf0d2afdfd78b0bfaa737ad0a67d3/shakespeare.txt"
)
CORE_SPECIAL_TOKEN_KEYS = {"bos_token", "eos_token", "unk_token", "pad_token"}
VOCAB_SIZE = 420


def _load_module_from_tokenizer_dir(module_name, filename):
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(TOKENIZER_DIR, filename),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


train_tokenizer_sentencepiece = _load_module_from_tokenizer_dir(
    "tokenizer_train_sentencepiece_mod", "train_tokenizer_sentencepiece.py"
)
train_tokenizer_tokenizers = _load_module_from_tokenizer_dir(
    "tokenizer_train_tokenizers_mod", "train_tokenizer_tokenizers.py"
)
tokenizer_eval = _load_module_from_tokenizer_dir("tokenizer_eval_mod", "tokenizer_eval.py")

print("All imports OK ✅")


def _read_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_sample_corpus(path: str | Path, min_chars: int = 40_000) -> None:
    fallback_text = "\n".join([
        "To be, or not to be, that is the question.",
        "Whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune.",
        "The tokenizer should keep <think>reasoning</think> and <tool_call>calls</tool_call> visible.",
        "Code indentation matters:\n    def example():\n        return '<answer>done</answer>'",
    ])

    try:
        with urllib.request.urlopen(SHAKESPEARE_URL, timeout=10) as response:
            text = response.read(250_000).decode("utf-8")
    except Exception:
        text = fallback_text

    text = text + "\n" + fallback_text
    while len(text) < min_chars:
        text += "\n" + text

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text[:min_chars])


def _base_training_args(output_dir: str, corpus_path: str, cache_dir: str) -> argparse.Namespace:
    return argparse.Namespace(
        data_path=corpus_path,
        data_type="txt",
        cache_dir=cache_dir,
        num_proc=1,
        batch_size=100,
        text_column="text",
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|pad|>",
        unk_token="<|unk|>",
        padding_side="right",
        truncation_side="right",
        add_bos_token=True,
        add_eos_token=False,
        model_max_length=4096,
        vocab_size=VOCAB_SIZE,
        output_dir=output_dir,
        hub_repo_id=None,
        token=None,
        private=True,
        byte_fallback=True,
    )


def _sentencepiece_training_args(output_dir: str, corpus_path: str) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_file=corpus_path,
        train_dataset_dir=None,
        dataset_type="txt",
        text_column="text",
        num_threads=1,
        output_dir=output_dir,
        seed=42,
        token=None,
        cache_dir=None,
        num_samples=None,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        tokenizer_name=None,
        unk_token="<|unk|>",
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|pad|>",
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        add_prefix_space=False,
        padding_side="right",
        truncation_side="right",
    )


def _assert_raises(expected_exceptions, fn, expected_message: str | None = None) -> None:
    try:
        fn()
    except expected_exceptions as error:
        if expected_message is not None:
            assert expected_message in str(error), str(error)
        return
    raise AssertionError(f"Expected {expected_exceptions} to be raised")


def _assert_special_token_contract(output_dir: str) -> None:
    special_tokens_map = _read_json(os.path.join(output_dir, "special_tokens_map.json"))
    assert set(special_tokens_map.keys()) == CORE_SPECIAL_TOKEN_KEYS
    assert "additional_special_tokens" not in special_tokens_map

    tokenizer_json = _read_json(os.path.join(output_dir, "tokenizer.json"))
    added_tokens = tokenizer_json.get("added_tokens", [])
    added_by_content = {token["content"]: token for token in added_tokens}

    for token in EXTRA_TOKENS:
        assert token in added_by_content, f"Missing extra token in tokenizer.json: {token!r}"
        assert added_by_content[token]["special"] is False, f"Extra token should not be special: {token!r}"

    for token_key in CORE_SPECIAL_TOKEN_KEYS:
        token_value = special_tokens_map[token_key]["content"] if isinstance(special_tokens_map[token_key], dict) else special_tokens_map[token_key]
        assert added_by_content[token_value]["special"] is True

    tokenizer = AutoTokenizer.from_pretrained(output_dir, use_fast=True)
    assert len(tokenizer) == VOCAB_SIZE
    assert not tokenizer.special_tokens_map.get("additional_special_tokens")

    extra_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in EXTRA_TOKENS]
    assert None not in extra_token_ids
    assert tokenizer.unk_token_id not in extra_token_ids
    assert len(set(extra_token_ids)) == len(EXTRA_TOKENS)

    for token in EXTRA_TOKENS:
        input_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
        assert input_ids == [tokenizer.convert_tokens_to_ids(token)], f"Extra token is not atomic: {token!r}"

    encoded = tokenizer("<think> visible reasoning </think>", add_special_tokens=True)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
    assert "<think>" in decoded
    assert "</think>" in decoded


def test_01_load_text_dataset_reads_plain_text_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "sample.txt")
        _write_sample_corpus(corpus_path, min_chars=1_000)

        dataset = load_text_dataset(corpus_path, "txt", cache_dir=os.path.join(tmpdir, "cache"), num_proc=1)

        assert len(dataset) >= 1
        assert "text" in dataset.column_names
        assert any(row.strip() for row in dataset["text"])
    print("Test 1 — load_text_dataset txt: OK ✅")


def test_02_train_tokenizers_bpe_saves_regular_extra_tokens():
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "sample.txt")
        output_dir = os.path.join(tmpdir, "bpe_tokenizer")
        _write_sample_corpus(corpus_path)

        args = _base_training_args(output_dir, corpus_path, os.path.join(tmpdir, "cache"))
        train_tokenizer_tokenizers.main(args)

        assert os.path.exists(os.path.join(output_dir, "tokenizer.json"))
        assert os.path.exists(os.path.join(output_dir, "tokenizer_config.json"))
        _assert_special_token_contract(output_dir)
    print("Test 2 — train_tokenizer_tokenizers BPE: OK ✅")


def test_03_train_sentencepiece_saves_regular_extra_tokens():
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "sample.txt")
        output_dir = os.path.join(tmpdir, "sentencepiece_tokenizer")
        _write_sample_corpus(corpus_path)

        args = _sentencepiece_training_args(output_dir, corpus_path)
        train_tokenizer_sentencepiece.main(args)

        assert os.path.exists(os.path.join(output_dir, "spm_tokenizer.model"))
        assert os.path.exists(os.path.join(output_dir, "tokenizer.model"))
        _assert_special_token_contract(output_dir)
    print("Test 3 — train_tokenizer_sentencepiece BPE: OK ✅")


def test_04_tokenizer_eval_writes_expected_metrics_for_local_tokenizer():
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "sample.txt")
        output_dir = os.path.join(tmpdir, "bpe_tokenizer")
        eval_path = os.path.join(tmpdir, "metrics.json")
        _write_sample_corpus(corpus_path)

        training_args = _base_training_args(output_dir, corpus_path, os.path.join(tmpdir, "cache"))
        train_tokenizer_tokenizers.main(training_args)

        eval_args = argparse.Namespace(
            tokenizers_to_evaluate=[output_dir],
            input_file=corpus_path,
            output_file=eval_path,
            cache_dir=None,
            token=None,
        )
        tokenizer_eval.main(eval_args)

        results = _read_json(eval_path)
        assert len(results) == 1
        assert results[0]["tokenizer_name"] == os.path.basename(output_dir)
        assert results[0]["total_num_words"] > 0
        assert results[0]["total_tokens"] > 0
        assert results[0]["vocab_size"] == VOCAB_SIZE
        assert results[0]["fertility"] > 0
        assert results[0]["unk_token_count"] == 0
    print("Test 4 — tokenizer_eval local tokenizer: OK ✅")


def test_05_tokenizers_bpe_rejects_unreachable_vocab_size():
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "sample.txt")
        output_dir = os.path.join(tmpdir, "too_small_tokenizer")
        _write_sample_corpus(corpus_path, min_chars=4_000)

        args = _base_training_args(output_dir, corpus_path, os.path.join(tmpdir, "cache"))
        args.vocab_size = len(EXTRA_TOKENS) + 3

        _assert_raises(
            AssertionError,
            lambda: train_tokenizer_tokenizers.main(args),
            expected_message="Expected vocab size",
        )
    print("Test 5 — train_tokenizer_tokenizers vocab mismatch: OK ✅")


def test_06_sentencepiece_rejects_unreachable_vocab_size():
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "sample.txt")
        output_dir = os.path.join(tmpdir, "too_small_sentencepiece")
        _write_sample_corpus(corpus_path, min_chars=4_000)

        args = _sentencepiece_training_args(output_dir, corpus_path)
        args.vocab_size = len(EXTRA_TOKENS) + 3

        _assert_raises(
            (AssertionError, RuntimeError, ValueError),
            lambda: train_tokenizer_sentencepiece.main(args),
        )
    print("Test 6 — train_tokenizer_sentencepiece vocab mismatch: OK ✅")


if __name__ == "__main__":
    tests = [
        test_01_load_text_dataset_reads_plain_text_files,
        test_02_train_tokenizers_bpe_saves_regular_extra_tokens,
        test_03_train_sentencepiece_saves_regular_extra_tokens,
        test_04_tokenizer_eval_writes_expected_metrics_for_local_tokenizer,
        test_05_tokenizers_bpe_rejects_unreachable_vocab_size,
        test_06_sentencepiece_rejects_unreachable_vocab_size,
    ]
    for test in tests:
        test()
    print("\n" + "=" * 50)
    print("All tests passed ✅")
    print("=" * 50)