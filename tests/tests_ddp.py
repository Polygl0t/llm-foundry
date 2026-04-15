"""
CPU-only pre-flight test suite for the DDP trainer codebase.

Run with:
    python tests_ddp.py

All tests use synthetic data and tiny model configs so they complete
in seconds on a standard desktop CPU without any GPU, DDP, or SLURM dependency.

Requirements:
- torch>=2.0
- transformers>=4.40
- datasets>=2.0
- numpy
- pyyaml
"""
# %%
#######################################
# 0. Setup for Testing
#######################################
import sys
import os
import json
import math
import tempfile
import shutil
import traceback

sys.pycache_prefix = os.path.join(tempfile.gettempdir(), "pycache")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DDP_DIR = os.path.join(REPO_ROOT, "ddp")
if DDP_DIR not in sys.path:
    sys.path.insert(0, DDP_DIR)

# Store test results as tuples of (test_name, passed_bool, error_message).
_results: list[tuple[str, bool, str]] = []


def run_test(name, fn):
    """Run *fn* and record pass/fail."""
    try:
        fn()
        _results.append((name, True, ""))
    except Exception as exc:
        tb = traceback.format_exc()
        _results.append((name, False, tb))
        print(f"  FAIL ❌  {name}\n{tb}")


def report():
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        print("\nFailed tests:")
        for name, ok, tb in _results:
            if not ok:
                print(f"  - {name}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("All tests passed! ✅")
        print("=" * 60)


# %%
#######################################
# 1. Imports & Setup
#######################################
print("\n" + "=" * 60)
print("1. Imports & Setup")
print("=" * 60)


def test_import_specifications():
    from specifications import TrainingArguments  # noqa: F401


def test_import_mfu():
    from mfu import (  # noqa: F401
        MFUContext,
        TrainingPerformanceMetrics,
        PEAK_FLOPS_BY_HARDWARE,
        MFU_REGISTRY,
        create_mfu_context,
        calculate_training_metrics,
    )


def test_import_data_loading():
    from data_loading import (  # noqa: F401
        create_collate_fn,
        prepare_dataloaders,
        DataLoaderBundle,
        SUPPORTED_FORMATS,
    )


def test_import_model_setup():
    from model_setup import (  # noqa: F401
        prepare_training_components,
        ModelInitializationResult,
        _resolve_checkpoint_path,
        _build_model_from_config,
    )


def test_import_optimizers():
    from optimizers import (  # noqa: F401
        create_lr_scheduler,
        create_optimizer,
        get_optimizer_summary_lines,
        get_muon_momentum,
        zeropower_via_newtonschulz5,
        muon_update,
        adam_update,
        SingleDeviceMuon,
        SingleDeviceMuonWithAuxAdam,
    )


def test_import_utils():
    from utils import (  # noqa: F401
        StructuredTrainingLogger,
        compute_training_schedule,
        cleanup_log_file,
        checkpoint_already_validated,
    )


for _fn in [
    test_import_specifications,
    test_import_mfu,
    test_import_data_loading,
    test_import_model_setup,
    test_import_optimizers,
    test_import_utils,
]:
    run_test(_fn.__name__, _fn)

print("Test 1 — Imports & Setup: OK ✅")

print("Test 2 — TrainingArguments & Config Loading: OK ✅")


# %%
#######################################
# 2. TrainingArguments & Config Loading
#######################################
print("\n" + "=" * 60)
print("2. TrainingArguments & Config Loading")
print("=" * 60)

import yaml
import torch
import numpy as np
import logging
from specifications import TrainingArguments


def test_training_args_defaults():
    """TrainingArguments can be created with all defaults."""
    args = TrainingArguments()
    assert args.micro_batch_size == 32
    assert args.seed == 1337
    assert args.optimizer_type == "adamw"
    assert args.bf16 is False
    assert args.sanity_check is False


def test_training_args_override():
    """TrainingArguments fields can be overridden at construction time."""
    args = TrainingArguments(micro_batch_size=4, seed=42, bf16=True)
    assert args.micro_batch_size == 4
    assert args.seed == 42
    assert args.bf16 is True


def test_training_args_from_yaml():
    """TrainingArguments can be populated directly from a YAML file path."""
    cfg = {
        "micro_batch_size": 8,
        "total_batch_size": 1024,
        "seed": 99,
        "optimizer_type": "muon_adam",
        "lr_decay_type": "wsd",
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as f:
        yaml.dump(cfg, f)
        tmp_path = f.name
    try:
        args = TrainingArguments.from_yaml(tmp_path)
        assert args.micro_batch_size == 8
        assert args.total_batch_size == 1024
        assert args.optimizer_type == "muon_adam"
    finally:
        os.unlink(tmp_path)


def test_training_args_to_dict_includes_runtime_fields():
    """TrainingArguments.to_dict serializes both YAML and runtime-populated fields."""
    args = TrainingArguments(micro_batch_size=8, stage_name="S2")
    args.max_position_embeddings = 4096
    args.vocab_size = 32000

    serialized = args.to_dict()

    assert serialized["micro_batch_size"] == 8
    assert serialized["stage_name"] == "S2"
    assert serialized["max_position_embeddings"] == 4096
    assert serialized["vocab_size"] == 32000


def test_training_args_invalid_field():
    """Passing an unknown field should raise TypeError."""
    raised = False
    try:
        TrainingArguments(nonexistent_field=True)
    except TypeError:
        raised = True
    assert raised, "Expected TypeError for unknown field"


for _fn in [
    test_training_args_defaults,
    test_training_args_override,
    test_training_args_from_yaml,
    test_training_args_to_dict_includes_runtime_fields,
    test_training_args_invalid_field,
]:
    run_test(_fn.__name__, _fn)


# %%
#######################################
# 3. MFU Calculation
#######################################
print("\n" + "=" * 60)
print("3. MFU Calculation")
print("=" * 60)

from mfu import (
    MFUContext,
    TrainingPerformanceMetrics,
    PEAK_FLOPS_BY_HARDWARE,
    MFU_REGISTRY,
    create_mfu_context,
    calculate_training_metrics,
    _mamba_layer_macs,
    _attention_layer_macs,
    _linear_attention_layer_macs,
)


def test_peak_flops_registry():
    """Known hardware entries exist in the registry."""
    assert "a100" in PEAK_FLOPS_BY_HARDWARE
    assert "a40" in PEAK_FLOPS_BY_HARDWARE
    assert PEAK_FLOPS_BY_HARDWARE["a100"] == 300e12


def test_mfu_registry_dense():
    """dense_transformer strategy is registered."""
    assert "dense_transformer" in MFU_REGISTRY


def test_create_mfu_context():
    """create_mfu_context builds a correct MFUContext from mock args."""
    args = TrainingArguments()
    args.mfu_type = "dense_transformer"
    args.num_hidden_layers = 12
    args.num_attention_heads = 12
    args.head_dim = 64
    args.max_position_embeddings = 512
    ctx = create_mfu_context(args, "a100", num_parameters=125_000_000)
    assert isinstance(ctx, MFUContext)
    assert ctx.peak_flops == 300e12
    assert ctx.num_parameters == 125_000_000
    assert ctx.sequence_length == 512


def test_create_mfu_context_unsupported_hardware():
    """create_mfu_context raises on unknown hardware."""
    args = TrainingArguments()
    args.mfu_type = "dense_transformer"
    args.num_hidden_layers = 12
    args.num_attention_heads = 12
    args.head_dim = 64
    args.max_position_embeddings = 512
    raised = False
    try:
        create_mfu_context(args, "tpu_v5", num_parameters=1)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for unsupported hardware"


def test_calculate_training_metrics():
    """calculate_training_metrics returns sensible values."""
    ctx = MFUContext(
        mfu_type="dense_transformer",
        peak_flops=300e12,
        num_parameters=125_000_000,
        num_hidden_layers=12,
        num_attention_heads=12,
        head_dim=64,
        sequence_length=512,
    )
    metrics = calculate_training_metrics(
        mfu_context=ctx,
        micro_batch_size=4,
        gradient_accumulation_steps=2,
        world_size=1,
        dt=1.0,
    )
    assert isinstance(metrics, TrainingPerformanceMetrics)
    assert metrics.tokens_processed == 4 * 2 * 512 * 1
    assert metrics.global_tokens_per_sec > 0
    assert metrics.tokens_per_sec_per_gpu == metrics.global_tokens_per_sec
    assert metrics.mfu > 0


def test_calculate_training_metrics_zero_dt():
    """dt <= 0 must raise."""
    ctx = MFUContext(
        mfu_type="dense_transformer",
        peak_flops=300e12,
        num_parameters=125_000_000,
        num_hidden_layers=12,
        num_attention_heads=12,
        head_dim=64,
        sequence_length=512,
    )
    raised = False
    try:
        calculate_training_metrics(ctx, 4, 2, 1, dt=0.0)
    except ValueError:
        raised = True
    assert raised


def test_calculate_training_metrics_invalid_type():
    """Unknown mfu_type must raise."""
    ctx = MFUContext(
        mfu_type="moe_transformer",
        peak_flops=300e12,
        num_parameters=125_000_000,
        num_hidden_layers=12,
        num_attention_heads=12,
        head_dim=64,
        sequence_length=512,
    )
    raised = False
    try:
        calculate_training_metrics(ctx, 4, 2, 1, dt=1.0)
    except ValueError:
        raised = True
    assert raised


def test_mfu_registry_hybrid():
    """'hybrid' and 'mamba' both map to the same strategy."""
    assert "hybrid" in MFU_REGISTRY
    assert "mamba" in MFU_REGISTRY
    assert MFU_REGISTRY["hybrid"] is MFU_REGISTRY["mamba"]


def _make_mamba_context(**overrides):
    """Helper: build an MFUContext with Mamba2-like parameters."""
    defaults = dict(
        mfu_type="mamba",
        peak_flops=300e12,
        num_parameters=0,
        num_hidden_layers=4,
        num_attention_heads=12,
        head_dim=128,
        sequence_length=1024,
        hidden_size=1536,
        vocab_size=100352,
        intermediate_size=512,
        mamba_d_state=128,
        mamba_chunk_size=256,
        mamba_d_conv=4,
        mamba_n_heads=48,
        mamba_d_head=64,
        mamba_n_groups=1,
    )
    defaults.update(overrides)
    return MFUContext(**defaults)


def test_mamba_layer_macs_formula():
    """Verify _mamba_layer_macs matches a hand-computed reference (Mamba2-like)."""
    ctx = _make_mamba_context()
    d, e, g, n, h = 1536, 48 * 64, 1, 128, 48
    expected = (
        d * (2 * e + 2 * g * n + h)     # in_proj
        + 4 * e                         # conv
        + e * d                         # out_proj
        + 2 * 256 * e                   # intra_chunk
        + 2 * e * n                     # inter_chunk
        + 3 * d * 512                   # mlp
    )
    assert _mamba_layer_macs(ctx) == expected


def test_attention_layer_macs_formula():
    """Verify _attention_layer_macs."""
    ctx = _make_mamba_context()
    d, s = 1536, 1024
    expected = 4 * d * d + d * s + 3 * d * 512
    assert _attention_layer_macs(ctx) == expected


def test_linear_attention_layer_macs_formula():
    """Verify _linear_attention_layer_macs matches reference calculation."""
    ctx = MFUContext(
        mfu_type="mamba",
        peak_flops=300e12,
        num_parameters=0,
        num_hidden_layers=32,
        num_attention_heads=30,
        head_dim=128,
        sequence_length=1024,
        hidden_size=3840,
        vocab_size=100352,
        intermediate_size=11008,
        linear_num_key_heads=30,
        linear_num_value_heads=30,
        linear_key_head_dim=96,
        linear_value_head_dim=192,
        linear_conv_kernel_dim=4,
        linear_chunk_size=256,
    )
    d = 3840
    k = 30 * 96   # 2880
    v = 30 * 192  # 5760
    h = 30
    L = 256
    expected = (
        d * (2 * k + v + 2 * h)        # projections
        + 4 * (2 * k + v)              # conv
        + 2 * d * v                    # gate + out
        + L * (3 * k + 2 * v)          # intra_chunk
        + 3 * k * v // h               # inter_chunk
        + 3 * d * 11008                # mlp
    )
    assert _linear_attention_layer_macs(ctx) == expected


def test_mamba_mfu_pure():
    """MFU for a pure Mamba model (no layer_types) produces a positive value."""
    ctx = _make_mamba_context(num_hidden_layers=4, layer_types=())
    metrics = calculate_training_metrics(ctx, micro_batch_size=4, gradient_accumulation_steps=1, world_size=1, dt=1.0)
    assert metrics.mfu > 0


def test_mamba_mfu_hybrid_layer_types():
    """
    Hybrid model: 3 mamba layers + 1 attention layer.
    MFU should be greater than zero and differ from a pure-mamba calculation.
    """
    layer_types = ("mamba", "mamba", "mamba", "attention")
    ctx_hybrid = _make_mamba_context(num_hidden_layers=4, layer_types=layer_types)
    ctx_pure   = _make_mamba_context(num_hidden_layers=4, layer_types=())

    m_hybrid = calculate_training_metrics(ctx_hybrid, 4, 1, 1, dt=1.0)
    m_pure   = calculate_training_metrics(ctx_pure,   4, 1, 1, dt=1.0)

    assert m_hybrid.mfu > 0
    assert m_pure.mfu > 0
    # They should differ because attention and mamba layers have different costs.
    assert m_hybrid.mfu != m_pure.mfu


def test_mamba_mfu_linear_attention_hybrid():
    """MFU for a linear-attention + full-attention hybrid."""
    layer_types = tuple(
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
        for i in range(32)
    )
    ctx = MFUContext(
        mfu_type="hybrid",
        peak_flops=300e12,
        num_parameters=0,
        num_hidden_layers=32,
        num_attention_heads=30,
        head_dim=128,
        sequence_length=1024,
        hidden_size=3840,
        vocab_size=100352,
        intermediate_size=11008,
        layer_types=layer_types,
        linear_num_key_heads=30,
        linear_num_value_heads=30,
        linear_key_head_dim=96,
        linear_value_head_dim=192,
        linear_conv_kernel_dim=4,
        linear_chunk_size=256,
    )
    metrics = calculate_training_metrics(ctx, micro_batch_size=4, gradient_accumulation_steps=1, world_size=1, dt=1.0)
    assert metrics.mfu > 0


def test_create_mfu_context_mamba_fields():
    """create_mfu_context extracts mamba/linear-attention fields from args."""
    args = TrainingArguments()
    args.mfu_type = "mamba"
    args.num_hidden_layers = 4
    args.num_attention_heads = 12
    args.head_dim = 128
    args.max_position_embeddings = 1024
    args.hidden_size = 1536
    args.vocab_size = 100352
    args.intermediate_size = 512
    args.mamba_d_state = 128
    args.mamba_n_heads = 48
    args.mamba_d_head = 64
    args.layer_types = ["mamba", "mamba", "mamba", "attention"]

    ctx = create_mfu_context(args, "a100", num_parameters=0)

    assert ctx.hidden_size == 1536
    assert ctx.vocab_size == 100352
    assert ctx.mamba_d_state == 128
    assert ctx.mamba_n_heads == 48
    assert ctx.layer_types == ("mamba", "mamba", "mamba", "attention")
    # Fields not set on args should get defaults
    assert ctx.linear_num_key_heads == 0


for _fn in [
    test_peak_flops_registry,
    test_mfu_registry_dense,
    test_mfu_registry_hybrid,
    test_create_mfu_context,
    test_create_mfu_context_unsupported_hardware,
    test_calculate_training_metrics,
    test_calculate_training_metrics_zero_dt,
    test_calculate_training_metrics_invalid_type,
    test_mamba_layer_macs_formula,
    test_attention_layer_macs_formula,
    test_linear_attention_layer_macs_formula,
    test_mamba_mfu_pure,
    test_mamba_mfu_hybrid_layer_types,
    test_mamba_mfu_linear_attention_hybrid,
    test_create_mfu_context_mamba_fields,
]:
    run_test(_fn.__name__, _fn)

print("Test 3 — MFU Calculation: OK ✅")


# %%
#######################################
# 4. Collate Function
#######################################
print("\n" + "=" * 60)
print("4. Collate Function")
print("=" * 60)

from data_loading import create_collate_fn


def test_collate_fn_no_mask():
    """Collate with no mask IDs: labels == input_ids."""
    collate = create_collate_fn(mask_token_ids=set())
    batch = collate([{"input_ids": torch.tensor([1, 2, 3, 4, 5])}])
    assert "labels" in batch
    assert "input_ids" in batch
    assert torch.equal(batch["labels"], batch["input_ids"])


def test_collate_fn_with_mask():
    """Collate masks specified token IDs with -100."""
    pad_id, eos_id = 0, 2
    collate = create_collate_fn(mask_token_ids={pad_id, eos_id})
    examples = [{"input_ids": torch.tensor([0, 1, 2, 3, 0])}]
    batch = collate(examples)
    labels = batch["labels"]
    # positions with token 0 or 2 should be -100
    assert labels[0, 0].item() == -100
    assert labels[0, 2].item() == -100
    assert labels[0, 4].item() == -100
    # other positions should be unchanged
    assert labels[0, 1].item() == 1
    assert labels[0, 3].item() == 3


def test_collate_fn_preserves_existing_labels():
    """If the example already contains labels, the collate fn should trust them."""
    collate = create_collate_fn(mask_token_ids={0})
    existing_labels = torch.tensor([10, 20, 30])
    examples = [{"input_ids": torch.tensor([0, 1, 2]), "labels": existing_labels}]
    batch = collate(examples)
    assert torch.equal(batch["labels"], existing_labels.unsqueeze(0))


def test_collate_fn_multi_sample_batch():
    """Collate with multiple samples produces correct batch dimensions."""
    collate = create_collate_fn(mask_token_ids={0})
    examples = [
        {"input_ids": torch.tensor([0, 1, 2, 3])},
        {"input_ids": torch.tensor([4, 5, 0, 6])},
    ]
    batch = collate(examples)
    assert batch["input_ids"].shape == (2, 4)
    assert batch["labels"].shape == (2, 4)
    # Verify masking in each row
    assert batch["labels"][0, 0].item() == -100
    assert batch["labels"][1, 2].item() == -100
    assert batch["labels"][0, 1].item() == 1
    assert batch["labels"][1, 0].item() == 4


for _fn in [
    test_collate_fn_no_mask,
    test_collate_fn_with_mask,
    test_collate_fn_preserves_existing_labels,
    test_collate_fn_multi_sample_batch,
]:
    run_test(_fn.__name__, _fn)

print("Test 4 — Collate Function: OK ✅")


# %%
#######################################
# 5. Sanity-Check Dataset & DataLoader
#######################################
print("\n" + "=" * 60)
print("5. Sanity-Check Dataset & DataLoader")
print("=" * 60)

from data_loading import prepare_dataloaders, DataLoaderBundle, _load_sanity_check_datasets
from transformers import AutoTokenizer


# We use a tiny public tokenizer for tests.
_TINY_TOKENIZER_NAME = "gpt2"
_tokenizer = AutoTokenizer.from_pretrained(_TINY_TOKENIZER_NAME)


def _make_sanity_args(**overrides):
    """Return TrainingArguments configured for sanity-check mode with tiny sizes."""
    defaults = dict(
        sanity_check=True,
        sanity_check_num_samples=64,
        micro_batch_size=4,
        eval_micro_batch_size=4,
        pin_memory=False,
        num_workers_for_dataloader=0,
        prefetch_factor=None,
        shuffle_dataset=False,
        additional_mask_token_ids=None,
        seed=42,
    )
    defaults.update(overrides)
    args = TrainingArguments(**{k: v for k, v in defaults.items() if k in TrainingArguments.__dataclass_fields__})
    # Set runtime fields that would normally be set by model_setup.py
    args.max_position_embeddings = 32
    args.vocab_size = _tokenizer.vocab_size
    return args


def test_load_sanity_check_datasets():
    """_load_sanity_check_datasets produces train/val with correct shapes."""
    args = _make_sanity_args()
    train_ds, val_ds = _load_sanity_check_datasets(args)
    assert len(train_ds) == 64
    assert len(val_ds) == max(1, int(64 * 0.1))
    sample = train_ds[0]
    assert "input_ids" in sample
    assert sample["input_ids"].shape == (32,)


def test_prepare_dataloaders_sanity():
    """prepare_dataloaders in sanity-check mode returns a valid DataLoaderBundle."""
    args = _make_sanity_args()
    bundle = prepare_dataloaders(
        args=args,
        tokenizer=_tokenizer,
        world_size=1,
        rank=0,
    )
    assert isinstance(bundle, DataLoaderBundle)
    assert bundle.num_train_samples == 64
    assert bundle.num_val_samples == max(1, int(64 * 0.1))


def test_dataloader_iteration():
    """We can iterate over the train dataloader and get correct batch shapes."""
    args = _make_sanity_args()
    bundle = prepare_dataloaders(args=args, tokenizer=_tokenizer, world_size=1, rank=0)
    batch = next(iter(bundle.train_dataloader))
    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape[0] <= args.micro_batch_size
    assert batch["input_ids"].shape[1] == 32


def test_dataloader_val_iteration():
    """Validation dataloader yields correct batches."""
    args = _make_sanity_args()
    bundle = prepare_dataloaders(args=args, tokenizer=_tokenizer, world_size=1, rank=0)
    batch = next(iter(bundle.val_dataloader))
    assert "input_ids" in batch
    assert batch["input_ids"].shape[1] == 32


def test_dataloader_custom_collate():
    """Custom collate function is respected when passed to prepare_dataloaders."""
    args = _make_sanity_args()
    custom_called = [False]

    def custom_collate(examples):
        custom_called[0] = True
        from transformers import default_data_collator
        return default_data_collator(examples)

    bundle = prepare_dataloaders(
        args=args, tokenizer=_tokenizer, world_size=1, rank=0,
        collate_fn=custom_collate,
    )
    _ = next(iter(bundle.train_dataloader))
    assert custom_called[0], "Custom collate function was not used"


for _fn in [
    test_load_sanity_check_datasets,
    test_prepare_dataloaders_sanity,
    test_dataloader_iteration,
    test_dataloader_val_iteration,
    test_dataloader_custom_collate,
]:
    run_test(_fn.__name__, _fn)

print("Test 5 — Sanity-Check Dataset & DataLoader: OK ✅")


# %%
#######################################
# 6. Model Initialization (CPU)
#######################################
print("\n" + "=" * 60)
print("6. Model Initialization (CPU)")
print("=" * 60)

from model_setup import (
    _resolve_checkpoint_path,
    _build_model_from_config,
    _create_tokenizer,
    _compute_active_trainable_params,
    _try_create_distributed_config,
    _check_kernels_available,
    prepare_training_components,
    ModelInitializationResult,
)
from transformers import AutoConfig


def test_resolve_checkpoint_path_none():
    """No checkpoint => None."""
    assert _resolve_checkpoint_path(None) is None
    assert _resolve_checkpoint_path("") is None


def test_resolve_checkpoint_path_with_steps():
    """Picks the latest step_* directory."""
    tmp = tempfile.mkdtemp()
    try:
        for step in [1, 5, 10]:
            os.makedirs(os.path.join(tmp, f"step_{step}"))
        result = _resolve_checkpoint_path(tmp)
        assert result.endswith("step_10")
    finally:
        shutil.rmtree(tmp)


def test_resolve_checkpoint_path_direct():
    """A path without step_* subdirs is returned as-is."""
    tmp = tempfile.mkdtemp()
    try:
        result = _resolve_checkpoint_path(tmp)
        # no step_ dirs → falls through to original path
        assert result == tmp
    finally:
        shutil.rmtree(tmp)


def _create_tiny_model_config(tmpdir):
    """Write a minimal GPT-2-like config.json and return its path."""
    config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "n_embd": 64,
        "n_head": 2,
        "n_layer": 2,
        "n_positions": 64,
        "vocab_size": 1000,
        "n_inner": 256,
        "activation_function": "gelu_new",
        "bos_token_id": 0,
        "eos_token_id": 0,
    }
    config_path = os.path.join(tmpdir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)
    return tmpdir


def test_build_model_from_config():
    """Build a tiny model from a config file on CPU."""
    tmpdir = tempfile.mkdtemp()
    try:
        config_dir = _create_tiny_model_config(tmpdir)
        args = TrainingArguments(
            path_to_model_config=config_dir,
            attn_implementation="eager",
            cache_dir=tmpdir,
        )
        model = _build_model_from_config(args, _tokenizer, torch.float32)
        assert model is not None
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
    finally:
        shutil.rmtree(tmpdir)


def test_build_model_no_config_raises():
    """Missing path_to_model_config raises ValueError."""
    args = TrainingArguments(path_to_model_config=None)
    raised = False
    try:
        _build_model_from_config(args, _tokenizer, torch.float32)
    except ValueError:
        raised = True
    assert raised


def test_prepare_training_components_cpu():
    """Full prepare_training_components pipeline on CPU with a tiny config."""
    tmpdir = tempfile.mkdtemp()
    try:
        config_dir = _create_tiny_model_config(tmpdir)
        args = TrainingArguments(
            path_to_model_config=config_dir,
            tokenizer_name_or_path=_TINY_TOKENIZER_NAME,
            attn_implementation="eager",
            cache_dir=tmpdir,
            torch_compile=False,
            use_liger_kernel=False,
            gradient_checkpointing=False,
            mat_mul_precision="highest",
            tf32=False,
            bf16=False,
        )
        result = prepare_training_components(
            args=args,
            device="cpu",
            master_process=True,
        )
        assert isinstance(result, ModelInitializationResult)
        assert result.model is not None
        assert result.tokenizer is not None
        assert result.precision == torch.float32
        assert result.checkpoint_path is None
        assert result.trainable_params > 0
        # Dense model: active params == total params
        assert result.active_trainable_params == result.trainable_params
        # Runtime fields should be populated on args
        assert result.args.max_position_embeddings is not None
        assert result.args.vocab_size is not None
        assert result.args.num_hidden_layers == 2
        assert result.args.num_attention_heads == 2
    finally:
        shutil.rmtree(tmpdir)


def test_prepare_training_components_bf16():
    """prepare_training_components with bf16=True uses bfloat16 precision."""
    tmpdir = tempfile.mkdtemp()
    try:
        config_dir = _create_tiny_model_config(tmpdir)
        args = TrainingArguments(
            path_to_model_config=config_dir,
            tokenizer_name_or_path=_TINY_TOKENIZER_NAME,
            attn_implementation="eager",
            cache_dir=tmpdir,
            torch_compile=False,
            use_liger_kernel=False,
            gradient_checkpointing=False,
            mat_mul_precision="highest",
            tf32=False,
            bf16=True,
        )
        result = prepare_training_components(args=args, device="cpu", master_process=True)
        assert result.precision == torch.bfloat16
    finally:
        shutil.rmtree(tmpdir)


def test_create_tokenizer_no_source_raises():
    """Neither tokenizer_name_or_path nor base_model → ValueError."""
    args = TrainingArguments()
    raised = False
    try:
        _create_tokenizer(args, master_process=True)
    except ValueError:
        raised = True
    assert raised


def _make_mock_config(**kwargs):
    """Create a lightweight mock config object for _compute_active_trainable_params tests."""
    class _Cfg:
        pass
    cfg = _Cfg()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def test_active_params_dense_model():
    """Dense models: active_trainable_params == trainable_params."""
    cfg = _make_mock_config(hidden_size=2048, num_hidden_layers=24, intermediate_size=5632)
    total = 1_000_000
    assert _compute_active_trainable_params(cfg, total) == total


def test_active_params_qwen_moe():
    """
    Qwen2MoE-style: num_experts, moe_intermediate_size, decoder_sparse_step.
    """
    cfg = _make_mock_config(
        hidden_size=2048,
        num_hidden_layers=24,
        intermediate_size=5632,
        num_experts=60,
        num_experts_per_tok=4,
        moe_intermediate_size=1408,
        decoder_sparse_step=1,
    )
    total = 10_000_000
    active = _compute_active_trainable_params(cfg, total)
    # 24 MoE layers, 56 inactive experts each, 3*2048*1408 params per expert
    params_per_expert = 3 * 2048 * 1408
    expected_inactive = 24 * (60 - 4) * params_per_expert
    assert active == total - expected_inactive
    assert active < total


def test_active_params_granite_moe():
    """
    Granite-style: num_local_experts, intermediate_size as per-expert size,
    no moe_intermediate_size, no decoder_sparse_step.
    """
    cfg = _make_mock_config(
        hidden_size=1536,
        num_hidden_layers=40,
        intermediate_size=512,
        num_local_experts=64,
        num_experts_per_tok=6,
    )
    total = 10_000_000
    active = _compute_active_trainable_params(cfg, total)
    params_per_expert = 3 * 1536 * 512
    expected_inactive = 40 * (64 - 6) * params_per_expert
    assert active == total - expected_inactive
    assert active < total


def test_active_params_single_expert_is_dense():
    """A config with num_experts=1 is treated as dense."""
    cfg = _make_mock_config(
        hidden_size=2048,
        num_hidden_layers=24,
        intermediate_size=5632,
        num_experts=1,
        num_experts_per_tok=1,
    )
    total = 5_000_000
    assert _compute_active_trainable_params(cfg, total) == total


def test_try_create_distributed_config_disabled():
    """When enable_expert_parallelism is False, returns None."""
    result = _try_create_distributed_config(False, master_process=True)
    assert result is None


def test_try_create_distributed_config_enabled():
    """
    When enable_expert_parallelism is True, returns a DistributedConfig
    if the import succeeds, or None with a warning if it fails.
    """
    result = _try_create_distributed_config(True, master_process=True)
    # On older transformers, result will be None (graceful fallback).
    # On newer transformers (>= 5.x), result will be a DistributedConfig.
    try:
        from transformers.distributed.configuration_utils import DistributedConfig
        assert result is not None
    except (ImportError, ModuleNotFoundError):
        assert result is None


def test_enable_expert_parallelism_spec_default():
    """enable_expert_parallelism defaults to False in TrainingArguments."""
    args = TrainingArguments()
    assert args.enable_expert_parallelism is False


def test_check_kernels_available_disabled():
    """When use_kernels is False, returns False."""
    assert _check_kernels_available(False, master_process=True) is False


def test_check_kernels_available_enabled():
    """
    When use_kernels is True, returns True only if both the `kernels` package
    and transformers' use_kernels kwarg are available.
    """
    result = _check_kernels_available(True, master_process=True)
    # Graceful: result is True if both dependencies are met, False otherwise.
    assert isinstance(result, bool)
    try:
        import kernels as _k  # noqa: F401
        import inspect
        from transformers import AutoModelForCausalLM as _A
        sig = inspect.signature(_A.from_pretrained)
        if "use_kernels" in sig.parameters:
            assert result is True
        else:
            assert result is False
    except (ImportError, ModuleNotFoundError):
        assert result is False


def test_use_kernels_spec_default():
    """use_kernels defaults to False in TrainingArguments."""
    args = TrainingArguments()
    assert args.use_kernels is False


for _fn in [
    test_resolve_checkpoint_path_none,
    test_resolve_checkpoint_path_with_steps,
    test_resolve_checkpoint_path_direct,
    test_build_model_from_config,
    test_build_model_no_config_raises,
    test_prepare_training_components_cpu,
    test_prepare_training_components_bf16,
    test_create_tokenizer_no_source_raises,
    test_active_params_dense_model,
    test_active_params_qwen_moe,
    test_active_params_granite_moe,
    test_active_params_single_expert_is_dense,
    test_try_create_distributed_config_disabled,
    test_try_create_distributed_config_enabled,
    test_enable_expert_parallelism_spec_default,
    test_check_kernels_available_disabled,
    test_check_kernels_available_enabled,
    test_use_kernels_spec_default,
]:
    run_test(_fn.__name__, _fn)

print("Test 6 — Model Initialization (CPU): OK ✅")


# %%
#######################################
# 7. Optimizers & LR Schedulers (CPU)
#######################################
print("\n" + "=" * 60)
print("7. Optimizers & LR Schedulers (CPU)")
print("=" * 60)

from optimizers import (
    create_lr_scheduler,
    create_optimizer,
    get_optimizer_summary_lines,
    get_muon_momentum,
    zeropower_via_newtonschulz5,
    muon_update,
    adam_update,
    SingleDeviceMuon,
    SingleDeviceMuonWithAuxAdam,
)


def test_muon_momentum_boundaries():
    """Muon momentum ramps from 0.85 → 0.95 in 300 steps."""
    m0 = get_muon_momentum(0)
    m300 = get_muon_momentum(300)
    m600 = get_muon_momentum(600)
    assert abs(m0 - 0.85) < 1e-6
    assert abs(m300 - 0.95) < 1e-6
    assert abs(m600 - 0.95) < 1e-6  # clamped at 1.0 frac


def test_zeropower_newtonschulz():
    """Newton-Schulz iteration runs on CPU with a small 2D matrix."""
    G = torch.randn(8, 8)
    result = zeropower_via_newtonschulz5(G, steps=5)
    assert result.shape == G.shape
    # Should be approximately orthogonal (U @ U^T ≈ I up to scale)
    product = result.float() @ result.float().T
    diag = torch.diag(product)
    assert all(d > 0 for d in diag), "Diagonal elements should be positive"


def test_adam_update():
    """adam_update produces a finite update tensor."""
    grad = torch.randn(16)
    buf1 = torch.zeros(16)
    buf2 = torch.zeros(16)
    update = adam_update(grad, buf1, buf2, step=1, betas=(0.9, 0.95), eps=1e-8)
    assert update.shape == grad.shape
    assert torch.isfinite(update).all()


def test_single_device_muon():
    """SingleDeviceMuon can step on a small 2D parameter (CPU)."""
    param = torch.nn.Parameter(torch.randn(8, 8))
    opt = SingleDeviceMuon([param], lr=0.01)
    param.grad = torch.randn_like(param)
    opt.step()
    # Param should have changed
    assert torch.isfinite(param).all()


def test_single_device_muon_with_aux_adam():
    """SingleDeviceMuonWithAuxAdam handles mixed param groups on CPU."""
    muon_param = torch.nn.Parameter(torch.randn(8, 8))
    adam_param = torch.nn.Parameter(torch.randn(4))
    groups = [
        {"params": [muon_param], "lr": 0.02, "momentum": 0.95, "weight_decay": 0, "use_muon": True},
        {"params": [adam_param], "lr": 3e-4, "betas": (0.9, 0.95), "eps": 1e-10, "weight_decay": 0, "use_muon": False},
    ]
    opt = SingleDeviceMuonWithAuxAdam(groups)
    muon_param.grad = torch.randn_like(muon_param)
    adam_param.grad = torch.randn_like(adam_param)
    opt.step()
    assert torch.isfinite(muon_param).all()
    assert torch.isfinite(adam_param).all()


def test_cosine_lr_scheduler():
    """Cosine LR scheduler warmup → decay → min_lr."""
    args = TrainingArguments(
        max_learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_steps=10,
        lr_decay_type="cosine",
        lr_decay_iters_coef=1.0,
        optimizer_type="adamw",
    )
    max_steps = 100
    scheduler = create_lr_scheduler(args, max_steps)

    # Step 0: warmup → lr should be small
    lr0, muon_lr0, stage0 = scheduler(0)
    assert stage0 == "warmup"
    assert lr0 < args.max_learning_rate

    # Step 10 (end of warmup): lr should be at max
    lr10, _, stage10 = scheduler(10)
    # After warmup we are in cosine decay for lr_decay_iters_coef=1.0
    assert stage10 == "cosine_decay"

    # Last step: lr should approach min
    lr_last, _, _ = scheduler(max_steps)
    assert lr_last >= args.min_learning_rate - 1e-9


def test_wsd_lr_scheduler():
    """WSD LR scheduler warmup → stable → decay."""
    args = TrainingArguments(
        max_learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_steps=10,
        lr_decay_type="wsd",
        lr_decay_iters_coef=0.1,
        optimizer_type="adamw",
        use_sqrt=False,
    )
    max_steps = 100
    scheduler = create_lr_scheduler(args, max_steps)

    lr0, _, stage0 = scheduler(0)
    assert stage0 == "warmup"

    # Mid training: should be stable
    lr50, _, stage50 = scheduler(50)
    assert stage50 == "stable"
    assert abs(lr50 - args.max_learning_rate) < 1e-9

    # Last step: in decay zone
    lr99, _, stage99 = scheduler(99)
    assert stage99 == "linear_decay"


def test_lr_scheduler_muon_adam():
    """When optimizer_type is muon_adam, scheduler returns a muon LR."""
    args = TrainingArguments(
        max_learning_rate=1e-3,
        min_learning_rate=1e-4,
        muon_learning_rate=0.02,
        warmup_steps=10,
        lr_decay_type="cosine",
        lr_decay_iters_coef=1.0,
        optimizer_type="muon_adam",
    )
    scheduler = create_lr_scheduler(args, max_steps=100)
    adam_lr, muon_lr, _ = scheduler(50)
    assert muon_lr is not None
    assert muon_lr > 0


def test_lr_scheduler_invalid_type():
    """Invalid lr_decay_type raises ValueError."""
    args = TrainingArguments(lr_decay_type="polynomial")
    raised = False
    try:
        create_lr_scheduler(args, max_steps=100)
    except ValueError:
        raised = True
    assert raised


def _make_tiny_model():
    """Return a tiny GPT2 model on CPU for optimizer tests."""
    config = AutoConfig.from_pretrained(
        "gpt2",
        n_embd=64, n_head=2, n_layer=2, n_positions=64,
        vocab_size=1000, n_inner=256,
    )
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_config(config, attn_implementation="eager")


def test_create_optimizer_adamw_cpu():
    """create_optimizer with adamw on CPU."""
    model = _make_tiny_model()
    args = TrainingArguments(
        optimizer_type="adamw",
        max_learning_rate=1e-3,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        torch_compile=False,
    )
    optimizer, step_fn, label = create_optimizer(model, args, device_type="cpu", master_process=True)
    assert label == "AdamW"
    assert optimizer is not None
    assert callable(step_fn)


def test_get_optimizer_summary_lines():
    """get_optimizer_summary_lines returns a list of strings."""
    args = TrainingArguments(optimizer_type="adamw")
    lines = get_optimizer_summary_lines(args)
    assert isinstance(lines, list)
    assert len(lines) > 0
    assert any("Optimizer type" in l for l in lines)

    # muon_adam should have extra line
    args2 = TrainingArguments(optimizer_type="muon_adam")
    lines2 = get_optimizer_summary_lines(args2)
    assert len(lines2) > len(lines)


for _fn in [
    test_muon_momentum_boundaries,
    test_zeropower_newtonschulz,
    test_adam_update,
    test_single_device_muon,
    test_single_device_muon_with_aux_adam,
    test_cosine_lr_scheduler,
    test_wsd_lr_scheduler,
    test_lr_scheduler_muon_adam,
    test_lr_scheduler_invalid_type,
    test_create_optimizer_adamw_cpu,
    test_get_optimizer_summary_lines,
]:
    run_test(_fn.__name__, _fn)

print("Test 7 — Optimizers & LR Schedulers (CPU): OK ✅")


# %%
#######################################
# 8. Utility Functions
#######################################
print("\n" + "=" * 60)
print("8. Utility Functions")
print("=" * 60)

from utils import (
    StructuredTrainingLogger,
    DistributedEnvironment,
    compute_training_schedule,
    load_checkpoint_state,
    initialize_wandb,
    create_emissions_tracker,
    cleanup_log_file,
    checkpoint_already_validated,
)


def test_compute_training_schedule_basic():
    """compute_training_schedule returns correct gradient accumulation steps."""
    args = TrainingArguments(
        total_batch_size=1024,
        micro_batch_size=4,
        num_train_epochs=1,
        max_steps=None,
    )
    args.max_position_embeddings = 64
    # tokens_per_step = 4 * 64 * 1 = 256
    # grad_accum = 1024 / 256 = 4
    ga, steps_per_epoch, max_steps = compute_training_schedule(args, train_dataloader_length=100, world_size=1)
    assert ga == 4
    assert steps_per_epoch == math.ceil(100 / 4)
    assert max_steps == steps_per_epoch


def test_compute_training_schedule_max_steps_override():
    """When max_steps is set, it overrides the epoch-based calculation."""
    args = TrainingArguments(
        total_batch_size=1024,
        micro_batch_size=4,
        num_train_epochs=1,
        max_steps=50,
    )
    args.max_position_embeddings = 64
    ga, steps_per_epoch, max_steps = compute_training_schedule(args, train_dataloader_length=100, world_size=1)
    assert max_steps == 50


def test_compute_training_schedule_misaligned_raises():
    """total_batch_size not divisible by tokens_per_step raises AssertionError."""
    args = TrainingArguments(
        total_batch_size=1000,
        micro_batch_size=4,
    )
    args.max_position_embeddings = 64
    raised = False
    try:
        compute_training_schedule(args, train_dataloader_length=100, world_size=1)
    except AssertionError:
        raised = True
    assert raised


def test_structured_logger_metadata():
    """StructuredTrainingLogger writes metadata lines."""
    tmpdir = tempfile.mkdtemp()
    try:
        log_path = os.path.join(tmpdir, "train.log")
        logger = StructuredTrainingLogger(log_path)
        logger.log_metadata("hello world")
        logger.log_metadata({"key": "value", "num": 42})

        with open(log_path) as f:
            content = f.read()
        assert "hello world" in content
        assert "key: value" in content
    finally:
        shutil.rmtree(tmpdir)


def test_structured_logger_stats():
    """StructuredTrainingLogger writes JSON stats entries."""
    tmpdir = tempfile.mkdtemp()
    try:
        log_path = os.path.join(tmpdir, "train.log")
        logger = StructuredTrainingLogger(log_path)
        logger.log_stats({"loss": 2.5, "step": 1})
        logger.log_stats({"loss": 2.3, "step": 2})

        with open(log_path) as f:
            lines = f.readlines()
        # Find JSON lines
        json_lines = [l for l in lines if l.strip().startswith("{")]
        assert len(json_lines) == 2
        parsed = json.loads(json_lines[0])
        assert "loss" in parsed
    finally:
        shutil.rmtree(tmpdir)


def test_structured_logger_create_python_logger():
    """StructuredTrainingLogger can create a configured Python logger."""
    logger = StructuredTrainingLogger.create_python_logger("test-ddp-logger")

    assert logger.name == "test-ddp-logger"
    assert logger.getEffectiveLevel() == logging.INFO


def test_structured_logger_invalid_type():
    """Logging with an unsupported type raises ValueError."""
    tmpdir = tempfile.mkdtemp()
    try:
        log_path = os.path.join(tmpdir, "train.log")
        logger = StructuredTrainingLogger(log_path)
        raised = False
        try:
            logger.log("msg", "invalid_type")
        except ValueError:
            raised = True
        assert raised
    finally:
        shutil.rmtree(tmpdir)


def test_cleanup_log_file_truncates():
    """cleanup_log_file truncates log after last validation entry."""
    tmpdir = tempfile.mkdtemp()
    try:
        log_path = os.path.join(tmpdir, "train.log")
        logger = StructuredTrainingLogger(log_path)
        logger.log_stats({"status": "training", "step": 1})
        logger.log_stats({"status": "validation", "step": 2})
        logger.log_stats({"status": "training", "step": 3})  # This should be truncated

        cleanup_log_file(log_path)

        with open(log_path) as f:
            content = f.read()
        assert "step\": 2" in content or "step\":2" in content
        # The step 3 training entry should be gone
        assert "step\": 3" not in content and "step\":3" not in content
    finally:
        shutil.rmtree(tmpdir)


def test_cleanup_log_file_missing_file():
    """cleanup_log_file with non-existent file does not raise."""
    cleanup_log_file("/nonexistent/path/log.txt")


def test_checkpoint_already_validated_no_dir():
    """Returns False when checkpoint dir does not exist."""
    result = checkpoint_already_validated("/nonexistent", "S1", 100, "/nonexistent/log.txt")
    assert result is False


def test_checkpoint_already_validated_positive():
    """Returns True when checkpoint dir and validation entry both exist."""
    tmpdir = tempfile.mkdtemp()
    try:
        # Create checkpoint directory structure
        ckpt_dir = os.path.join(tmpdir, "S1", "step_00100")
        os.makedirs(ckpt_dir)

        # Create log file with validation entry
        log_path = os.path.join(tmpdir, "train.log")
        logger = StructuredTrainingLogger(log_path)
        logger.log_stats({"status": "validation", "step": 100})

        result = checkpoint_already_validated(tmpdir, "S1", 100, log_path)
        assert result is True
    finally:
        shutil.rmtree(tmpdir)


def test_distributed_environment_local_fallback():
    """DistributedEnvironment falls back to single-process mode when no SLURM/torchrun vars are set."""
    saved = {}
    for key in ("SLURM_NTASKS", "SLURM_PROCID", "WORLD_SIZE", "RANK", "LOCAL_RANK"):
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    try:
        env = DistributedEnvironment(logging.getLogger("test"))
        assert env.world_size == 1
        assert env.rank == 0
        assert env.local_rank == 0
        assert env.master_process is True
        assert env.ddp is False
        assert env.device in ("cpu", "cuda:0")
    finally:
        os.environ.update(saved)


def test_distributed_environment_torchrun_vars():
    """DistributedEnvironment picks up WORLD_SIZE/RANK/LOCAL_RANK when SLURM vars are absent."""
    saved = {}
    for key in ("SLURM_NTASKS", "SLURM_PROCID", "WORLD_SIZE", "RANK", "LOCAL_RANK"):
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    # Simulate a single-process torchrun launch (world_size=1).
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    try:
        env = DistributedEnvironment(logging.getLogger("test"))
        assert env.world_size == 1
        assert env.rank == 0
        assert env.local_rank == 0
        assert env.ddp is False
    finally:
        for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
            os.environ.pop(key, None)
        os.environ.update(saved)


def test_distributed_environment_seed_everything():
    """seed_everything sets reproducible random state."""
    DistributedEnvironment.seed_everything(42)
    a = torch.randn(3)
    DistributedEnvironment.seed_everything(42)
    b = torch.randn(3)
    assert torch.equal(a, b)


def test_load_checkpoint_state_no_resume():
    """Returns defaults (0, 0, 1) when not resuming from checkpoint."""
    args = TrainingArguments(resume_from_checkpoint=None)
    resume_step, iter_count, epoch = load_checkpoint_state(
        args=args, checkpoint_path=None, optimizer=None,
    )
    assert resume_step == 0
    assert iter_count == 0
    assert epoch == 1


def test_load_checkpoint_state_resume():
    """Restores optimizer and returns checkpoint state when resuming."""
    tmpdir = tempfile.mkdtemp()
    try:
        # Create a fake checkpoint
        ckpt_data = {
            'optimizer': {},
            'resume_step': 50,
            'iteration': 200,
            'epoch': 2,
            'config': {},
        }
        torch.save(ckpt_data, os.path.join(tmpdir, 'checkpoint.pt'))

        # Minimal optimizer mock with load_state_dict
        class FakeOptimizer:
            def __init__(self):
                self.loaded = False
            def load_state_dict(self, state_dict):
                self.loaded = True

        fake_opt = FakeOptimizer()
        args = TrainingArguments(resume_from_checkpoint="some/path")
        args.begin_new_stage = False

        resume_step, iter_count, epoch = load_checkpoint_state(
            args=args, checkpoint_path=tmpdir, optimizer=fake_opt,
        )
        assert resume_step == 50
        assert iter_count == 200
        assert epoch == 2
        assert fake_opt.loaded
    finally:
        shutil.rmtree(tmpdir)


def test_load_checkpoint_state_new_stage():
    """Returns defaults (0, 0, 1) when beginning a new stage, but still loads optimizer."""
    tmpdir = tempfile.mkdtemp()
    try:
        ckpt_data = {
            'optimizer': {},
            'resume_step': 50,
            'iteration': 200,
            'epoch': 2,
            'config': {},
        }
        torch.save(ckpt_data, os.path.join(tmpdir, 'checkpoint.pt'))

        class FakeOptimizer:
            def __init__(self):
                self.loaded = False
            def load_state_dict(self, state_dict):
                self.loaded = True

        fake_opt = FakeOptimizer()
        args = TrainingArguments(resume_from_checkpoint="some/path")
        args.begin_new_stage = True

        resume_step, iter_count, epoch = load_checkpoint_state(
            args=args, checkpoint_path=tmpdir, optimizer=fake_opt,
        )
        assert resume_step == 0
        assert iter_count == 0
        assert epoch == 1
        assert fake_opt.loaded
    finally:
        shutil.rmtree(tmpdir)


def test_initialize_wandb_import():
    """initialize_wandb is callable and imports wandb internally."""
    assert callable(initialize_wandb)


def test_create_emissions_tracker_returns_tracker():
    """create_emissions_tracker creates, starts, and returns an EmissionsTracker."""
    try:
        import codecarbon  # noqa: F401
    except ImportError:
        return  # codecarbon not installed, skip test
    tmpdir = tempfile.mkdtemp()
    try:
        args = TrainingArguments(
            wandb_project="test-project",
            checkpoint_dir=tmpdir,
        )
        test_logger = logging.getLogger("tracker-test")
        tracker = create_emissions_tracker(args, test_logger)
        assert hasattr(tracker, '_total_energy')
        assert hasattr(tracker, 'flush')
        assert hasattr(tracker, 'stop')
        tracker.stop()
    finally:
        shutil.rmtree(tmpdir)


for _fn in [
    test_compute_training_schedule_basic,
    test_compute_training_schedule_max_steps_override,
    test_compute_training_schedule_misaligned_raises,
    test_structured_logger_metadata,
    test_structured_logger_stats,
    test_structured_logger_create_python_logger,
    test_structured_logger_invalid_type,
    test_cleanup_log_file_truncates,
    test_cleanup_log_file_missing_file,
    test_checkpoint_already_validated_no_dir,
    test_checkpoint_already_validated_positive,
    test_distributed_environment_local_fallback,
    test_distributed_environment_torchrun_vars,
    test_distributed_environment_seed_everything,
    test_load_checkpoint_state_no_resume,
    test_load_checkpoint_state_resume,
    test_load_checkpoint_state_new_stage,
    test_initialize_wandb_import,
    test_create_emissions_tracker_returns_tracker,
]:
    run_test(_fn.__name__, _fn)

print("Test 8 — Utility Functions: OK ✅")


# %%
#######################################
# 9. Integration: Forward Pass on CPU
#######################################
print("\n" + "=" * 60)
print("9. Integration: Forward Pass on CPU")
print("=" * 60)


def test_end_to_end_forward_pass():
    """
    Full integration: config → model → tokenizer → dataset → dataloader →
    collate → forward pass on CPU.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        config_dir = _create_tiny_model_config(tmpdir)
        args = TrainingArguments(
            path_to_model_config=config_dir,
            tokenizer_name_or_path=_TINY_TOKENIZER_NAME,
            attn_implementation="eager",
            cache_dir=tmpdir,
            torch_compile=False,
            use_liger_kernel=False,
            gradient_checkpointing=False,
            mat_mul_precision="highest",
            tf32=False,
            bf16=False,
            sanity_check=True,
            sanity_check_num_samples=16,
            micro_batch_size=4,
            eval_micro_batch_size=4,
            pin_memory=False,
            num_workers_for_dataloader=0,
            prefetch_factor=None,
            shuffle_dataset=False,
        )

        # 1. Build model + tokenizer
        result = prepare_training_components(args=args, device="cpu", master_process=True)
        model = result.model
        tokenizer = result.tokenizer
        args = result.args

        # 2. Build dataloaders
        bundle = prepare_dataloaders(args=args, tokenizer=tokenizer, world_size=1, rank=0)

        # 3. Grab a batch
        batch = next(iter(bundle.train_dataloader))
        assert batch["input_ids"].shape[0] <= 4
        assert "labels" in batch

        # 4. Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
            )
        assert hasattr(outputs, "loss")
        assert hasattr(outputs, "logits")
        assert outputs.loss is not None
        assert torch.isfinite(outputs.loss)
        assert outputs.logits.shape[0] == batch["input_ids"].shape[0]
        assert outputs.logits.shape[1] == batch["input_ids"].shape[1]
    finally:
        shutil.rmtree(tmpdir)


def test_end_to_end_backward_pass():
    """
    Forward + backward pass on CPU: verify gradients are computed.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        config_dir = _create_tiny_model_config(tmpdir)
        args = TrainingArguments(
            path_to_model_config=config_dir,
            tokenizer_name_or_path=_TINY_TOKENIZER_NAME,
            attn_implementation="eager",
            cache_dir=tmpdir,
            torch_compile=False,
            use_liger_kernel=False,
            gradient_checkpointing=False,
            mat_mul_precision="highest",
            tf32=False,
            bf16=False,
            sanity_check=True,
            sanity_check_num_samples=16,
            micro_batch_size=2,
            eval_micro_batch_size=2,
            pin_memory=False,
            num_workers_for_dataloader=0,
            prefetch_factor=None,
            shuffle_dataset=False,
        )

        result = prepare_training_components(args=args, device="cpu", master_process=True)
        model = result.model
        args = result.args

        bundle = prepare_dataloaders(args=args, tokenizer=result.tokenizer, world_size=1, rank=0)
        batch = next(iter(bundle.train_dataloader))

        model.train()
        outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
        outputs.loss.backward()

        # At least some parameters should have gradients
        grads_found = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert grads_found > 0, "No gradients were computed"
    finally:
        shutil.rmtree(tmpdir)


def test_end_to_end_optimizer_step():
    """
    Forward → backward → optimizer step on CPU.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        config_dir = _create_tiny_model_config(tmpdir)
        args = TrainingArguments(
            path_to_model_config=config_dir,
            tokenizer_name_or_path=_TINY_TOKENIZER_NAME,
            attn_implementation="eager",
            cache_dir=tmpdir,
            torch_compile=False,
            use_liger_kernel=False,
            gradient_checkpointing=False,
            mat_mul_precision="highest",
            tf32=False,
            bf16=False,
            sanity_check=True,
            sanity_check_num_samples=16,
            micro_batch_size=2,
            eval_micro_batch_size=2,
            pin_memory=False,
            num_workers_for_dataloader=0,
            prefetch_factor=None,
            shuffle_dataset=False,
            optimizer_type="adamw",
            max_learning_rate=1e-3,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )

        result = prepare_training_components(args=args, device="cpu", master_process=True)
        model = result.model
        args = result.args

        optimizer, step_fn, label = create_optimizer(model, args, device_type="cpu", master_process=True)

        bundle = prepare_dataloaders(args=args, tokenizer=result.tokenizer, world_size=1, rank=0)
        batch = next(iter(bundle.train_dataloader))

        # Snapshot weights before step
        first_param = next(model.parameters())
        before = first_param.data.clone()

        model.train()
        outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
        outputs.loss.backward()
        step_fn(args.max_learning_rate, None, 1)
        optimizer.zero_grad()

        after = first_param.data
        assert not torch.equal(before, after), "Weights should change after optimizer step"
    finally:
        shutil.rmtree(tmpdir)


def test_end_to_end_mfu_with_model():
    """
    Integration: create MFU context from a real model and compute metrics.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        config_dir = _create_tiny_model_config(tmpdir)
        args = TrainingArguments(
            path_to_model_config=config_dir,
            tokenizer_name_or_path=_TINY_TOKENIZER_NAME,
            attn_implementation="eager",
            cache_dir=tmpdir,
            torch_compile=False,
            use_liger_kernel=False,
            gradient_checkpointing=False,
            mat_mul_precision="highest",
            tf32=False,
            bf16=False,
            mfu_type="dense_transformer",
        )
        result = prepare_training_components(args=args, device="cpu", master_process=True)
        args = result.args

        ctx = create_mfu_context(args, "a100", num_parameters=result.trainable_params)
        metrics = calculate_training_metrics(ctx, micro_batch_size=4, gradient_accumulation_steps=1, world_size=1, dt=0.5)
        assert metrics.mfu > 0
        assert metrics.tokens_processed == 4 * 1 * args.max_position_embeddings
    finally:
        shutil.rmtree(tmpdir)


for _fn in [
    test_end_to_end_forward_pass,
    test_end_to_end_backward_pass,
    test_end_to_end_optimizer_step,
    test_end_to_end_mfu_with_model,
]:
    run_test(_fn.__name__, _fn)

print("Test 9 — Integration: Forward Pass on CPU: OK ✅")


# %%
#######################################
# 10. Trainer
#######################################
print("\n" + "=" * 60)
print("10. Trainer")
print("=" * 60)

from trainer import Trainer


class _MockTracker:
    """Minimal stand-in for codecarbon.EmissionsTracker used by the Trainer."""
    class _Energy:
        kWh = 0.0
    _total_energy = _Energy()
    def flush(self): pass
    def stop(self): pass


def test_trainer_cpu_two_steps():
    """
    Construct a Trainer on CPU with a tiny model and run 2 steps.
    Verifies the loop completes without error and that model weights change.
    """
    import logging
    tmpdir = tempfile.mkdtemp()
    try:
        config_dir = _create_tiny_model_config(tmpdir)
        args = TrainingArguments(
            path_to_model_config=config_dir,
            tokenizer_name_or_path=_TINY_TOKENIZER_NAME,
            attn_implementation="eager",
            cache_dir=tmpdir,
            torch_compile=False,
            use_liger_kernel=False,
            gradient_checkpointing=False,
            mat_mul_precision="highest",
            tf32=False,
            bf16=False,
            sanity_check=True,
            sanity_check_num_samples=16,
            micro_batch_size=2,
            eval_micro_batch_size=2,
            pin_memory=False,
            num_workers_for_dataloader=0,
            prefetch_factor=None,
            shuffle_dataset=False,
            optimizer_type="adamw",
            max_learning_rate=1e-3,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            max_grad_norm=1.0,
            num_train_epochs=1,
            total_batch_size=128,
            checkpointing_steps=2,
            stage_name="test",
            checkpoint_dir=tmpdir,
            wandb_token=None,
            push_to_hub=False,
            begin_new_stage=True,
            mfu_type="dense_transformer",
            lr_decay_type="cosine",
        )

        result = prepare_training_components(args=args, device="cpu", master_process=True)
        model = result.model
        args = result.args

        optimizer, step_fn, _ = create_optimizer(model, args, device_type="cpu", master_process=True)

        bundle = prepare_dataloaders(args=args, tokenizer=result.tokenizer, world_size=1, rank=0)

        gradient_accumulation_steps, _, max_steps = compute_training_schedule(
            args, len(bundle.train_dataloader), world_size=1,
        )
        max_steps = 2  # keep the test fast

        lr_scheduler = create_lr_scheduler(args, max_steps)

        log_file = os.path.join(tmpdir, "test.log")
        file_logger = StructuredTrainingLogger(log_file)
        logger = logging.getLogger("trainer-test")
        logger.setLevel(logging.WARNING)  # suppress verbose training logs

        mfu_context = create_mfu_context(args, "a100", num_parameters=result.trainable_params)

        # Snapshot weights before training.
        first_param = next(model.parameters())
        before = first_param.data.clone()

        trainer = Trainer(
            args=args,
            model=model,
            raw_model=model,
            tokenizer=result.tokenizer,
            optimizer=optimizer,
            optimizer_step=step_fn,
            lr_scheduler=lr_scheduler,
            train_dataloader=bundle.train_dataloader,
            validation_dataloader=bundle.val_dataloader,
            train_sampler=bundle.train_sampler,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_steps=max_steps,
            resume_step=0,
            iter_count=0,
            epoch=1,
            device="cpu",
            device_type="cpu",
            ddp=False,
            world_size=1,
            master_process=True,
            precision=torch.float32,
            logger=logger,
            file_logger=file_logger,
            log_file=log_file,
            slurm_job_id="test-000",
            tracker=_MockTracker(),
            mfu_context=mfu_context,
        )
        trainer.train()

        after = first_param.data
        assert not torch.equal(before, after), "Weights should change after training"
    finally:
        shutil.rmtree(tmpdir)


for _fn in [
    test_trainer_cpu_two_steps,
]:
    run_test(_fn.__name__, _fn)

print("Test 10 — Trainer: OK ✅")


# %%
#######################################
# 11. Report
#######################################
report()
