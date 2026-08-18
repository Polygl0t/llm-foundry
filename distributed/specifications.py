"""
Training Configuration Specifications for the distributed training scripts.

Dataclass-based training arguments for large-scale transformer model training.
Supports distributed training, gradient accumulation, mixed precision, and various
optimization strategies.

Provides:
    - `TrainingArguments` dataclass that encapsulates all training configuration options.
"""

from dataclasses import dataclass, field, fields
from typing import Any

import yaml


@dataclass
class TrainingArguments:
    """Class to hold the training arguments."""

    @staticmethod
    def load_yaml(specs_path: str) -> dict[str, Any]:
        """Load training arguments from a YAML file."""
        with open(specs_path, encoding="utf-8") as stream:
            loaded_args = yaml.safe_load(stream)

        if loaded_args is None:
            return {}

        if not isinstance(loaded_args, dict):
            raise ValueError(
                "Training specifications YAML must define a mapping of argument names to values."
            )

        return loaded_args

    @classmethod
    def from_yaml(cls, specs_path: str) -> "TrainingArguments":
        """Create TrainingArguments directly from a YAML specifications file."""
        return cls(**cls.load_yaml(specs_path))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the current TrainingArguments state, including runtime fields."""
        return {
            dataclass_field.name: getattr(self, dataclass_field.name)
            for dataclass_field in fields(self)
        }

    @property
    def wandb_enabled(self) -> bool:
        """Whether W&B (or trackio, in offline mode) logging is enabled.

        True when either a `wandb_token` is provided (real W&B) or `offline_mode`
        is set (trackio, which needs no token).
        """
        return self.wandb_token is not None or bool(self.offline_mode)

    # Directory settings
    checkpoint_dir: str | None = field(
        default="./checkpoints",
        metadata={
            "help": (
                "The directory to save the model checkpoints."
                "As a general rule, try to remember to set this to scratch if you are running on a cluster."
            )
        },
    )
    train_dataset_dir: str | list[str] | None = field(
        default="./dataset/train",
        metadata={
            "help": (
                "The directory or list of directories where the training dataset is stored."
                "This can be a string path or a list of string paths to directories of files ending in `dataset_type` (e.g., `parquet`, `jsonl`)."
                "If the directory contains other folders, it will concatenate all files in each folder."
            )
        },
    )
    val_dataset_dir: str | None = field(
        default="./dataset/val",
        metadata={
            "help": (
                "The directory where the validation dataset is stored."
                "This has to be a directory of files ending in `dataset_type` (e.g., `parquet`, `jsonl`)."
                "We expect that all validation files are in the same directory."
            )
        },
    )
    dataset_type: str | None = field(
        default="parquet",
        metadata={"help": "The type of dataset to use. Options: `jsonl`, `parquet`."},
    )
    cache_dir: str | None = field(
        default="./cache",
        metadata={"help": "The directory to save the cache files."},
    )

    # Data loading settings
    num_workers_for_dataloader: int | None = field(
        default=4,
        metadata={"help": "The number of workers for the dataloader."},
    )
    prefetch_factor: int | None = field(
        default=4,
        metadata={"help": "The prefetch factor for the dataloader."},
    )
    pin_memory: bool | None = field(
        default=True,
        metadata={"help": "Whether to pin the memory for faster data transfer on the dataloader."},
    )
    shuffle_dataset: bool | None = field(
        default=False,
        metadata={
            "help": (
                "Whether to shuffle the paths of the dataset files before loading them."
                "This only applies to the training dataset."
                "If set to True, it will also set the `shuffle` argument of the `DistributedSampler` to True."
            )
        },
    )
    additional_mask_token_ids: list[int] | None = field(
        default=None,
        metadata={
            "help": (
                "A list of extra token IDs to mask (set to -100) in the labels during training. "
                "Pad, EOS, and BOS tokens are always masked automatically when defined in the tokenizer."
            )
        },
    )

    # Model and tokenizer settings
    path_to_model_config: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to a Hugging Face-compatible model config file (e.g., config.json) or directory. "
                "Used to initialize the model architecture via AutoConfig.from_pretrained(). "
                "Required for training from scratch. For continual pretraining, the config is loaded from `base_model`."
            )
        },
    )
    base_model: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path or Hugging Face hub ID of the base model. "
                "Used for continual pretraining (loads pretrained weights) and as a "
                "fallback for tokenizer loading when `tokenizer_name_or_path` is not set. "
                "Not needed when training from scratch — use `path_to_model_config` instead."
            )
        },
    )
    tokenizer_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "The name or path of the tokenizer to use. "
                "Optional for training from scratch when your datasets are already tokenized; "
                "in that case the trainer skips tokenizer-dependent masking and does not save a tokenizer. "
                "For continual pretraining, the tokenizer is still required and defaults to `base_model` when omitted."
            )
        },
    )
    chat_template_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "The path to a chat template jinja2 file."
                "If specified, the chat template will be added to the tokenizer."
            )
        },
    )
    attn_implementation: str | None = field(
        default="eager",
        metadata={
            "help": "The attention implementation to use. Options: `eager`, `sdpa`, `flash_attention_2`, `flash_attention_3`, and `flash_attention_4`."
        },
    )
    continual_pretraining: bool | None = field(
        default=False,
        metadata={
            "help": (
                "Whether to do continual pretraining from the `base_model`."
                "If set to True, the model will be initialized with pretrained weights from `base_model`."
                "If set to False, the model will be initialized from scratch using `path_to_model_config`."
            )
        },
    )
    new_max_position_embeddings: int | None = field(
        default=None,
        metadata={
            "help": (
                "Override the max_position_embeddings from the model config for context extension. "
                "If set, this value replaces the config's max_position_embeddings. "
                "Takes priority over `rope_scale_factor`. Only relevant for continual pretraining."
            )
        },
    )
    new_rope_theta: float | None = field(
        default=None,
        metadata={
            "help": (
                "Override the rope_theta from the model config for context extension. "
                "When performing RoPE scaling, you typically need to increase both "
                "max_position_embeddings and rope_theta."
            )
        },
    )
    rope_scale_factor: int | None = field(
        default=None,
        metadata={
            "help": (
                "Multiplier to scale the config's max_position_embeddings for context extension. "
                "If set to a positive integer (> 1), the config's max_position_embeddings will be "
                "multiplied by this factor. Ignored if `new_max_position_embeddings` is set. "
                "E.g., 4096 * 4 = 16384"
            )
        },
    )

    # Training settings
    total_batch_size: int | None = field(
        default=524288,
        metadata={"help": "The total batch size in tokens."},
    )
    micro_batch_size: int | None = field(
        default=32,
        metadata={"help": "The micro batch size."},
    )
    eval_micro_batch_size: int | None = field(
        default=32,
        metadata={"help": "The evaluation micro batch size."},
    )
    eval_only: bool | None = field(
        default=False,
        metadata={"help": "Whether to run a single validation pass and exit without training."},
    )
    num_train_epochs: float | int | None = field(
        default=1,
        metadata={"help": "The number of training epochs."},
    )
    max_steps: int | None = field(
        default=None,
        metadata={
            "help": (
                "The maximum number of training steps."
                "If None, it will be calculated based on the size of the dataset, the dataloader, and the number of epochs."
                "If specified, it will override the in-built calculation."
            )
        },
    )
    seed: int | None = field(
        default=1337,
        metadata={"help": "The seed for PyTorch to ensure reproducibility."},
    )

    # Optimizer settings
    optimizer_type: str | None = field(
        default="adamw",
        metadata={
            "help": (
                "The optimizer configuration to use. "
                "Options: `adamw` for standard AdamW, `muon_adam` for hybrid Muon + Adam."
            )
        },
    )
    max_learning_rate: float | None = field(
        default=1e-3,
        metadata={"help": "The initial maximum learning rate."},
    )
    min_learning_rate: float | None = field(
        default=1e-4,
        metadata={"help": "The minimum learning rate."},
    )
    muon_learning_rate: float | None = field(
        default=0.02,
        metadata={"help": "The learning rate for the Muon optimizer."},
    )
    warmup_steps: int | None = field(
        default=1000,
        metadata={"help": "The number of warmup steps."},
    )
    lr_decay_type: str | None = field(
        default="cosine",
        metadata={"help": "The type of learning rate decay to use. Options: `cosine` and `wsd`."},
    )
    use_sqrt: bool | None = field(
        default=False,
        metadata={
            "help": (
                "Whether to use 1 - sqrt learning rate decay instead of linear decay."
                "This is only applicable if `lr_decay_type` is set to `wsd`."
            )
        },
    )
    lr_decay_iters_coef: float | None = field(
        default=0,
        metadata={
            "help": (
                "The percentage of the total number of steps (minus warmup steps) over which the learning rate will decay."
                "If the value is 0, no decay will be applied."
            )
        },
    )
    weight_decay: float | None = field(
        default=0.0,
        metadata={"help": "The weight decay to apply."},
    )
    beta1: float | None = field(
        default=0.9,
        metadata={"help": "The beta1 parameter for the Adam optimizer."},
    )
    beta2: float | None = field(
        default=0.95,
        metadata={"help": "The beta2 parameter for the Adam optimizer."},
    )
    eps: float | None = field(
        default=1e-8,
        metadata={"help": "The epsilon parameter for the Adam optimizer."},
    )
    max_grad_norm: float | None = field(
        default=1.0,
        metadata={"help": "The maximum gradient norm for gradient clipping."},
    )

    # Precision and optimization settings
    torch_compile: bool | None = field(
        default=False,
        metadata={"help": "Whether to use `torch.compile` for optimization."},
    )
    mat_mul_precision: str | None = field(
        default="highest",
        metadata={
            "help": ("The precision for matrix multiplication. Options: highest, high, medium.")
        },
    )
    tf32: bool | None = field(
        default=False,
        metadata={"help": "Whether to use tf32 mode (requires Ampere GPU)."},
    )
    bf16: bool | None = field(
        default=False,
        metadata={"help": "Whether to use bf16 mode."},
    )
    fp8: bool | None = field(
        default=False,
        metadata={
            "help": (
                "Whether to use fp8 mixed precision training via `torchao.float8`. "
                "When True, eligible `torch.nn.Linear` modules are swapped for `Float8Linear`, "
                "so the forward/backward matmuls run in fp8 while parameters, gradients, "
                "optimizer states, and all non-linear ops stay in bf16/fp32. "
                "Requires an fp8-capable GPU (compute capability >= 8.9, i.e. Ada / Hopper / "
                "Grace Hopper / Blackwell) and the `torchao` package. If `torchao` is not "
                "installed, or the hardware does not support fp8, a warning is logged and "
                "training falls back to the default configuration (tf32 / bf16)."
                "\n\nNOTES: fp8 is applied AFTER every other model-level optimization "
                "(Liger kernels, gradient checkpointing) so it wraps the final module tree. "
                "The `lm_head` is never converted (it is either fused by Liger's "
                "fused_linear_cross_entropy or numerically sensitive), and linears whose "
                "`in_features`/`out_features` are not divisible by 16 are skipped because "
                "the fp8 gemms require 16-element alignment. fp8 speedups grow with GEMM "
                "size and are largest when combined with `torch_compile`."
            )
        },
    )
    fp8_recipe: str | None = field(
        default="tensorwise",
        metadata={
            "help": (
                "The `torchao.float8` scaling recipe to use when `fp8` is True. "
                "Options: `tensorwise` (fastest), `rowwise` (more robust to outliers), "
                "`rowwise_with_gw_hp` (most accurate, keeps grad_weight in high precision)."
            )
        },
    )
    gradient_checkpointing: bool | None = field(
        default=False,
        metadata={"help": "Whether to use gradient checkpointing."},
    )
    use_liger_kernel: bool | None = field(
        default=False,
        metadata={
            "help": (
                "Whether to use the Liger kernels for training."
                "The promise is to increase multi-GPU training throughput by 20% and reduce memory usage by 60%."
                "WARNING: Not all models are compatible with this set of kernels."
                "Check the documentation for more information."
                "https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py#L1853"
            )
        },
    )

    static_graph: bool | None = field(
        default=False,
        metadata={
            "help": (
                "Whether to use a static graph for training in the DDP setup."
                "WARNING: This breaks the training loop if we are doing gradient accumulation."
                "There is an incompatibility with the `model.no_sync()` context manager."
                "Learn more here: https://github.com/pytorch/pytorch/issues/143580"
            )
        },
    )
    enable_expert_parallelism: bool | None = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable expert parallelism for MoE models via transformers' DistributedConfig. "
                "When True, the model will be loaded with `distributed_config=DistributedConfig(enable_expert_parallel=True)`. "
                "Requires a compatible version of transformers (>= 5.x). If the import fails, "
                "a warning is logged and training continues without expert parallelism."
                "\n\nCAVEAT: EP shards experts across ranks and inserts two all-to-all collectives "
                "per MoE layer (dispatch tokens to the rank owning each chosen expert, then gather "
                "the outputs). This only pays off when expert replication is the actual bottleneck, "
                "i.e. (a) the model is large enough that DDP-replicated experts dominate memory, "
                "(b) you have many ranks so each rank owns a small slice of experts, and "
                "(c) per-step compute >> all-to-all latency. On few GPUs (e.g. 2) and small MoE "
                "models that already fit comfortably in VRAM, EP tends to be neutral or slightly "
                "slower than plain DDP because you pay communication overhead for memory savings "
                "you didn't need. Prefer EP for large MoE models on many ranks; skip it otherwise."
            )
        },
    )
    use_kernels: bool | None = field(
        default=False,
        metadata={
            "help": (
                "Whether to use optimized HF Hub kernels via the `kernels` library. "
                "When True, the model will be loaded with `use_kernels=True`, letting transformers "
                "automatically find and apply the best available kernel implementations. "
                "Requires the `kernels` package (>= 0.11.0) and a compatible version of transformers. "
                "If the import fails, a warning is logged and training continues without kernels."
                "\n\nCAVEAT: Do NOT stack `use_kernels=True` on top of `use_liger_kernel=True` for the "
                "same ops. Liger monkey-patches RMSNorm, SwiGLU, RoPE, cross-entropy, etc. with tightly "
                "fused Triton kernels. The HF Hub `kernels` path then re-wraps those modules with "
                "community kernels that are typically standalone (not fused with neighboring ops), "
                "effectively downgrading Liger's fused stack to a looser one with more kernel launches "
                "and larger saved activations. In practice this has been observed to INCREASE step "
                "time and VRAM compared to Liger alone. Use `use_kernels` when (a) you "
                "are not using Liger, or (b) you scope it to ops Liger does not patch."
            )
        },
    )

    # FSDP specific settings.
    fsdp_mixed_precision: bool | None = field(
        default=True,
        metadata={"help": "Whether to use mixed precision training with FSDP."},
    )
    sequence_parallel: bool = field(
        default=False,
        metadata={"help": "Whether to enable intra-node sequence parallelism with FSDP2."},
    )
    sp_shard: int | None = field(
        default=None,
        metadata={"help": "Optional sequence-parallel size; defaults to the local world size."},
    )
    full_shard: bool | None = field(
        default=True,
        metadata={
            "help": (
                "If True, then this reshards parameters after forward and re-all-gathers in backward."
                "This is equivalent to ZeRO stage 3."
                "If False, then this does not reshard parameters."
                "If False, then this keeps the unsharded parameters in memory after forward and avoids the all-gather in backward."
                "This is equivalent to ZeRO stage 2."
            )
        },
    )
    cpu_offload: bool | None = field(
        default=False,
        metadata={
            "help": "This offload policy offloads parameters, gradients, and optimizer states to CPU."
        },
    )
    explicit_prefetching: bool | None = field(
        default=False,
        metadata={
            "help": (
                "Whether to use explicit prefetching."
                "This can help to overlap data transfer and computation."
                "The number of layers to prefetch is set to 2."
            )
        },
    )

    # Checkpoint settings
    resume_from_checkpoint: str | None = field(
        default=None,
        metadata={"help": "The path to the checkpoint to resume from."},
    )
    checkpointing_steps: int | None = field(
        default=2000,
        metadata={
            "help": "The number of steps to save a checkpoint. Eval will be performed after each checkpoint."
        },
    )
    begin_new_stage: bool | None = field(
        default=False,
        metadata={
            "help": (
                "Whether to begin a new stage of training."
                "If set to True, the training will start from the beginning,"
                " i.e., all counters will be reset."
            )
        },
    )
    stage_name: str | None = field(
        default="S1",
        metadata={"help": "The name of the current training stage."},
    )

    # Hub settings
    push_to_hub: bool | None = field(
        default=False,
        metadata={"help": "Whether to push the model to the hub."},
    )
    hub_token: str | None = field(
        default=None,
        metadata={"help": "The token to the huggingface hub."},
    )
    hub_model_id: str | None = field(
        default=None,
        metadata={"help": "The model id to push to the hub (e.g., userName/modelName)."},
    )

    # Logging settings
    wandb_token: str | None = field(
        default=None,
        metadata={"help": "The token to your W&B account."},
    )
    wandb_id: str | None = field(
        default=None,
        metadata={"help": "The id of the W&B run."},
    )
    wandb_project: str | None = field(
        default=None,
        metadata={"help": "The name of the W&B project."},
    )
    wandb_desc: str | None = field(
        default=None,
        metadata={"help": "The description of the W&B run or project."},
    )

    # Offline mode settings (for HPC clusters without internet on compute nodes)
    offline_mode: bool | None = field(
        default=False,
        metadata={
            "help": (
                "Whether to run in offline mode, for HPC clusters without internet access on"
                " compute nodes. When True, uses CodeCarbon's OfflineEmissionsTracker instead of"
                " EmissionsTracker, and trackio instead of W&B."
            )
        },
    )
    trackio_dir: str | None = field(
        default=None,
        metadata={
            "help": (
                "Directory for the trackio database. When set, overrides the TRACKIO_DIR "
                "environment variable. If neither is provided, defaults to ~/.trackio so "
                "multiple runs share a single database."
            )
        },
    )
    codecarbon_country_iso_code: str | None = field(
        default="DEU",
        metadata={
            "help": (
                "3-letter ISO code of the country where the experiment is being run."
                " Only used by CodeCarbon's OfflineEmissionsTracker when `offline_mode` is True."
            )
        },
    )
    codecarbon_region: str | None = field(
        default="north rhine-westphalia",
        metadata={
            "help": (
                "Province/State/City where the infrastructure is hosted."
                " Only used by CodeCarbon's OfflineEmissionsTracker when `offline_mode` is True."
            )
        },
    )

    # Miscellaneous settings
    sanity_check: bool | None = field(
        default=False,
        metadata={"help": "Whether to run a sanity check on a small dummy dataset."},
    )
    sanity_check_num_samples: int | None = field(
        default=1_000_000,
        metadata={"help": "The number of samples to use for the sanity check."},
    )

    # Runtime fields (populated by model_setup.py after model initialization, not from YAML)
    max_position_embeddings: int | None = field(
        default=None,
        init=False,
        repr=False,
        metadata={
            "help": "Sequence length from the model config. Set automatically after model init."
        },
    )
    vocab_size: int | None = field(
        default=None,
        init=False,
        repr=False,
        metadata={
            "help": "Vocabulary size from the model config. Set automatically after model init."
        },
    )
    num_hidden_layers: int | None = field(
        default=None,
        init=False,
        repr=False,
        metadata={
            "help": "Number of hidden layers from the model config. Set automatically after model init."
        },
    )
    num_attention_heads: int | None = field(
        default=None,
        init=False,
        repr=False,
        metadata={
            "help": "Number of attention heads from the model config. Set automatically after model init."
        },
    )
    head_dim: int | None = field(
        default=None,
        init=False,
        repr=False,
        metadata={
            "help": "Head dimension from the model config. Set automatically after model init."
        },
    )
