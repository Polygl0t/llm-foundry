"""
Training Configuration Specifications for the distributed training scripts.

Dataclass-based training arguments for large-scale transformer model training.
Supports distributed training, gradient accumulation, mixed precision, and various
optimization strategies.

Provides:
    - `TrainingArguments` dataclass that encapsulates all training configuration options.
"""
from typing import Any, Optional, Union
from dataclasses import dataclass, field, fields

import yaml

@dataclass
class TrainingArguments:
    """Class to hold the training arguments."""

    @staticmethod
    def load_yaml(specs_path: str) -> dict[str, Any]:
        """Load training arguments from a YAML file."""
        with open(specs_path, "r", encoding="utf-8") as stream:
            loaded_args = yaml.safe_load(stream)

        if loaded_args is None:
            return {}

        if not isinstance(loaded_args, dict):
            raise ValueError("Training specifications YAML must define a mapping of argument names to values.")

        return loaded_args

    @classmethod
    def from_yaml(cls, specs_path: str) -> "TrainingArguments":
        """Create TrainingArguments directly from a YAML specifications file."""
        return cls(**cls.load_yaml(specs_path))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the current TrainingArguments state, including runtime fields."""
        return {dataclass_field.name: getattr(self, dataclass_field.name) for dataclass_field in fields(self)}
    
    # Directory settings
    checkpoint_dir: Optional[str] = field(
        default="./checkpoints",
        metadata={"help": (
            "The directory to save the model checkpoints."
            "As a general rule, try to remember to set this to scratch if you are running on a cluster."
        )},
    )
    train_dataset_dir: Optional[Union[str, list[str]]] = field(
        default="./dataset/train",
        metadata={"help": (
            "The directory or list of directories where the training dataset is stored."
            "This can be a string path or a list of string paths to directories of files ending in `dataset_type` (e.g., `parquet`, `jsonl`)."
            "If the directory contains other folders, it will concatenate all files in each folder."
        )}
    )
    val_dataset_dir: Optional[str] = field(
        default="./dataset/val",
        metadata={"help": (
            "The directory where the validation dataset is stored."
            "This has to be a directory of files ending in `dataset_type` (e.g., `parquet`, `jsonl`)."
            "We expect that all validation files are in the same directory."
        )}
    )
    dataset_type: Optional[str] = field(
        default="parquet",
        metadata={"help": "The type of dataset to use. Options: `jsonl`, `parquet`."},
    )
    cache_dir: Optional[str] = field(
        default="./cache",
        metadata={"help": "The directory to save the cache files."},
    )

    # Data loading settings
    num_workers_for_dataloader: Optional[int] = field(
        default=4,
        metadata={"help": "The number of workers for the dataloader."},
    )
    prefetch_factor: Optional[int] = field(
        default=4,
        metadata={"help": "The prefetch factor for the dataloader."},
    )
    pin_memory: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to pin the memory for faster data transfer on the dataloader."},
    )
    shuffle_dataset: Optional[bool] = field(
        default=False,
        metadata={"help": (
            "Whether to shuffle the paths of the dataset files before loading them."
            "This only applies to the training dataset."
            "If set to True, it will also set the `shuffle` argument of the `DistributedSampler` to True."
        )}  
    )
    additional_mask_token_ids: Optional[list[int]] = field(
        default=None,
        metadata={"help": (
            "A list of extra token IDs to mask (set to -100) in the labels during training. "
            "Pad, EOS, and BOS tokens are always masked automatically when defined in the tokenizer."
        )},
    )

    # Model and tokenizer settings
    path_to_model_config: Optional[str] = field(
        default=None,
        metadata={"help": (
            "Path to a Hugging Face-compatible model config file (e.g., config.json) or directory. "
            "Used to initialize the model architecture via AutoConfig.from_pretrained(). "
            "Required for training from scratch. For continual pretraining, the config is loaded from `base_model`."
        )},
    )
    base_model: Optional[str] = field(
        default=None,
        metadata={"help": (
            "Path or Hugging Face hub ID of the base model. "
            "Used for continual pretraining (loads pretrained weights) and as a "
            "fallback for tokenizer loading when `tokenizer_name_or_path` is not set. "
            "Not needed when training from scratch — use `path_to_model_config` instead."
        )},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The name or path of the tokenizer to use."},
    )
    chat_template_path: Optional[str] = field(
        default=None,
        metadata={"help": (
            "The path to a chat template jinja2 file."
            "If specified, the chat template will be added to the tokenizer."
        )},
    )
    attn_implementation: Optional[str] = field(
        default="eager",
        metadata={"help": "The attention implementation to use. Options: `eager`, `sdpa`, `flash_attention_2`."},
    )
    continual_pretraining: Optional[bool] = field(
        default=False,
        metadata={"help": (
            "Whether to do continual pretraining from the `base_model`."
            "If set to True, the model will be initialized with pretrained weights from `base_model`."
            "If set to False, the model will be initialized from scratch using `path_to_model_config`."
        )},
    )
    new_max_position_embeddings: Optional[int] = field(
        default=None,
        metadata={"help": (
            "Override the max_position_embeddings from the model config for context extension. "
            "If set, this value replaces the config's max_position_embeddings. "
            "Takes priority over `rope_scale_factor`. Only relevant for continual pretraining."
        )},
    )
    new_rope_theta: Optional[float] = field(
        default=None,
        metadata={"help": (
            "Override the rope_theta from the model config for context extension. "
            "When performing RoPE scaling, you typically need to increase both "
            "max_position_embeddings and rope_theta."
        )},
    )
    rope_scale_factor: Optional[int] = field(
        default=None,
        metadata={"help": (
            "Multiplier to scale the config's max_position_embeddings for context extension. "
            "If set to a positive integer (> 1), the config's max_position_embeddings will be "
            "multiplied by this factor. Ignored if `new_max_position_embeddings` is set. "
            "E.g., 4096 * 4 = 16384"
        )},
    )

    # Training settings
    total_batch_size: Optional[int] = field(
        default=524288,
        metadata={"help": "The total batch size in tokens."},
    )
    micro_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "The micro batch size."},
    )
    eval_micro_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "The evaluation micro batch size."},
    )
    num_train_epochs: Optional[Union[float, int]] = field(
        default=1,
        metadata={"help": "The number of training epochs."},
    )
    max_steps: Optional[int] = field(
        default=None,
        metadata={"help": (
            "The maximum number of training steps." 
            "If None, it will be calculated based on the size of the dataset, the dataloader, and the number of epochs."
            "If specified, it will override the in-built calculation."
        )},
    )
    seed: Optional[int] = field(
        default=1337,
        metadata={"help": "The seed for PyTorch to ensure reproducibility."},
    )
    mfu_type: Optional[str] = field(
        default="dense_transformer",
        metadata={"help": (
            "The MFU calculation strategy to use. "
            "Options currently include: `dense_transformer`, `mamba`, and `hybrid` (a combination of the previous two). "
            "This is intended to be extended for other architectures such as MoE or Mamba."
        )},
    )

    # Optimizer settings
    optimizer_type: Optional[str] = field(
        default="adamw",
        metadata={"help": (
            "The optimizer configuration to use. "
            "Options: `adamw` for standard AdamW, `muon_adam` for hybrid Muon + Adam."
        )},
    )
    max_learning_rate: Optional[float] = field(
        default=1e-3,
        metadata={"help": "The initial maximum learning rate."},
    )
    min_learning_rate: Optional[float] = field(
        default=1e-4,
        metadata={"help": "The minimum learning rate."},
    )
    muon_learning_rate: Optional[float] = field(
        default=0.02,
        metadata={"help": "The learning rate for the Muon optimizer."},
    )
    warmup_steps: Optional[int] = field(
        default=1000,
        metadata={"help": "The number of warmup steps."},
    )
    lr_decay_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The type of learning rate decay to use. Options: `cosine` and `wsd`."},
    )
    use_sqrt: Optional[bool] = field(
        default=False,
        metadata={"help": (
            "Whether to use 1 - sqrt learning rate decay instead of linear decay."
            "This is only applicable if `lr_decay_type` is set to `wsd`."
        )},
    )
    lr_decay_iters_coef: Optional[float] = field(
        default=0,
        metadata={"help": (
            "The percentage of the total number of steps (minus warmup steps) over which the learning rate will decay."
            "If the value is 0, no decay will be applied."  
        )},
    )
    weight_decay: Optional[float] = field(
        default=0.0,
        metadata={"help": "The weight decay to apply."},
    )
    beta1: Optional[float] = field(
        default=0.9,
        metadata={"help": "The beta1 parameter for the Adam optimizer."},
    )
    beta2: Optional[float] = field(
        default=0.95,
        metadata={"help": "The beta2 parameter for the Adam optimizer."},
    )
    eps: Optional[float] = field(
        default=1e-8,
        metadata={"help": "The epsilon parameter for the Adam optimizer."},
    )
    max_grad_norm: Optional[float] = field(
        default=1.0,
        metadata={"help": "The maximum gradient norm for gradient clipping."},
    )

    # Precision and optimization settings
    torch_compile: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use `torch.compile` for optimization."},
    )
    mat_mul_precision: Optional[str] = field(
        default="highest",
        metadata={"help": (
            "The precision for matrix multiplication. "
            "Options: highest, high, medium."
        )}
    )
    tf32: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use tf32 mode (requires Ampere GPU)."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use bf16 mode."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use gradient checkpointing."},
    )
    use_liger_kernel: Optional[bool] = field(
        default=False,
        metadata={"help": (
            "Whether to use the Liger kernels for training."
            "The promise is to increase multi-GPU training throughput by 20% and reduce memory usage by 60%"
            "WARNING: Not all models are compatible with this set of kernels."
            "Check the documentation for more information."
            "https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py#L1853"
        )},
    )

    static_graph: Optional[bool] = field(
        default=False,
        metadata={"help": (
            "Whether to use a static graph for training in the DDP setup."
            "WARNING: This breaks the training loop if we are doing gradient accumulation."
            "There is an incompatibility with the `model.no_sync()` context manager."
            "Learn more here: https://github.com/pytorch/pytorch/issues/143580"
        )},
    )
    enable_expert_parallelism: Optional[bool] = field(
        default=False,
        metadata={"help": (
            "Whether to enable expert parallelism for MoE models via transformers' DistributedConfig. "
            "When True, the model will be loaded with `distributed_config=DistributedConfig(enable_expert_parallel=True)`. "
            "Requires a compatible version of transformers (>= 5.x). If the import fails, "
            "a warning is logged and training continues without expert parallelism."
        )},
    )
    use_kernels: Optional[bool] = field(
        default=False,
        metadata={"help": (
            "Whether to use optimized HF Hub kernels via the `kernels` library. "
            "When True, the model will be loaded with `use_kernels=True`, letting transformers "
            "automatically find and apply the best available kernel implementations. "
            "Requires the `kernels` package (>= 0.11.0) and a compatible version of transformers. "
            "If the import fails, a warning is logged and training continues without kernels."
        )},
    )
    
    # FSDP specific settings.
    fsdp_mixed_precision: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use mixed precision training with FSDP."},
    )
    dp_shard: Optional[int] = field(
        default=None,
        metadata={"help": "The number of shards to use for Hybrid Sharding Data Parallel (HSDP)."},
    )
    full_shard: Optional[bool] = field(
        default=True,
        metadata={"help": (
            "If True, then this reshards parameters after forward and re-all-gathers in backward."
            "This is equivalent to ZeRO stage 3."
            "If False, then this does not reshard parameters."
            "If False, then this keeps the unsharded parameters in memory after forward and avoids the all-gather in backward."
            "This is equivalent to ZeRO stage 2."
        )}
    )
    cpu_offload: Optional[bool] = field(
        default=False,
        metadata={"help": "This offload policy offloads parameters, gradients, and optimizer states to CPU."},
    )
    explicit_prefetching: Optional[bool] = field(
        default=False,
        metadata={"help": (
            "Whether to use explicit prefetching."
            "This can help to overlap data transfer and computation."
            "The number of layers to prefetch is set to 2."
        )},
    )

    # Checkpoint settings
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the checkpoint to resume from."},
    )
    checkpointing_steps: Optional[int] = field(
        default=2000,
        metadata={"help": "The number of steps to save a checkpoint. Eval will be performed after each checkpoint."},
    )
    begin_new_stage: Optional[bool] = field(
        default=False,
        metadata={"help": (
            "Whether to begin a new stage of training."
            "If set to True, the training will start from the beginning,"
            " i.e., all counters will be reset."
        )},
    )
    stage_name: Optional[str] = field(
        default="S1",
        metadata={"help": "The name of the current training stage."},
    )

    # Hub settings
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to push the model to the hub."},
    )
    hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "The token to the huggingface hub."},
    )
    hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "The model id to push to the hub (e.g., userName/modelName)."},
    )

    # Logging settings
    wandb_token: Optional[str] = field(
        default=None,
        metadata={"help": "The token to your W&B account."},
    )
    wandb_id: Optional[str] = field(
        default=None,
        metadata={"help": "The id of the W&B run."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the W&B project."},
    )
    wandb_desc: Optional[str] = field(
        default=None,
        metadata={"help": "The description of the W&B run or project."},
    )

    # Miscellaneous settings
    sanity_check: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run a sanity check on a small dummy dataset."},
    )
    sanity_check_num_samples: Optional[int] = field(
        default=1_000_000,
        metadata={"help": "The number of samples to use for the sanity check."},
    )

    # Runtime fields (populated by model_setup.py after model initialization, not from YAML)
    max_position_embeddings: Optional[int] = field(
        default=None, init=False, repr=False,
        metadata={"help": "Sequence length from the model config. Set automatically after model init."},
    )
    vocab_size: Optional[int] = field(
        default=None, init=False, repr=False,
        metadata={"help": "Vocabulary size from the model config. Set automatically after model init."},
    )
    num_hidden_layers: Optional[int] = field(
        default=None, init=False, repr=False,
        metadata={"help": "Number of hidden layers from the model config. Set automatically after model init."},
    )
    num_attention_heads: Optional[int] = field(
        default=None, init=False, repr=False,
        metadata={"help": "Number of attention heads from the model config. Set automatically after model init."},
    )
    head_dim: Optional[int] = field(
        default=None, init=False, repr=False,
        metadata={"help": "Head dimension from the model config. Set automatically after model init."},
    )
