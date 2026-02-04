"""
Training Configuration Specifications for Distributed Data Parallel (DDP)

Dataclass-based training arguments for large-scale transformer model training.
Supports distributed training, gradient accumulation, mixed precision, and various
optimization strategies.s
"""
from typing import Optional, Union
from dataclasses import dataclass, field

@dataclass
class TrainingArguments:
    """Class to hold the training arguments."""
    
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
    mask_eos_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to mask the loss of the eos token during training."},
    )
    mask_pad_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to mask the loss of the pad token during training."},
    )

    # Model architecture settings
    vocab_size: Optional[int] = field(
        default=32000,
        metadata={"help": (
            "The vocab size of the tokenizer."
            "This will ALWAYS be overridden by the tokenizer's vocab size."
            "If you want to use a different vocab size, use a different tokenizer."
            "We are just leaving this here for compatibility with older configurations."
        )},
    )
    num_hidden_layers: Optional[int] = field(
        default=12,
        metadata={"help": "The number of layers in the model."},
    )
    num_attention_heads: Optional[int] = field(
        default=12,
        metadata={"help": "The number of attention heads in the model."},
    )
    num_key_value_heads: Optional[int] = field(
        default=12,
        metadata={"help": (
            "The number of key-value heads for the attention mechanism."
            "If you want to do vanilla attention, set this to the same value as `num_attention_heads`."
            "If you want to do multi-query attention, set this to 1."
            "If you want to do grouped query attention, set this to a value that divides `num_attention_heads` evenly."
        )},
    )
    head_dim: Optional[int] = field(
        default=None,
        metadata={"help": (
            "The dimension of each attention head."
            "If not specified, it will be calculated as `hidden_size // num_attention_heads`."
        )},
    )
    hidden_size: Optional[int] = field(
        default=768,
        metadata={"help": "The embedding size of the model."},
    )
    intermediate_size: Optional[int] = field(
        default=None,
        metadata={"help": "The intermediate size of the model. Defaults to 4 * hidden_size."},
    )
    max_position_embeddings: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum context size of the model."},
    )
    tie_word_embeddings: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to tie the word embeddings."},
    )
    hidden_act: Optional[str] = field(
        default="silu",
        metadata={"help": "The activation function to use."},
    )
    output_hidden_states: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to output the hidden states."},
    )
    attn_implementation: Optional[str] = field(
        default="eager",
        metadata={"help": "The attention implementation to use. Options: `eager`, `sdpa`, `flash_attention_2`."},
    )
    use_cache: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use cache for faster inference."},
    )
    no_rope_layer_interval: Optional[int] = field(
        default=None,
        metadata={"help": (
            "IMPORTANT: This is a new setting and is only available for certain `model_type`s."
            "The interval at which to disable RoPE in the model."
            "If set to None, RoPE will be applied to all layers."
            "If set to a positive integer, RoPE will be applied to every Nth layer."
            "E.g., Smollm3 uses 4."
        )},
    )
    rope_theta : Optional[float] = field(
        default=10000.0,
        metadata={"help": (
            "This is the scaling factor for the RoPE embeddings."
            "Setting this number is still something of a black box (to me at least)."
            "Llama uses 10000.0, and Smollm3 uses 50000.0."
            "What I do know is that when we are doing context extension,"
            "we need to increase this number to avoid the embeddings being too close together."
            "If in doubt, check how Smollm3 does it for every stage of context extension."
            "E.g., https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs"
            "They jump from 50k to 2M, and then to 5M, for 4K -> 32K -> 64K context size increases."
        )},
    )
    rope_scale_factor: Optional[int] = field(
        default=None,
        metadata={"help": (
            "This is the scaling factor for the RoPE embeddings."
            "If set to None, no scaling will be applied."
            "If set to a positive integer (> 1), the embeddings will be scaled by this factor."
            "E.g., 4096 * 4 = 16384"
        )},
    )
    rms_norm_eps: Optional[float] = field(
        default=1e-6,
        metadata={"help": (
            "The epsilon value for the RMS normalization layer."
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
    warmup_steps: Optional[int] = field(
        default=1000,
        metadata={"help": "The number of warmup steps."},
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
            "The percentage of the toal number of steps (minus warmup steps) over which the learning rate will decay."
            "If the value is 0, no decay will be applied."  
        )},
    )
    seed: Optional[int] = field(
        default=1337,
        metadata={"help": "The seed for PyTorch to ensure reproducibility."},
    )
    max_steps: Optional[int] = field(
        default=None,
        metadata={"help": (
            "The maximum number of training steps." 
            "If None, it will be calculated based on the size of the dataset, the dataloader, and the number of epochs."
            "If specified, it will override the in-built calculation."
        )},
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
            "WARNING: This breaks the training loop and if we are doing gradieent accumulation."
            "There is an incompatibility with the `model.no_sync()` context manager."
            "Learn more here: https://github.com/pytorch/pytorch/issues/143580"
        )},
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

    # Tokenizer and Reference model settings
    reference_model: Optional[str] = field(
        default=None,
        metadata={"help": (
            "The name of the reference model to use for training. "
            "This allows the AutoConfig to loadthe proper configuration for the model."
            "If not specified, it will use a default configuration (llama)"
            "You can also specify a path to a local json file with the configuration."
            "The only thing this file has to acctually contain is the `model_type` field."
            "This script will work with almost any decoder model listed here: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoConfig"
            "e.g., llama, mistral, mixtral, olmo, phi, qwen3, smollm3"
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
    continual_pretraining: Optional[bool] = field(
        default=False,
        metadata={"help": (
            "Whether to do continual pretraining from a reference model."
            "If set to True, the model will be initialized from the reference model."
            "If set to False, the model will be initialized from `AutoConfig`."
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

    # Miscellaneous settings
    sanity_check: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run a sanity check on a small dummy dataset."},
    )
    sanity_check_num_samples: Optional[int] = field(
        default=1_000_000,
        metadata={"help": "The number of samples to use for the sanity check."},
    )
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
