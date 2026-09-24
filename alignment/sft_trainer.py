"""
Supervised Fine-Tuning (SFT) Trainer for Large Language Models

This script fine-tunes LLMs using the Hugging Face Transformers and TRL libraries.

Expected Dataset Format (always RAW; tokenising and packing are left to TRL):
{
    "messages": [
        {"role": "user", "content": "User message here."},
        {"role": "assistant", "content": "Assistant response here."},
        ...
    ]
}

Example usage:
    python sft_trainer.py \\
        --model_name_or_path meta-llama/Llama-3.1-8B \\
        --train_dataset_dir data/train \\
        --checkpoint_dir checkpoints/llama-sft \\
        --max_length 4096 \\
        --packing --assistant_only_loss \\
        --per_device_train_batch_size 4 \\
        --gradient_accumulation_steps 4 \\
        --learning_rate 3e-4 \\
        --num_train_epochs 3 \\
        --bf16 --gradient_checkpointing

"""

import argparse
import os

import torch
import trl
from datasets import DatasetDict
from utils import (
    get_logger,
    load_tokenizer,
    load_training_dataset,
    resolve_checkpoint_path,
    run_training,
    setup_distributed_state,
    split_dataset,
)


def main(args):
    logger = get_logger("SFT-Trainer")

    # Initialize the partial state for distributed training
    state, master_process = setup_distributed_state(logger)

    dataset = load_training_dataset(
        args.train_dataset_dir,
        args.dataset_type,
        args.num_proc,
        args.cache_dir,
        state,
        prebuilt_dataset_dir=args.prebuilt_dataset_dir,
        logger=logger,
    )

    if args.shuffle_dataset:
        dataset = dataset.shuffle(seed=args.seed)

    # Split the dataset into train and test sets if a test size is specified
    dataset = split_dataset(
        dataset,
        args.test_size,
        args.seed,
        args.checkpoint_dir,
        args.save_test_set,
        master_process,
        state,
    )

    # Load the tokenizer and validate it
    tokenizer = load_tokenizer(
        args.model_name_or_path, args.max_length, args.cache_dir, args.chat_template_path
    )

    # This filter drops samples that exceed max_length once the chat template is applied, which
    # would otherwise be truncated during packing.
    def filter_by_length(example):
        """Apply chat template and check if token count exceeds max_length.
        Also filters out samples where the last message is not from the assistant.
        """
        try:
            # Check if the last message is from the assistant
            # This will prevent issues with samples that do not have an assistant response
            if not example["messages"] or example["messages"][-1]["role"] != "assistant":
                return False

            token_ids = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
            )
            # Return True to keep the sample, False to filter it out
            return len(token_ids) <= args.max_length
        except Exception as e:
            logger.warning(f"Error processing sample: {e}")
            return False

    # Only main process runs the filter; others wait and load from cache
    if master_process:
        dataset = dataset.filter(
            filter_by_length,
            num_proc=args.num_proc,
            load_from_cache_file=True,
            desc=f"Filtering samples exceeding {args.max_length} tokens",
        )
        sizes = (
            ", ".join(f"{name}={split.num_rows:,}" for name, split in dataset.items())
            if isinstance(dataset, DatasetDict)
            else f"{dataset.num_rows:,}"
        )
        logger.info(f"Rows kept after the >{args.max_length}-token filter: {sizes}")

    # Wait for main process to finish filtering
    state.wait_for_everyone()

    # Non-main processes reload from cache
    if not master_process:
        dataset = dataset.filter(
            filter_by_length,
            num_proc=args.num_proc,
            load_from_cache_file=True,  # Will load from cache created by main process
            desc=f"Filtering samples exceeding {args.max_length} tokens",
        )

    if isinstance(dataset, DatasetDict):
        train_dataset, eval_dataset = dataset["train"], dataset.get("test")
    else:
        train_dataset, eval_dataset = dataset, None
    has_eval = eval_dataset is not None

    # Get the job ID from the environment variable or set it to "local" if not available
    jobid = os.getenv("SLURM_JOB_ID", "local")

    # Set the `WANDB_PROJECT` to args.wandb_project
    os.environ["WANDB_PROJECT"] = args.wandb_project

    # transformers >= 5.x replaced `TrainingArguments.warmup_ratio` with
    # `warmup_steps`.
    if args.warmup_ratio is not None:
        if args.warmup_steps:
            raise SystemExit(
                "--warmup_ratio and --warmup_steps are mutually exclusive. "
                "--warmup_ratio is deprecated; pass --warmup_steps instead."
            )
        args.warmup_steps = args.warmup_ratio
        if master_process:
            logger.warning(
                f"--warmup_ratio={args.warmup_ratio} is deprecated (transformers >= 5.x "
                f"removed it). Forwarding as --warmup_steps={args.warmup_ratio}: a float "
                f"in [0, 1) means the same ratio of total steps."
            )

    # See https://huggingface.co/docs/trl/main/en/sft_trainer#trl.SFTConfig
    # See https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
    training_args = trl.SFTConfig(
        model_init_kwargs={
            "cache_dir": args.cache_dir,
            "attn_implementation": args.attn_implementation,
            "dtype": torch.bfloat16 if args.bf16 else torch.float32,
            "trust_remote_code": True,
            "device_map": {"": state.process_index},
            "use_cache": not args.gradient_checkpointing,  # Disable cache if using gradient checkpointing
        },
        output_dir=args.checkpoint_dir,
        max_length=args.max_length,
        assistant_only_loss=args.assistant_only_loss,
        eos_token=tokenizer.eos_token,
        dataset_num_proc=args.num_proc,
        shuffle_dataset=args.shuffle_dataset,
        use_liger_kernel=args.use_liger_kernel,
        activation_offloading=args.activation_offloading,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}
        if torch.cuda.device_count() > 1 and args.gradient_checkpointing
        else None,
        packing=args.packing,  # Enable packing to optimize training (see https://huggingface.co/docs/trl/main/en/sft_trainer#packing-dataset)
        packing_strategy="bfd",  # Best-Fit Decreasing packing strategy (good default)
        seed=args.seed,
        eval_strategy="steps" if has_eval else "no",
        save_strategy="steps",
        eval_steps=args.eval_steps if has_eval else None,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=-1 if args.max_steps is None else args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size if has_eval else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters
        if torch.cuda.device_count() > 1
        else None,
        bf16=args.bf16,
        tf32=args.tf32,
        hub_token=args.hub_token,
        hub_model_id=args.hub_model_id,
        push_to_hub=bool(args.hub_token is not None and args.hub_model_id is not None),
        report_to=args.report_to,
        pad_to_multiple_of=args.pad_to_multiple_of,
        include_num_input_tokens_seen=True,
        hub_private_repo=True,  # If you want to push to a private repo
        run_name=f"{args.model_name_or_path.split('/')[-1]}-jobid-{jobid}-bs-{args.per_device_train_batch_size}-acumulation-{args.gradient_accumulation_steps}-ngpu-{torch.cuda.device_count()}-epochs-{args.num_train_epochs}",
    )

    # See https://huggingface.co/docs/trl/main/en/sft_trainer#trl.SFTTrainer
    trainer = trl.SFTTrainer(
        model=args.model_name_or_path,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Make sure every process is synced before training
    state.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    checkpoint_path = None
    if args.resume_from_checkpoint:
        checkpoint_path = resolve_checkpoint_path(
            args.resume_from_checkpoint, master_process, logger
        )

    # Start training, save final model (or 'last' on error)
    run_training(trainer, checkpoint_path, args.checkpoint_dir, master_process, logger)
    # Done!


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core dataset/model args
    parser.add_argument(
        "--dataset_type",
        choices=["jsonl", "parquet"],
        default="parquet",
        help="Type of the dataset files. Can be either 'jsonl' or 'parquet'.",
    )
    parser.add_argument(
        "--train_dataset_dir",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Path(s) to the training dataset directory or file. Can be a single "
            "directory/file or a list of directories/files. Required unless "
            "--prebuilt_dataset_dir is set."
        ),
    )
    parser.add_argument(
        "--prebuilt_dataset_dir",
        type=str,
        default=None,
        help=(
            "Path to a dataset already materialised on disk with Dataset.save_to_disk, laid "
            "out as <root>/train and <root>/validation. The Arrow files are memory-mapped, so "
            "no `datasets` build runs on the compute nodes and no per-rank worker pool is "
            "forked. When set, --train_dataset_dir is ignored and --test_size must not be "
            "used: the prebuilt dataset already defines both splits."
        ),
    )
    parser.add_argument(
        "--shuffle_dataset",
        action="store_true",
        help="If set, the dataset will be shuffled before training.",
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument(
        "--save_test_set",
        action="store_true",
        help="If set, the test set will be saved to a file in the checkpoint directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--chat_template_path",
        type=str,
        default=None,
        help="Path to the chat template file to use for the training.",
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action="store_true",
        help="Set the `find_unused_parameters` flag in DDP. Useful when some model parameters are not used during the forward pass.",
    )
    # Tokenization / packing / padding
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum sequence length for tokenization / model.",
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        help="If set, the dataset will be packed to optimize training. This is useful for large datasets.",
    )
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=32,
        help="Pad sequences to a multiple of this value.",
    )
    # Loss configuration
    # The `assistant_only_loss` requires that the chat template supports returning the assistant tokens mask via the {% generation %} keyword.
    parser.add_argument(
        "--assistant_only_loss",
        action="store_true",
        help="If set, the loss will only be computed on the assistant's responses, ignoring the user inputs.",
    )
    # Training and optimizer
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help=(
            "Maximum number of `checkpoint-*` directories kept in --checkpoint_dir; the "
            "oldest are deleted as new ones are written (the best one is always kept). "
            "None (default) keeps every checkpoint."
        ),
    )
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="Type of learning rate scheduler to use. Options: 'linear', 'cosine', and all the other types listed here: https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.SchedulerType",
    )
    parser.add_argument(
        "--warmup_steps",
        type=float,
        default=0.0,
        help=(
            "Warmup length: an int >= 1 is an exact number of steps, a float in [0, 1) is a "
            "RATIO of total steps. Replaces the removed --warmup_ratio."
        ),
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=None,
        help=(
            "DEPRECATED: transformers >= 5.x removed `warmup_ratio` from TrainingArguments. "
            "Kept so existing job scripts still parse; the value is forwarded to "
            "--warmup_steps unchanged, which means the same ratio."
        ),
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If set, overrides num_train_epochs.",
    )
    # Precision / performance
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 precision for training. Requires a GPU that supports bfloat16 (e.g., A100)",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Use TensorFloat-32 precision for training. Requires a GPU that supports TF32 (e.g., A100)",
    )
    parser.add_argument(
        "--activation_offloading",
        action="store_true",
        help="Use activation offloading to CPU to save GPU memory. This will slow down training but reduce memory usage.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory. This will slow down training but reduce memory usage.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        help=(
            "Attention implementation to use: 'eager', 'sdpa', 'flash_attention_2', "
            "'flash_attention_3' or 'flash_attention_4'. Which flash variant works depends "
            "on the installed package - flash-attn-4 only provides 'flash_attention_4' "
            "(it has no v2 API) and wants --bf16 so the dtype is explicit."
        ),
    )
    # Data loader / batch sizes
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # Hub / reporting
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument(
        "--report_to",
        type=str,
        nargs="+",
        default="none",
        help=(
            "Integrations to report results and logs to (default 'none'). Common values: "
            "'tensorboard', 'wandb', 'codecarbon', 'trackio'. See "
            "https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.report_to"
        ),
    )
    parser.add_argument("--wandb_project", type=str, default="Polyglot")
    # Experimental / other
    parser.add_argument(
        "--use_liger_kernel",
        action="store_true",
        help="Use the Liger kernel for training. This is an experimental feature that may improve performance on some GPUs.",
    )

    args = parser.parse_args()

    main(args)
