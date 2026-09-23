"""
Direct Preference Optimization (DPO) Training Pipeline

Implements DPO training for aligning language models with human preferences using the
TRL (Transformer Reinforcement Learning) library.

Expected Dataset Format:
{
    "prompt": "User question or instruction",
    "chosen": [{"role": "assistant", "content": "Good response"}],
    "rejected": [{"role": "assistant", "content": "Bad response"}]
}

Or with conversation history:
{
    "chosen": [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Good response"}
    ],
    "rejected": [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Bad response"}
    ]
}

Usage:
    # Basic DPO training
    python dpo_trainer.py \\
        --train_dataset_dir data/preferences.jsonl \\
        --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \\
        --checkpoint_dir checkpoints/ \\
        --loss_type sigmoid --beta 0.1 \\
        --per_device_train_batch_size 4 \\
        --num_train_epochs 1

"""

import argparse
import os

import torch
import transformers
import trl
from datasets import DatasetDict
from utils import (
    get_logger,
    has_column,
    load_tokenizer,
    load_training_dataset,
    resolve_checkpoint_path,
    run_training,
    setup_distributed_state,
    split_dataset,
)

LOSS_DESCRIPTIONS = {
    "sigmoid": "sigmoid loss from the original DPO paper.",
    "hinge": "hinge loss on the normalized likelihood from the SLiC paper.",
    "ipo": "IPO loss from the IPO paper.",
    "exo_pair": "pairwise EXO loss from the EXO paper.",
    "nca_pair": "pairwise NCA loss from the NCA paper.",
    "robust": "unbiased estimate of the DPO loss that is robust to preference noise from the Robust DPO paper.",
    "bco_pair": "pairwise BCO loss from the BCO paper.",
    "sppo_hard": "SPPO loss with hard label from the SPPO paper.",
    "aot": "AOT loss for paired datasets from the AOT paper.",
    "aot_pair": "AOT loss for unpaired datasets from the AOT paper.",
    "discopop": "DiscoPOP (a.k.a Log-Ratio Modulated Loss, LRML) loss from the DiscoPOP paper.",
    "apo_zero": "APO-zero loss from the APO paper.",
    "apo_down": "APO-down loss from the APO paper.",
    "sft": "Negative log-likelihood loss (standard supervised fine-tuning loss).",
}


def main(args):
    logger = get_logger("DPO-Trainer")

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

    # Convert the dataset so that the prompt is explicitly defined.
    # Why? -> https://huggingface.co/docs/trl/main/en/dpo_trainer#expected-dataset-type
    if not has_column(dataset, "prompt"):
        if master_process:
            dataset = dataset.map(
                trl.extract_prompt,
                num_proc=args.num_proc,
                desc="Extracting prompt from the data",
                load_from_cache_file=False,
            )

        state.wait_for_everyone()
        if not master_process:
            dataset = dataset.map(
                trl.extract_prompt,
                num_proc=args.num_proc,
                desc="Extracting prompt from the data",
                load_from_cache_file=True,
            )

    # Split into train/test sets if requested
    dataset = split_dataset(
        dataset,
        args.test_size,
        args.seed,
        args.checkpoint_dir,
        args.save_test_set,
        master_process,
        state,
    )

    if isinstance(dataset, DatasetDict):
        train_dataset, eval_dataset = dataset["train"], dataset.get("test")
    else:
        train_dataset, eval_dataset = dataset, None
    has_eval = eval_dataset is not None

    # Load the tokenizer and validate it
    tokenizer = load_tokenizer(
        args.model_name_or_path, args.max_length, args.cache_dir, args.chat_template_path
    )

    # Initialize the model and reference model explicitly
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model_kwargs = {
        "cache_dir": args.cache_dir,
        "attn_implementation": args.attn_implementation,
        "dtype": dtype,
        "trust_remote_code": True,
        "device_map": {"": state.process_index},
        "use_cache": not args.gradient_checkpointing,
    }

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, **model_kwargs
    )

    # Load reference model if specified, otherwise set to None
    if args.ref_model_name_or_path is not None:
        ref_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.ref_model_name_or_path, **model_kwargs
        )
    else:
        ref_model = None

    # Check if the `loss_type ` is valid.
    if isinstance(args.loss_type, str):
        assert args.loss_type in LOSS_DESCRIPTIONS, (
            f"Invalid loss type '{args.loss_type}'. Supported loss types are: {', '.join(LOSS_DESCRIPTIONS.keys())}."
        )
    elif isinstance(args.loss_type, list):
        for loss_type in args.loss_type:
            assert loss_type in LOSS_DESCRIPTIONS, (
                f"Invalid loss type '{loss_type}'. Supported loss types are: {', '.join(LOSS_DESCRIPTIONS.keys())}."
            )
    else:
        raise ValueError(
            f"Invalid loss type '{args.loss_type}'. Supported loss types are: {', '.join(LOSS_DESCRIPTIONS.keys())}."
        )

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

    # TRL >= 0.29 removed the prompt/completion split from `DPOConfig`, so the
    # prompt can no longer be capped separately.
    if args.truncation_mode == "keep_end" and master_process:
        logger.warning(
            "truncation_mode='keep_end' is deprecated by TRL (removal in TRL 2.0.0). "
            "It keeps the completion and drops the start of the prompt; the "
            "forward-compatible 'keep_start' does the opposite. Choose deliberately."
        )

    # See https://huggingface.co/docs/trl/main/en/dpo_trainer#trl.DPOConfig
    # See https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
    training_args = trl.DPOConfig(
        dataset_num_proc=args.num_proc,
        max_length=args.max_length,
        truncation_mode=args.truncation_mode,
        padding_free=args.padding_free,
        precompute_ref_log_probs=args.precompute_ref_log_probs,
        precompute_ref_batch_size=args.precompute_ref_batch_size
        if args.precompute_ref_batch_size
        else args.per_device_train_batch_size * 2,
        loss_type=args.loss_type,
        beta=args.beta,
        loss_weights=args.loss_weights,
        sync_ref_model=args.sync_ref_model if not args.precompute_ref_log_probs else False,
        ref_model_sync_steps=args.ref_model_sync_steps
        if not args.precompute_ref_log_probs
        else None,
        output_dir=args.checkpoint_dir,
        use_liger_kernel=args.use_liger_kernel,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}
        if torch.cuda.device_count() > 1 and args.gradient_checkpointing
        else None,
        seed=args.seed,
        eval_strategy="steps" if has_eval else "no",
        save_strategy="steps",
        eval_steps=args.eval_steps if has_eval else None,
        save_steps=args.save_steps,
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
        include_num_input_tokens_seen=True,
        hub_private_repo=True,
        run_name=f"{args.model_name_or_path.split('/')[-1]}-jobid-{jobid}-bs-{args.per_device_train_batch_size}-acumulation-{args.gradient_accumulation_steps}-ngpu-{torch.cuda.device_count()}-epochs-{args.num_train_epochs}",
    )

    # See https://huggingface.co/docs/trl/main/en/dpo_trainer#trl.DPOTrainer
    trainer = trl.DPOTrainer(
        model=model,
        ref_model=ref_model,
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
        help="If set, shuffle the dataset files before loading.",
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
        "--ref_model_name_or_path",
        type=str,
        default=None,
        help="Optional reference model to use. If not provided, the trainer may initialize or use the model_name_or_path as reference.",
    )
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
    # Tokenization / truncation / padding
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum sequence length for tokenization / model.",
    )
    parser.add_argument(
        "--truncation_mode",
        type=str,
        choices=["keep_start", "keep_end"],
        default="keep_start",
        help=(
            "Which end of an over-long sequence survives truncation. TRL's only "
            "forward-compatible value is 'keep_start' (its default); 'keep_end' is "
            "deprecated by TRL and will be removed in TRL 2.0. NOTE the behaviour: with "
            "'keep_start' a long prompt can push the completion out of the window, "
            "whereas 'keep_end' keeps the completion and drops the prompt's beginning."
        ),
    )
    parser.add_argument(
        "--padding_free",
        action="store_true",
        help="If set, use padding-free training to reduce memory overhead.",
    )
    # Reference precompute options
    parser.add_argument(
        "--precompute_ref_log_probs",
        action="store_true",
        help="Precompute reference log probabilities for efficiency.",
    )
    parser.add_argument(
        "--precompute_ref_batch_size",
        type=int,
        default=None,
        help="Batch size for precomputing reference log probabilities. If not set, defaults are used.",
    )
    # Loss configuration
    parser.add_argument(
        "--loss_type",
        type=str,
        nargs="+",
        default=["apo_zero"],
        help="Loss type(s) to use. Can pass multiple values. See LOSS_DESCRIPTIONS in the script.",
    )
    parser.add_argument(
        "--loss_weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional weights for each loss when providing multiple loss types.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Beta parameter controlling deviation from reference model for certain losses.",
    )
    # Reference model syncing
    parser.add_argument(
        "--sync_ref_model",
        action="store_true",
        help="Whether to synchronize the reference model with the active model periodically. If `precompute_ref_log_probs` is set, this will be set to False automatically.",
    )
    parser.add_argument(
        "--ref_model_sync_steps",
        type=int,
        default=100,
        help="Number of steps between synchronizing the reference model (if sync_ref_model is set).",
    )
    # Training and optimizer
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="Type of learning rate scheduler to use.",
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
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision for training.")
    parser.add_argument(
        "--tf32", action="store_true", help="Use TensorFloat-32 precision for training."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory.",
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
        help="Use the Liger kernel for training (experimental).",
    )

    args = parser.parse_args()

    main(args)
