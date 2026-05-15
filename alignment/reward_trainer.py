"""
Reward Model Training Pipeline

Implements reward modeling for preference datasets using the TRL RewardTrainer.

Expected Dataset Format:
{
    "chosen": "Preferred response",
    "rejected": "Less preferred response"
}

Or with an explicit prompt:
{
    "prompt": "User question or instruction",
    "chosen": "Preferred response",
    "rejected": "Less preferred response"
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
    python reward_trainer.py \\
        --train_dataset_dir data/preferences.jsonl \\
        --model_name_or_path Qwen/Qwen3-0.6B \\
        --checkpoint_dir checkpoints/reward-model \\
        --per_device_train_batch_size 4 \\
        --num_train_epochs 1
"""
import argparse
import os

import torch
import trl

from utils import (
    get_logger,
    load_tokenizer,
    load_training_dataset,
    resolve_checkpoint_path,
    run_training,
    setup_distributed_state,
    split_dataset,
)


def uses_conversational_format(dataset):
    """Return True when the dataset stores prompt/chosen/rejected as messages."""
    sample = dataset["train"][0] if "train" in dataset else dataset[0]

    for field_name in ("prompt", "chosen", "rejected"):
        value = sample.get(field_name)
        if isinstance(value, list) and value:
            first_item = value[0]
            if isinstance(first_item, dict) and "role" in first_item and "content" in first_item:
                return True

    return False


def main(args):
    logger = get_logger("Reward-Trainer")

    state, master_process = setup_distributed_state(logger)

    dataset = load_training_dataset(
        args.train_dataset_dir,
        args.dataset_type,
        args.num_proc,
        args.cache_dir,
        state,
    )

    if args.shuffle_dataset:
        dataset = dataset.shuffle(seed=args.seed)

    dataset = split_dataset(
        dataset,
        args.test_size,
        args.seed,
        args.checkpoint_dir,
        args.save_test_set,
        master_process,
        state,
    )

    tokenizer = load_tokenizer(
        args.model_name_or_path,
        args.max_length,
        args.cache_dir,
        args.chat_template_path,
        allow_eos_pad_token=True,
    )

    if uses_conversational_format(dataset) and tokenizer.chat_template is None:
        raise ValueError(
            "Conversational reward datasets require a tokenizer chat template. "
            "Provide --chat_template_path or use a tokenizer with a built-in chat template."
        )

    jobid = os.getenv("SLURM_JOB_ID", "local")
    os.environ["WANDB_PROJECT"] = args.wandb_project

    model_dtype = torch.bfloat16 if args.bf16 else torch.float32
    training_args = trl.RewardConfig(
        output_dir=args.checkpoint_dir,
        max_length=args.max_length,
        dataset_num_proc=args.num_proc,
        center_rewards_coefficient=args.center_rewards_coefficient,
        model_init_kwargs={
            "cache_dir": args.cache_dir,
            "attn_implementation": args.attn_implementation,
            "dtype": model_dtype,
            "trust_remote_code": True,
            "device_map": {"": state.process_index},
            "use_cache": False if args.gradient_checkpointing else True,
        },
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if torch.cuda.device_count() > 1 and args.gradient_checkpointing else None,
        seed=args.seed,
        eval_strategy="steps" if "test" in dataset else "no",
        save_strategy="steps",
        eval_steps=args.eval_steps if "test" in dataset else None,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=-1 if args.max_steps is None else args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size if "test" in dataset else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters if torch.cuda.device_count() > 1 else None,
        bf16=args.bf16,
        tf32=args.tf32,
        hub_token=args.hub_token,
        hub_model_id=args.hub_model_id,
        push_to_hub=True if args.hub_token is not None and args.hub_model_id is not None else False,
        report_to=args.report_to,
        include_tokens_per_second=True,
        hub_private_repo=True,
        run_name=f"{args.model_name_or_path.split('/')[-1]}-jobid-{jobid}-bs-{args.per_device_train_batch_size}-acumulation-{args.gradient_accumulation_steps}-ngpu-{torch.cuda.device_count()}-epochs-{args.num_train_epochs}",
    )

    trainer = trl.RewardTrainer(
        model=args.model_name_or_path,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"] if "train" in dataset else dataset,
        eval_dataset=dataset["test"] if "test" in dataset else None,
    )

    state.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    checkpoint_path = None
    if args.resume_from_checkpoint:
        checkpoint_path = resolve_checkpoint_path(args.resume_from_checkpoint, master_process, logger)

    run_training(trainer, checkpoint_path, args.checkpoint_dir, master_process, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--dataset_type", choices=["jsonl", "parquet"], default="parquet", help="Type of the dataset files. Can be either 'jsonl' or 'parquet'.")
    parser.add_argument("--train_dataset_dir", type=str, nargs="+", required=True, help="Path(s) to the training dataset directory or file. Can be a single directory/file or a list of directories/files.")
    parser.add_argument("--shuffle_dataset", action="store_true", help="If set, shuffle the dataset files before loading.")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--save_test_set", action="store_true", help="If set, the test set will be saved to a file in the checkpoint directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--chat_template_path", type=str, default=None, help="Path to the chat template file to use for conversational datasets.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--ddp_find_unused_parameters", action="store_true", help="Set the `find_unused_parameters` flag in DDP. Useful when some model parameters are not used during the forward pass.")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length for tokenization / model.")
    parser.add_argument("--center_rewards_coefficient", type=float, default=1e-2, help="Auxiliary coefficient that encourages reward scores to stay centered around zero.")
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Type of learning rate scheduler to use.")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None, help="Total number of training steps to perform. If set, overrides num_train_epochs.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision for training.")
    parser.add_argument("--tf32", action="store_true", help="Use TensorFloat-32 precision for training.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory.")
    parser.add_argument("--attn_implementation", type=str, default="eager", help="Attention implementation to use. Options: 'eager', 'sdpa', and 'flash_attention_2'.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--report_to", type=str, nargs="+", default=None, help="The list of integrations to report the results and logs to. Supported platforms are 'tensorboard', 'wandb', 'comet_ml', 'mlflow', 'clearml', and more.")
    parser.add_argument("--wandb_project", type=str, default="Polyglot")

    args = parser.parse_args()
    main(args)