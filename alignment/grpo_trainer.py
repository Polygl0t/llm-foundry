"""
Group Relative Policy Optimization (GRPO) Trainer

This script trains LLMs with TRL's GRPOTrainer using verifier-based rewards.

Expected Dataset Format:
{
    "prompt": "User question or instruction",
    "verifier_id_list": ["math:answer_check"],
    "kwargs": ["{\"expected_answer\": \"42\", \"relaxed\": true}"]
}

The reward is computed by alignment/gym/verifier.py. Each verifier returns a
boolean, and the final reward is the fraction of verifiers passed, always in
the [0, 1] range.

Example usage:
    python grpo_trainer.py \\
        --model_name_or_path Qwen/Qwen3-0.6B-Instruct \\
        --train_dataset_dir path/to/dataset.jsonl \\
        --dataset_type jsonl \\
        --checkpoint_dir checkpoints/qwen-grpo \\
        --max_prompt_length 2048 \\
        --max_completion_length 1024 \\
        --num_generations 8 \\
        --per_device_train_batch_size 4 \\
        --gradient_accumulation_steps 4 \\
        --learning_rate 1e-6 \\
        --bf16
"""

import argparse
import importlib.util
import os

import torch
import trl
from datasets import DatasetDict
from gym.verifier import Verifier
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

GRPO_LOSS_TYPES = ("grpo", "bnpo", "dr_grpo", "dapo", "sapo")
SCALE_REWARD_VALUES = {
    "group": "group",
    "true": "group",
    "batch": "batch",
    "none": "none",
    "false": "none",
}


def _completion_to_text(completion):
    """Return generated text from either standard or conversational GRPO output."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        message = completion[0]
        if isinstance(message, dict):
            return message.get("content", "")
    return str(completion)


def _normalize_verifier_ids(verifier_ids):
    """Normalize verifier_id_list to a list of strings."""
    if verifier_ids is None:
        return []
    if isinstance(verifier_ids, str):
        return [verifier_ids]
    return list(verifier_ids)


def _normalize_verifier_kwargs(verifier_kwargs):
    """Normalize verifier_kwargs to a list of dictionaries."""
    if verifier_kwargs is None:
        return []
    if isinstance(verifier_kwargs, str | dict):
        return [verifier_kwargs]
    return list(verifier_kwargs)


def verifier_reward_func(
    completions,
    verifier_id_list,
    kwargs,
    verifier_enable_thinking=False,
    verifier_strict=True,
    log_extra=None,
    log_metric=None,
    **unused_kwargs,
):
    """Reward completions with the fraction of verifier checks passed."""
    rewards = []
    passed_counts = []
    total_counts = []

    for completion, verifier_ids, verifier_kwargs in zip(
        completions, verifier_id_list, kwargs, strict=False
    ):
        completion_text = _completion_to_text(completion)
        verifier_ids = _normalize_verifier_ids(verifier_ids)
        verifier_kwargs = _normalize_verifier_kwargs(verifier_kwargs)

        try:
            # See alignment/gym/verifier.py for details on the Verifier class and
            # how it processes the completion and kwargs.
            verifier = Verifier(
                verifier_id_list=verifier_ids,
                kwargs=verifier_kwargs,
                completion=completion_text,
                enable_thinking=verifier_enable_thinking,
                strict=verifier_strict,
            )
            results = verifier.verify()
            passed = sum(bool(result) for result in results)
            total = len(results)
            reward = passed / total if total else 0.0
        except Exception:
            passed = 0
            total = len(verifier_ids)
            reward = 0.0

        reward = max(0.0, min(1.0, float(reward)))
        rewards.append(reward)
        passed_counts.append(passed)
        total_counts.append(total)

    if log_extra:
        log_extra("verifier_reward", rewards)
        log_extra("verifier_passed", passed_counts)
        log_extra("verifier_total", total_counts)

    if log_metric and rewards:
        log_metric("verifier/mean_reward", sum(rewards) / len(rewards))

    return rewards


def validate_dataset_columns(dataset):
    """Ensure the dataset contains the required columns for GRPO training.

    `has_column` accepts both a bare `Dataset` and the `DatasetDict` a prebuilt
    dataset comes back as.
    """
    required_columns = {"prompt", "verifier_id_list", "kwargs"}
    missing_columns = {column for column in required_columns if not has_column(dataset, column)}
    if missing_columns:
        raise ValueError(
            "GRPO verifier datasets must contain columns: "
            f"{', '.join(sorted(required_columns))}. Missing: {', '.join(sorted(missing_columns))}."
        )


def parse_scale_rewards(value):
    """Parse the scale_rewards argument and validate its value."""
    value = value.lower()
    if value not in SCALE_REWARD_VALUES:
        raise argparse.ArgumentTypeError("scale_rewards must be one of: group, batch, none")
    return SCALE_REWARD_VALUES[value]


def main(args):
    logger = get_logger("GRPO-Trainer")

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
    validate_dataset_columns(dataset)

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

    if isinstance(dataset, DatasetDict):
        train_dataset, eval_dataset = dataset["train"], dataset.get("test")
    else:
        train_dataset, eval_dataset = dataset, None
    has_eval = eval_dataset is not None

    tokenizer = load_tokenizer(
        args.model_name_or_path,
        args.max_prompt_length + args.max_completion_length,
        args.cache_dir,
        args.chat_template_path,
    )

    jobid = os.getenv("SLURM_JOB_ID", "local")
    os.environ["WANDB_PROJECT"] = args.wandb_project

    model_dtype = torch.bfloat16 if args.bf16 else torch.float32

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

    # Without vLLM, GRPO generates with `model.generate()`. Refuse
    # --use_vllm up front rather than letting TRL fail after the model is loaded.
    if args.use_vllm and importlib.util.find_spec("vllm") is None:
        raise SystemExit(
            "--use_vllm was requested but vLLM is not installed in this environment. "
            "Rebuild the venv with WITH_VLLM=1 WITH_FLASH_ATTN=0 (vLLM and flash-attn-4 "
            "are mutually exclusive -- they pin different `apache-tvm-ffi` versions), or "
            "drop --use_vllm: GRPO falls back to `model.generate()` and runs without it."
        )

    model_init_kwargs = {
        "cache_dir": args.cache_dir,
        "attn_implementation": args.attn_implementation,
        "dtype": model_dtype,
        "trust_remote_code": True,
        "use_cache": not args.gradient_checkpointing,
    }
    if not args.use_vllm or args.vllm_mode != "server":
        model_init_kwargs["device_map"] = {"": state.process_index}

    # See https://huggingface.co/docs/trl/en/grpo_trainer#trl.GRPOConfig
    # See https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
    training_args = trl.GRPOConfig(
        model_init_kwargs=model_init_kwargs,
        output_dir=args.checkpoint_dir,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        num_iterations=args.num_iterations,
        beta=args.beta,
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards,
        mask_truncated_completions=args.mask_truncated_completions,
        reward_weights=[1.0],
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode if args.use_vllm else "colocate",
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_importance_sampling_correction=args.vllm_importance_sampling_correction,
        vllm_importance_sampling_clip_max=args.vllm_importance_sampling_clip_max,
        vllm_importance_sampling_mode=args.vllm_importance_sampling_mode,
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
        pad_to_multiple_of=8,
        hub_token=args.hub_token,
        hub_model_id=args.hub_model_id,
        push_to_hub=bool(args.hub_token is not None and args.hub_model_id is not None),
        report_to=args.report_to,
        hub_private_repo=True,
        run_name=f"{args.model_name_or_path.split('/')[-1]}-jobid-{jobid}-bs-{args.per_device_train_batch_size}-accumulation-{args.gradient_accumulation_steps}-ngpu-{torch.cuda.device_count()}-epochs-{args.num_train_epochs}",
    )

    def reward_func(
        completions, verifier_id_list, kwargs, log_extra=None, log_metric=None, **unused_kwargs
    ):
        return verifier_reward_func(
            completions=completions,
            verifier_id_list=verifier_id_list,
            kwargs=kwargs,
            verifier_enable_thinking=args.verifier_enable_thinking,
            verifier_strict=args.verifier_strict,
            log_extra=log_extra,
            log_metric=log_metric,
            **unused_kwargs,
        )

    reward_func.__name__ = "verifier_reward_func"

    # See https://huggingface.co/docs/trl/en/grpo_trainer#trl.GRPOTrainer
    trainer = trl.GRPOTrainer(
        model=args.model_name_or_path,
        reward_funcs=reward_func,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    state.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    checkpoint_path = None
    if args.resume_from_checkpoint:
        checkpoint_path = resolve_checkpoint_path(
            args.resume_from_checkpoint, master_process, logger
        )

    run_training(trainer, checkpoint_path, args.checkpoint_dir, master_process, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

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
            "Path(s) to the training dataset directory or file. Required unless "
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
        help="If set, shuffle the dataset before training.",
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
        help="Path to the chat template file to use for training.",
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
        help="Set the `find_unused_parameters` flag in DDP.",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=2048,
        help="Maximum token length reserved when loading the tokenizer.",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=256,
        help="Maximum generated completion length.",
    )
    parser.add_argument(
        "--num_generations", type=int, default=8, help="Number of completions sampled per prompt."
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=1,
        help="Number of optimization iterations per generation batch.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="KL coefficient. TRL GRPO commonly defaults this to 0.0.",
    )
    parser.add_argument(
        "--loss_type", choices=GRPO_LOSS_TYPES, default="dapo", help="GRPO loss variant."
    )
    parser.add_argument(
        "--scale_rewards",
        type=parse_scale_rewards,
        default="group",
        help="Reward scaling mode: group, batch, or none.",
    )
    parser.add_argument(
        "--verifier_enable_thinking",
        action="store_true",
        help="Require completions to include a valid <think>...</think> block before verifier checks.",
    )
    parser.add_argument(
        "--verifier_strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use strict verifier checks. Pass --no-verifier_strict to allow soft checker tolerances where implemented.",
    )
    parser.add_argument(
        "--mask_truncated_completions",
        action="store_true",
        help="Mask completions that hit max_completion_length without EOS.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
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
        help="Attention implementation to use. Options: 'eager', 'sdpa', 'flash_attention_2', 'flash_attention_3', and 'flash_attention_4'.",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for GRPO rollouts.")
    parser.add_argument(
        "--vllm_mode",
        choices=["colocate", "server"],
        default="colocate",
        help="Run vLLM inside the trainer or through a separate server.",
    )
    parser.add_argument(
        "--vllm_server_host",
        type=str,
        default="127.0.0.1",
        help="vLLM server host when --vllm_mode server is used.",
    )
    parser.add_argument(
        "--vllm_server_port",
        type=int,
        default=8000,
        help="vLLM server port when --vllm_mode server is used.",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.3,
        help="GPU memory fraction for colocated vLLM.",
    )
    parser.add_argument(
        "--vllm_importance_sampling_correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Correct vLLM training-inference mismatch with importance sampling.",
    )
    parser.add_argument(
        "--vllm_importance_sampling_clip_max",
        type=float,
        default=3.0,
        help=(
            "Upper bound C_max on the vLLM importance-sampling ratio (only used with "
            "--use_vllm). Renamed from --vllm_importance_sampling_cap, which TRL 1.13 "
            "deprecates and removes in 2.0. The default is unchanged."
        ),
    )
    parser.add_argument("--vllm_importance_sampling_mode", type=str, default="sequence_mask")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument(
        "--report_to",
        type=str,
        nargs="+",
        default="none",
        help="The list of integrations to report logs to.",
    )
    parser.add_argument("--wandb_project", type=str, default="Polyglot")
    parser.add_argument(
        "--use_liger_kernel",
        action="store_true",
        help="Use the Liger kernel for training (experimental).",
    )

    args = parser.parse_args()
    main(args)
