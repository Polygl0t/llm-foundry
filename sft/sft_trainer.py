"""
Supervised Fine-Tuning (SFT) Trainer for Large Language Models

This script fine-tunes LLMs using the Hugging Face Transformers and TRL libraries.

Expected Dataset Format:
{
    "messages": [
        {"role": "user", "content": "User message here."},
        {"role": "assistant", "content": "Assistant response here."},
        ...
    ]
}

If the dataset is already tokenized, it should contain:
{
    "input_ids": [...], # Required (list of token IDs)
    "seq_lengths": [...], # Required (list of sequence lengths)
    "assistant_tokens_mask": [...]  # Optional, required if assistant_only_loss is used
}

Example usage:
    python sft_trainer.py \
        --model_name_or_path meta-llama/Llama-3.1-8B \
        --train_dataset_dir data/train \
        --checkpoint_dir checkpoints/llama-sft \
        --max_length 4096 \
        --packing --assistant_only_loss \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 3e-4 \
        --num_train_epochs 3 \
        --bf16 --gradient_checkpointing
"""
import transformers
import accelerate
import datasets
import argparse
import torch
import glob
import trl
import os

def main(args):

    # Initialize the partial state for distributed training
    state = accelerate.PartialState()
    # print the state of every process
    master_process = int(state.process_index) == 0
    if master_process:
        print(f"{state}")

    # Load our training dataset.
    # We expect the dataset to be in a specific format: jsonl or parquet.
    assert args.dataset_type in ["jsonl", "parquet"], f"Dataset type must be either 'jsonl' or 'parquet', got {args.dataset_type}."

    train_dataset_files = []
    train_dirs = args.train_dataset_dir
    if isinstance(train_dirs, str):
        train_dirs = [train_dirs]

    # Below, we loop over all training directories and collect the dataset files that
    # have the correct file extension.
    for train_dir in train_dirs:
        if os.path.isfile(train_dir) and train_dir.endswith(f".{args.dataset_type}"):
            train_dataset_files.append(train_dir)
        elif os.path.isdir(train_dir):
            train_dataset_files += glob.glob(f"{train_dir}/*.{args.dataset_type}")

    # Ensure all processes have the same file list (synchronize before loading)
    train_dataset_files = sorted(train_dataset_files)
    state.wait_for_everyone()

    # Load the datasets from disk
    # [datasets.load_dataset](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset)
    dataset = datasets.load_dataset(
        "json" if args.dataset_type == "jsonl" else args.dataset_type,
        data_files=train_dataset_files,
        split='train',
        num_proc=min(len(train_dataset_files), args.num_proc),
        cache_dir=args.cache_dir,
    )
    # Set a flag indicating whether the dataset is already processed (tokenized)
    is_processed = "input_ids" in dataset.column_names

    if args.shuffle_dataset:
        dataset = dataset.shuffle(seed=args.seed)

    # Split the dataset into train and test sets if a test size is specified
    if args.test_size is not None:
        dataset = dataset.train_test_split(
            test_size=args.test_size,
            seed=args.seed,
        )
        if master_process:
        # Save the test set to a JSONL file if requested
            if args.save_test_set:
                # Skip if the file already exists
                if os.path.exists(os.path.join(args.checkpoint_dir, "test_set.jsonl")):
                    pass
                else:
                    test_file = os.path.join(args.checkpoint_dir, "test_set.jsonl")
                    dataset["test"].to_json(test_file, orient="records", lines=True)
        # Wait for main process to finish saving.
        state.wait_for_everyone()

    # We use the AutoTokenizer to load the tokenizer.
    # [AutoTokenizer](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer)
    # Make sure to set the `model_max_length` in a way that it does not exceed the model's max position embeddings.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        cache_dir=args.cache_dir,
        use_fast=True,
        trust_remote_code=True,
    )

    # We use the AutoTokenizer to load the tokenizer.
    # [AutoTokenizer](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer)
    # Make sure to set the `model_max_length` in a way that it does not exceed the model's max position embeddings.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        cache_dir=args.cache_dir,
        use_fast=True,
        trust_remote_code=True,
    )

    # Load a Jinja template for the chat model
    # Learn more about chat templates [here](https://huggingface.co/docs/transformers/main/chat_templating)
    if tokenizer.chat_template is None:
        assert args.chat_template_path is not None, "Tokenizer does not have a chat template. Please provide a chat template path."
        with open(args.chat_template_path, "r") as f:
            tokenizer.chat_template = f.read()
    
    # Check if the tokenizer has a pad token and if the pad token is different from the eos token
    assert tokenizer.pad_token is not None, "The tokenizer does not have a pad token. Please set a pad token before training."
    assert tokenizer.pad_token != tokenizer.eos_token, "The tokenizer's pad token is the same as the eos token. Please set a different pad token before training."
    
    # Filter out samples that exceed max_length after applying chat template
    # This prevents truncated samples during packing
    # We only apply this filtering if the dataset is not already processed
    if not is_processed:
        def filter_by_length(example):
            """Apply chat template and check if token count exceeds max_length.
            Also filters out samples where the last message is not from the assistant.
            """
            try:
                # Check if the last message is from the assistant
                # This will prevent issues with samples that do not have an assistant response
                if not example["messages"] or example["messages"][-1]["role"] != "assistant":
                    return False
                
                # Apply chat template and tokenize in one step
                token_ids = tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=True,
                    add_generation_prompt=False
                )
                # Return True to keep the sample, False to filter it out
                return len(token_ids) <= args.max_length
            except Exception as e:
                print(f"Error processing sample: {e}")
                return False


        # Only main process runs the filter; others wait and load from cache
        if master_process:
            dataset = dataset.filter(
                filter_by_length,
                num_proc=args.num_proc,
                load_from_cache_file=True,
                desc=f"Filtering samples exceeding {args.max_length} tokens",
            )
        
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
    
    # Get the job ID from the environment variable or set it to "local" if not available
    jobid = os.getenv("SLURM_JOB_ID", "local")

    # Set the `WANDB_PROJECT` to args.wandb_project
    os.environ["WANDB_PROJECT"] = args.wandb_project

    # [SFTConfig](https://huggingface.co/docs/trl/main/en/sft_trainer#trl.SFTConfig)
    # [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
    training_args = trl.SFTConfig(
        model_init_kwargs={
            "cache_dir": args.cache_dir,
            "attn_implementation": args.attn_implementation,
            "dtype": torch.bfloat16 if args.bf16 else torch.float32,
            "trust_remote_code": True,
            "device_map":{'':state.process_index},
            "use_cache": False if args.gradient_checkpointing else True,  # Disable cache if using gradient checkpointing
        },
        output_dir=args.checkpoint_dir,
        max_length=args.max_length,
        assistant_only_loss=args.assistant_only_loss,
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token,
        dataset_num_proc=args.num_proc,
        shuffle_dataset=args.shuffle_dataset,
        neftune_noise_alpha=args.neftune_noise_alpha,
        use_liger_kernel=args.use_liger_kernel,
        activation_offloading=args.activation_offloading,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if torch.cuda.device_count() > 1 and args.gradient_checkpointing else None,
        packing=args.packing, # Enable packing to optimize training (learn more [here](https://huggingface.co/docs/trl/main/en/sft_trainer#packing-dataset))
        packing_strategy="bfd", # Best-Fit Decreasing packing strategy (good default)
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
        pad_to_multiple_of=args.pad_to_multiple_of,
        include_tokens_per_second=True, # Include tokens per second in the logs
        hub_private_repo=True, # If you want to push to a private repo
        run_name=f"{args.model_name_or_path.split('/')[-1]}-jobid-{jobid}-bs-{args.per_device_train_batch_size}-acumulation-{args.gradient_accumulation_steps}-ngpu-{torch.cuda.device_count()}-epochs-{args.num_train_epochs}",
    )

    # [SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer#trl.SFTTrainer)
    trainer = trl.SFTTrainer(
        model=args.model_name_or_path,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"] if "train" in dataset else dataset,
        eval_dataset=dataset["test"] if "test" in dataset else None,
        
    )

    # Make sure every process is synced before training
    state.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Start the training
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except Exception as e:
        save_path = os.path.join(args.checkpoint_dir, "last")
        trainer.save_model(save_path)
        print(f"Training failed with error: {e}")
        print(f"Model saved to 'last' checkpoint at {save_path}")

    # Save the final model
    trainer.save_model(os.path.join(args.checkpoint_dir, "final"))
    # Done!
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Core dataset/model args
    parser.add_argument("--dataset_type", choices=["jsonl", "parquet"], default="parquet", help="Type of the dataset files. Can be either 'jsonl' or 'parquet'.")
    parser.add_argument("--train_dataset_dir", type=str, nargs="+", required=True, help="Path(s) to the training dataset directory or file. Can be a single directory/file or a list of directories/files.")
    parser.add_argument("--shuffle_dataset", action="store_true", help="If set, the dataset will be shuffled before training.")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--save_test_set", action="store_true", help="If set, the test set will be saved to a file in the checkpoint directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--chat_template_path", type=str, default=None, help="Path to the chat template file to use for the training.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--ddp_find_unused_parameters", action="store_true", help="Set the `find_unused_parameters` flag in DDP. Useful when some model parameters are not used during the forward pass.")
    # Tokenization / packing / padding
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length for tokenization / model.")
    parser.add_argument("--packing", action="store_true", help="If set, the dataset will be packed to optimize training. This is useful for large datasets.")
    parser.add_argument("--pad_to_multiple_of", type=int, default=32, help="Pad sequences to a multiple of this value.")
    # Loss configuration
    # The `assistant_only_loss` requires that the chat template supports returning the assistant tokens mask via the {% generation %} keyword.
    parser.add_argument("--assistant_only_loss", action="store_true", help="If set, the loss will only be computed on the assistant's responses, ignoring the user inputs.")
    parser.add_argument("--system_message", type=str, default=None, help="System message to prepend to the user messages. If not set, no system message will be used, and we will default to the chat template's default behavior.")
    parser.add_argument("--neftune_noise_alpha", type=int, default=5, help="Alpha value for the Neftune noise.")
    # Training and optimizer
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Type of learning rate scheduler to use. Options: 'linear', 'cosine', and all the other types listed [here](https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.SchedulerType).")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None, help="Total number of training steps to perform. If set, overrides num_train_epochs.")
    # Precision / performance
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision for training. Requires a GPU that supports bfloat16 (e.g., A100)")
    parser.add_argument("--tf32", action="store_true", help="Use TensorFloat-32 precision for training. Requires a GPU that supports TF32 (e.g., A100)")
    parser.add_argument("--activation_offloading", action="store_true", help="Use activation offloading to CPU to save GPU memory. This will slow down training but reduce memory usage.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory. This will slow down training but reduce memory usage.")
    parser.add_argument("--attn_implementation", type=str, default="eager", help="Attention implementation to use. Options: 'eager', 'sdpa', and 'flash_attention_2'.")
    # Data loader / batch sizes
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # Hub / reporting
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--report_to", type=str, nargs="+", default=["wandb", "codecarbon"])
    parser.add_argument("--wandb_project", type=str, default="Polyglot")
    # Experimental / other
    parser.add_argument("--use_liger_kernel", action="store_true", help="Use the Liger kernel for training. This is an experimental feature that may improve performance on some GPUs.")

    args = parser.parse_args()

    main(args)