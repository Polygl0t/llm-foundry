"""
Training Script using Transformers Trainer API

This script fine-tunes a causal language model using the HuggingFace Transformers
Trainer API. It supports:

Workflow:
- Loading datasets in JSONL or Parquet format from one or more directories
- Custom learning rate schedulers: Cosine decay and Warmup-Stable-Decay (WSD)
- Distributed training with DDP or FSDP backends
- Memory optimizations: gradient checkpointing, bf16/tf32, Liger kernels
- Integration with Weights & Biases and CodeCarbon for logging

Usage:
    accelerate launch train_fsdp_hf.py \
        --model_name_or_path <model> \
        --train_dataset_dir <path> \
        --checkpoint_dir <output_path> \
        --bf16 --gradient_checkpointing
"""
import transformers
import accelerate
import datasets
import argparse
import torch
import math
import glob
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
    train_dataset = datasets.load_dataset(
        "json" if args.dataset_type == "jsonl" else args.dataset_type,
        data_files=train_dataset_files,
        split='train',
        num_proc=min(len(train_dataset_files), args.num_proc),
        cache_dir=args.cache_dir,
    )

    if args.shuffle_dataset:
        train_dataset = train_dataset.shuffle(seed=args.seed)

    if args.val_dataset_dir is not None:
        # Validation dataset is expected to always be in a single directory.
        # Validation files should be of the same type as the training files.
        val_dataset_files = glob.glob(f"{args.val_dataset_dir}/*.{args.dataset_type}")

        val_dataset = datasets.load_dataset(
            args.dataset_type,
            data_files=val_dataset_files,
            split='train',
            num_proc=min(len(val_dataset_files), args.num_proc),
            cache_dir=args.cache_dir,
        )
    else:
        val_dataset = None
    
    # Define the initialization kwargs for the model
    model_init_kwargs={
        "cache_dir": args.cache_dir,
        "attn_implementation": args.attn_implementation,
        "dtype": torch.bfloat16 if args.bf16 else torch.float32,
        "trust_remote_code": True,
        "device_map":{'':state.process_index},
        "use_cache": False if args.gradient_checkpointing else True,  # Disable cache if using gradient checkpointing
    }

    # Check if we are performing RoPE extension/scaling.
    if args.new_max_position_embeddings is not None and args.new_rope_theta is not None:
        
        # Get the model configuration from the reference model.
        config = transformers.AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

        # Increase the max position embeddings to the new value.
        assert args.new_max_position_embeddings > model.config.max_position_embeddings, f"New max position embeddings must be greater than the current value ({model.config.max_position_embeddings})."
        config.max_position_embeddings = args.new_max_position_embeddings
        # Increase the rope theta to the new value.
        assert args.new_rope_theta > model.config.rope_theta, f"New RoPE theta must be greater than the current value ({model.config.rope_theta})."
        config.rope_theta = args.new_rope_theta

    # We use the AutoModelForCausalLM to load the model.
    # [AutoModelForCausalLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config if args.new_max_position_embeddings is not None and args.new_rope_theta is not None else None,
        **model_init_kwargs
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

    # Check if the tokenizer has a pad token and if the pad token is different from the eos token
    assert tokenizer.pad_token is not None, "The tokenizer does not have a pad token. Please set a pad token before training."
    assert tokenizer.pad_token != tokenizer.eos_token, "The tokenizer's pad token is the same as the eos token. Please set a different pad token before training."
    
    # If the dataset is not already tokenized, we need to preprocess it.
    if "text" in train_dataset.column_names:

        # Split the dataset into train and test sets if a test size is specified
        if args.test_size is not None:
            dataset = train_dataset.train_test_split(
                test_size=args.test_size,
                seed=args.seed,
            )

        # Add the end-of-sequence token to each example in the dataset
        # Optionally, and beginning-of-sequence token if args.add_bos_token is set.
        def add_special_tokens(example):
            text = example["text"] + tokenizer.eos_token
            if args.add_bos_token:
                text = tokenizer.bos_token + text
            return {"text": text}
    
        # Define which columns to remove (all except 'text')
        columns = dataset["train"].column_names if "train" in dataset else dataset.column_names
        columns_to_remove = [col for col in columns if col != "text"]

        # Only preprocess on main process
        if state.is_main_process:
            dataset = dataset.map(
                add_special_tokens,
                num_proc=args.num_proc,
                load_from_cache_file=True,
                remove_columns=columns_to_remove,
                desc=f"Adding EOS token to the dataset" if not args.add_bos_token else "Adding BOS and EOS tokens to the dataset",
            )
        
        # Wait for main process to finish preprocessing
        state.wait_for_everyone()
        
        # Other processes load the cached result
        if not state.is_main_process:
            dataset = dataset.map(
                add_special_tokens,
                num_proc=args.num_proc,
                load_from_cache_file=True,
                remove_columns=columns_to_remove,
                desc="Loading preprocessed dataset from cache",
            )
        
        train_dataset, val_dataset = dataset["train"], dataset["test"] if "test" in dataset else None

    # Get the job ID from the environment variable or set it to "local" if not available
    jobid = os.getenv("SLURM_JOB_ID", "local")

    # Set the `WANDB_PROJECT` to args.wandb_project
    os.environ["WANDB_PROJECT"] = args.wandb_project

    # For the optimizer, we will split weights in two groups: 
    # - one with weight decay; 
    # - and the other without weight decay (biases, layer norms, and embeddings).
    #
    # Following OLMo 2, we remove weight decay from embedding layers to improve training stability. 
    # Source: https://arxiv.org/abs/2501.00656
    no_decay = ["bias", "layer_norm.weight", "embed_tokens.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
    # Create the optimizer with fused operations for better performance on CUDA devices
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        betas=(args.adam_beta1, args.adam_beta2),
        fused=True if torch.cuda.is_available() else False
    )

    # Calculate total training steps for learning rate scheduler
    # This is needed for custom LR schedulers like cosine and WSD
    if args.max_steps is not None:
        max_steps = args.max_steps
    else:
        # Estimate steps based on dataset size and batch configuration
        # Note: This is an approximation since we don't know exact dataset size here
        # The Trainer will handle this automatically, but we need it for custom schedulers
        max_steps = args.num_train_epochs * (len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()))

    # Create custom learning rate scheduler if using cosine or WSD
    # Otherwise, the Trainer will use the scheduler specified in TrainingArguments
    lr_scheduler = None
    if args.lr_scheduler_type.lower() == "cosine":
        # Cosine decay learning rate scheduler with warmup
        # Source: https://arxiv.org/abs/1608.03983
        def cosine_scheduler(step):
            if step < args.warmup_steps:
                # Linear warmup
                return step / args.warmup_steps
            else:
                # Cosine decay
                progress = (step - args.warmup_steps) / (max_steps - args.warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_scheduler)
        
    elif args.lr_scheduler_type.lower() == "wsd":
        # Warmup-Stable-Decay (WSD) learning rate scheduler
        # This scheduler has three phases:
        # 1. Warmup: linear increase from 0 to max_lr
        # 2. Stable: constant at max_lr
        # 3. Decay: cosine decay from max_lr to min_lr
        #
        # Source: https://arxiv.org/abs/2108.06084
        stable_steps = int(0.4 * max_steps)  # 40% of training at stable LR
        decay_steps = max_steps - args.warmup_steps - stable_steps
        
        def wsd_scheduler(step):
            if step < args.warmup_steps:
                # Warmup phase
                return step / args.warmup_steps
            elif step < args.warmup_steps + stable_steps:
                # Stable phase
                return 1.0
            else:
                # Decay phase
                progress = (step - args.warmup_steps - stable_steps) / decay_steps
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=wsd_scheduler)
    
    # [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
    training_args = transformers.TrainingArguments(
        output_dir=args.checkpoint_dir,
        do_eval=True if val_dataset is not None else False,
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=args.eval_steps if val_dataset is not None else None,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size if val_dataset is not None else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=-1 if args.max_steps is None else args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type if args.lr_scheduler_type.lower() not in ["cosine", "wsd"] else None,
        warmup_steps=args.warmup_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
        data_seed=args.seed,
        bf16=args.bf16,
        tf32=args.tf32,
        ddp_backend=args.ddp_backend,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor =args.dataloader_prefetch_factor,
        use_liger_kernel=args.use_liger_kernel,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if torch.cuda.device_count() > 1 and args.gradient_checkpointing else None,
        torch_compile=args.torch_compile,
        hub_token=args.hub_token,
        hub_model_id=args.hub_model_id,
        report_to=args.report_to,
        push_to_hub=True if args.hub_token is not None and args.hub_model_id is not None else False,
        hub_private_repo=True, # If you want to push to a private repo
        remove_unused_columns=True,
        run_name=f"{args.model_name_or_path.split('/')[-1]}-jobid-{jobid}-bs-{args.per_device_train_batch_size * args.gradient_accumulation_steps}-ngpu-{torch.cuda.device_count()}-lr-{args.learning_rate}",
    )

    # [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer)
    # Pass the custom optimizer and learning rate scheduler to the Trainer
    trainer = transformers.Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, lr_scheduler) if lr_scheduler is not None else (optimizer, None),
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
    parser = argparse.ArgumentParser(
        description="Training Script using Transformers Trainer API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ============================================================================
    # Dataset Configuration
    # ============================================================================
    dataset_group = parser.add_argument_group('Dataset Configuration')
    dataset_group.add_argument(
        "--dataset_type", 
        type=str,
        choices=["jsonl", "parquet"], 
        default="parquet", 
        help="Type of the dataset files."
    )
    dataset_group.add_argument(
        "--train_dataset_dir", 
        type=str, 
        nargs="+", 
        required=True, 
        help="Path(s) to the training dataset directory or file. Can be a single directory/file or a list."
    )
    dataset_group.add_argument(
        "--val_dataset_dir", 
        type=str, 
        default=None, 
        help="Path to the validation dataset directory. If not provided, no validation will be performed."
    )
    dataset_group.add_argument(
        "--shuffle_dataset", 
        action="store_true", 
        help="Shuffle the dataset before training."
    )
    dataset_group.add_argument(
        "--test_size", 
        type=int, 
        default=None, 
        help="Number of samples to use for test set when splitting train dataset."
    )
    dataset_group.add_argument(
        "--add_bos_token", 
        action="store_true", 
        help="Add a beginning-of-sequence token to each example in the dataset."
    )
    dataset_group.add_argument(
        "--cache_dir", 
        type=str, 
        default=None,
        help="Directory to cache datasets and models."
    )
    dataset_group.add_argument(
        "--num_proc", 
        type=int, 
        default=16,
        help="Number of processes for dataset preprocessing."
    )
    
    # ============================================================================
    # Model Configuration
    # ============================================================================
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--model_name_or_path", 
        type=str, 
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    model_group.add_argument(
        "--max_length", 
        type=int, 
        default=4096,
        help="Maximum sequence length for tokenization."
    )
    model_group.add_argument(
        "--attn_implementation", 
        type=str, 
        default="eager", 
        help="Attention implementation to use. Options: 'eager', 'sdpa', 'flash_attention_2'."
    )
    model_group.add_argument(
        "--new_max_position_embeddings", 
        type=int, 
        default=None, 
        help="Extend the model's max position embeddings to this value (for context extension)."
    )
    model_group.add_argument(
        "--new_rope_theta", 
        type=int, 
        default=None, 
        help="Extend the model's RoPE theta to this value (for context extension)."
    )
    
    # ============================================================================
    # Training Configuration
    # ============================================================================
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility."
    )
    training_group.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=1,
        help="Total number of training epochs to perform."
    )
    training_group.add_argument(
        "--max_steps", 
        type=int, 
        default=None, 
        help="Total number of training steps to perform. If set, overrides num_train_epochs."
    )
    training_group.add_argument(
        "--per_device_train_batch_size", 
        type=int, 
        default=8,
        help="Batch size per GPU/TPU core for training."
    )
    training_group.add_argument(
        "--per_device_eval_batch_size", 
        type=int, 
        default=8,
        help="Batch size per GPU/TPU core for evaluation."
    )
    training_group.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass."
    )
    training_group.add_argument(
        "--max_grad_norm", 
        type=float, 
        default=1.0, 
        help="Maximum gradient norm for gradient clipping."
    )
    
    # ============================================================================
    # Optimizer Configuration
    # ============================================================================
    optimizer_group = parser.add_argument_group('Optimizer Configuration')
    optimizer_group.add_argument(
        "--learning_rate", 
        type=float, 
        default=3e-4,
        help="Initial learning rate for AdamW optimizer."
    )
    optimizer_group.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0,
        help="Weight decay coefficient for AdamW optimizer."
    )
    optimizer_group.add_argument(
        "--adam_beta1", 
        type=float, 
        default=0.9,
        help="Beta1 hyperparameter for AdamW optimizer."
    )
    optimizer_group.add_argument(
        "--adam_beta2", 
        type=float, 
        default=0.95,
        help="Beta2 hyperparameter for AdamW optimizer."
    )
    optimizer_group.add_argument(
        "--adam_epsilon", 
        type=float, 
        default=1e-8,
        help="Epsilon hyperparameter for AdamW optimizer."
    )
    
    # ============================================================================
    # Learning Rate Scheduler Configuration
    # ============================================================================
    scheduler_group = parser.add_argument_group('Learning Rate Scheduler Configuration')
    scheduler_group.add_argument(
        "--lr_scheduler_type", 
        type=str, 
        default="cosine", 
        help="Type of learning rate scheduler. Options: 'linear', 'cosine', 'wsd', or any from transformers.SchedulerType."
    )
    scheduler_group.add_argument(
        "--warmup_steps", 
        type=int, 
        default=0,
        help="Number of steps for learning rate warmup."
    )
    
    # ============================================================================
    # Precision and Performance Configuration
    # ============================================================================
    performance_group = parser.add_argument_group('Precision and Performance Configuration')
    performance_group.add_argument(
        "--bf16", 
        action="store_true", 
        help="Use bfloat16 mixed precision training. Requires GPU support (e.g., A100, H100)."
    )
    performance_group.add_argument(
        "--tf32", 
        action="store_true", 
        help="Use TensorFloat-32 precision for matmul operations. Requires GPU support (e.g., A100, H100)."
    )
    performance_group.add_argument(
        "--gradient_checkpointing", 
        action="store_true", 
        help="Enable gradient checkpointing to save memory at the cost of slower backward pass."
    )
    performance_group.add_argument(
        "--torch_compile", 
        action="store_true", 
        help="Enable torch.compile for faster training (experimental)."
    )
    performance_group.add_argument(
        "--use_liger_kernel", 
        action="store_true", 
        help="Use Liger kernel for training. May improve performance on some GPUs (experimental)."
    )
    
    # ============================================================================
    # Distributed Training Configuration
    # ============================================================================
    distributed_group = parser.add_argument_group('Distributed Training Configuration')
    distributed_group.add_argument(
        "--ddp_backend", 
        type=str, 
        default="nccl", 
        help="Distributed data parallel backend. Options: 'nccl', 'gloo', 'mpi'."
    )
    
    # ============================================================================
    # DataLoader Configuration
    # ============================================================================
    dataloader_group = parser.add_argument_group('DataLoader Configuration')
    dataloader_group.add_argument(
        "--dataloader_num_workers", 
        type=int, 
        default=16, 
        help="Number of subprocesses to use for data loading."
    )
    dataloader_group.add_argument(
        "--dataloader_prefetch_factor", 
        type=int, 
        default=4, 
        help="Number of batches to prefetch per worker."
    )
    
    # ============================================================================
    # Checkpointing and Evaluation Configuration
    # ============================================================================
    checkpoint_group = parser.add_argument_group('Checkpointing and Evaluation Configuration')
    checkpoint_group.add_argument(
        "--checkpoint_dir", 
        type=str, 
        required=True,
        help="Directory to save model checkpoints."
    )
    checkpoint_group.add_argument(
        "--resume_from_checkpoint", 
        type=str, 
        default=None, 
        help="Path to a checkpoint to resume training from."
    )
    checkpoint_group.add_argument(
        "--save_steps", 
        type=int, 
        default=1000,
        help="Save checkpoint every X update steps."
    )
    checkpoint_group.add_argument(
        "--eval_steps", 
        type=int, 
        default=1000,
        help="Run evaluation every X update steps."
    )
    checkpoint_group.add_argument(
        "--logging_steps", 
        type=int, 
        default=1,
        help="Log metrics every X update steps."
    )
    
    # ============================================================================
    # Logging and Monitoring Configuration
    # ============================================================================
    logging_group = parser.add_argument_group('Logging and Monitoring Configuration')
    logging_group.add_argument(
        "--report_to", 
        type=str, 
        nargs="+", 
        default=None,
        help="The list of integrations to report the results and logs to. Supported platforms are 'tensorboard', 'wandb', 'comet_ml', 'mlflow', 'clearml', 'wandb' etc. See [here](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.report_to) for more details."
    )
    logging_group.add_argument(
        "--wandb_project", 
        type=str, 
        default="Polyglot",
        help="Weights & Biases project name for logging."
    )
    
    # ============================================================================
    # HuggingFace Hub Configuration
    # ============================================================================
    hub_group = parser.add_argument_group('HuggingFace Hub Configuration')
    hub_group.add_argument(
        "--hub_token", 
        type=str, 
        default=None,
        help="HuggingFace Hub authentication token for pushing models."
    )
    hub_group.add_argument(
        "--hub_model_id", 
        type=str, 
        default=None,
        help="HuggingFace Hub model repository ID for pushing trained model."
    )

    args = parser.parse_args()

    main(args)