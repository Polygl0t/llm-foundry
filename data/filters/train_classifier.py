"""
Sequence Classification Model Training Pipeline

Trains regression-based classifiers for scoring text quality, educational value, toxicity, etc.
Designed for creating custom filters for dataset curation pipelines.

Input data format:
- Dataset with text_column and target_column (scores in range [1, 5])
- Automatically converts scores to [0, 4] for training
- Supports JSONL, Parquet, or HuggingFace datasets

Output:
- Trained model saved to checkpoint_dir/final/
- Optional push to Hugging Face Hub
- Evaluation metrics and confusion matrix

Usage:
    # Train edu-score classifier
    python train_classifier.py --train_dataset_dir scored_data.jsonl \
        --dataset_type jsonl \
        --shuffle_dataset \
        --model_name microsoft/deberta-v3-base \
        --text_column text --target_column score --id_label Edu-Score \
        --checkpoint_dir checkpoints/ --max_length 512 \
        --per_device_train_batch_size 32 --num_train_epochs 20 \
        --learning_rate 3e-4 --bf16
    
    # Train with frozen layers (faster)
    python train_classifier.py --train_dataset_dir data.jsonl \
        --dataset_type jsonl \
        --shuffle_dataset \
        --model_name Qwen/Qwen2-1.5B --freeze \
        --checkpoint_dir ckpt/ --gradient_checkpointing \
        --hub_token TOKEN --hub_model_id username/my-classifier
"""
from transformers import (
    AutoTokenizer,
    AutoConfig,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
)

from sklearn.metrics import classification_report, confusion_matrix
import accelerate
import datasets
import numpy as np
import evaluate
import argparse
import torch
import glob
import os

def compute_metrics(eval_pred):
    """Compute metrics for the evaluation step"""

    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    preds = np.round(logits.squeeze()).clip(0, 4).astype(int) # Clip the predictions to the range [0, 4]
    labels = np.round(labels.squeeze()).astype(int)

    precision = precision_metric.compute(predictions=preds, references=labels, average="macro")["precision"]
    recall = recall_metric.compute(predictions=preds, references=labels, average="macro")["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]

    # [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
    report = classification_report(labels, preds)
    # [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
    cm = confusion_matrix(labels, preds)

    print("Validation Report:\n" + report)
    print("Confusion Matrix:\n" + str(cm))

    return {
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "accuracy": accuracy,
    }

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

    if args.shuffle_dataset:
        dataset = dataset.shuffle(seed=args.seed)

    # Given that the scores we generated in `llm_filter.py` are in the range [1, 5], 
    # we need to convert them to the range [0, 4] for training.
    dataset = dataset.map(
        lambda x: {args.target_column: np.clip(int(x[args.target_column])-1, 0, 4)},
        num_proc=args.num_proc,
    )
    # Cast the target column to ClassLabel.
    dataset = dataset.cast_column(
        args.target_column, datasets.ClassLabel(names=[str(i) for i in range(0, 5)])
    )
    # Split the dataset into train and test sets.
    dataset = dataset.train_test_split(
        test_size=min(args.test_size, len(dataset) * 0.1),  # Ensure test_size doesn't exceed dataset size
        seed=args.seed, 
        stratify_by_column=args.target_column
    )

    # We use the AutoConfig to infer what is the model type and how to configure the model.
    # [AutoConfig](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoConfig)
    config = AutoConfig.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )

    model_args = {
        "num_labels": 1,
        "hidden_dropout_prob": 0.0,
        "output_hidden_states": False,
        "cache_dir": args.cache_dir,
        "id2label": {0: args.id_label},
        "label2id": {args.id_label: 0},
        "trust_remote_code": True,
        "use_cache": True if not args.gradient_checkpointing else False,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float32,
        "device_map":{'':state.process_index},
    }
    
    # Add classifier_dropout for non-DeBERTa models (will be ignored by DeBERTa)
    if "deberta" not in config.model_type:
        model_args["classifier_dropout"] = 0.0

    # Remove the `hidden_dropout_prob` and `classifier_dropout` for Qwen and Llama models
    if "qwen" in config.model_type or "llama" in config.model_type:
        if "hidden_dropout_prob" in model_args:
            del model_args["hidden_dropout_prob"]
        if "classifier_dropout" in model_args:
            del model_args["classifier_dropout"]

    # We use the AutoModelForSequenceClassification to load the model.
    # [AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSequenceClassification)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        attn_implementation=args.attn_implementation,
        **model_args
    )

    # We use the AutoTokenizer to load the tokenizer.
    # [AutoTokenizer](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer)
    # Make sure to set the `model_max_length` in a way that it does not exceed the model's max position embeddings.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=min(model.config.max_position_embeddings, args.max_length),
        cache_dir=args.cache_dir,
        use_fast=True,
        trust_remote_code=True,
    )

    # Set the pad token if it is not already set.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if args.chat_template_path is not None:
        # Load a Jinja template for the chat model
        # Learn more about chat templates [here](https://huggingface.co/docs/transformers/main/chat_templating)
        with open(args.chat_template_path, "r") as f:
            tokenizer.chat_template = f.read()

    # Preprocess the dataset.
    if args.text_column not in dataset['train'].column_names:
        raise ValueError(f"Text column '{args.text_column}' not found in the dataset. Available columns: {dataset['train'].column_names}")
    
    # Only main process runs the preprocessing and mapping
    if master_process:
        def preprocess(examples):
            batch = tokenizer(examples[args.text_column], truncation=True)
            batch["labels"] = np.float32(examples[args.target_column])
            return batch

        dataset = dataset.map(
            preprocess, 
            batched=True,
            num_proc=args.num_proc,
            load_from_cache_file=True,
            desc="Tokenizing dataset",
        )
    # Wait for main process to finish preprocessing
    state.wait_for_everyone()

    # Non-main processes reload from cache
    if not master_process:
        dataset = dataset.map(
            preprocess,
            num_proc=args.num_proc,
            load_from_cache_file=True,
            desc=f"Loading preprocessed dataset on process {state.process_index}",
        )

    # Create a simple data collator that pads the inputs to the maximum length in the batch
    # [DataCollatorWithPadding](https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorWithPadding)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if args.freeze:
        # Freeze the embeddings and the encoder (we only want to train the classifier head)
        if config.model_type == "electra":
            for param in model.electra.embeddings.parameters():
                param.requires_grad = False
            for param in model.electra.encoder.parameters():
                param.requires_grad = False
        
        elif config.model_type == "bert":
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False
            for param in model.bert.encoder.parameters():
                param.requires_grad = False
        
        elif "deberta" in config.model_type:
            for param in model.deberta.embeddings.parameters():
                param.requires_grad = False
            for param in model.deberta.encoder.parameters():
                param.requires_grad = False

        elif "roberta" in config.model_type:
            for param in model.roberta.embeddings.parameters():
                param.requires_grad = False
            for param in model.roberta.encoder.parameters():
                param.requires_grad = False

        # For the decoder family, we freeze the embedding layer but keep the rest of the model trainable.
        elif "qwen" in config.model_type or "llama" in config.model_type:
            if hasattr(model, "model"):
                if hasattr(model.model, "embed_tokens"):
                    for param in model.model.embed_tokens.parameters():
                        param.requires_grad = False
                #if hasattr(model.model, "layers"):
                #    for param in model.model.layers.parameters():
                #        param.requires_grad = False
            else:
                print(f"Warning: model.model not found in {type(model)}. No encoder/embedding frozen.")
            
        else:
            raise ValueError(f"Model type {model.config.model_type} not supported")

    # Get the job ID from the environment variable or set it to "local" if not available
    jobid = os.getenv("SLURM_JOB_ID", "local")

    # Set the `WANDB_PROJECT` to args.wandb_project
    os.environ["WANDB_PROJECT"] = args.wandb_project

    # [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if torch.cuda.device_count() > 1 and args.gradient_checkpointing else None,
        push_to_hub=True if args.hub_token is not None and args.hub_model_id is not None else False,
        hub_token=args.hub_token,
        hub_model_id=args.hub_model_id,
        report_to=args.report_to,
        save_total_limit=args.save_total_limit,
        eval_on_start=args.eval_on_start, # Do not evaluate on the first epoch
        load_best_model_at_end=True, # Load the best model at the end of training
        metric_for_best_model="f1_macro", # Use F1 macro as the metric for the best model
        greater_is_better=True, # Higher F1 macro is better
        hub_private_repo=True, # Push to a private repository if the user has provided a token
        run_name=f"{args.model_name.split('/')[-1]}-jobid-{jobid}-bs-{args.per_device_train_batch_size}-acumulation-{args.gradient_accumulation_steps}-ngpu-{torch.cuda.device_count()}-epochs-{args.num_train_epochs}",
    )

    #[Trainer](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer)
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )

    # Make sure every process is synced before training
    state.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    if args.resume_from_checkpoint:

        # We expect the `resume_from_checkpoint` argument to be the path to a directory of checkpoints,
        # the logic below will find the latest checkpoint in that directory.
        checkpoint_path = args.resume_from_checkpoint

        try:
            # We try to find the latest checkpoint in the directory.
            checkpoint_dirs = os.listdir(checkpoint_path)
            checkpoint_dirs = [dir for dir in checkpoint_dirs if dir.startswith(f"checkpoint-")] # Checkpoints are intended to be named like "checkpoint-"
            checkpoint_path = os.path.join(checkpoint_path, sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[-1].split(".")[0]))[-1])
        except:
            # If the checkpoint directory does not contain any checkpoints, we will assume
            # that `resume_from_checkpoint` is already set to the latest checkpoint.
            pass
        if master_process:
            print(f"Resuming training from checkpoint: {checkpoint_path}")

    # Start the training
    try:
        trainer.train(resume_from_checkpoint=checkpoint_path if args.resume_from_checkpoint else None)
    except Exception as e:
        save_path = os.path.join(args.checkpoint_dir, "last")
        trainer.save_model(save_path)
        if master_process:
            print(f"Training failed with error: {e}")
            print(f"Model saved to 'last' checkpoint at {save_path}")

    try:
        trainer.evaluate()
    except Exception as e:
        if master_process:
            print(f"Evaluation failed with error: {e}")
            print("Skipping final evaluation...")

    # Save the final model
    trainer.save_model(os.path.join(args.checkpoint_dir, "final"))
    # Done!
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Core dataset/model args
    parser.add_argument("--dataset_type", choices=["jsonl", "parquet"], default="parquet", help="Type of the dataset files. Can be either 'jsonl' or 'parquet'.")
    parser.add_argument("--train_dataset_dir", type=str, nargs="+", required=True, help="Path(s) to the training dataset directory or file. Can be a single directory/file or a list of directories/files.")
    parser.add_argument("--shuffle_dataset", action="store_true", help="If set, shuffle the dataset files before loading.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Path to a directory to use for caching the datasets and models.")
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--target_column", type=str, default="score")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--test_size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--chat_template_path", type=str, default=None, help="Path to a Jinja template.")
    parser.add_argument("--id_label", type=str, default="Score", help="Label for the classification task, e.g., 'Edu-Score', 'Toxicity', etc.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from.")
    # Tokenization / model configuration
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization / model.")
    parser.add_argument("--freeze", action="store_true", help="Freeze the embeddings and decoder/encoder layers. Only the classifier head will be trained.")
    # Training and optimizer
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Type of learning rate scheduler to use. Options: 'linear', 'cosine', and all the other types listed [here](https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.SchedulerType).")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--eval_on_start", action="store_true")
    # Precision / performance
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision for training. Requires a GPU that supports bfloat16 (e.g., A100)")
    parser.add_argument("--tf32", action="store_true", help="Use TensorFloat-32 precision for training. Requires a GPU that supports TF32 (e.g., A100)")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory. This will slow down training but reduce memory usage.")
    parser.add_argument("--attn_implementation", type=str, default="eager", help="Attention implementation to use. Options: 'eager', 'sdpa', and 'flash_attention_2'.")
    # Data loader / batch sizes
    parser.add_argument("--per_device_train_batch_size", type=int, default=256)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # Hub / reporting
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--report_to", type=str, nargs="+", default=None , help="The list of integrations to report the results and logs to. Supported platforms are 'tensorboard', 'wandb', 'comet_ml', 'mlflow', 'clearml', 'wandb' etc. See [here](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.report_to) for more details.")
    parser.add_argument("--wandb_project", type=str, default="Polyglot")

    args = parser.parse_args()

    main(args)

# How to use the trained model:
#
#from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
#model_id = "my-username/my-model-name"
#tokenizer = AutoTokenizer.from_pretrained(model_id)
#model = AutoModelForSequenceClassification.from_pretrained(model_id)
#
#text = "This is a test sentence."
#inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
#outputs = model(**inputs)
#logits = outputs.logits.squeeze(-1).float().detach().numpy()
#score = logits.item() + 1 # scores are produced in the range [0, 4]. To convert to the range [1, 5], we add 1 to the score.
#result = {
#    "text": text,
#    "score": score,
#    "edu_score": int(round(max(0, min(score, 4)))) + 1, # scores are produced in the range [0, 4]. To convert to the range [1, 5], we add 1 to the rounded score.
#}
#
#print(result)