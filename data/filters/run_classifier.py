"""
Sequence Classification Inference for Dataset Scoring

Runs inference with HuggingFace sequence classification models to score datasets.
Commonly used for educational content scoring, quality assessment, or toxicity detection.

Methodology:
- Loads pre-trained classifier (trained with train_classifier.py)
- Applies optional chat template formatting to text
- Runs batched inference with configurable batch size
- Outputs both float scores (raw logits + 1) and rounded integer scores [1-5]
- Preserves original dataset structure with added score columns

Score mapping:
- Model outputs logits in range [0, 4]
- float_score: logits + 1 → [1, 5] range with decimals
- int_score: round(clip(logits, 0, 4)) + 1 → integer [1, 5]

Usage:
    # Score dataset with edu classifier
    python run_classifier.py --model_name username/edu-classifier \
        --dataset_path data/ --text_column text \
        --output_folder scored/ --batch_size 32 \
        --float_score edu_score_float --int_score edu_score
    
    # Score chat dataset with template
    python run_classifier.py --model_name username/quality-classifier \
        --dataset_path conversations.jsonl --text_column messages \
        --apply_chat_template --output_folder scored/ \
        --max_length 1024
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import argparse
import torch
import glob
import os


def load_dataset_files(dataset_path, cache_dir):
    """
    Load dataset from a file or directory containing JSONL or Parquet files.
    
    Args:
        dataset_path: Path to dataset file or directory
        cache_dir: Cache directory for datasets
        
    Returns:
        tuple: (dataset, dataset_type, dataset_files)
    """
    if os.path.isdir(dataset_path):
        print(f"Loading dataset from directory: {dataset_path}")
        dataset_files = sorted(glob.glob(os.path.join(dataset_path, "*.jsonl")))
        dataset_type = "json"
        
        if not dataset_files:
            dataset_files = sorted(glob.glob(os.path.join(dataset_path, "*.parquet")))
            dataset_type = "parquet"
            
        if not dataset_files:
            raise ValueError(f"No JSONL or Parquet files found in {dataset_path}")
            
    elif os.path.isfile(dataset_path):
        assert dataset_path.endswith((".jsonl", ".parquet")), "Dataset file must be either .jsonl or .parquet"
        print(f"Loading dataset from file: {dataset_path}")
        dataset_files = [dataset_path]
        dataset_type = "json" if dataset_path.endswith(".jsonl") else "parquet"
        
    dataset = load_dataset(
        dataset_type,
        data_files=dataset_files,
        split='train',
        num_proc=len(dataset_files),
        cache_dir=cache_dir,
    )
    
    print(f"Loaded {len(dataset_files)} file(s) with {len(dataset)} total examples")
    
    return dataset, dataset_type, dataset_files


def apply_chat_template_to_dataset(dataset, tokenizer, text_column, num_proc):
    """
    Apply chat template formatting to the text column of the dataset.
    
    Args:
        dataset: The input dataset
        tokenizer: The tokenizer with chat template
        text_column: Name of the column containing text
        num_proc: Number of processes for parallel processing
        
    Returns:
        tuple: (formatted_dataset, new_text_column_name)
    """
    if tokenizer.chat_template is None:
        raise ValueError(
            "The tokenizer does not have a chat template. "
            "Please use a tokenizer that supports chat templates."
        )
    
    def format_messages(example):
        formatted_text = tokenizer.apply_chat_template(
            example[text_column],
            tokenize=False  # Returns string instead of token IDs
        )
        example['formatted_text'] = formatted_text
        return example
    
    formatted_dataset = dataset.map(
        format_messages,
        num_proc=num_proc,
        desc="Formatting messages with chat template"
    )
    
    return formatted_dataset, 'formatted_text'


def save_dataset_split(dataset, dataset_type, dataset_files, output_folder):
    """
    Split dataset evenly and save with same filenames as input.
    Outputs the same number of files with matching names.
    
    Args:
        dataset: The dataset to save
        dataset_type: Type of dataset ('json' or 'parquet')
        dataset_files: List of original dataset file paths
        output_folder: Output directory path
    """
    os.makedirs(output_folder, exist_ok=True)
    
    n_files = len(dataset_files)
    total_examples = len(dataset)
    chunk_size = total_examples // n_files
    
    print(f"Splitting {total_examples} examples into {n_files} file(s)...")
    
    for i, file_path in enumerate(dataset_files):
        # Calculate indices for this chunk
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < n_files - 1 else total_examples
        
        # Select chunk
        chunk = dataset.select(range(start_idx, end_idx))
        
        # Get filename and save
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_folder, filename)
        
        if dataset_type == "json":
            chunk.to_json(output_path)
        elif dataset_type == "parquet":
            chunk.to_parquet(output_path)
        
        print(f"  Saved {filename}: {len(chunk)} examples")
    
    print(f"\nSaved {total_examples} total examples to '{output_folder}'")


def main(args):
    """Main execution function for the classifier script."""
    
    # Validate that input and output directories are different
    input_path = os.path.abspath(args.dataset_path)
    output_path = os.path.abspath(args.output_folder)
    
    if os.path.isdir(input_path) and input_path == output_path:
        raise ValueError(
            f"Input and output directories must be different.\n"
            f"Input: {input_path}\n"
            f"Output: {output_path}"
        )
    
    if os.path.isfile(input_path):
        input_dir = os.path.dirname(input_path)
        if input_dir == output_path:
            raise ValueError(
                f"Input file directory and output directory must be different.\n"
                f"Input directory: {input_dir}\n"
                f"Output directory: {output_path}"
            )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir if args.cache_dir else "./.cache",
        token=args.token if args.token else None
    )

    # Load sequence classification model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir if args.cache_dir else "./.cache",
        token=args.token if args.token else None,
        attn_implementation="eager",  # Use eager attention if SDPA doesn't work
    )

    # Setup device and move model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load dataset
    dataset, dataset_type, dataset_files = load_dataset_files(
        args.dataset_path,
        args.cache_dir
    )

    # Apply chat template if requested
    text_column = args.text_column
    if args.apply_chat_template:
        dataset, text_column = apply_chat_template_to_dataset(
            dataset,
            tokenizer,
            args.text_column,
            args.num_proc
        )

    def run_annotator(batch):
        """Annotate a batch of examples with classification scores."""
        # Tokenize batch
        encoded_input = tokenizer(
            batch[text_column],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)

        # Run inference
        with torch.no_grad():
            model_output = model(**encoded_input)
            logits = model_output.logits.squeeze(-1).float().cpu().numpy()

        # Convert logits to scores in range [1, 5]
        batch[args.float_score] = [x + 1 for x in logits.tolist()]
        batch[args.int_score] = [
            int(round(max(0, min(score, 4)))) + 1 for score in logits
        ]

        return batch

    # Run classification on dataset
    dataset = dataset.map(
        run_annotator,
        batched=True,
        batch_size=args.batch_size if args.batch_size else 1,
        num_proc=None, # Disable multiprocessing for model inference
        desc="Classifying dataset",
    )

    # Remove temporary formatted text column if it was added
    if args.apply_chat_template:
        dataset = dataset.remove_columns(['formatted_text'])

    # Save dataset with same number and naming as input files
    save_dataset_split(
        dataset,
        dataset_type,
        dataset_files,
        args.output_folder
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to be used.")
    parser.add_argument("--apply_chat_template", action='store_true', help="Whether to apply a chat template to the text column.")
    parser.add_argument("--dataset_path", type=str, required=True, help="The path to the directory containing the dataset or a specific file (supports jsonl and parquet).")
    parser.add_argument("--token", type=str, default=None, help="The token to access the dataset.")
    parser.add_argument("--cache_dir", type=str, default="./.cache", help="The directory to store the dataset.")
    parser.add_argument("--text_column", type=str, default="text", help="The name of the text column in the dataset.")
    parser.add_argument("--num_proc", type=int, default=1, help="The number of processes to use.")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size.")
    parser.add_argument("--max_length", type=int, default=512, help="The maximum length of the text to be tokenized.")
    parser.add_argument("--float_score", type=str, default="float_score", help="The name of the column to store the float scores.")
    parser.add_argument("--int_score", type=str, default="int_score", help="The name of the column to store the integer scores.")
    parser.add_argument("--output_folder", type=str, required=True, help="The directory to store the output files (must be different from input directory).")

    args = parser.parse_args()

    main(args)
