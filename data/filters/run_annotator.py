"""
Inference Pipeline for Dataset Annotation

Runs inference with HuggingFace sequence classification models to annotate datasets.

Methodology:
- Loads pre-trained classifier (trained with train_classifier.py)
- Applies optional chat template formatting to text
- Runs batched inference with configurable batch size
- Outputs both float scores (raw logits + 1) and rounded integer scores (e.g., 1-5)
- Preserves original dataset structure with added score columns

Annotation mapping:
- Model outputs logits in range [0, 4]
- float_score: logits + 1 -> [1, 5] range with decimals
- int_score: round(clip(logits, 0, 4)) + 1 -> integer [1, 5]

Usage:
    # Annotate dataset with edu classifier
    python run_annotator.py --model_name username/edu-classifier \\
        --dataset_path data/ --text_column text \\
        --output_folder scored/ --batch_size 32 \\
        --float_score edu_score_float --int_score edu_score

    # Annotate chat dataset with template
    python run_annotator.py --model_name username/quality-classifier \\
        --dataset_path conversations.jsonl --text_column messages \\
        --apply_chat_template --output_folder scored/ \\
        --max_length 1024
"""

import argparse
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import DatasetLoader, apply_chat_template_to_dataset, get_logger, save_dataset

logger = get_logger("RunAnnotator")


def main(args):
    input_path = os.path.abspath(args.dataset_path)
    output_path = os.path.abspath(args.output_folder)

    if os.path.isdir(input_path) and input_path == output_path:
        logger.warning(
            "Input and output paths are the same. We will add the suffix '_annotated' to the output folder to avoid overwriting the input dataset."
        )
        output_path += "_annotated"

    if os.path.isfile(input_path):
        input_dir = os.path.dirname(input_path)
        if input_dir == output_path:
            logger.warning(
                "Input and output paths are the same. We will add the suffix '_annotated' to the output folder to avoid overwriting the input dataset."
            )
            output_path += "_annotated"

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir if args.cache_dir else "./.cache",
        token=args.token if args.token else None,
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

    # Load dataset using unified loader
    loader = DatasetLoader(
        path=args.dataset_path,
        cache_dir=args.cache_dir,
        num_proc=args.num_proc,
    )
    dataset = loader.load()
    logger.info("Loaded %d examples", len(dataset))

    # Apply chat template if requested
    text_column = args.text_column
    if args.apply_chat_template:
        dataset, text_column = apply_chat_template_to_dataset(
            dataset, tokenizer, args.text_column, args.num_proc
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
        batch[args.int_score] = [int(round(max(0, min(score, 4)))) + 1 for score in logits]

        return batch

    # Run the annotator over the dataset in batches
    dataset = dataset.map(
        run_annotator,
        batched=True,
        batch_size=args.batch_size if args.batch_size else 1,
        num_proc=None,  # Disable multiprocessing for model inference
        desc="Classifying dataset",
    )

    # Remove temporary formatted text column if it was added
    if args.apply_chat_template:
        dataset = dataset.remove_columns(["formatted_text"])

    # Save annotated dataset using utility
    save_dataset(
        dataset,
        args.output_folder,
        output_type="parquet",
        tokens_per_chunk=0,
        token_count=0,
        n_chunks=1,
    )
    logger.info("Saved %d examples to '%s'", len(dataset), args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model_name", type=str, required=True, help="The name of the model to be used."
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Whether to apply a chat template to the text column.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="The path to the directory containing the dataset or a specific file (supports jsonl and parquet).",
    )
    parser.add_argument("--token", type=str, default=None, help="The token to access the dataset.")
    parser.add_argument(
        "--cache_dir", type=str, default="./.cache", help="The directory to store the dataset."
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="The name of the text column in the dataset.",
    )
    parser.add_argument("--num_proc", type=int, default=1, help="The number of processes to use.")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size.")
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="The maximum length of the text to be tokenized.",
    )
    parser.add_argument(
        "--float_score",
        type=str,
        default="float_score",
        help="The name of the column to store the float scores.",
    )
    parser.add_argument(
        "--int_score",
        type=str,
        default="int_score",
        help="The name of the column to store the integer scores.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="The directory to store the output files (must be different from input directory).",
    )

    args = parser.parse_args()

    main(args)
