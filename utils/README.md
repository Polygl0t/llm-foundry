# Utilities

Miscellaneous utility scripts and helpers for working with models and training pipelines.

## Overview

This folder contains miscellaneous utility scripts and helpers for working with models and training pipelines.

## Contents

- `convert_dataset_to_hf.py` — A script to convert JSONL or Parquet files into Hugging Face Dataset format.
- `count_tokens.py` — Creates a report of token counts for a given pretraining corpus.
- `download.py` — A helper script to download and cache repositories from Hugging Face Hub.
- `inference_test.py` — A script to test model inference on a set of samples.
- `inspect_model.py` — Analyze model configurations and compute parameter counts (including MoE expert routing).
- `marvin_create_workspace.sh` — A helper script to create a new workspace in the Marvin cluster.
- `pdf2markdown.sh` — Convert PDF documents to Markdown format using Marker.
- `parse_run.py` — Parse and summarize training run logs for the distributed training scripts.
- `resize_embedding_layer.py` — Check for size mismatches between a tokenizer and a model's embedding layer, and optionally resize the embeddings to match (e.g., after running tokensurgeon).
- `upload_ckpts_to_hf.py` — A script to upload model checkpoints from a local directory to Hugging Face Hub, creating branches for each training step and optionally uploading extra files in the root directory.
- `upload.py` — A helper script to upload directories to Hugging Face Hub.

