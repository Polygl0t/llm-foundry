"""
This script checks for size mismatches between a tokenizer and a model's embedding layer.
It is useful when adapting models to new tokenizers that may have different vocab sizes.
# e.g., with tokensurgeon.
"""
import argparse
import json
import os
import sys
from typing import Tuple, Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_path: str, dtype: str = "bfloat16") -> Tuple[Any, Any]:
	"""
	Load tokenizer and model from the specified path.
	
	Args:
		model_path: Path to the model directory
		dtype: Data type to load the model in (default: bfloat16)
	
	Returns:
		Tuple of (tokenizer, model)
	
	Raises:
		ValueError: If model or tokenizer cannot be loaded
	"""
	try:
		print(f"Loading tokenizer from {model_path} ...")
		tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
		
		print(f"Loading model from {model_path} (dtype={dtype}) ...")
		torch_dtype = getattr(torch, dtype)
		model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
		
		return tokenizer, model
	except Exception as e:
		raise ValueError(f"Failed to load model/tokenizer from {model_path}: {e}")


def get_sizes(tokenizer: Any, model: Any) -> Tuple[int, int]:
	"""
	Get the tokenizer size and embedding size.
	
	Args:
		tokenizer: The loaded tokenizer
		model: The loaded model
	
	Returns:
		Tuple of (tokenizer_size, embedding_size)
	
	Raises:
		ValueError: If sizes cannot be determined
	"""
	# Tokenizer size (number of token ids recognized)
	try:
		tokenizer_size = len(tokenizer)
	except Exception:
		# fallback
		tokenizer_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer.get_vocab())
	
	# Embedding matrix size (number of embedding rows)
	emb = model.get_input_embeddings()
	if emb is None:
		raise ValueError("Model has no input embeddings (get_input_embeddings() returned None).")
	
	embed_size = emb.weight.shape[0]
	
	return tokenizer_size, embed_size


def report_mismatch(tokenizer: Any, tokenizer_size: int, embed_size: int, 
					save_missing: bool, out_dir: str) -> None:
	"""
	Report details about tokenizer/embedding size mismatches.
	
	Args:
		tokenizer: The loaded tokenizer
		tokenizer_size: Size of the tokenizer vocabulary
		embed_size: Size of the embedding matrix
		save_missing: Whether to save missing tokens/embeddings to a file
		out_dir: Output directory for saved files
	"""
	if tokenizer_size > embed_size:
		# Tokens without embeddings: ids >= embed_size
		missing_ids = list(range(embed_size, tokenizer_size))
		missing_tokens = tokenizer.convert_ids_to_tokens(missing_ids)
		
		print(f"\nTokenizer has {tokenizer_size} ids but embedding has {embed_size} rows.")
		print(f"Tokens without embeddings: {len(missing_tokens)}")
		print("\nShowing first 20 examples:")
		for i, t in enumerate(missing_tokens[:20]):
			print(f"  id={missing_ids[i]:6d}  token={t!r}")
		
		if save_missing:
			os.makedirs(out_dir, exist_ok=True)
			out_path = os.path.join(out_dir, "tokens_without_embeddings.json")
			with open(out_path, "w", encoding="utf-8") as f:
				json.dump({
					"ids": missing_ids,
					"tokens": missing_tokens,
					"count": len(missing_tokens)
				}, f, ensure_ascii=False, indent=2)
			print(f"\nWrote full list to {out_path}")
	
	else:
		# Embedding rows without tokenizer tokens: embed_size > tokenizer_size
		extra_embed_ids = list(range(tokenizer_size, embed_size))
		
		print(f"\nEmbedding has {embed_size} rows but tokenizer has {tokenizer_size} ids.")
		print(f"Embedding rows without tokenizer tokens: {len(extra_embed_ids)}")
		print("\nShowing first 20 examples:")
		for eid in extra_embed_ids[:20]:
			print(f"  embed_id={eid}")
		
		if save_missing:
			os.makedirs(out_dir, exist_ok=True)
			out_path = os.path.join(out_dir, "embeddings_without_tokens.json")
			with open(out_path, "w", encoding="utf-8") as f:
				json.dump({
					"embed_ids": extra_embed_ids,
					"count": len(extra_embed_ids)
				}, f, ensure_ascii=False, indent=2)
			print(f"\nWrote full list to {out_path}")


def resize_and_save(model: Any, tokenizer: Any, tokenizer_size: int, 
					embed_size: int, out_dir: str) -> None:
	"""
	Resize model embeddings to match tokenizer and save both.
	
	Args:
		model: The loaded model
		tokenizer: The loaded tokenizer
		tokenizer_size: Target size (tokenizer vocabulary size)
		embed_size: Current embedding size
		out_dir: Directory to save the resized model
	
	Raises:
		RuntimeError: If resizing or saving fails
	"""
	try:
		print(f"\nResizing model embeddings from {embed_size} -> {tokenizer_size} ...")
		
		# This updates input embeddings (and tied LM head if applicable)
		model.resize_token_embeddings(tokenizer_size)
		
		# Verify new embedding size
		new_emb = model.get_input_embeddings()
		new_embed_size = new_emb.weight.shape[0]
		
		if new_embed_size != tokenizer_size:
			raise RuntimeError(
				f"Resize failed: expected {tokenizer_size} rows, got {new_embed_size}"
			)
		
		print(f"Successfully resized to {new_embed_size} rows")
		
		# Save resized model + tokenizer
		os.makedirs(out_dir, exist_ok=True)
		print(f"\nSaving resized model and tokenizer to {out_dir} ...")
		model.save_pretrained(out_dir)
		tokenizer.save_pretrained(out_dir)
		print("Successfully saved resized model and tokenizer.")
		
	except Exception as e:
		raise RuntimeError(f"Failed to resize and save model: {e}")


def main(args: argparse.Namespace) -> int:
	"""
	Main function to check and optionally resize model embeddings.
	
	Args:
		args: Parsed command-line arguments
	
	Returns:
		Exit code (0 for success, 1 for error)
	"""
	try:
		# Load model and tokenizer
		tokenizer, model = load_model_and_tokenizer(args.model_path, args.dtype)
		
		# Get sizes
		tokenizer_size, embed_size = get_sizes(tokenizer, model)
		
		print(f"\ntokenizer_size = {tokenizer_size}")
		print(f"embedding_rows = {embed_size}")
		
		# Check if sizes match
		if tokenizer_size == embed_size:
			print("\n✓ Sizes match: every tokenizer id has a corresponding embedding row.")
			return 0
		
		# Determine output directory
		if args.output_dir is None:
			args.output_dir = os.path.join(args.model_path, "model_with_resized_embedding")
		
		# Report mismatch details
		report_mismatch(tokenizer, tokenizer_size, embed_size, args.save_missing, args.output_dir)
		
		# Resize and save if requested
		if args.resize:
			resize_and_save(model, tokenizer, tokenizer_size, embed_size, args.output_dir)
		else:
			print("\n⚠ Skipping resize (use --resize to resize and save the model)")
		
		return 0
		
	except Exception as e:
		print(f"\n✗ Error: {e}", file=sys.stderr)
		return 1


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(
		description="Check and resize model embeddings to match tokenizer vocabulary size.",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Check for size mismatches (no changes)
  python resize_embedding_layer.py /path/to/model

  # Check and save mismatch details to JSON
  python resize_embedding_layer.py /path/to/model --save-missing

  # Check and resize, saving to default output directory
  python resize_embedding_layer.py /path/to/model --resize

  # Check and resize, saving to custom output directory
  python resize_embedding_layer.py /path/to/model --resize --output-dir /path/to/output

  # Use float16 instead of bfloat16
  python resize_embedding_layer.py /path/to/model --dtype float16 --resize
		"""
	)
	
	parser.add_argument(
		"model_path",
		type=str,
		help="Path to the model directory containing model and tokenizer files"
	)
	
	parser.add_argument(
		"-o", "--output-dir",
		type=str,
		default=None,
		help="Output directory for resized model (default: MODEL_PATH/model_with_resized_embedding)"
	)
	
	parser.add_argument(
		"--resize",
		action="store_true",
		help="Resize the model embeddings and save (default: only check, don't resize)"
	)
	
	parser.add_argument(
		"--save-missing",
		action="store_true",
		help="Save list of missing tokens/embeddings to JSON file"
	)
	
	parser.add_argument(
		"--dtype",
		type=str,
		default="bfloat16",
		choices=["float32", "float16", "bfloat16"],
		help="Data type to load the model in (default: bfloat16)"
	)
	
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	sys.exit(main(args))