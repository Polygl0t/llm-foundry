"""
Dataset Decontamination via K-token ("n-gram") Matching

Removes examples from a training dataset that contain k-token sequences found in reference
datasets (e.g., test/validation sets) to prevent data leakage and contamination.

- Note: This decontamination operates at the token level, matching sequences of token IDs
rather than raw text or characters.

Workflow:
- Builds k-gram indices from reference datasets for lengths [min_k, max_k]
- Scans input dataset for contiguous token subsequences matching reference k-grams
- Supports exact matching and approximate matching (1-token substitution)
- Trims first/last tokens from reference examples before indexing

Input format:
- Dataset with an "input_ids" field containing tokenized sequences (List[int])
- Both input and reference datasets must have this format

Output:
- Decontaminated dataset with contaminated examples removed

Usage:
	python decontaminate.py --input_dir data/tokenized_train \\
        --reference_path eval_set/ \\
        --output_dir cleaned_data \\
        --min_k 8 --max_k 32 --allow_one_token_mismatch
"""
import argparse
from collections import defaultdict
import os
from numbers import Integral

from utils import DatasetLoader, get_logger, list_matching_files, save_dataset, save_metadata

logger = get_logger("Decontaminate")


def validate_input_ids(dataset, dataset_name: str) -> None:
	"""Validate that a dataset contains `input_ids` as lists of integers."""
	if "input_ids" not in dataset.column_names:
		raise ValueError(
			f"{dataset_name} must contain an 'input_ids' column. "
			f"Available columns: {dataset.column_names}"
		)

	sample_size = min(len(dataset), 5)
	for row_idx in range(sample_size):
		input_ids = dataset[row_idx]["input_ids"]
		if not isinstance(input_ids, list):
			raise ValueError(
				f"{dataset_name} row {row_idx} has invalid 'input_ids' type "
				f"{type(input_ids).__name__}; expected a list of integers."
			)
		if any(not isinstance(token, Integral) or isinstance(token, bool) for token in input_ids):
			raise ValueError(
				f"{dataset_name} row {row_idx} contains non-integer values in 'input_ids'."
			)


def build_reference_indices(reference_ids, min_k: int, max_k: int, approx_max_k: int, allow_one_token_mismatch: bool):
	"""Build exact and optional 1-token-mismatch indices from reference examples."""
	ref_kgrams = defaultdict(set)
	masked_map = defaultdict(set) if allow_one_token_mismatch else None

	for ids in reference_ids:
		total_length = len(ids)
		if total_length == 0:
			continue

		local_max_k = min(max_k, total_length)
		for k in range(min_k, local_max_k + 1):
			for i in range(total_length - k + 1):
				kgram = ids[i : i + k]
				ref_kgrams[k].add(kgram)

				if allow_one_token_mismatch and k <= approx_max_k:
					for j in range(k):
						masked = list(kgram)
						masked[j] = None
						masked_map[k].add(tuple(masked))

	return ref_kgrams, masked_map


def create_clean_filter(ref_kgrams, masked_map, approx_max_k: int, allow_one_token_mismatch: bool):
	"""Create a batched filter function for contamination checks."""
	lengths = sorted(ref_kgrams.keys())
	min_ref_k = lengths[0] if lengths else None

	def is_clean_batch_contiguous(batch):
		ids_batch = batch["input_ids"]
		if not ref_kgrams:
			return [True] * len(ids_batch)

		out = []
		for ids in ids_batch:
			ids_len = len(ids)
			contaminated = False

			if ids_len < min_ref_k:
				out.append(True)
				continue

			for length in lengths:
				if length > ids_len:
					break

				exact_kgrams = ref_kgrams[length]
				for start in range(ids_len - length + 1):
					window = tuple(ids[start : start + length])
					if window in exact_kgrams:
						contaminated = True
						break

					if allow_one_token_mismatch and length <= approx_max_k:
						for masked_idx in range(length):
							masked = list(window)
							masked[masked_idx] = None
							if tuple(masked) in masked_map[length]:
								contaminated = True
								break
						if contaminated:
							break

				if contaminated:
					break

			out.append(not contaminated)

		return out

	return is_clean_batch_contiguous


def main(args):
	if not os.path.isdir(args.input_dir):
		raise ValueError(
			f"--input_dir must point to a local directory containing .jsonl or .parquet shards. "
			f"Received: '{args.input_dir}'"
		)

	dataset = DatasetLoader(path=args.input_dir, cache_dir=args.cache_dir).load()
	reference_dataset = DatasetLoader(path=args.reference_path, cache_dir=args.cache_dir).load()

	validate_input_ids(dataset, "Input dataset")
	validate_input_ids(reference_dataset, "Reference dataset")

	logger.info(f"Loaded input dataset: {len(dataset):,} examples.")
	logger.info(f"Loaded reference dataset: {len(reference_dataset):,} examples.")

	reference_dataset = reference_dataset.filter(
		lambda ex: len(ex["input_ids"]) > (args.min_k + 1),
		num_proc=args.num_proc,
		desc="Filtering short reference examples",
	)
	reference_dataset = reference_dataset.map(
		lambda example: {"input_ids": example["input_ids"][1:-1]},
		num_proc=args.num_proc,
		desc="Trimming reference boundary tokens",
	)

	unique_reference_ids = {tuple(ids) for ids in reference_dataset["input_ids"]}
	ref_kgrams, masked_map = build_reference_indices(
		unique_reference_ids,
		args.min_k,
		args.max_k,
		args.approx_max_k,
		args.allow_one_token_mismatch,
	)

	total_kgrams = sum(len(kgrams) for kgrams in ref_kgrams.values())
	logger.info(
		"Built reference k-grams: lengths=%s, total_kgrams=%s",
		sorted(ref_kgrams.keys()),
		f"{total_kgrams:,}",
	)
	if args.allow_one_token_mismatch:
		total_masks = sum(len(masked_entries) for masked_entries in masked_map.values())
		logger.info(
			"Built masked entries for 1-token mismatch (approx_max_k=%s): total_masks=%s",
			args.approx_max_k,
			f"{total_masks:,}",
		)

	cleaned = dataset.filter(
		create_clean_filter(
			ref_kgrams,
			masked_map,
			args.approx_max_k,
			args.allow_one_token_mismatch,
		),
		batched=True,
		batch_size=args.batch_size,
		num_proc=args.num_proc,
		desc="Removing contaminated examples",
	)

	original_samples = len(dataset)
	cleaned_samples = len(cleaned)
	removed_samples = original_samples - cleaned_samples
	token_count = int(sum(len(ids) for ids in cleaned["input_ids"]))

	logger.info("Original size: %s", f"{original_samples:,}")
	logger.info("Cleaned size: %s", f"{cleaned_samples:,}")
	logger.info("Removed contaminated examples: %s", f"{removed_samples:,}")

	input_shards = list_matching_files(args.input_dir, "*.jsonl", "*.json", "*.parquet")
	n_chunks = save_dataset(
		cleaned,
		args.output_dir,
		args.output_type,
		tokens_per_chunk=None,
		token_count=None,
		n_chunks=max(1, len(input_shards)),
	)

	save_metadata(
		args.output_dir,
		samples=cleaned_samples,
		tokens=token_count,
		original_samples=original_samples,
		removed_samples=removed_samples,
		chunks=n_chunks,
		tokens_per_chunk=token_count // max(n_chunks, 1) if n_chunks else 0,
		min_k=args.min_k,
		max_k=args.max_k,
		allow_one_token_mismatch=args.allow_one_token_mismatch,
		approx_max_k=args.approx_max_k,
		output_type=args.output_type,
		reference_path=args.reference_path,
	)

	logger.info("Decontamination complete.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
        description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

	parser.add_argument(
		"--input_dir",
		type=str,
		required=True,
		help="Directory containing the contaminated dataset shards (.jsonl, .json, or .parquet).",
	)
	parser.add_argument(
		"--reference_path",
		type=str,
		required=True,
		help="Reference dataset: a local file, local directory, or HuggingFace dataset id.",
	)
	parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for datasets.")
	parser.add_argument("--num_proc", type=int, default=8, help="Number of processes for dataset operations.")
	parser.add_argument(
		"--batch_size",
		type=int,
		default=10000,
		help="Batch size for batched filter/map operations. Increasing reduces Python overhead.",
	)
	parser.add_argument("--output_dir", type=str, required=True, help="Output directory for cleaned shards.")
	parser.add_argument(
		"--output_type",
		type=str,
		default="jsonl",
		choices=["jsonl", "parquet"],
		help="Output shard format.",
	)
	parser.add_argument("--min_k", type=int, default=8, help="Minimum token span considered a match.")
	parser.add_argument(
		"--max_k",
		type=int,
		default=32,
		help="Maximum token span to index from references. Smaller is faster.",
	)
	parser.add_argument(
		"--allow_one_token_mismatch",
		action="store_true",
		help="Allow detection of matches with a single token substituted.",
	)
	parser.add_argument(
		"--approx_max_k",
		type=int,
		default=10,
		help="Maximum k for masked one-token-mismatch matching. Smaller is faster.",
	)

	args = parser.parse_args()
	if args.min_k < 1:
		raise ValueError("--min_k must be at least 1.")
	if args.max_k < args.min_k:
		raise ValueError("--max_k must be greater than or equal to --min_k.")
	if args.approx_max_k < args.min_k:
		raise ValueError("--approx_max_k must be greater than or equal to --min_k.")

	logger.info("Starting decontamination.")
	main(args)