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
- JSONL files with an "ids" field containing tokenized sequences (List[int])
- Both input and reference datasets must have this format

Output:
- Cleaned JSONL files with contaminated examples removed
- "ids" and "text" fields are dropped from output
- Files split into chunks matching input file count

Usage:
    python decontaminate.py --input_pattern "data/*.jsonl" \
        --reference_files eval_set.jsonl test_set.jsonl \
        --output_dir cleaned_data \
        --min_k 8 --max_k 32 --allow_one_token_mismatch
"""
import datasets
import glob
from collections import defaultdict
import argparse
import numpy as np
import os

def main(args):
	# Parse arguments
	cache_dir = args.cache_dir
	files = glob.glob(args.input_pattern)
	references = args.reference_files
	num_proc = args.num_proc
	output_dir = args.output_dir
	min_k = args.min_k
	max_k = args.max_k
	allow_one_token_mismatch = args.allow_one_token_mismatch
	approx_max_k = args.approx_max_k
	batch_size = args.batch_size

	# If num_proc not provided, use available CPUs minus one to avoid overcommit
	if num_proc is None or num_proc < 1:
		cpus = os.cpu_count() or 1
		num_proc = max(1, cpus - 1)

	# Load dataset (to be cleaned)
	# This dataset should have an "ids" field with token ids.
	# These ids will be checked against reference k-grams.
	dataset = datasets.load_dataset(
		"json",
		data_files=files,
		cache_dir=cache_dir,
		split="train",
	)

	# Load & preprocess reference dataset
	# This dataset should have an "ids" field with token ids.
	# These ids will be the source of k-grams for decontamination.
	ref_ds = datasets.load_dataset(
		"json",
		data_files=references,
		cache_dir=cache_dir,
		split="train",
	)

	# Keep those with at least min_k+2 tokens (after trimming)
	ref_ds = ref_ds.filter(lambda ex: len(ex["ids"]) > (min_k + 1))

	# Trim first and last token from ids
	def trim_ids(example):
		example["ids"] = example["ids"][1:-1]
		return example

	ref_ds = ref_ds.map(trim_ids, num_proc=num_proc)

	# Turn into list of lists: List[List[int]]
	reference_ids_list = list(ref_ds["ids"])

	# Deduplicate reference examples to avoid redundant indexing (can hugely cut work!)
	if reference_ids_list:
		unique_tuples = set(tuple(ids) for ids in reference_ids_list)
		reference_ids_list = [list(t) for t in unique_tuples]

	# Build k-gram lookup for fast membership checks
	# We'll create:
	#  - ref_kgrams: length -> set(of tuples of token ids) for all k-grams with k in [min_k, max_k]
	#  - masked_map (optional): length -> dict(mask_tuple -> True), where mask_tuple has one position set to None
	ref_kgrams = defaultdict(set)
	masked_map = defaultdict(dict) if allow_one_token_mismatch else None

	for ids in reference_ids_list:
		Ltot = len(ids)
		if Ltot == 0:
			continue
		# generate all contiguous k-grams for k in [min_k, min(max_k, Ltot)]
		local_max_k = min(max_k, Ltot)
		for k in range(min_k, local_max_k + 1):
			for i in range(Ltot - k + 1):
				kg = tuple(ids[i : i + k])
				ref_kgrams[k].add(kg)
				if allow_one_token_mismatch and k <= approx_max_k:
					# build masked entries (one masked position) for fast 1-token substitution check
					# store masked tuple with None at masked position
					for j in range(k):
						m = list(kg)
						m[j] = None
						m_t = tuple(m)
						masked_map[k][m_t] = True

	# Quick diagnostics for logging
	total_kgrams = sum(len(s) for s in ref_kgrams.values())
	print(f"Built reference k-grams: lengths={sorted(ref_kgrams.keys())}, total_kgrams={total_kgrams}")
	if allow_one_token_mismatch:
		total_masks = sum(len(d) for d in masked_map.values())
		print(f"Built masked entries for 1-token mismatch (approx_max_k={approx_max_k}): total_masks={total_masks}")

	# Decontamination method: contiguous subsequence match
	# For each example in the dataset, check if any contiguous subsequence of length k in [min_k, max_k]
	# matches any of the reference k-grams. If so, mark as contaminated.
	def is_clean_batch_contiguous(batch):
		out = []
		ids_batch = batch["ids"]
		if not ref_kgrams:
			# no references -> everything is clean
			return [True] * len(ids_batch)

		lengths = sorted(ref_kgrams.keys())
		min_ref_k = lengths[0]

		for ids in ids_batch:
			ids_len = len(ids)
			contaminated = False
			# Skip if example shorter than the smallest considered k
			if ids_len < min_ref_k:
				out.append(True)
				continue

			# For each candidate reference length L, slide a window
			for L in lengths:
				if L > ids_len:
					break
				s = ref_kgrams[L]
				# Slide window
				for i in range(ids_len - L + 1):
					window = tuple(ids[i : i + L])
					if window in s:
						contaminated = True
						break
					# Check single-token substitution via masked map
					if allow_one_token_mismatch and L <= approx_max_k:
						# generate masks for the window and test membership
						found_mask = False
						for j in range(L):
							m = list(window)
							m[j] = None
							if tuple(m) in masked_map[L]:
								contaminated = True
								found_mask = True
								break
						if found_mask:
							break
				if contaminated:
					break
			out.append(not contaminated)
		return out

	# Use configured num_proc and batch_size
	cleaned = dataset.filter(
		is_clean_batch_contiguous,
		batched=True,
		batch_size=batch_size,
		num_proc=num_proc
	)

	# Remove the "ids" and "text" fields before saving
	cleaned = cleaned.remove_columns(["ids", "text"])

	print("Original size:", len(dataset))
	print("Cleaned size:", len(cleaned))

	indices = np.array_split(np.arange(len(cleaned)), len(files))
	chunks = [cleaned.select(idx) for idx in indices]
	os.makedirs(output_dir, exist_ok=True)
	for i, chunk in enumerate(chunks):
		chunk.to_json(f"{output_dir}/train-{i:05d}-of-{len(chunks):05d}.jsonl")

if __name__ == "__main__":
	parser = argparse.ArgumentParser("Decontaminate JSONL dataset using contiguous k-token matching against reference datasets.")
	parser.add_argument("--input_pattern", type=str, required=True, help="Glob pattern for input contaminated JSONL files.")
	parser.add_argument("--reference_files", type=str, nargs='+', required=True, help="List of reference JSONL files.")
	parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for datasets.")
	parser.add_argument("--num_proc", type=int, default=None, help="Number of processes for dataset operations (default: auto-detect).")
	parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for batched filter/map operations (default 10000). Increasing reduces Python overhead.")
	parser.add_argument("--output_dir", type=str, required=True, help="Output directory for cleaned JSONL files.")
	parser.add_argument("--min_k", type=int, default=8, help="Minimum contiguous token length to consider a match (default 8).")
	parser.add_argument("--max_k", type=int, default=32, help="Maximum contiguous token length to index from references (default 32). Smaller -> faster.")
	parser.add_argument("--allow_one_token_mismatch", action="store_true", help="Allow detection of matches with a single token substituted (approximate match).")
	parser.add_argument("--approx_max_k", type=int, default=10, help="Maximum k length for which to build masked entries to detect single-token mismatches (default 10). Smaller -> faster.")

	args = parser.parse_args()

	print("Starting decontamination! 🧹")
	main(args)
	print("Decontamination complete! ✅")