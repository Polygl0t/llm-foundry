# Evaluation

Evaluation scripts for running language model evaluations on multiple language benchmarks using the Language Model Evaluation Harness.

> **Run evaluations on Colab for free 🤗** <a href="https://colab.research.google.com/drive/1A37MAJ9SU3bMukdLgOW-uNd4v8S_juIT" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

## Contents

- [`eval_harness_bn.sh`](eval_harness_bn.sh) — SLURM job submission script for evaluating models on Bengali language benchmarks.
- [`eval_harness_hi.sh`](eval_harness_hi.sh) — SLURM job submission script for evaluating models on Hindi language benchmarks.
- [`eval_harness_pt.sh`](eval_harness_pt.sh) — SLURM job submission script for evaluating models on Portuguese language benchmarks.
- [`eval_parallel.sh`](eval_parallel.sh) — generic SLURM job submission script for evaluating multiple models in parallel on a list of tasks.
- [`eval_sequential.sh`](eval_sequential.sh) — generic SLURM job submission script for evaluating multiple models sequentially on a list of tasks.

## Usage Summary

### `eval_parallel.sh`

Evaluates multiple models in parallel, one per GPU (up to `MAX_GPUS_PER_NODE`). Each model's raw JSON output is post-processed into a `results-{timestamp}.yaml` file saved inside the model's folder.

Example:
```bash
sbatch eval_parallel.sh
```

Configure the following in the script before submission:
- `--partition` — Target GPU partition
- `--gpus` — Number of GPUs to allocate
- `MODELS` — Bash array of local model folder paths to evaluate
- `TASKS` — Comma-separated list of benchmark tasks
- `NUM_FEWSHOT` — Number of few-shot examples (typically 0, 5, or 10)
- `BATCH_SIZE` — Batch size (`"auto"` or an integer)
- `MODE` — Inference backend (`hf` for HuggingFace, `vllm` for vLLM)
- `MAX_GPUS_PER_NODE` — Maximum concurrent evaluation jobs (one GPU each)

### `eval_sequential.sh`

Evaluates models one at a time on a single GPU, looping through a list of local model folders. Results are saved identically as YAML in each model's folder.

Example:
```bash
sbatch eval_sequential.sh
```

Configuration options are the same as `eval_parallel.sh`, except `MAX_GPUS_PER_NODE` is not applicable (only one GPU is used).

## Notes

- The evaluation harness uses EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework to benchmark model performance.
- Results are post-processed from JSON to YAML and saved as `*.yaml` files for readability reasons.
- Ensure model folders exist and are accessible before submission.
- GPU memory and runtime requirements depend on model size and task complexity; adjust `--gpus`, `--mem`, and `--time` accordingly.
