# Gym

Generation, validation, and verification utilities for gym-based (Reinforcement Learning-style) training and evaluation pipelines. This folder contains scripts to generate instruction, long-context, math, email, and tool-call samples from templates, as well as verifiers for validating sample correctness and formatting.

These generators are very lightweight and do not require significant compute to run. You can use them to generate thousands of samples in a few minutes on a single CPU. The generated samples are intended for use in training and evaluation pipelines, and are formatted to work with our GRPO implementation (see [`alignment/`](../)).

## Contents

- [`assets/`](./assets/) — Input assets for generation workflows, including documents, email data, and tool definitions.
- [`generate_from_instruction_templates.py`](./generate_from_instruction_templates.py) — Builds instruction-following samples from template definitions.
- [`generate_from_long_context_templates.py`](./generate_from_long_context_templates.py) — Creates long-context and haystack-style samples from local documents.
- [`generate_from_math_dataset.py`](./generate_from_math_dataset.py) — Builds math answer/check samples from a math dataset.
- [`generate_from_email_templates.py`](./generate_from_email_templates.py) — Builds structured email extraction samples.
- [`generate_from_tool_call_templates.py`](./generate_from_tool_call_templates.py) — Builds tool-call and tool-usage samples.
- [`instruction_templates.py`](./instruction_templates.py) — Instruction-task templates used by the instruction generator.
- [`long_context_templates.py`](./long_context_templates.py) — Long-context and haystack task templates used by the long-context generator.
- [`tasks_metadata.py`](./tasks_metadata.py) — Verifier IDs, compatibility rules, kwargs templates, and helper utilities.
- [`verifier.py`](./verifier.py) — Verifier entry point and registry usage.
- [`verifiers.py`](./verifiers.py) — Concrete verifier implementations.
- [`utils.py`](./utils.py) — Shared helper functions used across gym scripts.

## Usage Summary

### `generate_from_instruction_templates.py`
Generate instruction-following examples from template definitions.

Example:
```bash
python generate_from_instruction_templates.py \
  --output_file ./outputs/instruction_samples.json \
  --num_samples 1000 \
  --min_modifiers 1 \
  --max_modifiers 4 \
  --seed 123 \
  --verbose
```

Main parameters:
- `--output_file`: output JSON file path.
- `--num_samples`: total number of samples to generate.
- `--seed`: random seed for reproducible generation.
- `--min_modifiers`: minimum number of instruction modifiers per sample.
- `--max_modifiers`: maximum number of instruction modifiers per sample.
- `--verbose`: print detailed validation warnings.

### `generate_from_long_context_templates.py`
Generate long-context and haystack-style samples.

Example:
```bash
python generate_from_long_context_templates.py \
  --output_file ./outputs/long_context_samples.json \
  --num_samples 200 \
  --max_seq_length 1024 2048 \
  --tokenizer gpt2 \
  --docs_dir ./assets \
  --task_types haystack question_answer \
  --seed 42 \
  --verbose
```

Main parameters:
- `--output_file`: output JSON or JSONL file path.
- `--num_samples`: number of samples to generate per task type.
- `--max_seq_length`: target token lengths for generated haystack tasks.
- `--tokenizer`: Hugging Face tokenizer name or path (required for haystack tasks).
- `--num_context_words`: approximate number of words in generated word-list tasks.
- `--docs_dir`: directory containing `.txt` documents (default [`./assets`](./assets)).
- `--task_types`: restrict generation to specific task types.
- `--seed`: random seed for reproducibility.
- `--verbose`: print detailed validation info.
- `--cache_dir`: Hugging Face cache directory for tokenizer models.

### `generate_from_math_dataset.py`
Build math problems and answer/check samples.

Example:
```bash
python generate_from_math_dataset.py \
  --output_file ./outputs/math_samples.json \
  --num_samples 1000 \
  --num_synthetic 100 \
  --seed 42 \
  --verbose
```

Main parameters:
- `--output_file`: output JSON or JSONL file path.
- `--num_samples`: number of dataset samples from [`assets/math-problems.jsonl`](./assets/math-problems.jsonl).
- `--num_synthetic`: number of synthetic math problems to generate.
- `--seed`: random seed for reproducibility.
- `--verbose`: print detailed validation warnings.

### `generate_from_email_templates.py`
Create structured email extraction samples.

Example:
```bash
python generate_from_email_templates.py \
  --emails_file ./assets/emails.jsonl \
  --output_file ./outputs/email_samples.json \
  --num_samples 300 \
  --min_fields 5 \
  --max_fields 10 \
  --seed 42 \
  --verbose
```

Main parameters:
- `--emails_file`: path to the source JSONL emails file (one email per line with an `email` field).
- `--output_file`: output JSON or JSONL file path.
- `--num_samples`: number of samples to generate.
- `--min_fields`: minimum number of JSON fields requested per sample.
- `--max_fields`: maximum number of JSON fields requested per sample.
- `--seed`: random seed for reproducibility.
- `--verbose`: print per-sample validation warnings.

### `generate_from_tool_call_templates.py`
Generate tool-call examples, including refusal samples.

Example:
```bash
python generate_from_tool_call_templates.py \
  --output_file ./outputs/tool_call_samples.json \
  --num_samples 500 \
  --min_tools 1 \
  --max_tools 3 \
  --refusal_ratio 0.5 \
  --seed 42 \
  --verbose
```

Main parameters:
- `--output_file`: output JSON or JSONL file path.
- `--num_samples`: total number of samples to generate.
- `--seed`: random seed for reproducibility.
- `--min_tools`: minimum number of tools included in each prompt.
- `--max_tools`: maximum number of tools included in each prompt.
- `--refusal_ratio`: fraction of samples that should be refusal tasks.
- `--min_refusal_words`: minimum number of words expected in a refusal response.
- `--data_file`: optional tools file path (defaults to [`assets/tools.json`](./assets/tools.json)).
- `--verbose`: print detailed validation warnings.

## Data Assets

The [`assets/`](./assets/) folder includes:

- [`doc-0.txt ... doc-22.txt`](./assets/) — Long-context source documents.
- [`emails.jsonl`](./assets/emails.jsonl) — Email examples for structured extraction tasks.
- [`tools.json`](./assets/tools.json) — Tool definitions and examples for tool-call tasks.
- [`math-problems.jsonl`](./assets/math-problems.jsonl) — Math problems and solutions for math task generation.

## Notes

- The whole gym is designed to create samples in Portuguese. In the future, we plan to modularize the generation scripts to support multiple languages.
- Long-context and haystack generation use local documents in [`assets/`](./assets/).
- `generate_from_email_templates.py` requires [`assets/emails.jsonl`](./assets/emails.jsonl).
- `generate_from_tool_call_templates.py` expects [`assets/tools.json`](./assets/tools.json) by default.
- Verification logic is centralized in [`tasks_metadata.py`](./tasks_metadata.py), [`verifier.py`](./verifier.py), and [`verifiers.py`](./verifiers.py).
