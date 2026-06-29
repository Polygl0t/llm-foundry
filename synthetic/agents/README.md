# Agents

Agent-based trace generation using [smolagents](https://github.com/huggingface/smolagents). This folder contains scripts and tools to run `CodeAgent` instances against a dataset of prompts, recording multi-step reasoning traces (thoughts, code executions, observations, and final answers) for downstream training or evaluation.

## Contents

- [`data/`](./data/) — Sample input datasets for trace generation.
- [`prompts/`](./prompts/) — YAML system-prompt templates for the CodeAgent.
- [`generate_agent_traces.py`](./generate_agent_traces.py) — Main entry point for generating CodeAgent execution traces from a dataset.
- [`generate_agent_traces.sh`](./generate_agent_traces.sh) — SLURM batch script for launching distributed trace generation on HPC clusters.
- [`tools.py`](./tools.py) — Custom `smolagents.Tool` implementations for filesystem operations (read, write, edit, list, search, grep) and mathematical problem solving.
- [`utils.py`](./utils.py) — Shared utilities including `TraceRecord`, `OutputManager` (with JSON-array file rotation), trace formatting/validation, and vLLM compatibility patches.

## Usage Summary

### `generate_agent_traces.py`

Run a [CodeAgent](https://huggingface.co/docs/smolagents) over a dataset of prompts and save the resulting traces.

Example:
```bash
python generate_agent_traces.py \
  --model-type vllm \
  --model-id /path/to/model \
  --max-new-tokens 16000 \
  --dataset data/sample.jsonl \
  --prompt-column prompt \
  --max-steps 20 \
  --executor-timeout 120 \
  --output-dir ./traces \
  --language pt \
  --temperature 0.7 \
  --top-p 0.80 \
  --top-k 20
```

Main parameters:
- `--model-type`: Model backend — `litellm` (remote API), `transformers` (local HF), or `vllm` (local vLLM server).
- `--model-id`: Model identifier (HF name, LiteLLM ID, or local path).
- `--max-new-tokens`: Maximum tokens per agent step.
- `--dataset`: Path to input dataset (`.json`, `.jsonl`, or `.parquet`).
- `--prompt-column`: Column name containing the task prompt.
- `--id-column`: Column name for trace IDs (defaults to SHA-256 hash of the prompt).
- `--ground-truth-column`: Column name for ground-truth answers for evaluation.
- `--max-steps`: Maximum agent steps per example.
- `--executor-timeout`: Maximum seconds per tool execution step.
- `--output-dir`: Base directory for saving trace files.
- `--language`: Language code for search tools (`en` or `pt`). You can extend this to other languages. Portuguese serve as an example of "how to do it" for other languages.
- `--temperature`: Sampling temperature (optional).
- `--top-p`: Nucleus sampling top-p (optional).
- `--top-k`: Top-k sampling cutoff (optional).
- `--api-key`: API key for LiteLLM models (optional, also read from `.env`).
- `--api-base`: API base URL for LiteLLM (optional, also read from `.env`).
- `--system-prompt-file`: Path to custom YAML prompt templates (defaults to smolagents built-in).
- `--enable-thinking`: Enable thinking/reasoning mode (vLLM/Transformers only).
- `--enable-planning`: Run a planning step before agent execution.
- `--code-block-opening-tag` / `--code-block-closing-tag`: Custom code-block delimiters (default `<code>` / `</code>`).
- `--disable-formatting`: Skip saving formatted conversation traces (raw traces only).
- `--max-entries-per-file`: Max JSON objects per consolidated output file before rotation (default 50,000).
- `--no-resume`: Disable auto-resume (always start fresh).

### `generate_agent_traces.sh`

SLURM batch script for launching distributed trace generation on HPC clusters. Configures environment variables for model, dataset, agent, and output parameters, then invokes `generate_agent_traces.py`.

Example:
```bash
sbatch generate_agent_traces.sh
```

## Custom Tools

[`tools.py`](./tools.py) provides custom `smolagents.Tool` subclasses that extend the agent's capabilities beyond the built-in defaults:

| Tool                   | Description                                                                                    |
|------------------------|------------------------------------------------------------------------------------------------|
| `ReadFileTool`         | Read file contents with optional line-range limiting.                                          |
| `WriteFileTool`        | Create or overwrite a file, creating parent directories as needed.                             |
| `EditFileTool`         | Replace an exact string in a file (single-occurrence match required).                          |
| `ListDirectoryTool`    | List directory contents with type indicators and file sizes.                                   |
| `SearchFilesTool`      | Find files matching a glob pattern (supports `**` recursion).                                  |
| `GrepFilesTool`        | Search inside files using a regex pattern, returning matches with file, line number, and text. |
| `SympyTool`            | Solve mathematical problems using sympy (equations, calculus, linear algebra, etc.).           |
| `DuckDuckGoSearchTool` | Web search via DuckDuckGo (re-exported from `smolagents.default_tools`).                       |

These tools are loaded via `get_custom_tools()` which returns a list ready for `CodeAgent` initialization. These serve as examples of how to implement custom tools for other domains or tasks. Feel free to extend or modify them as needed for your use case.

## Trace Output

The `OutputManager` (in [`utils.py`](./utils.py)) saves traces in two formats under the output directory:

- **Raw traces** (`raw_traces/raw_traces_00001.json`, ...): Full trace records including all action steps, system prompts, metadata, errors, and timing information—stored as proper JSON arrays (`[{...}, {...}]`).
- **Formatted traces** (`formatted_traces_00001.json`, ...): Conversation-formatted traces (messages array) compatible with TRL-style chat training, saved only for successful runs. Traces with unclosed `<think>` tags are automatically discarded as malformed.

Files rotate automatically once they reach the `--max-entries-per-file` threshold (default 50,000 entries), and the manager supports auto-resume by scanning existing files on startup.

## Prompt Templates

The [`prompts/`](./prompts/) folder contains YAML system-prompt templates for the CodeAgent:

- [`SYSTEM.yaml`](./prompts/SYSTEM.yaml) — Default English system prompt.
- [`SISTEMA.yaml`](./prompts/SISTEMA.yaml) — Portuguese (Português) system prompt.

These files use Jinja2 placeholders (`{{tools}}`, `{{authorized_imports}}`, `{{code_block_opening_tag}}`, etc.) that are populated at runtime by smolagents. Pass them via `--system-prompt-file` (CLI) or `SYSTEM_PROMPT_FILE` (shell script).

## Data Assets

The [`data/`](./data/) folder includes:

- [`sample.jsonl`](./data/sample.jsonl) — Example input dataset with `id`, `prompt`, and `ground_truth` fields. Format your own datasets similarly for trace generation. Each line should be a JSON object with at least a `prompt` field. Optional fields include `id` (unique identifier) and `ground_truth` (optional).

## Notes

- The [`utils.py`](./utils.py) module includes a `_PatchedVLLMModel` class that fixes sampling-parameter handling for vLLM when used with smolagents 1.26.0+.
- A vLLM tokenizer compatibility shim (`_ensure_vllm_tokenizer_compat`) is applied at startup for vLLM ≥ 0.11.
