# Agents

Agent-based trace generation using [smolagents](https://github.com/huggingface/smolagents). This folder contains scripts and tools to run `CodeAgent` instances against a dataset of prompts, recording multi-step reasoning traces (thoughts, code executions, observations, and final answers) for downstream training or evaluation.

## Contents

- [`data/`](./data/) — Sample input datasets for trace generation.
- [`prompts/`](./prompts/) — YAML system-prompt templates for the CodeAgent.
- [`/slurm`](./slurm) — Folder containing SLURM job scripts for cluster-managed environments. Before submitting, update the scripts with your cluster-specific settings and correct paths for your artifacts/workspace. **These are templates, not ready-to-run scripts.**
- [`generate_agent_traces.py`](./generate_agent_traces.py) — Main entry point for generating CodeAgent execution traces from a dataset.
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

# Or, via the SLURM batch script:
sbatch generate_agent_traces.sh
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
- `--language`: Language code (`en` or `pt`) that configures the whole system — system prompt, Wikipedia language, DuckDuckGo search region, and formatter translations. Defaults to `en`. Extend via `LANGUAGE_CONFIGS` in `utils.py` plus a matching `prompts/<language>.yaml`.
- `--temperature`: Sampling temperature (optional).
- `--top-p`: Nucleus sampling top-p (optional).
- `--top-k`: Top-k sampling cutoff (optional).
- `--api-key`: API key for LiteLLM models (optional, also read from `.env`).
- `--api-base`: API base URL for LiteLLM (optional, also read from `.env`).
- `--system-prompt-file`: Optional override — path to custom YAML prompt templates. If unset, the bundled `prompts/<language>.yaml` for the selected `--language` is used.
- `--enable-thinking`: Enable thinking/reasoning mode (vLLM/Transformers only).
- `--enable-planning`: Run a planning step before agent execution.
- `--code-block-opening-tag` / `--code-block-closing-tag`: Custom code-block delimiters (default `<code>` / `</code>`).
- `--save-raw-traces`: Also save raw traces (full trace records). Formatted conversation traces are always saved.
- `--max-entries-per-file`: Max JSON objects per consolidated output file before rotation (default 50,000).
- `--no-resume`: Disable auto-resume (always start fresh).

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

The `OutputManager` (in [`utils.py`](./utils.py)) always saves formatted traces, and optionally saves raw traces under the output directory:

- **Formatted traces** (`formatted_traces/formatted_traces_00001.json`, ...): Conversation-formatted traces (messages array) compatible with TRL-style chat training, saved for successful runs. Traces with unclosed `<think>` tags are automatically discarded as malformed.
- **Raw traces** (`raw_traces/raw_traces_00001.json`, ...): Full trace records including all action steps, system prompts, metadata, errors, and timing information—stored as proper JSON arrays (`[{...}, {...}]`). Saved only with `--save-raw-traces`.

Files rotate automatically once they reach the `--max-entries-per-file` threshold (default 50,000 entries), and the manager supports auto-resume by scanning existing files on startup.

## Prompt Templates

The [`prompts/`](./prompts/) folder contains YAML system-prompt templates for the CodeAgent:

- [`en.yaml`](./prompts/en.yaml) — English system prompt (default).
- [`pt.yaml`](./prompts/pt.yaml) — Portuguese (Português) system prompt.

These files use Jinja2 placeholders (`{{tools}}`, `{{authorized_imports}}`, `{{code_block_opening_tag}}`, etc.) that are populated at runtime by smolagents. They are selected automatically by `--language` (CLI) or `LANGUAGE` (shell script). An explicit `--system-prompt-file` / `SYSTEM_PROMPT_FILE` overrides the language selection.

## Data Assets

The [`data/`](./data/) folder includes:

- [`sample.jsonl`](./data/sample.jsonl) — Example input dataset with `id`, `prompt`, and `ground_truth` fields. Format your own datasets similarly for trace generation. Each line should be a JSON object with at least a `prompt` field. Optional fields include `id` (unique identifier) and `ground_truth` (optional).

## Notes

- The current version of smolagents is `1.26.0`. This version had some bugs that we have patched in our [`patches.py`](patches.py) module of this folder. If you are using smolagents 1.26.0 or later, please ensure that you apply the patches in `patches.py` to avoid the certain issues (e.g., timeout handling, sampling parameter handling, etc.) that were present in the original smolagents release. The patches are applied automatically when you lunch `generate_agent_traces.py`, so you don't need to worry about it.
