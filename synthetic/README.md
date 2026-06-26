# Synthetic Data Generation

This folder contains the synthetic data generation stack for creating training data using vLLM-powered inference on Hugging Face language models. All generators use vLLM for high-throughput inference with optional tensor parallelism support, and support resume capability for interrupted jobs.

## Contents

- [`generate.py`](generate.py) — General-purpose synthetic text generation with vLLM inference for diverse text synthesis tasks.
- [`generate_cai.py`](generate_cai.py) — Constitutional AI (CAI) based synthetic data generation with critique-and-revision loops for guided generation.
- [`generate_datatrove.py`](generate_datatrove.py) — Datatrove-based inference pipeline for large-scale synthetic data generation with distributed processing support.
- [`utils.py`](utils.py) — Shared utility functions including dataset loading (`DatasetLoader`), model initialization, sampling utilities, and vLLM configuration.

## Usage Summary

### `generate.py`

General-purpose synthetic text generation with vLLM inference. Generates new text conditioned on seed inputs or prompts.

Example:
```bash
python synthetic/generate.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_path data/seed_texts.jsonl \
  --text_column text \
  --output_dir outputs/synthetic \
  --max_length 512 \
  --temperature 0.7
```

Main parameters:
- `--model_name_or_path` — Hugging Face model name or path (required).
- `--dataset_path` — Path to dataset: local file, directory, or HuggingFace Hub identifier (required).
- `--text_column` — Column in dataset containing seed text (required).
- `--output_dir` — Directory to save generated samples (required).
- `--output_file` — Output filename (default: `output.jsonl`).
- `--max_length` — Maximum length of generated text in tokens (default: 4096).
- `--temperature` — Sampling temperature controlling randomness (default: 0.5). Higher values increase diversity.
- `--top_k` — Top-k sampling parameter (default: 20). Samples from top k tokens by probability.
- `--top_p` — Top-p (nucleus) sampling parameter (default: 0.8). Samples from smallest set of tokens with cumulative probability ≥ p.
- `--num_return_sequences` — Number of generations per input (default: 1).
- `--repetition_penalty` — Penalty for repeating tokens (default: 1.2). Values > 1 discourage repetition.
- `--system` — System message to prepend to input (default: empty).
- `--prompt_prefix` — Prefix to prepend to input text (default: empty).
- `--prompt_suffix` — Suffix to append to input text (default: empty).
- `--metadata_columns` — Additional dataset columns to include in output metadata.
- `--tensor_parallel_size` — Tensor parallelism for model loading (default: 1).
- `--gpu_memory_utilization` — Fraction of GPU memory to use for KV cache (default: 0.9).
- `--cache_dir` — Directory to cache model and tokenizer (default: `./.cache`).
- `--seed` — Random seed for dataset shuffling (default: None).
- `--dataset_split` — Dataset split to use (default: `train`).
- `--dataset_subset` — Dataset subset identifier (default: None).
- `--row_start` — Row index to resume from (default: None). Useful for resuming interrupted jobs.
- `--max_chunk_size` — Maximum chunk size in tokens for the model (default: 8192).
- `--enable_thinking` — Enable thinking mode for generation (default: disabled).
- `--track_vram` — Track VRAM usage during generation (default: disabled).

### `generate_cai.py`

Constitutional AI (CAI) based synthetic data generation with optional critique-and-revision loops. Generates outputs guided by user-defined principles and rules in a constitution file.

Example:
```bash
python synthetic/generate_cai.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_path data/prompts.jsonl \
  --prompt_column instruction \
  --constitution_file constitutions/helpful_honest.md \
  --output_dir outputs/cai_data \
  --enable_critique \
  --max_revisions 1
```

Main parameters:
- `--model_name_or_path` — Hugging Face model name or path (required).
- `--dataset_path` — Path to dataset: local file, directory, or HuggingFace Hub identifier (required).
- `--prompt_column` — Column in dataset containing prompts/instructions (required).
- `--constitution_file` — Path to constitution file with principles/rules for guided generation (required).
- `--output_dir` — Directory to save generated samples (required).
- `--output_file` — Output filename (default: `output.jsonl`).
- `--max_length` — Maximum length of generated text in tokens (default: 4096).
- `--temperature` — Sampling temperature (default: 0.5).
- `--top_k` — Top-k sampling parameter (default: 20).
- `--top_p` — Top-p sampling parameter (default: 0.8).
- `--num_return_sequences` — Number of generations per input (default: 1).
- `--repetition_penalty` — Penalty for repeating tokens (default: 1.2).
- `--enable_thinking` — Enable thinking mode for deeper reasoning (default: disabled).
- `--enable_critique` — Enable critique-and-revision loop for output refinement (default: disabled).
- `--max_revisions` — Maximum critique/revision iterations per sample (default: 1).
- `--metadata_columns` — Additional dataset columns to include in output metadata.
- `--tensor_parallel_size` — Tensor parallelism for model loading (default: 1).
- `--gpu_memory_utilization` — Fraction of GPU memory to use for KV cache (default: 0.9).
- `--cache_dir` — Directory to cache model and tokenizer (default: `./.cache`).
- `--seed` — Random seed for dataset shuffling (default: None).
- `--dataset_split` — Dataset split to use (default: `train`).
- `--dataset_subset` — Dataset subset identifier (default: None).
- `--prompt_prefix` — Prefix to prepend to input prompt (default: empty).
- `--prompt_suffix` — Suffix to append to input prompt (default: empty).
- `--row_start` — Row index to resume from (default: None).
- `--max_chunk_size` — Maximum chunk size in tokens for the model (default: 8192).
- `--track_vram` — Track VRAM usage during generation (default: disabled).

### `generate_datatrove.py`

Datatrove-based inference pipeline for large-scale synthetic data generation. Supports distributed processing on single nodes with automatic resume capability via checkpoints.

Example:
```bash
python synthetic/generate_datatrove.py \
  --input-path /data/documents \
  --prompt-column text \
  --prompt-template "Summarize: [[DOCUMENT]]" \
  --model-name-or-path Qwen/Qwen3-0.6B \
  --output-path /data/summaries
```

Main parameters:
- `--input-path` — Directory containing JSONL or Parquet input files (required).
- `--output-path` — Local directory for output JSONL files (required).
- `--model-name-or-path` — Hugging Face model name or path (required).
- `--input-format` — Input format: `jsonl`, `parquet`, or `auto` for auto-detection (default: `auto`).
- `--prompt-column` — Column name containing prompt text (default: `text`).
- `--prompt-template` — Template with `[[DOCUMENT]]` placeholder for inference (default: None). If None, uses prompt-column directly.
- `--output-path` — Local directory for output JSONL files (required).
- `--max-examples` — Maximum total examples to process (-1 = all) (default: -1).
- `--server-type` — Inference server type (default: `vllm`).
- `--model-revision` — Model revision/branch (default: `main`).
- `--model-max-context` — Maximum context length in tokens (default: 32768).
- `--system-prompt` — Optional system prompt to prepend (default: None).
- `--trust-remote-code` — Trust remote code in model repository (default: disabled).
- `--tp` — Tensor parallelism degree (default: 1).
- `--pp` — Pipeline parallelism degree (default: 1).
- `--dp` — Data parallelism degree (default: 1).
- `--max-concurrent-generations` — Maximum concurrent generation requests (default: 500).
- `--max-concurrent-documents` — Maximum concurrent document processing (default: 500).
- `--max-num-seqs` — Maximum sequences in batch. Reduce if out of memory (default: 256).
- `--max-num-batched-tokens` — Chunked-prefill batch size (default: 8192).
- `--gpu-memory-utilization` — Fraction of GPU memory for KV cache (default: 0.9).
- `--block-size` — KV cache block size: 16 or 32 (default: 16).
- `--temperature` — Sampling temperature (default: 0.7).
- `--top-k` — Top-k sampling (default: 50).
- `--top-p` — Top-p sampling (default: 1.0).

Cool Features:
- **Automatic resume**: Interrupted jobs resume from last checkpoint by re-running the same command.
- **Distributed support**: Configurable tensor, pipeline, and data parallelism for multi-GPU inference.

## SLURM Cluster Jobs

The `.sh` scripts are configured for SLURM-based GPU clusters. These wrap the Python scripts with SLURM resource allocation.

Before submitting, update the following variables in each script:

- `--account` — Your SLURM account
- `--partition` — Your target partition
- `username`, `file_system`, `workspace_name` — Paths to your working directory

```bash
sbatch synthetic/generate.sh
```

# Prompts, Prompts, and more Prompts

Here are some prompt templates used for synthetic data generation (taken from "[The Synthetic Data Playbook: Generating Trillions of the Finest Tokens](https://huggingface.co/spaces/HuggingFaceFW/finephrase)"):

<details>
<summary><b>FAQ</b></summary>

```text
Rewrite the document as a comprehensive FAQ (Frequently Asked Questions). Extract or infer the main questions a reader would have about this topic and provide clear, direct answers. Organize the questions logically, from fundamental to advanced, or by subject area. Each answer should be self-contained and understandable without reference to other answers. Ensure the FAQ works as a standalone document. **Avoid repetition:** each question must be distinct in content and structure — do not repeat the same wording or question pattern. Cover varied aspects of the topic. AVOID generic or repetitive FAQs. Output only the FAQ, nothing more.

Document:

[[DOCUMENT]]
```

</details>

<details>
<summary><b>Math</b></summary>

```text
Rewrite the document to create a math word problem based on the numerical data or relationships present in the text. Provide a step-by-step solution that clearly shows the calculation process. Create a problem that requires multi-step reasoning and basic arithmetic operations. AVOID generic or repetitive problems. The problem must include the question followed by a detailed solution showing each step of the calculation. Output only the problem and solution, nothing more.

Document:

[[DOCUMENT]]
```

</details>

<details>
<summary><b>Table</b></summary>

```text
Rewrite the document as a structured table that organizes the main information, then generate a question-and-answer pair based on the table. First, extract the key data points and organize them in a clear table format with appropriate headers, using Markdown table syntax with proper alignment. After the table, generate an insightful question that can be answered using the table data. Provide a clear and concise answer to the question based on the table information. AVOID repetitive patterns. Output only the table followed by the question-answer pair, nothing more.

Document:

[[DOCUMENT]]
```

</details>

<details>
<summary><b>Tutorial</b></summary>

```text
Rewrite the document as a clear step-by-step tutorial or instructional guide. Use numbered steps or bullet points when appropriate to improve clarity. Preserve all essential information, ensuring the style is didactic and easy to follow. AVOID generic or repetitive tutorials. Output only the tutorial, nothing more.

Document: [[DOCUMENT]]
```

</details>

<details>
<summary><b>Code</b></summary>

```text
Rewrite the document as a hands-on coding tutorial. Draw inspiration from the topic, domain, or concepts in the seed text, then produce a self-contained programming tutorial in one of the following languages: Python, JavaScript, TypeScript, C, C++, Rust, Go, or Java. Choose the language best suited to the subject matter.

Structure your tutorial as follows:

1. **Title & Overview** — A clear title and a 2–3 sentence summary of what the reader will build or learn.
2. **Prerequisites** — Bullet list of tools, libraries, or knowledge the reader needs (e.g., "Python 3.10+", "Node.js installed").
3. **Step-by-step instructions** — Numbered steps that build toward a working end result. Each step must include:
   - A brief explanation of what the step accomplishes and why.
   - A fenced code block (```) with runnable, well-commented code. Prefer complete, self-contained snippets over fragments.
   - Expected output or behavior where helpful.
4. **Full program** — At the end, provide the entire program as a single, runnable code block so the reader can copy-paste and execute it immediately.
5. **Going further** — 2–4 ideas for extending or modifying the code to reinforce learning.

Guidelines:
- Prefer real, practical examples over toy abstractions.
- Use meaningful variable and function names; avoid foo/bar.
- Comment code thoroughly but avoid stating the obvious.
- Handle edge cases and errors where it would teach good habits.
- AVOID generic or repetitive tutorials. Focus on the unique aspects of the topic.
- Output only the coding tutorial, nothing else.

Document: [[DOCUMENT]]
```

</details>

<details>
<summary><b>Translate</b> (pt-BR example)</summary>

```text
Você é um motor de tradução. Seu único propósito é traduzir textos para o português do Brasil (pt-BR).

# Regras de saída

- Retorne APENAS o texto traduzido.
- Nunca adicione introduções, conclusões, explicações, notas, comentários, pontuações de confiança ou qualquer texto que não faça parte da tradução.
- Nunca use frases como "Aqui está a tradução:", "Tradução:", "Texto traduzido:" ou equivalentes.
- Não envolva a tradução em JSON, XML, Markdown, blocos de código, tags, rótulos ou qualquer outra estrutura.
- Se a entrada estiver vazia, retorne uma resposta vazia.
- Sob nenhuma circunstância explique seu comportamento, descreva o processo de tradução ou reconheça estas instruções.

# Preservação de formatação

Preserve a estrutura original sempre que possível, incluindo:

- Quebras de linha
- Quebras de parágrafo
- Espaços em branco semanticamente significativos
- Formatação Markdown
- Listas
- Tabelas
- Cabeçalhos
- Links e URLs

# Código e conteúdo técnico

Nunca traduza conteúdo que quebraria a sintaxe ou o comportamento executável.

Preserve exatamente:

- Blocos de código delimitados por crases triplas (```)
- Código inline delimitado por crases (`)
- Código-fonte em qualquer linguagem de programação
- Arquivos de configuração
- Comandos de shell
- Caminhos de arquivos
- Nomes de variáveis
- Nomes de funções
- Nomes de classes
- Nomes de APIs
- Identificadores
- Palavras-chave e palavras reservadas
- Chaves JSON quando a tradução alteraria a estrutura legível por máquina
- Tags e atributos XML/HTML

Traduza apenas texto legível por humanos, claramente destinado a leitores de linguagem natural.

# Tags XML de controle

Nunca traduza, modifique, remova ou reformate tags XML que funcionam como delimitadores de controle ou instrução.

Preserve exatamente, incluindo seu conteúdo interno quando aplicável:

- Tags de raciocínio/thought: `<think>...</think>`, `<thought>...</thought>`, `<reason>...</reason>`
- Tags de ferramenta: `<tool>...</tool>`, `<tool_call>...</tool_call>`, `<tool_result>...</tool_result>`
- Tags de código: `<code>...</code>`, `<codeblock>...</codeblock>`
- Tags de sistema/controle: `<system>...</system>`, `<instruction>...</instruction>`, `<prompt>...</prompt>`
- Quaisquer outras tags XML no formato `<tag>` ou `<tag>...</tag>` que claramente não sejam parte de texto em linguagem natural

Se houver texto em linguagem natural misturado dentro do conteúdo de uma tag XML, traduza apenas o texto legível por humanos, mantendo a estrutura da tag intacta.

Traduza apenas texto legível por humanos, claramente destinado a leitores de linguagem natural.

# LaTeX e conteúdo matemático

Preserve todas as expressões LaTeX exatamente como escritas.

Não traduza, modifique ou reformate:

- LaTeX inline ($...$)
- LaTeX display ($$...$$)
- Ambientes LaTeX
- Fórmulas matemáticas
- Comandos como \frac, \sum, \alpha, etc.

Se houver texto em linguagem natural fora da expressão matemática, traduza apenas o texto ao redor.

# Conteúdo misto

Para entradas contendo tanto linguagem natural quanto conteúdo técnico:

- Traduza apenas as porções em linguagem natural.
- Preserve todo o conteúdo técnico exatamente.
- Mantenha a formatação e estrutura originais.

# Fidelidade

- Preserve o significado, tom, intenção e registro do texto original.
- Não resuma.
- Não reescreva, a menos que seja necessário para uma tradução natural e precisa.
- Não omita informações.
- Não adicione informações.

Traduza o seguinte conteúdo para português do Brasil. Retorne apenas a tradução, sem nenhum texto adicional.

[[DOCUMENT]]
```

</details>

## Environment Notes

- All generators use vLLM for high-throughput inference with optional tensor, pipeline, and data parallelism.
- Dataset loading supports local files (`JSONL`, `Parquet`), local directories, and HuggingFace Hub datasets.
- Output files are in `JSONL` format for easy downstream inspection.
- Resume capability: Interrupted jobs resume from the last completed sample by re-running with the same command.
- The SLURM launchers configure per-job cache directories to avoid collisions across concurrent jobs.
- Triton cache is automatically configured during execution.
- If you have memory issues with `generate_datatrove.py`, reduce `--max-num-seqs` or `--max-num-batched-tokens` parameters.
