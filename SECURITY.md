# Security Policy

## Hugging Face Hub, remote artifacts, and remote code

LLM Foundry is an open-source research codebase for pretraining and post-training large language models. Many of its pipelines (data processing, tokenization, training, evaluation, synthetic data generation, and model merging) are tightly coupled to the Hugging Face ecosystem, and therefore download models, datasets, tokenizers, and checkpoints from the Hugging Face Hub and other sources.

When downloading artifacts uploaded by others on any platform, you expose yourself to risks. Please read the security recommendations below to keep your runtime and local environment safe.

### Remote artefacts

Models uploaded on the Hugging Face Hub come in different formats. We heavily recommend uploading and downloading models in the [`safetensors`](https://github.com/huggingface/safetensors) format, which was developed specifically to prevent arbitrary code execution on your system.

To avoid loading models from unsafe formats (e.g., [pickle](https://docs.python.org/3/library/pickle.html)), pass `use_safetensors=True` when loading models with `transformers`. If no `.safetensors` file is present, `transformers` will raise an error instead of silently falling back to an unsafe format.

### Remote code

`transformers` also bridges your Python runtime and models stored in repositories on the Hugging Face Hub, and it can execute code that ships inside a model repository. This is required for some custom architectures, but it also means the remote repository can run arbitrary Python on your machine.

Only set `trust_remote_code=True` after you have read and understood the modeling code you are about to execute, and pin the repository to a specific revision so that later changes to the repository cannot silently change what runs. The same caution applies to any other tool used in this repository that executes remote or repository-supplied code.

## Research software disclaimer

LLM Foundry is research software developed for the Polyglot project at the University of Bonn. It is primarily designed to run on the University of Bonn HPC clusters, is provided "as is", and comes with no security guarantees. Always review the code, datasets, and checkpoints you download or execute before using them on a cluster where you have access to sensitive data or shared resources.

## Reporting a Vulnerability

This is a research project without a dedicated security team. If you believe you have found a security issue, please report it privately to the maintainers at [kluge@uni-bonn.de](mailto:kluge@uni-bonn.de) rather than opening a public issue, so that it can be addressed responsibly. General questions and non-sensitive bug reports are welcome as [GitHub issues](https://github.com/Polygl0t/llm-foundry/issues).
