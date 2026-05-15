"""
Inference Testing

Runs a model on a set of samples and records the results in a JSON file and markdown report.
Supports two modes: chat (for instruction-tuned models) and completion (for base models).

Chat Mode (default):
    Samples must contain a "messages" list following the chat format. The tokenizer's chat
    template is applied before generation. EOS token presence is checked and reported.

    Sample format:
    ```json
    [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What were the main causes of the French Revolution?"
                }
            ],
            "task_type": "History",
            "tool": []
        }
    ]
    ```

    Example usage:
        python inference_test.py \\
            --model_path checkpoints/llama-sft/final \\
            --samples_file samples.json \\
            --output_file results/evaluation.json \\
            --max_new_tokens 1024 \\
            --temperature 0.1

Completion Mode (--completion):
    For base models. Samples must contain a "prompt" string that is fed directly to the
    model without any chat templating. EOS checking is skipped.

    Sample format:
    ```json
    [
        {
            "prompt": "The main causes of the French Revolution were",
            "task_type": "History"
        }
    ]
    ```

    Example usage:
        python inference_test.py \\
            --model_path checkpoints/base-model \\
            --samples_file samples.json \\
            --output_file results/evaluation.json \\
            --max_new_tokens 256 \\
            --temperature 0.7 \\
            --mode completion

Output:
    - JSON file with results
    - Markdown report
"""
import json
import torch
import argparse
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)


def load_samples(samples_file: str) -> List[Dict[str, Any]]:
    """Load samples from a JSON file."""
    
    # Try relative to script directory first, then absolute path
    samples_path = Path(__file__).parent / samples_file
    if not samples_path.exists():
        samples_path = Path(samples_file)
    
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_file}")
    
    print(f"Loading samples from: {samples_path}")
    with open(samples_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    if not isinstance(samples, list):
        raise ValueError("Samples file must contain a JSON array of sample objects.")
    
    print(f"Loaded {len(samples)} samples.")
    return samples


def indent_text(text: str, spaces: int = 4) -> str:
    """Indent text to create a code block in markdown without using backticks."""
    indent = ' ' * spaces
    lines = text.split('\n')
    return '\n'.join(indent + line for line in lines)


def generate_markdown_report(samples: list, output_path: str):
    """Generate a markdown report from inference samples."""
    md_lines = []

    md_lines.append("# Inference Samples Report\n")
    md_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("\n---\n")

    # Summary statistics
    md_lines.append("## Summary Statistics\n")
    total_tokens = sum(s.get('num_generated_tokens', 0) for s in samples)
    has_eos_count = sum(1 for s in samples if s.get('has_eos', False))
    md_lines.append(f"- **Total Samples:** {len(samples)}")
    md_lines.append(f"- **Total Generated Tokens:** {total_tokens}")
    md_lines.append(f"- **Average Tokens per Sample:** {total_tokens / len(samples):.2f}")
    if has_eos_count > 0 or any('has_eos' in s for s in samples):
        md_lines.append(f"- **Samples with EOS:** {has_eos_count} ({has_eos_count/len(samples)*100:.1f}%)")
    md_lines.append("\n---\n")

    md_lines.append("## Samples\n")

    for sample in samples:
        task_type = sample.get('task_type', 'Unknown')
        num_tokens = sample.get('num_generated_tokens', 0)

        md_lines.append(f"### {task_type}\n")
        if 'has_eos' in sample:
            eos_status = "✅ EOS" if sample['has_eos'] else "❌ No EOS"
            md_lines.append(f"**Status:** {eos_status} | **Tokens:** {num_tokens}\n")
        else:
            md_lines.append(f"**Tokens:** {num_tokens}\n")

        prompt = sample.get('prompt', '').replace('#', r'\#').strip()
        md_lines.append("**Prompt:**\n")
        md_lines.append(indent_text(prompt))
        md_lines.append("\n")

        md_lines.append("**Response:**\n")
        md_lines.append(indent_text(sample.get('generated_text', '').strip()))
        md_lines.append("\n")

        md_lines.append("\n---\n")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    return output_path



def main(args):
    # Load samples
    samples = load_samples(args.samples_file)
    
    # Load model and tokenizer
    print(f"\nLoading model: {args.model_path}")
    print(f"Mode: {args.mode}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load external chat template if provided (ensures train-inference consistency)
    if args.mode == "chat":
        if args.chat_template_path is not None:
            print(f"Loading chat template from: {args.chat_template_path}")
            with open(args.chat_template_path, "r") as f:
                tokenizer.chat_template = f.read()
        elif tokenizer.chat_template is None:
            print("WARNING: Tokenizer has no chat_template! This may cause inference issues.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=args.temperature,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.2,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        renormalize_logits=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Helper functions
    def format_chat(messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None) -> str:
        """Apply the model chat template safely."""
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": args.enable_thinking,
        }
        
        if tools:
            kwargs["tools"] = tools
        
        return tokenizer.apply_chat_template(
            messages,
            **kwargs
        )

    def generate_one(prompt: str) -> Dict[str, Any]:
        """Run a single generation and return text + metadata."""
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
            )

        full_sequence = outputs.sequences[0]
        generated_ids = full_sequence[len(inputs.input_ids[0]):]

        text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
        )

        result = {
            "prompt": prompt,
            "generated_text": text,
            "num_generated_tokens": len(generated_ids),
        }
        if args.mode == "chat":
            result["has_eos"] = tokenizer.eos_token_id in generated_ids.tolist()
        return result

    # Main evaluation loop
    results = []
    print(f"\nRunning inference on {len(samples)} samples...")

    for idx, sample in enumerate(samples):
        task_type = sample.get("task_type", "unknown")
        print(f"\n=== Running sample {idx} | task={task_type} ===")

        if args.mode == "completion":
            prompt = sample["prompt"]
        else:
            tools = sample.get("tool", [])
            if tools:
                print(f"Using {len(tools)} tool(s)")
            prompt = format_chat(sample["messages"], tools=tools if tools else None)

        generation = generate_one(prompt)

        result = {
            "task_type": task_type,
            **generation,
        }

        results.append(result)
        if args.mode == "chat":
            print(f"has_eos={generation['has_eos']} | tokens={generation['num_generated_tokens']}")
        else:
            print(f"tokens={generation['num_generated_tokens']}")
        print("Output:\n", generation["generated_text"])

    # Save results
    print(f"\nSaving results to: {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nAll tasks completed.")
    
    # Generate markdown report
    print("\nGenerating markdown analysis report...")
    output_path = Path(args.output_file)
    markdown_path = output_path.parent / f"{output_path.stem}.md"
    
    try:
        generate_markdown_report(results, str(markdown_path))
        print(f"✅ Markdown report saved to: {markdown_path}")
    except Exception as e:
        print(f"⚠️ Failed to generate markdown report: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path or identifier for the model to load (e.g., 'Polygl0t/Tucano2-qwen-0.5B-Instruct')"
    )
    parser.add_argument(
        "--samples_file",
        type=str,
        required=True,
        help="Path to a JSON file containing test samples."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="model_task_outputs.json",
        help="Path to save the output results JSON file (default: model_task_outputs.json)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate (default: 1024)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)"
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable 'thinking' mode in chat template."
    )
    parser.add_argument(
        "--chat_template_path",
        type=str,
        default=None,
        help="Path to a jinja chat template file. If provided, overrides the tokenizer's chat template."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["chat", "completion"],
        default="chat",
        help="Inference mode: 'chat' applies the chat template (default), 'completion' feeds the prompt directly to the model (for base models)."
    )
    args = parser.parse_args()

    main(args)
