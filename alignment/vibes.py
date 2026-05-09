"""
Inference Testing

This script performs inference testing on trained language models across multiple task types.
Test samples are loaded from an external JSON file (default: default_samples.json).

Input Format:
    - JSON array of sample objects:
    ```json
    [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Liste os principais eventos envolvendo a Revolução Farroupilha."
                }
            ],
            "task_type": "History",
            "tool" : []
        }
    ]
    ```

Example usage:
    # Run with default samples
    python inference_test.py --model_path checkpoints/model-sft/final
    
    # Run with custom samples and settings
    python inference_test.py \
        --model_path checkpoints/llama-sft/final \
        --samples_file custom_samples.json \
        --output_file results/evaluation.json \
        --max_new_tokens 1024 \
        --temperature 0.1 \
        --enable_thinking

Output:
    - JSON file with detailed results and task-specific checks
    - Markdown report with formatted analysis and statistics
"""
import json
import re
import torch
import argparse
import json_repair
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on a language model with various task samples."
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
    return parser.parse_args()


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


def clean_text(text: str, escape_markdown: bool = False) -> str:
    """Clean and format text for display.
    
    Args:
        text: Text to clean
        escape_markdown: If True, escape # to prevent markdown heading interpretation
    """
    if escape_markdown:
        text = text.replace('#', r'\#')
    return text.strip()


def fix_json_encoding(json_str: str) -> str:
    """Fix double-escaped unicode characters in JSON string."""
    import json as json_module
    
    try:
        # Fix double-escaped unicode: \\u00e7 -> \u00e7
        json_str = json_str.replace('\\\\u', '\\u')
        
        # Parse and re-serialize to ensure proper formatting
        data = json_module.loads(json_str)
        return json_module.dumps(data, ensure_ascii=False, indent=2)
    except:
        # If JSON parsing fails, return original
        return json_str


def indent_text(text: str, spaces: int = 4) -> str:
    """Indent text to create a code block in markdown without using backticks."""
    indent = ' ' * spaces
    lines = text.split('\n')
    return '\n'.join(indent + line for line in lines)


def build_status_indicators(sample: Dict[str, Any]) -> tuple:
    """Build status indicators and notes for a sample.
    
    Returns:
        tuple: (status_list, notes_list) for the sample
    """
    checks = sample.get('checks', {})
    has_eos = checks.get('has_eos', False)
    empty_output = checks.get('empty_output', False)
    num_tokens = sample.get('num_generated_tokens', 0)
    
    status = ["✅ Has EOS" if has_eos else "❌ No EOS"]
    if empty_output:
        status.append("⚠️ Empty Output")
    
    notes = []
    if not has_eos:
        notes.append("⚠️ Missing EOS token")
    if num_tokens >= 1024:
        notes.append("📏 Hit max token limit")
    if empty_output:
        notes.append("❌ Empty output")
    
    return status, notes


def generate_markdown_report(samples: list, output_path: str):
    """
    Generate a markdown report from inference samples.
    
    Args:
        samples: List of inference sample dictionaries
        output_path: Path to save the markdown report
    """
    
    # Start building the markdown content
    md_lines = []
    
    # Header
    md_lines.append("# Inference Samples Report\n")
    md_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("\n---\n")
    
    # Summary statistics
    md_lines.append("## Summary Statistics\n")
    
    task_types = {}
    has_eos_count = 0
    empty_output_count = 0
    total_tokens = 0
    
    for sample in samples:
        task_type = sample.get('task_type', 'Unknown')
        task_types[task_type] = task_types.get(task_type, 0) + 1
        
        if sample.get('checks', {}).get('has_eos', False):
            has_eos_count += 1
        if sample.get('checks', {}).get('empty_output', False):
            empty_output_count += 1
        
        total_tokens += sample.get('num_generated_tokens', 0)
    
    md_lines.append(f"- **Total Generated Tokens:** {total_tokens}")
    md_lines.append(f"- **Average Tokens per Sample:** {total_tokens / len(samples):.2f}")
    md_lines.append(f"- **Samples with EOS:** {has_eos_count} ({has_eos_count/len(samples)*100:.1f}%)")
    md_lines.append(f"- **Samples with Empty Output:** {empty_output_count} ({empty_output_count/len(samples)*100:.1f}%)")
    
    md_lines.append("\n---\n")
    
    # Detailed sample analysis
    md_lines.append("## Detailed Sample Analysis\n")
    
    for sample in samples:
        task_type = sample.get('task_type', 'Unknown')
        num_tokens = sample.get('num_generated_tokens', 0)
        
        md_lines.append(f"### {task_type}\n")
        
        # Build status indicators and notes
        status, notes = build_status_indicators(sample)
        
        md_lines.append(f"**Status:** {' | '.join(status)} | **Tokens:** {num_tokens}")
        
        if notes:
            md_lines.append(f" | {', '.join(notes)}")
        
        md_lines.append("\n")
        
        # Task-specific checks
        checks = sample.get('checks', {})
        task_specific_info = []
        
        if task_type == "Classification":
            if 'has_classification_label' in checks:
                label_status = "✅ Has label" if checks['has_classification_label'] else "❌ No label"
                task_specific_info.append(f"- **Classification Label Check:** {label_status}")
                if checks.get('classification_label'):
                    task_specific_info.append(f"- **Detected Label:** {checks['classification_label']}")
        
        elif task_type == "Structured Output":
            if 'valid_json' in checks:
                json_status = "✅ Valid" if checks['valid_json'] else "❌ Invalid"
                task_specific_info.append(f"- **JSON Validation:** {json_status}")
                if checks['valid_json'] and checks.get('json_output'):
                    json_output = checks['json_output']
                    if isinstance(json_output, dict):
                        task_specific_info.append(f"- **JSON Keys:** {list(json_output.keys())}")
                    else:
                        task_specific_info.append(f"- **JSON Output:** {json_output}")
        
        elif task_type == "Function Call / Tool Use":
            if 'made_function_call' in checks:
                call_status = "✅ Made function call" if checks['made_function_call'] else "❌ No function call"
                task_specific_info.append(f"- **Function Call:** {call_status}")
        
        elif task_type == "Summarization":
            if 'output_shorter_than_prompt' in checks:
                shorter_status = "✅ Yes" if checks['output_shorter_than_prompt'] else "❌ No"
                task_specific_info.append(f"- **Output Shorter than Prompt:** {shorter_status}")
            if 'prompt_length' in checks and 'output_length' in checks:
                task_specific_info.append(f"- **Prompt Length:** {checks['prompt_length']} chars")
                task_specific_info.append(f"- **Output Length:** {checks['output_length']} chars")
        
        if task_specific_info:
            md_lines.append("**Task-Specific Checks:**\n")
            md_lines.extend(task_specific_info)
            md_lines.append("\n")
        
        # Prompt
        prompt = clean_text(sample.get('prompt', ''), escape_markdown=True)
        md_lines.append("**Prompt:**\n")
        # Indent the prompt to preserve formatting and prevent # from being headings
        md_lines.append(indent_text(prompt))
        md_lines.append("\n")
        
        # Generated response
        response = sample.get('generated_text', '')
        
        # Only apply encoding fixes for Structured Output tasks
        if task_type == "Structured Output":
            # Try to extract and fix JSON content
            try:
                # Find JSON content in response
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    json_content = json_match.group(0)
                    fixed_json = fix_json_encoding(json_content)
                    response = response.replace(json_content, fixed_json)
            except:
                pass
        
        response = response.strip()
        
        # Truncate very long responses for readability (only if no EOS token)
        if len(response) > 2000 and not sample.get('checks', {}).get('has_eos', True):
            md_lines.append(f"**Response:** *(truncated - showing first 2000 of {len(response)} characters)*\n")
            response = response[:2000] + "\n\n[... truncated ...]"
        else:
            md_lines.append("**Response:**\n")
        
        # Indent the response to preserve formatting
        md_lines.append(indent_text(response))
        md_lines.append("\n")
        
        md_lines.append("\n---\n")
    
    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    return output_path



def main():
    args = parse_args()
    
    # Load samples
    samples = load_samples(args.samples_file)
    
    # Load model and tokenizer
    print(f"\nLoading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load external chat template if provided (ensures train-inference consistency)
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

    def generate_one(messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a single generation and return text + metadata."""
        prompt = format_chat(messages, tools=tools)
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

        return {
            "prompt": prompt,
            "generated_text": text,
            "generated_ids": generated_ids.tolist(),
            "eos_generated": tokenizer.eos_token_id in generated_ids.tolist(),
            "num_generated_tokens": len(generated_ids),
        }

    def run_task_checks(task_type: str, output: Dict[str, Any]) -> Dict[str, Any]:
        """Run task-specific checks on generated output."""
        text = output["generated_text"]
        checks = {
            "has_eos": output["eos_generated"],
            "empty_output": len(text.strip()) == 0
        }

        # Task-specific checks
        if task_type == "Classification":
            labels = {"positiva": "Positiva", "negativa": "Negativa", "neutra": "Neutra"}
            detected_label = next((v for k, v in labels.items() if re.search(rf"\b{k}\b", text.lower())), None)
            checks["has_classification_label"] = detected_label is not None
            checks["classification_label"] = detected_label
        
        elif task_type == "Structured Output":
            try:
                json_string = json_repair.repair_json(text)
                checks["valid_json"] = True
                checks["json_output"] = json_repair.loads(json_string)
            except Exception:
                checks["valid_json"] = False
                checks["json_output"] = None
        
        elif task_type == "Function Call / Tool Use":
            checks["made_function_call"] = bool(re.search(r"<tool_call>.*</tool_call>", text, re.DOTALL))
        
        elif task_type == "Summarization":
            prompt_length = len(output.get("prompt", ""))
            output_length = len(text)
            checks.update({
                "output_shorter_than_prompt": output_length < prompt_length,
                "prompt_length": prompt_length,
                "output_length": output_length
            })
        
        elif "Reasoning" in task_type:
            think_matches = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
            checks.update({
                "has_think_tags": bool(think_matches),
                "has_non_empty_think": any(match.strip() for match in think_matches),
                "think_tag_count": len(think_matches)
            })

        return checks

    # Main evaluation loop
    results = []
    print(f"\nRunning inference on {len(samples)} samples...")

    for idx, sample in enumerate(samples):
        task_type = sample.get("task_type", "unknown")
        print(f"\n=== Running sample {idx} | task={task_type} ===")

        # Extract tools if they exist
        tools = sample.get("tool", [])
        if tools:
            print(f"Using {len(tools)} tool(s)")
        
        generation = generate_one(sample["messages"], tools=tools if tools else None)
        checks = run_task_checks(task_type, generation)

        result = {
            "task_type": task_type,
            "prompt": generation["prompt"],
            "checks": checks,
            "generated_text": generation["generated_text"],
            "num_generated_tokens": generation["num_generated_tokens"],
            "has_eos": generation["eos_generated"],
        }

        results.append(result)

        print("Checks:", checks)
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
    main()
