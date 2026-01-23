"""
AlpacaEval Portuguese Generation Script

Generates model responses for the alpaca-eval-pt benchmark dataset using chat templates
and configurable generation parameters. Designed for evaluating Portuguese instruction models.

Output Format:
- AlpacaEval JSON: [{"instruction": "...", "output": "...", "generator": "..."}]

Benchmark Details:
- Dataset: TucanoBR/alpaca-eval-pt (805 instructions)
- Language: Portuguese
- Task: Instruction following evaluation
- Compatible with alpaca_eval CLI tool

Usage:
    # Basic generation
    python alpaca_eval_generate.py \
        --model_id TucanoBR/Tucano-2b4-Instruct \
        --output_folder outputs/ \
        --batch_size 8
"""
import argparse
import torch
import tqdm
import json
from pathlib import Path
from typing import List, Dict
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def format_instruction_as_messages(instruction: str, system_prompt: str = None) -> List[Dict[str, str]]:
    """Format instruction as a message list for chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": instruction})
    return messages


def main(args):
  
    # Set up precision and CUDA backend options
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    # Load tokenizer and model
    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, 
        revision=args.revision, 
        token=args.token, 
        cache_dir=args.cache_dir
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        attn_implementation=args.attn_implementation,
        dtype=torch.bfloat16 if args.precision == "bfloat16" else torch.float16 if args.precision == "float16" else torch.float32,
        revision=args.revision,
        token=args.token,
        cache_dir=args.cache_dir,
    )
    # Set to evaluation mode
    model.eval()  

    # Set up generation configuration
    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        renormalize_logits=True,
    )

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Load evaluation dataset
    print("Loading evaluation dataset...")
    eval_set = load_dataset("TucanoBR/alpaca-eval-pt", split="eval", cache_dir=args.cache_dir)
    if "output" in eval_set.column_names:
        eval_set = eval_set.remove_columns(["output"])
    if "generator" in eval_set.column_names:
        eval_set = eval_set.remove_columns(["generator"])
    
    samples = eval_set['instruction']
    print(f"Loaded {len(samples)} samples")

    # Generate outputs
    outputs = []
    
    print("\nGenerating outputs...")
    for i in tqdm.tqdm(range(0, len(samples), args.batch_size), desc="Processing batches"):
        batch_instructions = samples[i:i + args.batch_size]
        
        # Format each instruction using chat template
        batch_prompts = []
        for instruction in batch_instructions:
            messages = format_instruction_as_messages(instruction, args.system_prompt)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking
            )
            batch_prompts.append(prompt)
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length
        ).to(device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                generation_config=generation_config,
            )
        
        # Decode only the generated portion (excluding input)
        for j, gen_ids in enumerate(generated_ids):
            input_length = inputs.input_ids[j].shape[0]
            generated_tokens = gen_ids[input_length:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            outputs.append(output_text)

    # Add output and generator columns to dataset and save
    print("\nSaving results...")
    generator_name = args.model_id.split('/')[-1]
    generator_names = [generator_name] * len(outputs)
    eval_set = eval_set.add_column("output", outputs)
    eval_set = eval_set.add_column("generator", generator_names)
    
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"alpaca_{generator_name}.json"
    
    data = [sample for sample in eval_set]

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving with UTF-8: {e}")
        print("Retrying with ASCII encoding...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=True, indent=2)
        print(f"Results saved to: {output_file} (ASCII encoding)")
    
    print(f"\nGeneration complete! Processed {len(outputs)} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run alpaca-eval-pt evaluation using proper chat templates."
    )
    # Model arguments
    parser.add_argument(
        "--model_id", 
        type=str, 
        required=True,
        help="Model identifier (e.g., 'TucanoBR/Tucano-2b4-Instruct')"
    )
    parser.add_argument("--revision", type=str, default="main", help="Model revision")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory")
    parser.add_argument(
        "--attn_implementation", 
        type=str, 
        default="flash_attention_2", 
        help="Attention implementation (e.g., 'flash_attention_2', 'sdpa', 'eager')"
    )
    parser.add_argument(
        "--precision", 
        type=str, 
        choices=["bfloat16", "float16", "float32"], 
        default="bfloat16", 
        help="Model precision type"
    )
    
    # Generation arguments
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=2048, 
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=4096,
        help="Maximum input length for tokenization"
    )
    parser.add_argument(
        "--do_sample", 
        action="store_true",
        default=True,
        help="Whether to use sampling for generation"
    )
    parser.add_argument(
        "--repetition_penalty", 
        type=float, 
        default=1.2, 
        help="Penalty for token repetition"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.1, 
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=50, 
        help="Top-k filtering for generation"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=1.0, 
        help="Top-p (nucleus) sampling"
    )
    
    # Chat template arguments
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Optional system prompt to prepend to each instruction"
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=False,
        help="Enable 'thinking' style prompts in generation"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_folder", 
        type=str, 
        default="outputs", 
        help="Folder to save generated outputs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="Batch size for generation"
    )

    args = parser.parse_args()
    main(args)
