"""
Helper script to initialize a model from a config file and print its size and other important information.

How to Use:
    python inspect_model.py --config_path <path_to_config> [--base_model <model_id>] [--precision {bfloat16,float32}]

Example:
    python inspect_model.py --config_path ./config.json
    python inspect_model.py --config_path ./config.json --base_model meta-llama/Llama-2-7b --precision bfloat16
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def format_bytes(num_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def compute_active_trainable_params(config, trainable_params: int) -> int:
    """
    Compute the number of active trainable parameters.
    
    For dense models, active_trainable_params == trainable_params.
    For MoE models, only the routed experts per token are counted.
    """
    num_experts = getattr(config, 'num_experts', None) or getattr(config, 'num_local_experts', None)
    if num_experts is None or num_experts <= 1:
        return trainable_params

    num_experts_per_tok = getattr(config, 'num_experts_per_tok', None)
    if num_experts_per_tok is None or num_experts_per_tok >= num_experts:
        return trainable_params

    hidden_size = config.hidden_size
    expert_intermediate_size = getattr(config, 'moe_intermediate_size', None) or config.intermediate_size
    
    # SwiGLU MLP per expert: gate_proj + up_proj + down_proj
    params_per_expert = 3 * hidden_size * expert_intermediate_size
    
    decoder_sparse_step = getattr(config, 'decoder_sparse_step', 1) or 1
    num_moe_layers = config.num_hidden_layers // decoder_sparse_step
    
    inactive_params = num_moe_layers * (num_experts - num_experts_per_tok) * params_per_expert
    return trainable_params - inactive_params


def estimate_model_memory(model: torch.nn.Module, precision: torch.dtype) -> int:
    """
    Estimate the model's memory footprint in bytes for inference.
    
    Includes model parameters but not KV cache or activations.
    """
    bytes_per_param = 2 if precision in [torch.float16, torch.bfloat16] else 4
    num_params = sum(p.numel() for p in model.parameters())
    return num_params * bytes_per_param


def inspect_model(
    config_path: str,
    base_model: Optional[str] = None,
    precision_str: str = "bfloat16",
    device: str = "cpu",
) -> None:
    """
    Initialize a model from a config file and print detailed information.
    
    Args:
        config_path: Path to the model config file (JSON or directory)
        base_model: Optional base model to inherit from
        precision_str: Precision for model ("bfloat16" or "float32")
        device: Device to load model on
    """
    precision = torch.bfloat16 if precision_str == "bfloat16" else torch.float32
    
    print("=" * 80)
    print("MODEL INSPECTION")
    print("=" * 80)
    
    # Load config
    print(f"\n[1] Loading config from: {config_path}")
    try:
        config = AutoConfig.from_pretrained(config_path)
        print(f"    ✓ Config loaded successfully")
    except Exception as e:
        print(f"    ✗ Failed to load config: {e}")
        return
    
    # Load tokenizer if base model is provided
    tokenizer = None
    if base_model:
        print(f"\n[2] Loading tokenizer from: {base_model}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
            # Update config with tokenizer token IDs
            config.bos_token_id = tokenizer.bos_token_id
            config.eos_token_id = tokenizer.eos_token_id
            config.pad_token_id = tokenizer.pad_token_id
            print(f"    ✓ Tokenizer loaded (vocab size: {len(tokenizer):,})")
        except Exception as e:
            print(f"    ✗ Failed to load tokenizer: {e}")
    else:
        print(f"\n[2] Tokenizer: Not specified (use --base_model to load)")
    
    # Update config with precision info
    config.torch_dtype = precision
    config.vocab_size = max(config.vocab_size, len(tokenizer) if tokenizer else config.vocab_size)
    
    # Initialize model
    print(f"\n[3] Initializing model from config (precision: {precision_str})")
    try:
        model = AutoModelForCausalLM.from_config(config)
        model = model.to(device)
        print(f"    ✓ Model initialized successfully")
    except Exception as e:
        print(f"    ✗ Failed to initialize model: {e}")
        return
    
    # Compute metrics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    active_trainable_params = compute_active_trainable_params(config, trainable_params)
    memory_estimate = estimate_model_memory(model, precision)
    
    # Print model information
    print(f"\n[4] Model Information")
    print(f"    Model type:               {config.model_type}")
    print(f"    Architecture:             {config.architectures[0] if config.architectures else 'Unknown'}")
    
    print(f"\n[5] Model Architecture")
    print(f"    Hidden size:              {config.hidden_size:,}")
    print(f"    Number of layers:         {config.num_hidden_layers}")
    print(f"    Number of attention heads: {config.num_attention_heads}")
    if hasattr(config, 'num_key_value_heads'):
        print(f"    Number of KV heads:       {config.num_key_value_heads}")
    print(f"    Intermediate size:        {config.intermediate_size:,}")
    print(f"    Vocabulary size:          {config.vocab_size:,}")
    if hasattr(config, 'max_position_embeddings'):
        print(f"    Max position embeddings:  {config.max_position_embeddings:,}")
    
    # MoE-specific info if applicable
    if hasattr(config, 'num_experts') or hasattr(config, 'num_local_experts'):
        num_experts = getattr(config, 'num_experts', None) or getattr(config, 'num_local_experts', None)
        print(f"\n[6] Mixture of Experts (MoE) Configuration")
        print(f"    Total experts:            {num_experts}")
        if hasattr(config, 'num_experts_per_tok'):
            print(f"    Experts per token:        {config.num_experts_per_tok}")
    
    print(f"\n[7] Parameter Counts")
    print(f"    Total parameters:         {total_params:,}")
    print(f"    Trainable parameters:     {trainable_params:,}")
    if active_trainable_params != trainable_params:
        print(f"    Active trainable params:  {active_trainable_params:,} (MoE model)")
    print(f"    Percentage trainable:     {100 * trainable_params / total_params:.2f}%")
    
    print(f"\n[8] Memory Footprint (approximate, {precision_str})")
    print(f"    Model weights:            {format_bytes(memory_estimate)}")
    print(f"    Per parameter:            {8 if precision_str == 'bfloat16' else 4} bytes")
    
    # Other config details
    print(f"\n[9] Other Configuration")
    print(f"    Attention implementation: {getattr(config, 'attn_implementation', 'default')}")
    print(f"    Use cache:                {config.use_cache}")
    print(f"    Tie word embeddings:      {config.tie_word_embeddings if hasattr(config, 'tie_word_embeddings') else 'N/A'}")
    if hasattr(config, 'rope_theta'):
        print(f"    RoPE theta:               {config.rope_theta}")
    
    print(f"\n[10] Model Layer Breakdown")
    layer_counts = {}
    for name, module in model.named_modules():
        module_type = module.__class__.__name__
        layer_counts[module_type] = layer_counts.get(module_type, 0) + 1
    
    for module_type, count in sorted(layer_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {module_type}: {count}")
    
    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a model initialized from a config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inspect_model.py --config_path ./config.json
  python inspect_model.py --config_path ./config.json --base_model meta-llama/Llama-2-7b
  python inspect_model.py --config_path ./config.json --precision float32
        """
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the model config file (JSON) or huggingface model directory",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model to load tokenizer from (optional)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="Precision for the model (default: bfloat16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on (default: cpu)",
    )
    
    args = parser.parse_args()
    
    # Verify config path exists
    if not Path(args.config_path).exists():
        print(f"Error: Config path does not exist: {args.config_path}")
        return
    
    inspect_model(
        config_path=args.config_path,
        base_model=args.base_model,
        precision_str=args.precision,
        device=args.device,
    )


if __name__ == "__main__":
    main()
