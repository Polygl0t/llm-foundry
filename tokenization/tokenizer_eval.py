"""
Tokenizer Evaluation Script

This script evaluates and compares multiple tokenizers on various metrics .

Metrics Computed:
-----------------
1. Subword Fertility (SF): Average tokens per word
   Formula: SF = Total Tokens / Total Words
   Lower is better (less splitting = more efficient)
   
2. Proportion of Continued Words (PCW): Fraction of words split into 2+ tokens
   Formula: PCW = Words Split into ≥2 Tokens / Total Words
   Lower is better (less fragmentation)
   
3. Characters per Token: Average character count per token
   Higher values suggest more efficient encoding
   
4. Unknown Token Count: Number of <unk> tokens
   Lower is better (better vocabulary coverage)

Example Usage:
--------------
python tokenizer_eval.py \
    --tokenizers_to_evaluate gpt2 bert-base-uncased meta-llama/Llama-2-7b-hf \
    --input_file sample_text.txt \
    --output_file tokenizer_comparison.json \
    --cache_dir ./.cache
"""
import argparse
import json
from transformers import AutoTokenizer
import pandas as pd

def main(args):
    """
    Main evaluation function that processes tokenizers and computes metrics.
    
    The evaluation workflow:
    1. Load reference text and split into words (whitespace-separated)
    2. For each tokenizer:
       a. Tokenize the full text to get total token count
       b. Calculate vocabulary size and unknown token count
       c. Tokenize each word individually to measure splitting behavior
       d. Compute all metrics (SF, PCW, Chars/Token)
    3. Sort results by fertility (most efficient first)
    4. Save to JSON and display as formatted table
    """
    
    # Load the reference text
    # We read the entire file as a single string for full-text tokenization
    with open(args.input_file, "r") as file:
        text = "".join(file.readlines())

    # Split into words using whitespace
    # This is our ground truth for what constitutes a "word"
    words = text.split()
    total_num_words = len(words)

    token = args.token
    cache_dir = args.cache_dir

    # Define column names for the output table
    columns = ["Tokenizer", "Total Words", f"Number of Generated Tokens", "Vocabulary size", "Fertility", "PCW", "Chars/Token", "UNK count"]
    results = []

    # Evaluate each tokenizer in the list
    for tokenizer_name in args.tokenizers_to_evaluate:
        print(f"\nEvaluating tokenizer: {tokenizer_name}")
        
        # Extract a short name for display purposes
        name = tokenizer_name.split('/')[-1]

        # Load the tokenizer from Hugging Face Hub or local path
        # use_fast=True enables the Rust-based fast tokenizer implementation
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=token, cache_dir=cache_dir, use_fast=True)
        
        # Set to a very large value to avoid truncation warnings
        # We want to tokenize the full text without length limits
        tokenizer.model_max_length = int(1000e9)
        
        # Tokenize the entire text at once
        # This gives us the total number of tokens the tokenizer produces
        # add_special_tokens=False ensures we only count content tokens, not [CLS], [SEP], etc.
        tokens = tokenizer(
            text,
            return_attention_mask=False,  # We don't need attention masks for evaluation
            return_token_type_ids=False,  # We don't need token type IDs
            add_special_tokens=False      # Exclude special tokens like [CLS], [SEP]
        )

        total_tokens = len(tokens['input_ids'])
        vocab_size = len(tokenizer.get_vocab())

        # ===================================================================
        # METRIC 1: Unknown Token Count
        # ===================================================================
        # Count how many times the tokenizer had to use <unk> (unknown token)
        # This indicates vocabulary coverage - fewer unknowns = better coverage
        unk_token_count = None
        
        # Some tokenizers use unk_token, others might not have it defined
        # In that case, we fall back to eos_token as a proxy
        if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
            unk_token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
        else:
            unk_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

        # Count occurrences of the unknown token ID
        if unk_token_id is not None:
            input_ids = tokens['input_ids']
            if isinstance(input_ids, list):
                unk_token_count = input_ids.count(unk_token_id)
            else:
                unk_token_count = input_ids.tolist().count(unk_token_id)

        # ===================================================================
        # METRIC 2: Subword Fertility (SF)
        # ===================================================================
        # Formula: SF = Total Tokens / Total Words
        # 
        # Interpretation:
        # - SF = 1.0: Perfect efficiency, each word = one token
        # - SF = 2.0: On average, each word is split into 2 tokens
        # - Lower is generally better (more efficient encoding)
        #
        # Why it matters: Higher fertility = more tokens = more computation
        # during training/inference and longer sequences
        fertility = total_tokens / total_num_words if total_num_words != 0 else 0

        # ===================================================================
        # METRIC 3 & 4: Proportion of Continued Words (PCW) & Chars/Token
        # ===================================================================
        # For PCW, we need to tokenize each word individually to see
        # how many are split into multiple subword tokens
        #
        # Formula: PCW = (Words Split into ≥2 Tokens) / Total Words
        #
        # Interpretation:
        # - PCW = 0.0: No words are split (all single tokens)
        # - PCW = 0.5: Half of the words are split
        # - PCW = 1.0: Every word is split into multiple tokens
        # - Lower is generally better (less fragmentation)
        continued_words = 0
        chars_per_token_list = []
        
        for word in words:
            # Tokenize this individual word
            word_tokens = tokenizer(
                word,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )
            
            # Check if this word was split into 2 or more tokens
            if len(word_tokens['input_ids']) >= 2:
                continued_words += 1
            
            # Calculate characters per token for this word
            # This helps us understand token granularity
            token_strings = tokenizer.convert_ids_to_tokens(word_tokens['input_ids'])
            for token_str in token_strings:
                chars_per_token_list.append(len(token_str))
        
        # Calculate final metrics
        pcw = continued_words / total_num_words if total_num_words != 0 else 0
        mean_chars_per_token = sum(chars_per_token_list) / len(chars_per_token_list) if chars_per_token_list else 0

        # Calculate final metrics
        pcw = continued_words / total_num_words if total_num_words != 0 else 0
        mean_chars_per_token = sum(chars_per_token_list) / len(chars_per_token_list) if chars_per_token_list else 0

        # Store all results for this tokenizer
        d = {
            "tokenizer_name": name,
            "total_num_words": total_num_words,
            "total_tokens": total_tokens,
            "vocab_size": vocab_size,
            "fertility": fertility,
            "pcw": pcw,
            "mean_chars_per_token": mean_chars_per_token,
            "unk_token_count": unk_token_count,
        }
        results.append(d)
        
        # Print summary for this tokenizer
        print(f"  ✓ Fertility: {fertility:.3f} | PCW: {pcw:.3f} | Chars/Token: {mean_chars_per_token:.2f}")

    # Sort results by fertility (most efficient tokenizers first)
    # Lower fertility = fewer tokens = more efficient
    results = sorted(results, key=lambda x: x["fertility"])
    
    # Save detailed results to JSON for further analysis
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Create a formatted table for easy comparison
    table = pd.DataFrame(results)
    table.columns = columns

    print("\n" + "=" * 80)
    print("TOKENIZER EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_file}")
    print(f"Text analyzed: {total_num_words:,} words, {len(text):,} characters")
    print("\nComparison Table (sorted by Fertility):")
    print(table.to_markdown(index=False))
    print("\n" + "=" * 80)
    print("\nInterpretation Guide:")
    print("  • Fertility: Lower = more efficient (fewer tokens per word)")
    print("  • PCW: Lower = less fragmentation (fewer words split)")
    print("  • Chars/Token: Higher = more information per token")
    print("  • UNK count: Lower = better vocabulary coverage")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate and compare tokenizers on efficiency and behavior metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare multiple tokenizers on a text sample
  python tokenizer_eval.py \
      --tokenizers_to_evaluate gpt2 bert-base-uncased meta-llama/Llama-2-7b-hf \
      --input_file sample.txt \
      --output_file results.json

  # Evaluate with custom cache and HF token for private models
  python tokenizer_eval.py \
      --tokenizers_to_evaluate username/my-tokenizer \
      --input_file data.txt \
      --output_file eval.json \
      --cache_dir ~/.cache/tokenizers \
      --token hf_xxx

Understanding the Metrics:
  Subword Fertility (SF): Lower values indicate more efficient tokenizers
  Proportion of Continued Words (PCW): Lower values mean less word fragmentation
  Chars/Token: Higher values suggest more information packed per token
  UNK count: Lower values indicate better vocabulary coverage
        """
    )
    
    parser.add_argument(
        "--tokenizers_to_evaluate", 
        type=str, 
        nargs='+', 
        required=True, 
        help="List of tokenizer names (HF Hub IDs) or local paths to evaluate. "
             "Example: gpt2 bert-base-uncased meta-llama/Llama-2-7b-hf"
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the input text file for evaluation. Should be plain text, "
             "representative of your target domain/language."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True, 
        help="Path to save the evaluation results as JSON. "
             "Contains detailed metrics for each tokenizer."
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        required=False, 
        default=None, 
        help="Directory to cache downloaded tokenizers. "
             "Speeds up repeated evaluations. Default: None (uses HF default cache)"
    )
    parser.add_argument(
        "--token", 
        type=str, 
        required=False, 
        default=None, 
        help="Hugging Face authentication token for accessing private/gated models. "
             "Get yours at https://huggingface.co/settings/tokens"
    )

    args = parser.parse_args()
    
    print("=" * 80)
    print("TOKENIZER EVALUATION TOOL")
    print("=" * 80)
    print(f"\nInput file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Tokenizers to evaluate: {len(args.tokenizers_to_evaluate)}")
    for tok in args.tokenizers_to_evaluate:
        print(f"  • {tok}")
    print("\nStarting evaluation... 🚀\n")
    
    main(args)
    
    print("\n✅ Tokenizers evaluated successfully! 🎉")