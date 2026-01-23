"""
AMD GPU Installation Validation Script

Validates the installation of Python packages for AMD GPU environments (ROCm).
Tests ML/DL frameworks, data processing libraries, and GPU-specific functionality.

Key packages tested:
- PyTorch with CUDA/ROCm support and GPU operations
- ML frameworks: TRL, Accelerate, Liger Kernel
- GPU acceleration: flash-attn, vLLM
- Data libraries: datasets, pandas, numpy, transformers
- Monitoring tools: codecarbon, wandb
- NLP tools: datatrove, stanza, spacy

Performs functional tests including:
- CUDA/ROCm availability and GPU device detection
- GPU tensor operations and matrix multiplication
- Transformers integration with Accelerate
- Dataset creation and manipulation

Exit codes:
    0: All tests passed
    1: One or more critical tests failed

Usage:
    python install_amd.py
"""

import sys
import traceback
from datetime import datetime

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class ValidationResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
    
    def add_pass(self, test_name, message=""):
        self.passed.append((test_name, message))
        print(f"{GREEN}✓{RESET} {test_name}: {message}")
    
    def add_fail(self, test_name, error):
        self.failed.append((test_name, str(error)))
        print(f"{RED}✗{RESET} {test_name}: {error}")
    
    def add_warning(self, test_name, message):
        self.warnings.append((test_name, message))
        print(f"{YELLOW}⚠{RESET} {test_name}: {message}")
    
    def print_summary(self):
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Passed: {len(self.passed)}")
        print(f"Failed: {len(self.failed)}")
        print(f"Warnings: {len(self.warnings)}")
        
        if self.failed:
            print(f"\n{RED}Failed Tests:{RESET}")
            for test_name, error in self.failed:
                print(f"  - {test_name}: {error}")
        
        if self.warnings:
            print(f"\n{YELLOW}Warnings:{RESET}")
            for test_name, message in self.warnings:
                print(f"  - {test_name}: {message}")
        
        print("="*60)
        return len(self.failed) == 0

def test_import(result, module_name, package_name=None):
    """Test if a module can be imported"""
    if package_name is None:
        package_name = module_name
    
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        result.add_pass(package_name, f"v{version}")
        return mod
    except Exception as e:
        result.add_fail(package_name, f"Import failed: {e}")
        return None

def test_torch_functionality(result, torch):
    """Test PyTorch basic functionality"""
    if torch is None:
        return
    
    try:
        # Test CUDA availability
        if torch.cuda.is_available():
            result.add_pass("PyTorch CUDA", f"Available - {torch.cuda.get_device_name(0)}")
            
            # Test basic tensor operations on GPU
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.matmul(x, y)
            assert z.shape == (100, 100)
            result.add_pass("PyTorch GPU Operations", "Basic tensor ops working")
        else:
            result.add_fail("PyTorch CUDA", "CUDA not available")
    except Exception as e:
        result.add_fail("PyTorch GPU Operations", str(e))

def test_flash_attn(result):
    """Test flash-attn installation"""
    try:
        import flash_attn
        version = getattr(flash_attn, '__version__', 'unknown')
        result.add_pass("flash-attn", f"v{version}")
        
        # Try to import the actual attention function
        try:
            from flash_attn import flash_attn_func
            result.add_pass("flash-attn functionality", "Functions accessible")
        except ImportError as e:
            result.add_warning("flash-attn functionality", f"Could not import functions: {e}")
    except Exception as e:
        result.add_fail("flash-attn", str(e))

def test_vllm(result):
    """Test vLLM installation"""
    try:
        import vllm
        version = getattr(vllm, '__version__', 'unknown')
        result.add_pass("vLLM", f"v{version}")
    except Exception as e:
        result.add_warning("vLLM", f"Import failed (optional): {e}")

def test_transformers_integration(result):
    """Test transformers with accelerate"""
    try:
        from transformers import AutoConfig
        from accelerate import Accelerator
        
        # Test basic config loading
        config = AutoConfig.from_pretrained("gpt2")
        result.add_pass("Transformers Integration", "Config loading works")
        
        # Test accelerator initialization
        accelerator = Accelerator()
        result.add_pass("Accelerate Integration", f"Device: {accelerator.device}")
    except Exception as e:
        result.add_fail("Transformers Integration", str(e))

def test_data_processing(result):
    """Test data processing libraries"""
    try:
        from datasets import Dataset
        import pandas as pd
        import numpy as np
        
        # Create a simple dataset
        data = {"text": ["test1", "test2"], "label": [0, 1]}
        df = pd.DataFrame(data)
        ds = Dataset.from_pandas(df)
        
        result.add_pass("Data Processing", "datasets, pandas, numpy working")
    except Exception as e:
        result.add_fail("Data Processing", str(e))

def main():
    print("\n" + "="*60)
    print("AMD INSTALLATION VALIDATION")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    result = ValidationResult()
    
    # Test core packages
    print(f"{BLUE}Testing Core Packages...{RESET}")
    numpy = test_import(result, "numpy")
    pandas = test_import(result, "pandas")
    test_import(result, "datasets")
    test_import(result, "transformers")
    test_import(result, "evaluate")
    test_import(result, "huggingface_hub")
    test_import(result, "sklearn", "scikit-learn")
    test_import(result, "matplotlib")
    test_import(result, "sentencepiece")
    test_import(result, "yaml", "pyyaml")
    
    # Test PyTorch
    print(f"\n{BLUE}Testing PyTorch...{RESET}")
    torch = test_import(result, "torch")
    test_torch_functionality(result, torch)
    
    # Test ML frameworks
    print(f"\n{BLUE}Testing ML Frameworks...{RESET}")
    test_import(result, "trl")
    test_import(result, "accelerate")
    test_import(result, "liger_kernel")
    
    # Test flash attention (can be problematic)
    print(f"\n{BLUE}Testing Flash Attention...{RESET}")
    test_flash_attn(result)
    
    # Test vLLM (optional)
    print(f"\n{BLUE}Testing vLLM...{RESET}")
    test_vllm(result)
    
    # Test monitoring tools
    print(f"\n{BLUE}Testing Monitoring Tools...{RESET}")
    test_import(result, "codecarbon")
    test_import(result, "wandb")
    
    # Test data processing libraries
    print(f"\n{BLUE}Testing Data Processing...{RESET}")
    test_import(result, "datatrove")
    test_import(result, "stanza")
    test_import(result, "spacy")
    test_import(result, "json_repair", "json-repair")
    
    # Test integrations
    print(f"\n{BLUE}Testing Integrations...{RESET}")
    test_transformers_integration(result)
    test_data_processing(result)
    
    # Print summary
    success = result.print_summary()
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{RED}CRITICAL ERROR:{RESET} Validation script crashed")
        print(traceback.format_exc())
        sys.exit(1)
