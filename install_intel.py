"""
Intel Installation Validation Script

Validates the installation of Python packages for CPU-only Intel environments.
Tests core data science and NLP libraries including numpy, pandas, transformers,
datasets, and various NLP tools (stanza, spacy, datatrove).

Performs functional tests beyond import checks:
- NumPy matrix operations
- Pandas DataFrame manipulations
- Transformers model config and tokenizer loading
- Datasets library functionality

Exit codes:
    0: All tests passed
    1: One or more tests failed

Usage:
    python install_intel.py
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

def test_numpy_operations(result, numpy):
    """Test numpy basic operations"""
    if numpy is None:
        return
    
    try:
        # Test basic array operations
        a = numpy.random.rand(1000, 1000)
        b = numpy.random.rand(1000, 1000)
        c = numpy.dot(a, b)
        assert c.shape == (1000, 1000)
        result.add_pass("NumPy Operations", "Matrix operations working")
    except Exception as e:
        result.add_fail("NumPy Operations", str(e))

def test_pandas_operations(result, pandas):
    """Test pandas basic operations"""
    if pandas is None:
        return
    
    try:
        # Test DataFrame operations
        df = pandas.DataFrame({
            'A': range(1000),
            'B': range(1000, 2000),
            'C': ['test'] * 1000
        })
        df['D'] = df['A'] + df['B']
        assert len(df) == 1000
        result.add_pass("Pandas Operations", "DataFrame operations working")
    except Exception as e:
        result.add_fail("Pandas Operations", str(e))

def test_transformers_functionality(result):
    """Test transformers basic functionality"""
    try:
        from transformers import AutoConfig, AutoTokenizer
        
        # Test config loading (doesn't download model)
        config = AutoConfig.from_pretrained("gpt2")
        result.add_pass("Transformers Config", "Config loading works")
        
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        text = "Hello, this is a test."
        tokens = tokenizer(text)
        result.add_pass("Transformers Tokenizer", "Tokenization works")
    except Exception as e:
        result.add_fail("Transformers Functionality", str(e))

def test_datasets_functionality(result):
    """Test datasets library"""
    try:
        from datasets import Dataset
        import pandas as pd
        
        # Create a simple dataset
        data = {
            "text": ["sample1", "sample2", "sample3"],
            "label": [0, 1, 0]
        }
        df = pd.DataFrame(data)
        ds = Dataset.from_pandas(df)
        
        assert len(ds) == 3
        result.add_pass("Datasets Functionality", "Dataset creation and manipulation works")
    except Exception as e:
        result.add_fail("Datasets Functionality", str(e))

def test_nlp_tools(result):
    """Test NLP processing tools"""
    try:
        import stanza
        result.add_pass("Stanza", f"v{stanza.__version__}")
    except Exception as e:
        result.add_fail("Stanza", str(e))
    
    try:
        import spacy
        result.add_pass("SpaCy", f"v{spacy.__version__}")
    except Exception as e:
        result.add_fail("SpaCy", str(e))

def test_data_processing(result):
    """Test datatrove"""
    try:
        import datatrove
        result.add_pass("Datatrove", f"v{datatrove.__version__}")
    except Exception as e:
        result.add_fail("Datatrove", str(e))

def main():
    print("\n" + "="*60)
    print("INTEL INSTALLATION VALIDATION")
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
    test_import(result, "json_repair", "json-repair")
    
    # Test numpy operations
    print(f"\n{BLUE}Testing NumPy Operations...{RESET}")
    test_numpy_operations(result, numpy)
    
    # Test pandas operations
    print(f"\n{BLUE}Testing Pandas Operations...{RESET}")
    test_pandas_operations(result, pandas)
    
    # Test transformers functionality
    print(f"\n{BLUE}Testing Transformers...{RESET}")
    test_transformers_functionality(result)
    
    # Test datasets
    print(f"\n{BLUE}Testing Datasets Library...{RESET}")
    test_datasets_functionality(result)
    
    # Test NLP tools
    print(f"\n{BLUE}Testing NLP Tools...{RESET}")
    test_nlp_tools(result)
    
    # Test data processing
    print(f"\n{BLUE}Testing Data Processing...{RESET}")
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
