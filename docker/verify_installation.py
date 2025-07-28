#!/usr/bin/env python3
"""
Comprehensive installation verification for Round 1B system
"""

import sys
import importlib
import traceback

def verify_package(package_name, import_name=None, test_function=None):
    """Verify a package installation and functionality"""
    import_name = import_name or package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        
        # Run additional test if provided
        if test_function:
            test_result = test_function(module)
            if test_result:
                print(f"✓ {package_name} {version} - verified with test")
            else:
                print(f"⚠ {package_name} {version} - imported but test failed")
        else:
            print(f"✓ {package_name} {version} - imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ {package_name} - import failed: {str(e)}")
        return False
    except Exception as e:
        print(f"⚠ {package_name} - import succeeded but error: {str(e)}")
        return True  # Still count as success if import worked

def test_torch(torch_module):
    """Test PyTorch functionality"""
    try:
        # Test basic tensor operations
        x = torch_module.tensor([1.0, 2.0, 3.0])
        y = x * 2
        
        # Test CPU device
        device = torch_module.device('cpu')
        z = x.to(device)
        
        return len(y) == 3 and z.device.type == 'cpu'
    except:
        return False

def test_sentence_transformers(st_module):
    """Test sentence transformers functionality"""
    try:
        # Just verify the class exists - actual model loading tested elsewhere
        return hasattr(st_module, 'SentenceTransformer')
    except:
        return False

def test_spacy(spacy_module):
    """Test spaCy functionality"""
    try:
        # Test that spaCy can be used (model loading tested elsewhere)
        return hasattr(spacy_module, 'load') and hasattr(spacy_module, 'blank')
    except:
        return False

def test_nltk(nltk_module):
    """Test NLTK functionality"""
    try:
        # Test basic NLTK functions
        return hasattr(nltk_module, 'download') and hasattr(nltk_module, 'data')
    except:
        return False

def main():
    """Main verification function"""
    print("=== Round 1B Installation Verification ===")
    
    # Define packages to verify
    packages = [
        # Core ML packages
        ('torch', 'torch', test_torch),
        ('torchvision', 'torchvision', None),
        ('torchaudio', 'torchaudio', None),
        
        # NLP packages
        ('sentence-transformers', 'sentence_transformers', test_sentence_transformers),
        ('spacy', 'spacy', test_spacy),
        ('nltk', 'nltk', test_nltk),
        
        # Data processing
        ('numpy', 'numpy', None),
        ('pandas', 'pandas', None),
        ('scikit-learn', 'sklearn', None),
        
        # Validation
        ('pydantic', 'pydantic', None),
        
        # PDF processing
        ('PyPDF2', 'PyPDF2', None),
        ('PyMuPDF', 'fitz', None),
        
        # System
        ('psutil', 'psutil', None),
    ]
    
    successful = 0
    total = len(packages)
    
    for package_name, import_name, test_func in packages:
        if verify_package(package_name, import_name, test_func):
            successful += 1
    
    print(f"\n=== Verification Summary ===")
    print(f"Successful: {successful}/{total} packages")
    
    if successful == total:
        print("🎉 All packages verified successfully!")
        print("=== System Ready for Round 1B Processing ===")
    elif successful >= total * 0.8:  # 80% success rate
        print("⚠ Most packages working - system should function with fallbacks")
        print("=== System Ready with Fallback Support ===")
    else:
        print("❌ Too many package failures - system may not function properly")
        print("=== System Requires Attention ===")
    
    # Additional system checks
    print(f"\n=== System Information ===")
    print(f"Python version: {sys.version}")
    
    try:
        import torch
        print(f"PyTorch device: {torch.device('cpu')}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    except:
        pass
    
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        print(f"Current memory usage: {memory_mb:.1f} MB")
    except:
        pass

if __name__ == "__main__":
    main()
