"""
Test script for production validation
Tests basic functionality without running full pipeline
"""

import sys
sys.path.append('.')

from pathlib import Path
import pandas as pd
import numpy as np

def test_basic_imports():
    """Test that all required modules can be imported"""
    print("Testing basic imports...")
    
    try:
        import yaml
        print("✓ yaml imported successfully")
    except ImportError as e:
        print(f"✗ yaml import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        from loguru import logger
        print("✓ loguru imported successfully")
    except ImportError as e:
        print(f"✗ loguru import failed: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    config_path = Path('config/config.yaml')
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['ceramic_systems', 'properties', 'targets', 'paths']
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing config section: {section}")
                return False
        
        print("✓ Configuration loaded successfully")
        print(f"  Ceramic systems: {config['ceramic_systems']['primary']}")
        print(f"  Mechanical R² target: {config['targets']['mechanical_r2']}")
        print(f"  Ballistic R² target: {config['targets']['ballistic_r2']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist or can be created"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'data/raw',
        'data/processed', 
        'data/features',
        'results/models',
        'results/figures',
        'results/reports'
    ]
    
    all_good = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"✓ Directory created: {dir_path}")
            except Exception as e:
                print(f"✗ Cannot create directory {dir_path}: {e}")
                all_good = False
    
    return all_good

def test_model_structure():
    """Test if any trained models exist"""
    print("\nTesting model structure...")
    
    models_dir = Path('results/models')
    if not models_dir.exists():
        print("⚠ No models directory found - this is expected for initial setup")
        return True
    
    # Look for any model files
    model_files = list(models_dir.rglob("*.pkl"))
    if model_files:
        print(f"✓ Found {len(model_files)} model files")
        
        # Show some examples
        for i, model_file in enumerate(model_files[:5]):
            print(f"  - {model_file.relative_to(models_dir)}")
        
        if len(model_files) > 5:
            print(f"  ... and {len(model_files) - 5} more")
    else:
        print("⚠ No trained models found - run training first")
    
    return True

def test_data_availability():
    """Test if any data files exist"""
    print("\nTesting data availability...")
    
    data_dirs = ['data/raw', 'data/processed', 'data/features']
    total_files = 0
    
    for data_dir in data_dirs:
        dir_path = Path(data_dir)
        if dir_path.exists():
            csv_files = list(dir_path.rglob("*.csv"))
            total_files += len(csv_files)
            print(f"✓ {data_dir}: {len(csv_files)} CSV files")
        else:
            print(f"⚠ {data_dir}: Directory not found")
    
    if total_files > 0:
        print(f"✓ Total data files found: {total_files}")
    else:
        print("⚠ No data files found - run data collection first")
    
    return True

def main():
    """Run all tests"""
    print("="*60)
    print("PRODUCTION VALIDATION - BASIC TESTS")
    print("="*60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration Loading", test_config_loading),
        ("Directory Structure", test_directory_structure),
        ("Model Structure", test_model_structure),
        ("Data Availability", test_data_availability)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! System is ready for production validation.")
    else:
        print("⚠️  Some tests failed. Address issues before running production validation.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)