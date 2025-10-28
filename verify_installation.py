#!/usr/bin/env python3
"""
Quick verification script to test that all dependencies are working
"""
import sys

def test_imports():
    """Test importing all critical packages"""
    print("🔍 Testing package imports...")
    
    packages_to_test = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('sklearn', 'scikit-learn'),
        ('xgboost', 'xgb'),
        ('catboost', 'cb'),
        ('lightgbm', 'lgb'),
        ('shap', 'shap'),
        ('pymatgen.core', 'pymatgen'),
        ('matplotlib.pyplot', 'plt'),
        ('loguru', 'loguru'),
        ('psutil', 'psutil'),
        ('pytest', 'pytest')
    ]
    
    failed_imports = []
    
    for package, alias in packages_to_test:
        try:
            if alias == 'np':
                import numpy as np
                print(f"✅ numpy v{np.__version__}")
            elif alias == 'pd':
                import pandas as pd
                print(f"✅ pandas v{pd.__version__}")
            elif alias == 'scikit-learn':
                import sklearn
                print(f"✅ scikit-learn v{sklearn.__version__}")
            elif alias == 'xgb':
                import xgboost as xgb
                print(f"✅ xgboost v{xgb.__version__}")
            elif alias == 'cb':
                import catboost as cb
                print(f"✅ catboost v{cb.__version__}")
            elif alias == 'lgb':
                import lightgbm as lgb
                print(f"✅ lightgbm v{lgb.__version__}")
            elif alias == 'shap':
                import shap
                print(f"✅ shap v{shap.__version__}")
            elif alias == 'pymatgen':
                from pymatgen.core import Structure
                import pymatgen
                print(f"✅ pymatgen v{pymatgen.__version__}")
            elif alias == 'plt':
                import matplotlib.pyplot as plt
                import matplotlib
                print(f"✅ matplotlib v{matplotlib.__version__}")
            elif alias == 'loguru':
                from loguru import logger
                print(f"✅ loguru (imported successfully)")
            elif alias == 'psutil':
                import psutil
                print(f"✅ psutil v{psutil.__version__}")
            elif alias == 'pytest':
                import pytest
                print(f"✅ pytest v{pytest.__version__}")
                
        except ImportError as e:
            print(f"❌ {package} - FAILED: {e}")
            failed_imports.append(package)
    
    return failed_imports

def main():
    """Main verification function"""
    print("🚀 Verifying Ceramic Armor ML Pipeline Installation")
    print("=" * 60)
    
    failed_imports = test_imports()
    
    print("\n" + "=" * 60)
    if not failed_imports:
        print("🎉 SUCCESS: All packages imported successfully!")
        print("✅ Installation is complete and ready to use")
        print("\nNext steps:")
        print("  - Run tests: python -m pytest tests/ -v")
        print("  - Configure API keys in config/api_keys.yaml (if needed)")
        return 0
    else:
        print("⚠️  ISSUES FOUND:")
        print(f"  - Failed imports: {', '.join(failed_imports)}")
        print("\n🔧 Try running the installation script again:")
        print("  python install_dependencies.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())