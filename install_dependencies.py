#!/usr/bin/env python3
"""
One-click installation script for Ceramic Armor ML Pipeline
Installs all dependencies with tested compatible versions
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main installation process"""
    print("🚀 Installing Ceramic Armor ML Pipeline Dependencies")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        print("⚠️  Pip upgrade failed, continuing anyway...")
    
    # Install requirements in one go
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing all dependencies"):
        print("❌ Failed to install requirements. Please check the error messages above.")
        sys.exit(1)
    
    # Verify critical packages
    critical_packages = [
        "numpy", "pandas", "scikit-learn", "xgboost", "catboost", 
        "lightgbm", "shap", "pymatgen", "matplotlib", "pytest", "loguru", "psutil"
    ]
    
    print("\n🔍 Verifying critical packages...")
    failed_imports = []
    for package in critical_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️  Some packages failed to import: {', '.join(failed_imports)}")
        print("You may need to check for conflicts or install them manually.")
        return 1
    else:
        print("\n🎉 All critical packages installed successfully!")
    
    print("\n📋 Installation Summary:")
    print("- All dependencies installed with tested compatible versions")
    print("- No version conflicts should occur")
    print("- Ready to run tests with: python -m pytest tests/ -v")
    print("- Ready to run pipeline")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())