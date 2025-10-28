#!/usr/bin/env python3
"""
Test runner script for the Ceramic Armor ML Pipeline.
Provides different test execution modes and reporting options.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install pytest")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for Ceramic Armor ML Pipeline")
    parser.add_argument(
        "--mode", 
        choices=["unit", "integration", "all", "fast", "slow", "coverage"],
        default="all",
        help="Test execution mode"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Generate coverage report (requires pytest-cov)"
    )
    parser.add_argument(
        "--html-report", 
        action="store_true",
        help="Generate HTML test report"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add test selection based on mode
    if args.mode == "unit":
        cmd.extend(["-m", "unit"])
    elif args.mode == "integration":
        cmd.extend(["-m", "integration"])
    elif args.mode == "fast":
        cmd.extend(["-m", "not slow"])
    elif args.mode == "slow":
        cmd.extend(["-m", "slow"])
    elif args.mode == "coverage":
        args.coverage = True  # Enable coverage for coverage mode
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    # Add coverage reporting
    if args.coverage:
        cmd.extend([
            "--cov=src",
            "--cov=data",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    # Add HTML report
    if args.html_report:
        cmd.extend(["--html=test_report.html", "--self-contained-html"])
    
    # Add test directory
    cmd.append("tests/")
    
    # Print test summary
    print("🧪 Ceramic Armor ML Pipeline - Test Runner")
    print(f"Mode: {args.mode}")
    print(f"Parallel: {args.parallel}")
    print(f"Coverage: {args.coverage}")
    print(f"HTML Report: {args.html_report}")
    
    # Run tests
    success = run_command(cmd, f"Tests ({args.mode} mode)")
    
    if success:
        print("\n🎉 All tests completed successfully!")
        
        if args.coverage:
            print("\n📊 Coverage report generated:")
            print("  - Terminal: See above")
            print("  - HTML: Open htmlcov/index.html")
        
        if args.html_report:
            print("\n📋 HTML test report: test_report.html")
    else:
        print("\n💥 Some tests failed. Check the output above for details.")
        sys.exit(1)


def run_specific_tests():
    """Run specific test categories with predefined configurations."""
    print("🧪 Running Ceramic Armor ML Pipeline Test Suite")
    print("="*60)
    
    test_configs = [
        {
            "name": "Unit Tests (Fast)",
            "cmd": ["python", "-m", "pytest", "-v", "-m", "unit", "tests/"],
            "description": "Fast unit tests for individual components"
        },
        {
            "name": "Integration Tests",
            "cmd": ["python", "-m", "pytest", "-v", "-m", "integration", "tests/"],
            "description": "Integration tests for component interactions"
        },
        {
            "name": "Data Collection Tests",
            "cmd": ["python", "-m", "pytest", "-v", "-k", "data_collection", "tests/"],
            "description": "Tests for data collection modules"
        },
        {
            "name": "Model Tests",
            "cmd": ["python", "-m", "pytest", "-v", "-k", "model", "tests/"],
            "description": "Tests for ML model implementations"
        }
    ]
    
    results = []
    for config in test_configs:
        success = run_command(config["cmd"], config["name"])
        results.append((config["name"], success))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name:<30} {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All test suites passed!")
        return 0
    else:
        print("\n💥 Some test suites failed!")
        return 1


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - run specific test categories
        sys.exit(run_specific_tests())
    else:
        # Arguments provided - use argument parser
        main()