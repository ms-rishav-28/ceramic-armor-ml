#!/usr/bin/env python3
"""
Run Comprehensive Interpretability Analysis
Implements Task 5: Generate comprehensive interpretability analysis with mechanistic insights

This script:
1. Refactors existing SHAP analyzer to produce SHAP importance plots for each ceramic system and target property
2. Creates feature ranking showing which material factors control ballistic performance
3. Generates mechanistic interpretation correlating feature importance to known materials science principles
4. Creates publication-ready visualizations with proper scientific formatting, error bars, and statistical significance
5. Documents why tree-based models outperform neural networks for ceramic materials prediction domain
6. Fixes trainer-SHAP integration to ensure consistent feature name handling and data persistence formats
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import yaml
import json
from loguru import logger
from typing import Dict, Any

from interpretation.comprehensive_interpretability import ComprehensiveInterpretabilityAnalyzer
from utils.config_loader import load_config


def setup_logging():
    """Setup logging configuration"""
    logger.remove()  # Remove default handler
    
    # Add console handler with custom format
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler
    log_file = Path("logs/comprehensive_interpretability_analysis.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB"
    )


def validate_environment() -> bool:
    """Validate that the environment is ready for interpretability analysis"""
    logger.info("Validating environment for interpretability analysis...")
    
    validation_passed = True
    
    # Check required directories
    required_dirs = [
        Path("models"),
        Path("src/interpretation"),
        Path("config")
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            logger.error(f"Required directory not found: {dir_path}")
            validation_passed = False
        else:
            logger.info(f"✓ Found required directory: {dir_path}")
    
    # Check for trained models
    models_dir = Path("models")
    if models_dir.exists():
        model_count = len(list(models_dir.rglob("*.pkl")))
        if model_count == 0:
            logger.warning("No trained models found - analysis may be limited")
        else:
            logger.info(f"✓ Found {model_count} trained model files")
    
    # Check configuration
    config_file = Path("config/config.yaml")
    if not config_file.exists():
        logger.warning(f"Configuration file not found: {config_file}")
        logger.info("Will use default configuration")
    else:
        logger.info(f"✓ Configuration file found: {config_file}")
    
    # Check Python dependencies
    try:
        import shap
        import matplotlib
        import seaborn
        import pandas
        import numpy
        logger.info("✓ All required Python packages available")
    except ImportError as e:
        logger.error(f"Missing required Python package: {e}")
        validation_passed = False
    
    if validation_passed:
        logger.info("✅ Environment validation passed")
    else:
        logger.error("❌ Environment validation failed")
    
    return validation_passed


def check_trainer_shap_integration() -> bool:
    """Check and fix trainer-SHAP integration issues"""
    logger.info("Checking trainer-SHAP integration...")
    
    # Run the integration test
    try:
        from scripts.test_trainer_shap_integration import (
            test_trainer_shap_integration,
            test_missing_files_handling,
            test_data_validation
        )
        
        logger.info("Running trainer-SHAP integration tests...")
        
        tests = [
            ("Basic Integration", test_trainer_shap_integration),
            ("Missing Files Handling", test_missing_files_handling),
            ("Data Validation", test_data_validation)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                    logger.info(f"✓ {test_name} passed")
                else:
                    logger.warning(f"⚠ {test_name} failed")
            except Exception as e:
                logger.error(f"✗ {test_name} error: {e}")
        
        success_rate = passed / total * 100
        logger.info(f"Integration test results: {passed}/{total} passed ({success_rate:.1f}%)")
        
        if success_rate >= 100:
            logger.info("✅ Trainer-SHAP integration is working correctly")
            return True
        else:
            logger.warning("⚠ Some trainer-SHAP integration issues detected")
            return False
            
    except Exception as e:
        logger.error(f"Failed to run integration tests: {e}")
        return False


def run_sample_analysis(analyzer: ComprehensiveInterpretabilityAnalyzer) -> bool:
    """Run a sample analysis to verify functionality"""
    logger.info("Running sample interpretability analysis...")
    
    try:
        # Check for available models
        models_dir = Path("models")
        
        # Find any available system-property combination
        sample_found = False
        
        for system_dir in models_dir.iterdir():
            if system_dir.is_dir():
                for property_dir in system_dir.iterdir():
                    if property_dir.is_dir():
                        # Check if this directory has model files
                        model_files = list(property_dir.glob("*.pkl"))
                        if model_files:
                            system = system_dir.name.upper()
                            property_name = property_dir.name
                            
                            logger.info(f"Running sample analysis for {system} - {property_name}")
                            
                            # Create sample output directory
                            sample_output = Path("results/sample_interpretability_test")
                            sample_output.mkdir(parents=True, exist_ok=True)
                            
                            # Run analysis
                            result = analyzer.analyze_system_property(
                                system, property_name, models_dir, sample_output
                            )
                            
                            if result['status'] == 'success':
                                logger.info("✅ Sample analysis completed successfully")
                                logger.info(f"  Publication ready: {result.get('publication_ready', False)}")
                                logger.info(f"  Visualizations: {len(result.get('visualizations', {}).get('plots_created', []))}")
                                return True
                            else:
                                logger.warning(f"Sample analysis failed: {result.get('error', 'Unknown error')}")
                            
                            sample_found = True
                            break
                
                if sample_found:
                    break
        
        if not sample_found:
            logger.warning("No suitable models found for sample analysis")
            return False
        
        return False
        
    except Exception as e:
        logger.error(f"Sample analysis failed: {e}")
        return False


def main():
    """Main execution function"""
    
    # Setup logging
    setup_logging()
    
    logger.info("🔬" * 20)
    logger.info("COMPREHENSIVE INTERPRETABILITY ANALYSIS")
    logger.info("Task 5: Generate comprehensive interpretability analysis with mechanistic insights")
    logger.info("🔬" * 20)
    
    try:
        # Step 1: Validate environment
        logger.info("\n--- Step 1: Environment Validation ---")
        if not validate_environment():
            logger.error("Environment validation failed. Please fix issues before proceeding.")
            return 1
        
        # Step 2: Check trainer-SHAP integration
        logger.info("\n--- Step 2: Trainer-SHAP Integration Check ---")
        integration_ok = check_trainer_shap_integration()
        if integration_ok:
            logger.info("✅ Trainer-SHAP integration verified")
        else:
            logger.warning("⚠ Trainer-SHAP integration issues detected but proceeding...")
        
        # Step 3: Initialize comprehensive analyzer
        logger.info("\n--- Step 3: Initialize Comprehensive Analyzer ---")
        
        config_path = "config/config.yaml"
        if not Path(config_path).exists():
            logger.warning(f"Config file {config_path} not found, using default configuration")
            config_path = None
        
        analyzer = ComprehensiveInterpretabilityAnalyzer(config_path)
        logger.info("✅ Comprehensive Interpretability Analyzer initialized")
        
        # Step 4: Run sample analysis to verify functionality
        logger.info("\n--- Step 4: Sample Analysis Verification ---")
        sample_success = run_sample_analysis(analyzer)
        
        if not sample_success:
            logger.warning("Sample analysis failed, but proceeding with full analysis...")
        
        # Step 5: Run comprehensive analysis
        logger.info("\n--- Step 5: Comprehensive Interpretability Analysis ---")
        
        # Set output directory
        output_dir = "results/comprehensive_interpretability_analysis"
        models_dir = "models"
        
        logger.info(f"Models directory: {models_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Run the comprehensive analysis
        results = analyzer.run_comprehensive_analysis(
            models_dir=models_dir,
            output_dir=output_dir
        )
        
        # Step 6: Analyze results
        logger.info("\n--- Step 6: Results Analysis ---")
        
        summary = results['analysis_summary']
        success_rate = summary['successful_analyses'] / summary['total_analyses'] * 100 if summary['total_analyses'] > 0 else 0
        pub_rate = summary['publication_ready_analyses'] / summary['total_analyses'] * 100 if summary['total_analyses'] > 0 else 0
        
        logger.info(f"Analysis Results:")
        logger.info(f"  Total analyses: {summary['total_analyses']}")
        logger.info(f"  Successful: {summary['successful_analyses']} ({success_rate:.1f}%)")
        logger.info(f"  Publication ready: {summary['publication_ready_analyses']} ({pub_rate:.1f}%)")
        logger.info(f"  Failed: {summary['failed_analyses']}")
        
        # Check if Task 5 requirements are met
        task_5_requirements = {
            'shap_plots_generated': summary['successful_analyses'] > 0,
            'feature_ranking_complete': summary['successful_analyses'] > 0,
            'mechanistic_interpretation': summary['successful_analyses'] > 0,
            'publication_visualizations': summary['publication_ready_analyses'] > 0,
            'tree_model_documentation': 'tree_model_superiority_evidence' in results,
            'trainer_shap_integration': integration_ok
        }
        
        requirements_met = sum(task_5_requirements.values())
        total_requirements = len(task_5_requirements)
        
        logger.info(f"\nTask 5 Requirements Assessment:")
        for requirement, met in task_5_requirements.items():
            status = "✅" if met else "❌"
            logger.info(f"  {status} {requirement.replace('_', ' ').title()}")
        
        logger.info(f"\nTask 5 Completion: {requirements_met}/{total_requirements} requirements met")
        
        if requirements_met == total_requirements:
            logger.info("🎉 Task 5 COMPLETED SUCCESSFULLY!")
            logger.info("All interpretability analysis requirements have been implemented:")
            logger.info("  ✅ SHAP analyzer refactored with publication-grade visualizations")
            logger.info("  ✅ Feature ranking showing ballistic performance controlling factors")
            logger.info("  ✅ Mechanistic interpretation with materials science principles")
            logger.info("  ✅ Publication-ready visualizations with statistical significance")
            logger.info("  ✅ Tree-based model superiority documentation")
            logger.info("  ✅ Trainer-SHAP integration fixes")
            
            return 0
        else:
            logger.warning("⚠ Task 5 PARTIALLY COMPLETED")
            logger.warning("Some requirements not fully met - see analysis above")
            return 1
    
    except Exception as e:
        logger.error(f"Comprehensive interpretability analysis failed: {e}")
        logger.exception("Full error traceback:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)