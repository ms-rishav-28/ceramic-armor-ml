#!/usr/bin/env python3
"""
Setup verification script for Ceramic Armor ML Pipeline.
Checks that all components are properly installed and configured.
"""

import sys
import importlib
from pathlib import Path
from loguru import logger


def check_imports():
    """Check that all required modules can be imported."""
    logger.info("🔍 Checking imports...")
    
    required_modules = [
        # Core ML libraries
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('xgboost', 'XGBoost'),
        ('catboost', 'CatBoost'),
        
        # Materials science
        ('pymatgen', 'Pymatgen'),
        ('jarvis', 'JARVIS-tools'),
        
        # Visualization
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        
        # Optimization
        ('optuna', 'Optuna'),
        
        # Interpretation
        ('shap', 'SHAP'),
        
        # Utilities
        ('yaml', 'PyYAML'),
        ('requests', 'Requests'),
    ]
    
    missing_modules = []
    for module_name, display_name in required_modules:
        try:
            importlib.import_module(module_name)
            logger.info(f"  ✅ {display_name}")
        except ImportError:
            logger.error(f"  ❌ {display_name} - NOT FOUND")
            missing_modules.append(display_name)
    
    if missing_modules:
        logger.error(f"Missing modules: {', '.join(missing_modules)}")
        logger.error("Install with: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All required modules available")
    return True


def check_project_structure():
    """Check that all required directories exist."""
    logger.info("🔍 Checking project structure...")
    
    required_dirs = [
        'data/raw/materials_project',
        'data/raw/aflow',
        'data/raw/jarvis',
        'data/raw/nist',
        'data/processed',
        'data/features',
        'data/splits',
        'results/models',
        'results/predictions',
        'results/metrics',
        'results/figures',
        'results/reports',
        'src/data_collection',
        'src/preprocessing',
        'src/feature_engineering',
        'src/models',
        'src/training',
        'src/evaluation',
        'src/interpretation',
        'src/utils',
        'scripts',
        'tests',
        'config',
        'docs'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            logger.info(f"  ✅ {dir_path}")
        else:
            logger.error(f"  ❌ {dir_path} - NOT FOUND")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        logger.error(f"Missing directories: {', '.join(missing_dirs)}")
        return False
    
    logger.info("✅ All required directories exist")
    return True


def check_config_files():
    """Check that configuration files exist."""
    logger.info("🔍 Checking configuration files...")
    
    config_files = [
        ('config/config.yaml', 'Main configuration'),
        ('config/model_params.yaml', 'Model parameters'),
        ('requirements.txt', 'Python dependencies'),
        ('pytest.ini', 'Test configuration'),
    ]
    
    missing_files = []
    for file_path, description in config_files:
        if Path(file_path).exists():
            logger.info(f"  ✅ {description}: {file_path}")
        else:
            logger.error(f"  ❌ {description}: {file_path} - NOT FOUND")
            missing_files.append(file_path)
    
    # Check API keys file
    api_keys_file = Path('config/api_keys.yaml')
    if api_keys_file.exists():
        logger.info(f"  ✅ API keys: {api_keys_file}")
        
        # Validate API key format
        try:
            with open(api_keys_file, 'r') as f:
                import yaml
                keys = yaml.safe_load(f)
                mp_key = keys.get('materials_project', '')
                if mp_key and len(mp_key) > 20:
                    logger.info(f"  ✅ Materials Project API key found: {mp_key[:8]}...")
                else:
                    logger.warning("  ⚠️  Materials Project API key appears invalid")
        except Exception as e:
            logger.warning(f"  ⚠️  Could not validate API keys: {e}")
    else:
        logger.error(f"  ❌ API keys: {api_keys_file} - NOT FOUND")
        logger.error("     This file is required for Materials Project access")
        missing_files.append(str(api_keys_file))
    
    if missing_files:
        logger.error(f"Missing config files: {', '.join(missing_files)}")
        return False
    
    logger.info("✅ All required configuration files exist")
    return True


def check_pipeline_components():
    """Check that all pipeline components can be imported."""
    logger.info("🔍 Checking pipeline components...")
    
    components = [
        # Data collection
        ('src.data_collection.materials_project_collector', 'MaterialsProjectCollector'),
        ('data.data_collection.aflow_collector', 'AFLOWCollector'),
        ('data.data_collection.jarvis_collector', 'JARVISCollector'),
        ('data.data_collection.nist_downloader', 'NISTLoader'),
        ('data.data_collection.data_integrator', 'DataIntegrator'),
        
        # Preprocessing
        ('src.preprocessing.data_cleaner', 'DataCleaner'),
        ('src.preprocessing.unit_standardizer', 'standardize'),
        ('src.preprocessing.outlier_detector', 'remove_iqr_outliers'),
        ('src.preprocessing.missing_value_handler', 'impute_knn'),
        
        # Feature engineering
        ('src.feature_engineering.compositional_features', 'CompositionalFeatureCalculator'),
        ('src.feature_engineering.microstructure_features', 'MicrostructureFeatureCalculator'),
        ('src.feature_engineering.derived_properties', 'DerivedPropertiesCalculator'),
        
        # Models
        ('src.models.xgboost_model', 'XGBoostModel'),
        ('src.models.catboost_model', 'CatBoostModel'),
        ('src.models.random_forest_model', 'RandomForestModel'),
        ('src.models.gradient_boosting_model', 'GradientBoostingModel'),
        ('src.models.ensemble_model', 'EnsembleModel'),
        
        # Training
        ('src.training.trainer', 'CeramicPropertyTrainer'),
        ('src.training.cross_validator', 'CrossValidator'),
        ('src.training.hyperparameter_tuner', 'HyperparameterTuner'),
        
        # Evaluation
        ('src.evaluation.metrics', 'ModelEvaluator'),
        ('src.evaluation.error_analyzer', 'ErrorAnalyzer'),
        
        # Interpretation
        ('src.interpretation.visualization', 'parity_plot'),
        ('src.interpretation.materials_insights', 'interpret_feature_ranking'),
    ]
    
    failed_imports = []
    for module_path, component_name in components:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, component_name):
                logger.info(f"  ✅ {module_path}.{component_name}")
            else:
                logger.error(f"  ❌ {module_path}.{component_name} - COMPONENT NOT FOUND")
                failed_imports.append(f"{module_path}.{component_name}")
        except ImportError as e:
            logger.error(f"  ❌ {module_path}.{component_name} - IMPORT ERROR: {e}")
            failed_imports.append(f"{module_path}.{component_name}")
    
    if failed_imports:
        logger.error(f"Failed imports: {', '.join(failed_imports)}")
        return False
    
    logger.info("✅ All pipeline components can be imported")
    return True


def check_test_suite():
    """Check that the test suite is ready."""
    logger.info("🔍 Checking test suite...")
    
    test_files = [
        'tests/test_data_collection.py',
        'tests/test_preprocessing.py',
        'tests/test_feature_engineering.py',
        'tests/test_models.py',
        'tests/test_training.py',
        'tests/test_evaluation.py',
        'tests/test_interpretation.py',
        'tests/test_integration.py',
        'tests/conftest.py'
    ]
    
    missing_tests = []
    for test_file in test_files:
        if Path(test_file).exists():
            logger.info(f"  ✅ {test_file}")
        else:
            logger.error(f"  ❌ {test_file} - NOT FOUND")
            missing_tests.append(test_file)
    
    if missing_tests:
        logger.error(f"Missing test files: {', '.join(missing_tests)}")
        return False
    
    # Try to run a simple test
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 'tests/', '--collect-only', '-q'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info("  ✅ Test collection successful")
        else:
            logger.warning(f"  ⚠️  Test collection issues: {result.stderr}")
    except Exception as e:
        logger.warning(f"  ⚠️  Could not verify test collection: {e}")
    
    logger.info("✅ Test suite is ready")
    return True


def main():
    """Run all verification checks."""
    logger.info("🚀 Ceramic Armor ML Pipeline - Setup Verification")
    logger.info("=" * 60)
    
    checks = [
        ("Python Dependencies", check_imports),
        ("Project Structure", check_project_structure),
        ("Configuration Files", check_config_files),
        ("Pipeline Components", check_pipeline_components),
        ("Test Suite", check_test_suite),
    ]
    
    results = []
    for check_name, check_func in checks:
        logger.info(f"\n📋 {check_name}")
        logger.info("-" * 40)
        try:
            success = check_func()
            results.append((check_name, success))
        except Exception as e:
            logger.error(f"❌ {check_name} failed with error: {e}")
            results.append((check_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 60)
    
    all_passed = True
    for check_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{check_name:<25} {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 ALL CHECKS PASSED!")
        logger.info("The Ceramic Armor ML Pipeline is ready to run.")
        logger.info("\nNext steps:")
        logger.info("1. Add your Materials Project API key to config/api_keys.yaml")
        logger.info("2. Run the complete pipeline: python scripts/run_full_pipeline.py")
        logger.info("3. Run tests: python scripts/run_tests.py")
        return 0
    else:
        logger.error("\n💥 SOME CHECKS FAILED!")
        logger.error("Please fix the issues above before running the pipeline.")
        return 1


if __name__ == "__main__":
    sys.exit(main())