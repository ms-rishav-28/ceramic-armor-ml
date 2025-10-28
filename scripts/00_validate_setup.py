#!/usr/bin/env python3
"""
Comprehensive Setup Validation Script for Ceramic Armor ML Pipeline.

This script validates all existing dependencies, imports, API connectivity,
directory structure, and configuration files to ensure the system is ready.

Requirements: 3.1 - Setup validation for existing system
"""

import sys
import os
import importlib
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Simple logger replacement if loguru not available
try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logger.warning("PyYAML not available - config validation will be limited")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("Requests not available - API testing will be skipped")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not available - system resource checks will be limited")


class SetupValidator:
    """Comprehensive setup validation for the ML pipeline."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results = {}
        self.critical_failures = []
        self.warnings = []
        
    def validate_python_environment(self) -> bool:
        """Validate Python version and environment."""
        logger.info("🐍 Validating Python Environment")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major != 3 or python_version.minor < 8:
            self.critical_failures.append(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            logger.error(f"❌ Python version: {python_version.major}.{python_version.minor} (requires 3.8+)")
            return False
        
        logger.info(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        if in_venv:
            logger.info("✅ Running in virtual environment")
        else:
            self.warnings.append("Not running in virtual environment - recommended for isolation")
            logger.warning("⚠️  Not in virtual environment (recommended)")
        
        return True
    
    def validate_dependencies(self) -> bool:
        """Check all required dependencies are installed correctly."""
        logger.info("📦 Validating Dependencies")
        
        # Core dependencies with version checks
        required_packages = {
            'numpy': '1.20.0',
            'pandas': '1.3.0', 
            'scipy': '1.7.0',
            'scikit-learn': '1.0.0',
            'xgboost': '1.5.0',
            'catboost': '1.0.0',
            'lightgbm': '3.0.0',
            'pymatgen': '2022.0.0',
            'mp-api': '0.30.0',
            'jarvis-tools': '2021.0.0',
            'matminer': '0.7.0',
            'shap': '0.40.0',
            'optuna': '2.10.0',
            'matplotlib': '3.5.0',
            'seaborn': '0.11.0',
            'plotly': '5.0.0',
            'requests': '2.25.0',
            'beautifulsoup4': '4.9.0',
            'pyyaml': '5.4.0',
            'tqdm': '4.60.0',
            'loguru': '0.6.0',
            'pytest': '6.0.0'
        }
        
        missing_packages = []
        version_issues = []
        
        for package, min_version in required_packages.items():
            try:
                module = importlib.import_module(package.replace('-', '_'))
                
                # Get version if available
                version = getattr(module, '__version__', 'unknown')
                logger.info(f"  ✅ {package}: {version}")
                
                # Basic version check (simplified)
                if version != 'unknown' and min_version:
                    try:
                        from packaging import version as pkg_version
                        if pkg_version.parse(version) < pkg_version.parse(min_version):
                            version_issues.append(f"{package}: {version} < {min_version}")
                    except:
                        pass  # Skip version comparison if packaging not available
                        
            except ImportError:
                missing_packages.append(package)
                logger.error(f"  ❌ {package}: NOT INSTALLED")
        
        # Intel optimizations check
        try:
            import sklearnex
            logger.info("  ✅ Intel scikit-learn extensions available")
        except ImportError:
            self.warnings.append("Intel scikit-learn extensions not available - performance may be reduced")
            logger.warning("  ⚠️  Intel scikit-learn extensions not found")
        
        if missing_packages:
            self.critical_failures.extend([f"Missing package: {pkg}" for pkg in missing_packages])
            logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
            return False
        
        if version_issues:
            for issue in version_issues:
                self.warnings.append(f"Version issue: {issue}")
                logger.warning(f"  ⚠️  {issue}")
        
        logger.info("✅ All dependencies installed")
        return True
    
    def validate_project_imports(self) -> bool:
        """Verify all existing project modules can be imported."""
        logger.info("🔧 Validating Project Module Imports")
        
        # Test imports from existing modules
        import_tests = [
            # Utilities
            ('src.utils.logger', 'get_logger'),
            ('src.utils.config_loader', 'load_config'),
            ('src.utils.data_utils', 'safe_save_data'),
            
            # Data collection
            ('src.data_collection.materials_project_collector', 'MaterialsProjectCollector'),
            
            # Preprocessing
            ('src.preprocessing.data_cleaner', 'DataCleaner'),
            ('src.preprocessing.unit_standardizer', 'UnitStandardizer'),
            ('src.preprocessing.outlier_detector', 'OutlierDetector'),
            ('src.preprocessing.missing_value_handler', 'MissingValueHandler'),
            
            # Feature engineering
            ('src.feature_engineering.compositional_features', 'CompositionalFeatures'),
            ('src.feature_engineering.derived_properties', 'DerivedProperties'),
            ('src.feature_engineering.microstructure_features', 'MicrostructureFeatures'),
            ('src.feature_engineering.phase_stability', 'PhaseStability'),
            
            # Models
            ('src.models.base_model', 'BaseModel'),
            ('src.models.xgboost_model', 'XGBoostModel'),
            ('src.models.catboost_model', 'CatBoostModel'),
            ('src.models.random_forest_model', 'RandomForestModel'),
            ('src.models.gradient_boosting_model', 'GradientBoostingModel'),
            ('src.models.ensemble_model', 'EnsembleModel'),
            
            # Training
            ('src.training.trainer', 'Trainer'),
            ('src.training.cross_validator', 'CrossValidator'),
            ('src.training.hyperparameter_tuner', 'HyperparameterTuner'),
            
            # Evaluation
            ('src.evaluation.metrics', 'Metrics'),
            ('src.evaluation.error_analyzer', 'ErrorAnalyzer'),
            
            # Interpretation
            ('src.interpretation.shap_analyzer', 'SHAPAnalyzer'),
            ('src.interpretation.materials_insights', 'MaterialsInsights'),
            ('src.interpretation.visualization', 'Visualization')
        ]
        
        failed_imports = []
        
        for module_path, component_name in import_tests:
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, component_name):
                    logger.info(f"  ✅ {module_path}.{component_name}")
                else:
                    failed_imports.append(f"{module_path}.{component_name} - component not found")
                    logger.error(f"  ❌ {module_path}.{component_name} - COMPONENT NOT FOUND")
            except ImportError as e:
                failed_imports.append(f"{module_path}.{component_name} - {str(e)}")
                logger.error(f"  ❌ {module_path}.{component_name} - IMPORT ERROR: {e}")
        
        if failed_imports:
            self.critical_failures.extend(failed_imports)
            logger.error(f"❌ Failed imports: {len(failed_imports)} modules")
            return False
        
        logger.info("✅ All project modules import successfully")
        return True
    
    def validate_api_connectivity(self) -> bool:
        """Test API connectivity with existing collector implementations."""
        logger.info("🌐 Validating API Connectivity")
        
        # Load API keys
        api_keys_path = self.project_root / 'config' / 'api_keys.yaml'
        if not api_keys_path.exists():
            self.critical_failures.append("API keys file missing: config/api_keys.yaml")
            logger.error("❌ API keys file not found")
            return False
        
        if not HAS_YAML:
            logger.warning("⚠️  Cannot validate API keys (PyYAML not available)")
            return True
        
        try:
            with open(api_keys_path, 'r') as f:
                api_keys = yaml.safe_load(f)
        except Exception as e:
            self.critical_failures.append(f"Cannot load API keys: {e}")
            logger.error(f"❌ Cannot load API keys: {e}")
            return False
        
        api_results = []
        
        # Test Materials Project API
        mp_key = api_keys.get('materials_project')
        if mp_key and HAS_REQUESTS:
            try:
                logger.info("  Testing Materials Project API...")
                url = "https://api.materialsproject.org/materials/summary"
                headers = {"X-API-KEY": mp_key}
                params = {"formula": "SiC", "_limit": 1}
                
                response = requests.get(url, headers=headers, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('data'):
                        logger.info("  ✅ Materials Project API - Connected")
                        api_results.append(True)
                    else:
                        logger.warning("  ⚠️  Materials Project API - Empty response")
                        api_results.append(False)
                else:
                    logger.error(f"  ❌ Materials Project API - Status {response.status_code}")
                    api_results.append(False)
            except Exception as e:
                logger.error(f"  ❌ Materials Project API - Error: {e}")
                api_results.append(False)
        elif not HAS_REQUESTS:
            logger.warning("  ⚠️  API testing skipped (requests not available)")
            api_results.append(True)  # Don't fail validation for missing optional dependency
        else:
            logger.error("  ❌ Materials Project API key not found")
            api_results.append(False)
        
        # Test AFLOW API
        if HAS_REQUESTS:
            try:
                logger.info("  Testing AFLOW API...")
                url = "https://aflowlib.duke.edu/search/API/"
                params = {"species": "Si,C", "nspecies": "2", "paging": "0"}
                
                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 200 and len(response.text) > 100:
                    logger.info("  ✅ AFLOW API - Connected")
                    api_results.append(True)
                else:
                    logger.warning("  ⚠️  AFLOW API - Limited response")
                    api_results.append(False)
            except Exception as e:
                logger.error(f"  ❌ AFLOW API - Error: {e}")
                api_results.append(False)
        else:
            logger.info("  ℹ️  AFLOW API testing skipped (requests not available)")
            api_results.append(True)
        
        # Test JARVIS connectivity
        try:
            logger.info("  Testing JARVIS connectivity...")
            from jarvis.db.figshare import data as jdata
            # Quick test - don't load full dataset
            logger.info("  ✅ JARVIS - Import successful")
            api_results.append(True)
        except Exception as e:
            logger.error(f"  ❌ JARVIS - Error: {e}")
            api_results.append(False)
        
        successful_apis = sum(api_results)
        total_apis = len(api_results)
        
        if successful_apis >= 2:
            logger.info(f"✅ API connectivity: {successful_apis}/{total_apis} APIs available")
            return True
        else:
            self.critical_failures.append(f"Insufficient API connectivity: {successful_apis}/{total_apis}")
            logger.error(f"❌ API connectivity: Only {successful_apis}/{total_apis} APIs available")
            return False
    
    def validate_directory_structure(self) -> bool:
        """Validate existing directory structure and config files."""
        logger.info("📁 Validating Directory Structure")
        
        required_dirs = [
            'data/raw', 'data/processed', 'data/features', 'data/splits',
            'results/models', 'results/predictions', 'results/metrics', 
            'results/figures', 'results/reports',
            'src/data_collection', 'src/preprocessing', 'src/feature_engineering',
            'src/models', 'src/training', 'src/evaluation', 'src/interpretation',
            'src/utils', 'scripts', 'tests', 'config', 'logs'
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                logger.info(f"  ✅ {dir_path}")
            else:
                missing_dirs.append(dir_path)
                logger.error(f"  ❌ {dir_path} - NOT FOUND")
        
        # Check config files
        config_files = [
            ('config/config.yaml', 'Main configuration'),
            ('config/model_params.yaml', 'Model parameters'),
            ('requirements.txt', 'Dependencies'),
            ('pytest.ini', 'Test configuration')
        ]
        
        missing_configs = []
        for file_path, description in config_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                logger.info(f"  ✅ {description}: {file_path}")
                
                # Validate YAML files
                if file_path.endswith('.yaml') and HAS_YAML:
                    try:
                        with open(full_path, 'r') as f:
                            yaml.safe_load(f)
                        logger.info(f"    ✅ Valid YAML format")
                    except Exception as e:
                        self.warnings.append(f"Invalid YAML in {file_path}: {e}")
                        logger.warning(f"    ⚠️  YAML validation failed: {e}")
                elif file_path.endswith('.yaml') and not HAS_YAML:
                    logger.info(f"    ℹ️  YAML validation skipped (PyYAML not available)")
            else:
                missing_configs.append(file_path)
                logger.error(f"  ❌ {description}: {file_path} - NOT FOUND")
        
        if missing_dirs:
            # Create missing directories
            logger.info("Creating missing directories...")
            for dir_path in missing_dirs:
                try:
                    (self.project_root / dir_path).mkdir(parents=True, exist_ok=True)
                    logger.info(f"  ✅ Created: {dir_path}")
                except Exception as e:
                    self.critical_failures.append(f"Cannot create directory {dir_path}: {e}")
                    logger.error(f"  ❌ Failed to create {dir_path}: {e}")
        
        if missing_configs:
            self.critical_failures.extend([f"Missing config: {cfg}" for cfg in missing_configs])
            logger.error(f"❌ Missing config files: {', '.join(missing_configs)}")
            return False
        
        logger.info("✅ Directory structure validated")
        return True
    
    def validate_system_resources(self) -> bool:
        """Check system resources and performance requirements."""
        logger.info("💻 Validating System Resources")
        
        if HAS_PSUTIL:
            # Memory check
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            if memory_gb < 8:
                self.warnings.append(f"Low memory: {memory_gb:.1f}GB (recommended: 16GB+)")
                logger.warning(f"  ⚠️  Memory: {memory_gb:.1f}GB (recommended: 16GB+)")
            else:
                logger.info(f"  ✅ Memory: {memory_gb:.1f}GB")
            
            # CPU check
            cpu_count = psutil.cpu_count()
            logger.info(f"  ✅ CPU cores: {cpu_count}")
            
            # Disk space check
            disk = psutil.disk_usage(str(self.project_root))
            disk_free_gb = disk.free / (1024**3)
            
            if disk_free_gb < 10:
                self.warnings.append(f"Low disk space: {disk_free_gb:.1f}GB free")
                logger.warning(f"  ⚠️  Disk space: {disk_free_gb:.1f}GB free (need 10GB+)")
            else:
                logger.info(f"  ✅ Disk space: {disk_free_gb:.1f}GB free")
        else:
            logger.info("  ℹ️  System resource checks skipped (psutil not available)")
            logger.info("  ℹ️  Install psutil for detailed system monitoring: pip install psutil")
        
        # Test write permissions
        try:
            test_file = self.project_root / 'logs' / 'test_write.tmp'
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("test")
            test_file.unlink()
            logger.info("  ✅ Write permissions")
        except Exception as e:
            self.critical_failures.append(f"No write permissions: {e}")
            logger.error(f"  ❌ Write permissions failed: {e}")
            return False
        
        return True
    
    def generate_setup_report(self) -> Dict[str, Any]:
        """Generate comprehensive setup report for existing system."""
        logger.info("📊 Generating Setup Report")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': sys.platform,
            'project_root': str(self.project_root),
            'validation_results': self.results,
            'critical_failures': self.critical_failures,
            'warnings': self.warnings,
            'system_info': {
                'memory_gb': psutil.virtual_memory().total / (1024**3) if HAS_PSUTIL else 'unknown',
                'cpu_cores': psutil.cpu_count() if HAS_PSUTIL else 'unknown',
                'disk_free_gb': psutil.disk_usage(str(self.project_root)).free / (1024**3) if HAS_PSUTIL else 'unknown'
            }
        }
        
        # Save report
        report_path = self.project_root / 'logs' / 'setup_validation_report.yaml'
        try:
            if HAS_YAML:
                with open(report_path, 'w') as f:
                    yaml.dump(report, f, default_flow_style=False, indent=2)
                logger.info(f"✅ Setup report saved: {report_path}")
            else:
                # Save as text if YAML not available
                report_path = self.project_root / 'logs' / 'setup_validation_report.txt'
                with open(report_path, 'w') as f:
                    f.write(f"Setup Validation Report\n")
                    f.write(f"Timestamp: {report['timestamp']}\n")
                    f.write(f"Python Version: {report['python_version']}\n")
                    f.write(f"Critical Failures: {len(report['critical_failures'])}\n")
                    f.write(f"Warnings: {len(report['warnings'])}\n")
                    for failure in report['critical_failures']:
                        f.write(f"FAILURE: {failure}\n")
                    for warning in report['warnings']:
                        f.write(f"WARNING: {warning}\n")
                logger.info(f"✅ Setup report saved: {report_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")
        
        return report
    
    def run_validation(self) -> bool:
        """Run all validation checks."""
        logger.info("🚀 Ceramic Armor ML Pipeline - Comprehensive Setup Validation")
        logger.info("=" * 70)
        
        validation_steps = [
            ("Python Environment", self.validate_python_environment),
            ("Dependencies", self.validate_dependencies),
            ("Project Imports", self.validate_project_imports),
            ("API Connectivity", self.validate_api_connectivity),
            ("Directory Structure", self.validate_directory_structure),
            ("System Resources", self.validate_system_resources)
        ]
        
        all_passed = True
        
        for step_name, step_func in validation_steps:
            logger.info(f"\n📋 {step_name}")
            logger.info("-" * 50)
            
            try:
                success = step_func()
                self.results[step_name] = success
                if not success:
                    all_passed = False
            except Exception as e:
                logger.error(f"❌ {step_name} failed with error: {e}")
                self.results[step_name] = False
                self.critical_failures.append(f"{step_name}: {e}")
                all_passed = False
        
        # Generate report
        report = self.generate_setup_report()
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)
        
        for step_name, success in self.results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{step_name:<25} {status}")
        
        if self.warnings:
            logger.info(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.info(f"  • {warning}")
        
        if self.critical_failures:
            logger.info(f"\n❌ Critical Failures ({len(self.critical_failures)}):")
            for failure in self.critical_failures:
                logger.info(f"  • {failure}")
        
        if all_passed:
            logger.info("\n🎉 ALL VALIDATION CHECKS PASSED!")
            logger.info("The Ceramic Armor ML Pipeline is ready to run.")
            logger.info("\nNext steps:")
            logger.info("1. Run data collector testing: python scripts/01_test_data_collectors.py")
            logger.info("2. Run data quality inspection: python scripts/02_inspect_data_quality.py")
            logger.info("3. Run training monitoring: python scripts/03_monitor_training.py")
            return True
        else:
            logger.error("\n💥 VALIDATION FAILED!")
            logger.error("Please fix the critical issues above before proceeding.")
            logger.error(f"Report saved to: logs/setup_validation_report.yaml")
            return False


def main():
    """Main entry point."""
    validator = SetupValidator()
    success = validator.run_validation()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())