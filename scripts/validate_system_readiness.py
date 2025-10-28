"""
System Readiness Validation Script
Validates that the ceramic armor ML system is ready for production testing
Uses only built-in Python modules for maximum compatibility
"""

import sys
import os
import json
from pathlib import Path
import time

class SystemReadinessValidator:
    """
    Validates system readiness for production testing
    Checks configuration, directory structure, and basic functionality
    """
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def log(self, message, level="INFO"):
        """Simple logging function"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def test_python_environment(self):
        """Test Python environment and basic modules"""
        self.log("Testing Python environment...")
        
        results = {
            'python_version': sys.version,
            'python_executable': sys.executable,
            'platform': sys.platform,
            'modules_available': {}
        }
        
        # Test critical modules
        critical_modules = [
            'pathlib', 'json', 'os', 'sys', 'time',
            'yaml', 'pandas', 'numpy', 'sklearn', 
            'xgboost', 'catboost', 'shap', 'loguru'
        ]
        
        for module in critical_modules:
            try:
                __import__(module)
                results['modules_available'][module] = True
                self.log(f"  ✓ {module} available")
            except ImportError:
                results['modules_available'][module] = False
                self.log(f"  ✗ {module} not available", "WARNING")
        
        # Calculate availability percentage
        available_count = sum(results['modules_available'].values())
        total_count = len(critical_modules)
        availability_pct = available_count / total_count * 100
        
        results['module_availability_percent'] = availability_pct
        results['critical_modules_available'] = availability_pct >= 80
        
        self.log(f"Module availability: {availability_pct:.1f}% ({available_count}/{total_count})")
        
        self.results['python_environment'] = results
        return results['critical_modules_available']
    
    def test_configuration_files(self):
        """Test that configuration files exist and are valid"""
        self.log("Testing configuration files...")
        
        config_files = {
            'config/config.yaml': 'Main configuration',
            'config/model_params.yaml': 'Model parameters',
            'config/api_keys.yaml': 'API keys (optional)',
            'requirements.txt': 'Python dependencies'
        }
        
        results = {
            'files_found': {},
            'files_valid': {},
            'total_files': len(config_files),
            'found_files': 0,
            'valid_files': 0
        }
        
        for file_path, description in config_files.items():
            path = Path(file_path)
            
            # Check if file exists
            if path.exists():
                results['files_found'][file_path] = True
                results['found_files'] += 1
                self.log(f"  ✓ Found: {file_path}")
                
                # Basic validation
                try:
                    if file_path.endswith('.yaml'):
                        # Try to read as text (can't parse YAML without pyyaml)
                        try:
                            with open(path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if len(content) > 0 and 'ceramic_systems' in content:
                                    results['files_valid'][file_path] = True
                                    results['valid_files'] += 1
                                    self.log(f"    ✓ Valid YAML structure")
                                else:
                                    results['files_valid'][file_path] = False
                                    self.log(f"    ✗ Invalid or empty YAML", "WARNING")
                        except UnicodeDecodeError:
                            # Try with different encoding
                            try:
                                with open(path, 'r', encoding='latin-1') as f:
                                    content = f.read()
                                    if len(content) > 0:
                                        results['files_valid'][file_path] = True
                                        results['valid_files'] += 1
                                        self.log(f"    ✓ YAML file readable (non-UTF8)")
                                    else:
                                        results['files_valid'][file_path] = False
                                        self.log(f"    ✗ Empty YAML file", "WARNING")
                            except Exception as e2:
                                results['files_valid'][file_path] = False
                                self.log(f"    ✗ Cannot read YAML: {e2}", "WARNING")
                    else:
                        # For non-YAML files, just check they're not empty
                        if path.stat().st_size > 0:
                            results['files_valid'][file_path] = True
                            results['valid_files'] += 1
                            self.log(f"    ✓ File not empty")
                        else:
                            results['files_valid'][file_path] = False
                            self.log(f"    ✗ File is empty", "WARNING")
                            
                except Exception as e:
                    results['files_valid'][file_path] = False
                    self.log(f"    ✗ Validation error: {e}", "ERROR")
            else:
                results['files_found'][file_path] = False
                self.log(f"  ✗ Missing: {file_path}", "WARNING")
        
        # Calculate success rate
        found_pct = results['found_files'] / results['total_files'] * 100
        valid_pct = results['valid_files'] / results['total_files'] * 100
        
        results['found_percentage'] = found_pct
        results['valid_percentage'] = valid_pct
        results['configuration_ready'] = found_pct >= 75 and valid_pct >= 50
        
        self.log(f"Configuration files: {results['found_files']}/{results['total_files']} found, {results['valid_files']}/{results['total_files']} valid")
        
        self.results['configuration'] = results
        return results['configuration_ready']
    
    def test_directory_structure(self):
        """Test directory structure for data pipeline"""
        self.log("Testing directory structure...")
        
        required_dirs = {
            'src/': 'Source code',
            'src/data_collection/': 'Data collection modules',
            'src/models/': 'Model implementations',
            'src/training/': 'Training modules',
            'src/evaluation/': 'Evaluation modules',
            'src/interpretation/': 'Interpretation modules',
            'data/': 'Data directory',
            'data/raw/': 'Raw data storage',
            'data/processed/': 'Processed data storage',
            'data/features/': 'Feature data storage',
            'results/': 'Results directory',
            'results/models/': 'Trained models',
            'results/figures/': 'Generated figures',
            'results/reports/': 'Analysis reports',
            'scripts/': 'Execution scripts',
            'config/': 'Configuration files'
        }
        
        results = {
            'directories_found': {},
            'directories_created': {},
            'total_directories': len(required_dirs),
            'existing_directories': 0,
            'created_directories': 0
        }
        
        for dir_path, description in required_dirs.items():
            path = Path(dir_path)
            
            if path.exists() and path.is_dir():
                results['directories_found'][dir_path] = True
                results['existing_directories'] += 1
                self.log(f"  ✓ Exists: {dir_path}")
            else:
                results['directories_found'][dir_path] = False
                
                # Try to create directory
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    results['directories_created'][dir_path] = True
                    results['created_directories'] += 1
                    self.log(f"  ✓ Created: {dir_path}")
                except Exception as e:
                    results['directories_created'][dir_path] = False
                    self.log(f"  ✗ Cannot create {dir_path}: {e}", "ERROR")
        
        # Calculate success rate
        total_ready = results['existing_directories'] + results['created_directories']
        ready_pct = total_ready / results['total_directories'] * 100
        
        results['ready_percentage'] = ready_pct
        results['structure_ready'] = ready_pct >= 90
        
        self.log(f"Directory structure: {total_ready}/{results['total_directories']} ready ({ready_pct:.1f}%)")
        
        self.results['directory_structure'] = results
        return results['structure_ready']
    
    def test_source_code_structure(self):
        """Test that source code files exist"""
        self.log("Testing source code structure...")
        
        critical_files = {
            'src/__init__.py': 'Main package init',
            'src/training/trainer.py': 'Main trainer',
            'src/evaluation/metrics.py': 'Evaluation metrics',
            'src/interpretation/shap_analyzer.py': 'SHAP analysis',
            'src/data_collection/materials_project_collector.py': 'Materials Project collector',
            'src/models/base_model.py': 'Base model class',
            'src/utils/config_loader.py': 'Configuration loader',
            'src/utils/logger.py': 'Logging utilities'
        }
        
        results = {
            'files_found': {},
            'files_size': {},
            'total_files': len(critical_files),
            'found_files': 0,
            'substantial_files': 0  # Files with meaningful content
        }
        
        for file_path, description in critical_files.items():
            path = Path(file_path)
            
            if path.exists() and path.is_file():
                results['files_found'][file_path] = True
                results['found_files'] += 1
                
                # Check file size
                file_size = path.stat().st_size
                results['files_size'][file_path] = file_size
                
                if file_size > 1000:  # At least 1KB suggests substantial content
                    results['substantial_files'] += 1
                    self.log(f"  ✓ Found: {file_path} ({file_size} bytes)")
                else:
                    self.log(f"  ⚠ Found but small: {file_path} ({file_size} bytes)", "WARNING")
            else:
                results['files_found'][file_path] = False
                results['files_size'][file_path] = 0
                self.log(f"  ✗ Missing: {file_path}", "WARNING")
        
        # Calculate success rate
        found_pct = results['found_files'] / results['total_files'] * 100
        substantial_pct = results['substantial_files'] / results['total_files'] * 100
        
        results['found_percentage'] = found_pct
        results['substantial_percentage'] = substantial_pct
        results['code_ready'] = found_pct >= 80 and substantial_pct >= 60
        
        self.log(f"Source code: {results['found_files']}/{results['total_files']} found, {results['substantial_files']} substantial")
        
        self.results['source_code'] = results
        return results['code_ready']
    
    def test_data_availability(self):
        """Test if any data files are available"""
        self.log("Testing data availability...")
        
        data_locations = {
            'data/raw/': 'Raw data files',
            'data/processed/': 'Processed data files',
            'data/features/': 'Feature data files'
        }
        
        results = {
            'locations': {},
            'total_files': 0,
            'total_size_mb': 0,
            'has_data': False
        }
        
        for location, description in data_locations.items():
            path = Path(location)
            location_info = {
                'exists': False,
                'file_count': 0,
                'csv_files': 0,
                'total_size': 0
            }
            
            if path.exists():
                location_info['exists'] = True
                
                # Count files
                all_files = list(path.rglob("*"))
                csv_files = list(path.rglob("*.csv"))
                
                location_info['file_count'] = len([f for f in all_files if f.is_file()])
                location_info['csv_files'] = len(csv_files)
                
                # Calculate total size
                total_size = sum(f.stat().st_size for f in all_files if f.is_file())
                location_info['total_size'] = total_size
                results['total_size_mb'] += total_size / (1024 * 1024)
                
                self.log(f"  ✓ {location}: {location_info['file_count']} files ({location_info['csv_files']} CSV)")
            else:
                self.log(f"  ✗ {location}: Directory not found")
            
            results['locations'][location] = location_info
            results['total_files'] += location_info['file_count']
        
        results['has_data'] = results['total_files'] > 0
        
        if results['has_data']:
            self.log(f"Data summary: {results['total_files']} files, {results['total_size_mb']:.1f} MB total")
        else:
            self.log("No data files found - data collection needed", "WARNING")
        
        self.results['data_availability'] = results
        return True  # Not critical for initial setup
    
    def test_model_availability(self):
        """Test if any trained models are available"""
        self.log("Testing model availability...")
        
        models_dir = Path('results/models')
        
        results = {
            'models_dir_exists': False,
            'model_files': 0,
            'systems_with_models': [],
            'properties_with_models': [],
            'has_models': False
        }
        
        if models_dir.exists():
            results['models_dir_exists'] = True
            
            # Find model files
            model_files = list(models_dir.rglob("*.pkl"))
            results['model_files'] = len(model_files)
            
            # Analyze model structure
            systems = set()
            properties = set()
            
            for model_file in model_files:
                parts = model_file.parts
                if len(parts) >= 3:  # results/models/system/property/model.pkl
                    system = parts[2]
                    if len(parts) >= 4:
                        prop = parts[3]
                        systems.add(system)
                        properties.add(prop)
            
            results['systems_with_models'] = list(systems)
            results['properties_with_models'] = list(properties)
            results['has_models'] = len(model_files) > 0
            
            if results['has_models']:
                self.log(f"  ✓ Found {results['model_files']} model files")
                self.log(f"  ✓ Systems: {len(systems)} ({', '.join(list(systems)[:3])}{'...' if len(systems) > 3 else ''})")
                self.log(f"  ✓ Properties: {len(properties)} ({', '.join(list(properties)[:3])}{'...' if len(properties) > 3 else ''})")
            else:
                self.log("  ⚠ No trained models found - training needed", "WARNING")
        else:
            self.log("  ⚠ Models directory not found - training needed", "WARNING")
        
        self.results['model_availability'] = results
        return True  # Not critical for initial setup
    
    def generate_readiness_report(self):
        """Generate comprehensive readiness report"""
        self.log("Generating readiness report...")
        
        total_time = time.time() - self.start_time
        
        # Determine overall readiness
        critical_tests = [
            self.results.get('python_environment', {}).get('critical_modules_available', False),
            self.results.get('configuration', {}).get('configuration_ready', False),
            self.results.get('directory_structure', {}).get('structure_ready', False),
            self.results.get('source_code', {}).get('code_ready', False)
        ]
        
        overall_ready = all(critical_tests)
        ready_count = sum(critical_tests)
        
        # Create report
        report_content = f"""# System Readiness Validation Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Validation Time:** {total_time:.2f} seconds

## Executive Summary

**Overall Status:** {'READY FOR PRODUCTION TESTING' if overall_ready else 'REQUIRES SETUP'}
**Critical Tests Passed:** {ready_count}/4

## Detailed Results

### 1. Python Environment
"""
        
        env_results = self.results.get('python_environment', {})
        report_content += f"""
- **Python Version:** {env_results.get('python_version', 'Unknown')}
- **Module Availability:** {env_results.get('module_availability_percent', 0):.1f}%
- **Status:** {'READY' if env_results.get('critical_modules_available', False) else 'NEEDS PACKAGES'}

### 2. Configuration Files
"""
        
        config_results = self.results.get('configuration', {})
        report_content += f"""
- **Files Found:** {config_results.get('found_files', 0)}/{config_results.get('total_files', 0)}
- **Files Valid:** {config_results.get('valid_files', 0)}/{config_results.get('total_files', 0)}
- **Status:** {'READY' if config_results.get('configuration_ready', False) else 'NEEDS CONFIG'}

### 3. Directory Structure
"""
        
        dir_results = self.results.get('directory_structure', {})
        report_content += f"""
- **Directories Ready:** {dir_results.get('existing_directories', 0) + dir_results.get('created_directories', 0)}/{dir_results.get('total_directories', 0)}
- **Ready Percentage:** {dir_results.get('ready_percentage', 0):.1f}%
- **Status:** {'READY' if dir_results.get('structure_ready', False) else 'NEEDS DIRS'}

### 4. Source Code
"""
        
        code_results = self.results.get('source_code', {})
        report_content += f"""
- **Files Found:** {code_results.get('found_files', 0)}/{code_results.get('total_files', 0)}
- **Substantial Files:** {code_results.get('substantial_files', 0)}/{code_results.get('total_files', 0)}
- **Status:** {'READY' if code_results.get('code_ready', False) else 'NEEDS CODE'}

### 5. Data Availability (Optional)
"""
        
        data_results = self.results.get('data_availability', {})
        report_content += f"""
- **Total Files:** {data_results.get('total_files', 0)}
- **Total Size:** {data_results.get('total_size_mb', 0):.1f} MB
- **Status:** {'DATA AVAILABLE' if data_results.get('has_data', False) else 'NO DATA (run collection first)'}

### 6. Model Availability (Optional)
"""
        
        model_results = self.results.get('model_availability', {})
        report_content += f"""
- **Model Files:** {model_results.get('model_files', 0)}
- **Systems with Models:** {len(model_results.get('systems_with_models', []))}
- **Status:** {'MODELS AVAILABLE' if model_results.get('has_models', False) else 'NO MODELS (run training first)'}

## Next Steps

"""
        
        if overall_ready:
            report_content += """**System is ready for production validation!**

You can now run:
1. `python scripts/07_production_validation.py` - Full production validation
2. `python scripts/07_2_validate_shap_publication.py` - SHAP analysis validation

"""
        else:
            report_content += """**System requires setup before production validation:**

"""
            if not env_results.get('critical_modules_available', False):
                report_content += "1. **Install Python packages:** `pip install -r requirements.txt`\n"
            
            if not config_results.get('configuration_ready', False):
                report_content += "2. **Set up configuration files** in the `config/` directory\n"
            
            if not dir_results.get('structure_ready', False):
                report_content += "3. **Fix directory structure** (some directories couldn't be created)\n"
            
            if not code_results.get('code_ready', False):
                report_content += "4. **Ensure source code files are complete**\n"
        
        report_content += f"""
## Technical Details

- **Validation Time:** {total_time:.2f} seconds
- **Python Executable:** {env_results.get('python_executable', 'Unknown')}
- **Platform:** {env_results.get('platform', 'Unknown')}

---
*Generated by System Readiness Validator*
"""
        
        # Save report
        report_dir = Path("results/reports/system_readiness")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / "system_readiness_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Save detailed results as JSON
        results_file = report_dir / "readiness_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.log(f"✓ Readiness report saved: {report_file}")
        self.log(f"✓ Detailed results saved: {results_file}")
        
        return str(report_file), overall_ready
    
    def run_full_validation(self):
        """Run complete system readiness validation"""
        self.log("="*60)
        self.log("CERAMIC ARMOR ML SYSTEM - READINESS VALIDATION")
        self.log("="*60)
        
        # Run all tests
        tests = [
            ("Python Environment", self.test_python_environment),
            ("Configuration Files", self.test_configuration_files),
            ("Directory Structure", self.test_directory_structure),
            ("Source Code Structure", self.test_source_code_structure),
            ("Data Availability", self.test_data_availability),
            ("Model Availability", self.test_model_availability)
        ]
        
        test_results = []
        for test_name, test_func in tests:
            self.log(f"\n--- {test_name} ---")
            try:
                result = test_func()
                test_results.append((test_name, result))
            except Exception as e:
                self.log(f"Test failed with exception: {e}", "ERROR")
                test_results.append((test_name, False))
        
        # Generate report
        report_path, overall_ready = self.generate_readiness_report()
        
        # Final summary
        self.log("\n" + "="*60)
        self.log("SYSTEM READINESS VALIDATION COMPLETE")
        self.log("="*60)
        
        passed_tests = sum(1 for _, result in test_results if result)
        self.log(f"Tests passed: {passed_tests}/{len(test_results)}")
        self.log(f"Overall ready: {'Yes' if overall_ready else 'No'}")
        self.log(f"Report: {report_path}")
        
        if overall_ready:
            self.log("🎉 SYSTEM IS READY FOR PRODUCTION VALIDATION! 🎉")
        else:
            self.log("⚠️ SYSTEM REQUIRES SETUP BEFORE PRODUCTION VALIDATION")
        
        return overall_ready


def main():
    """Main execution function"""
    try:
        validator = SystemReadinessValidator()
        ready = validator.run_full_validation()
        sys.exit(0 if ready else 1)
    except Exception as e:
        print(f"ERROR: System readiness validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()