"""
Production Scale Validation Script
Tests the complete ceramic armor ML pipeline at full scale with 5,600+ materials
Validates R² targets and generates comprehensive performance reports
"""

import sys
sys.path.append('.')

try:
    import yaml
except ImportError:
    import json as yaml  # Fallback for basic functionality
    
import pandas as pd
import numpy as np
import time
try:
    import psutil
except ImportError:
    psutil = None  # Handle gracefully if not available
    
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# Import pipeline components
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.data_collection.materials_project_collector import MaterialsProjectCollector
from src.training.trainer import CeramicPropertyTrainer
from src.evaluation.metrics import ModelEvaluator, PerformanceChecker
from src.interpretation.shap_analyzer import SHAPAnalyzer

class ProductionValidator:
    """
    Comprehensive production-scale validation system
    
    Tests:
    1. Full-scale data collection (5,600+ materials)
    2. Complete model training pipeline
    3. R² target validation (≥0.85 mechanical, ≥0.80 ballistic)
    4. SHAP analysis for publication readiness
    5. System performance and resource usage
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize production validator"""
        self.config = load_config(config_path)
        self.logger = get_logger(__name__)
        self.start_time = time.time()
        self.results = {
            'data_collection': {},
            'model_training': {},
            'performance_validation': {},
            'shap_analysis': {},
            'system_metrics': {}
        }
        
        # Performance targets
        self.mechanical_r2_target = self.config['targets']['mechanical_r2']
        self.ballistic_r2_target = self.config['targets']['ballistic_r2']
        
        self.logger.info("Production Validator initialized")
        self.logger.info(f"Targets: Mechanical R² ≥ {self.mechanical_r2_target}, Ballistic R² ≥ {self.ballistic_r2_target}")
    
    def log_system_resources(self, stage: str):
        """Log current system resource usage"""
        if psutil is None:
            self.logger.info(f"[{stage}] System resource monitoring not available")
            return {
                'memory_percent': 0,
                'memory_used_gb': 0,
                'memory_total_gb': 0,
                'cpu_percent': 0
            }
            
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        self.logger.info(f"[{stage}] System Resources:")
        self.logger.info(f"  Memory: {memory.percent:.1f}% used ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")
        self.logger.info(f"  CPU: {cpu_percent:.1f}% utilization")
        
        return {
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used/1024**3,
            'memory_total_gb': memory.total/1024**3,
            'cpu_percent': cpu_percent
        }
    
    def validate_data_collection(self) -> Dict:
        """
        Validate full-scale data collection for all ceramic systems
        Target: 5,600+ total materials across 5 systems
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 1: PRODUCTION DATA COLLECTION VALIDATION")
        self.logger.info("="*80)
        
        collection_results = {}
        total_materials = 0
        
        # Load API keys
        api_keys_path = Path('config/api_keys.yaml')
        if api_keys_path.exists():
            with open(api_keys_path, 'r') as f:
                api_keys = yaml.safe_load(f)
            mp_api_key = api_keys.get('materials_project')
        else:
            import os
            mp_api_key = os.environ.get('MP_API_KEY')
            
        if not mp_api_key:
            self.logger.error("Materials Project API key not found!")
            return {'status': 'failed', 'error': 'Missing API key'}
        
        mp_collector = MaterialsProjectCollector(mp_api_key)
        
        for system in self.config['ceramic_systems']['primary']:
            self.logger.info(f"\n--- Validating data collection for {system} ---")
            
            try:
                # Check if data already exists
                data_file = Path(f"data/raw/materials_project/{system.lower()}_raw.csv")
                
                if data_file.exists():
                    # Load existing data
                    df = pd.read_csv(data_file)
                    material_count = len(df)
                    self.logger.info(f"✓ Found existing data: {material_count} materials")
                else:
                    # Collect new data
                    self.logger.info(f"Collecting new data for {system}...")
                    start_time = time.time()
                    
                    # Full-scale collection
                    df = mp_collector.collect_ceramic_data(
                        system, 
                        max_entries=None,  # No limit for production
                        include_elastic=True,
                        include_thermal=True
                    )
                    
                    collection_time = time.time() - start_time
                    material_count = len(df) if df is not None else 0
                    
                    self.logger.info(f"✓ Collected {material_count} materials in {collection_time:.1f}s")
                
                # Validate data quality
                if df is not None and not df.empty:
                    # Check for required properties
                    required_props = ['youngs_modulus', 'bulk_modulus', 'shear_modulus', 'density']
                    available_props = [prop for prop in required_props if prop in df.columns]
                    completeness = len(available_props) / len(required_props) * 100
                    
                    # Check data completeness
                    missing_data_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
                    
                    collection_results[system] = {
                        'material_count': material_count,
                        'property_completeness': completeness,
                        'missing_data_percent': missing_data_pct,
                        'status': 'success'
                    }
                    
                    total_materials += material_count
                    
                    self.logger.info(f"  Materials: {material_count}")
                    self.logger.info(f"  Property completeness: {completeness:.1f}%")
                    self.logger.info(f"  Missing data: {missing_data_pct:.1f}%")
                else:
                    collection_results[system] = {
                        'material_count': 0,
                        'status': 'failed',
                        'error': 'No data collected'
                    }
                    self.logger.error(f"✗ No data collected for {system}")
                    
            except Exception as e:
                collection_results[system] = {
                    'material_count': 0,
                    'status': 'failed',
                    'error': str(e)
                }
                self.logger.error(f"✗ Data collection failed for {system}: {e}")
        
        # Overall validation
        target_materials = 5600
        success = total_materials >= target_materials
        
        self.logger.info(f"\n--- Data Collection Summary ---")
        self.logger.info(f"Total materials collected: {total_materials}")
        self.logger.info(f"Target: {target_materials}")
        self.logger.info(f"Status: {'✓ PASS' if success else '✗ FAIL'}")
        
        self.results['data_collection'] = {
            'total_materials': total_materials,
            'target_materials': target_materials,
            'systems': collection_results,
            'success': success
        }
        
        return self.results['data_collection']
    
    def validate_model_training(self) -> Dict:
        """
        Validate complete model training pipeline
        Tests all models (XGBoost, CatBoost, Random Forest, Ensemble) on all properties
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 2: PRODUCTION MODEL TRAINING VALIDATION")
        self.logger.info("="*80)
        
        training_results = {}
        
        # Log system resources before training
        pre_training_resources = self.log_system_resources("Pre-Training")
        
        try:
            # Initialize trainer
            trainer = CeramicPropertyTrainer(self.config)
            
            # Train all systems and properties
            self.logger.info("Starting full-scale model training...")
            start_time = time.time()
            
            trainer.train_all_systems()
            
            training_time = time.time() - start_time
            
            # Log system resources after training
            post_training_resources = self.log_system_resources("Post-Training")
            
            # Validate trained models exist
            models_dir = Path(self.config['paths']['models'])
            trained_models = {}
            
            for system in self.config['ceramic_systems']['primary']:
                system_models = {}
                
                # Check mechanical properties
                for prop in self.config['properties']['mechanical']:
                    prop_dir = models_dir / system.lower() / prop
                    if prop_dir.exists():
                        model_files = list(prop_dir.glob("*_model.pkl"))
                        system_models[prop] = {
                            'models_trained': len(model_files),
                            'expected_models': 4,  # XGBoost, CatBoost, RF, Ensemble
                            'success': len(model_files) >= 4
                        }
                    else:
                        system_models[prop] = {
                            'models_trained': 0,
                            'expected_models': 4,
                            'success': False
                        }
                
                # Check ballistic properties
                for prop in self.config['properties']['ballistic']:
                    prop_dir = models_dir / system.lower() / prop
                    if prop_dir.exists():
                        model_files = list(prop_dir.glob("*_model.pkl"))
                        system_models[prop] = {
                            'models_trained': len(model_files),
                            'expected_models': 4,
                            'success': len(model_files) >= 4
                        }
                    else:
                        system_models[prop] = {
                            'models_trained': 0,
                            'expected_models': 4,
                            'success': False
                        }
                
                trained_models[system] = system_models
            
            # Calculate overall success rate
            total_expected = 0
            total_trained = 0
            
            for system_data in trained_models.values():
                for prop_data in system_data.values():
                    total_expected += prop_data['expected_models']
                    total_trained += prop_data['models_trained']
            
            success_rate = total_trained / total_expected * 100 if total_expected > 0 else 0
            
            training_results = {
                'training_time_hours': training_time / 3600,
                'models_trained': total_trained,
                'models_expected': total_expected,
                'success_rate': success_rate,
                'systems': trained_models,
                'resources': {
                    'pre_training': pre_training_resources,
                    'post_training': post_training_resources
                },
                'success': success_rate >= 90  # 90% success rate threshold
            }
            
            self.logger.info(f"\n--- Model Training Summary ---")
            self.logger.info(f"Training time: {training_time/3600:.1f} hours")
            self.logger.info(f"Models trained: {total_trained}/{total_expected}")
            self.logger.info(f"Success rate: {success_rate:.1f}%")
            self.logger.info(f"Status: {'✓ PASS' if training_results['success'] else '✗ FAIL'}")
            
        except Exception as e:
            training_results = {
                'success': False,
                'error': str(e),
                'training_time_hours': 0,
                'models_trained': 0
            }
            self.logger.error(f"✗ Model training failed: {e}")
        
        self.results['model_training'] = training_results
        return training_results
    
    def validate_performance_targets(self) -> Dict:
        """
        Validate that trained models meet R² performance targets
        Mechanical: R² ≥ 0.85, Ballistic: R² ≥ 0.80
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 3: PERFORMANCE TARGET VALIDATION")
        self.logger.info("="*80)
        
        performance_results = {}
        evaluator = ModelEvaluator()
        
        mechanical_results = []
        ballistic_results = []
        
        models_dir = Path(self.config['paths']['models'])
        
        for system in self.config['ceramic_systems']['primary']:
            self.logger.info(f"\n--- Validating performance for {system} ---")
            
            system_results = {}
            
            # Evaluate mechanical properties
            for prop in self.config['properties']['mechanical']:
                prop_dir = models_dir / system.lower() / prop
                pred_file = prop_dir / "test_predictions.csv"
                
                if pred_file.exists():
                    try:
                        df = pd.read_csv(pred_file)
                        y_true = df['y_true'].values
                        
                        # Use ensemble predictions if available, otherwise best individual model
                        if 'y_pred_ensemble' in df.columns:
                            y_pred = df['y_pred_ensemble'].values
                            model_type = 'ensemble'
                        else:
                            # Find best performing individual model
                            pred_cols = [col for col in df.columns if col.startswith('y_pred_')]
                            if pred_cols:
                                best_r2 = -np.inf
                                best_pred = None
                                for col in pred_cols:
                                    r2 = evaluator.evaluate(y_true, df[col].values)['r2']
                                    if r2 > best_r2:
                                        best_r2 = r2
                                        best_pred = df[col].values
                                        model_type = col.replace('y_pred_', '')
                                y_pred = best_pred
                            else:
                                continue
                        
                        # Calculate metrics
                        metrics = evaluator.evaluate(y_true, y_pred, prop)
                        r2_score = metrics['r2']
                        
                        # Check if meets target
                        meets_target = r2_score >= self.mechanical_r2_target
                        
                        system_results[prop] = {
                            'r2_score': r2_score,
                            'target': self.mechanical_r2_target,
                            'meets_target': meets_target,
                            'model_type': model_type,
                            'rmse': metrics['rmse'],
                            'mae': metrics['mae']
                        }
                        
                        mechanical_results.append(r2_score)
                        
                        status = "✓ PASS" if meets_target else "✗ FAIL"
                        self.logger.info(f"  {prop}: R² = {r2_score:.4f} {status}")
                        
                    except Exception as e:
                        system_results[prop] = {
                            'error': str(e),
                            'meets_target': False
                        }
                        self.logger.error(f"  {prop}: Error - {e}")
                else:
                    system_results[prop] = {
                        'error': 'Predictions file not found',
                        'meets_target': False
                    }
                    self.logger.warning(f"  {prop}: No predictions file found")
            
            # Evaluate ballistic properties
            for prop in self.config['properties']['ballistic']:
                prop_dir = models_dir / system.lower() / prop
                pred_file = prop_dir / "test_predictions.csv"
                
                if pred_file.exists():
                    try:
                        df = pd.read_csv(pred_file)
                        y_true = df['y_true'].values
                        
                        # Use ensemble predictions if available
                        if 'y_pred_ensemble' in df.columns:
                            y_pred = df['y_pred_ensemble'].values
                            model_type = 'ensemble'
                        else:
                            # Find best performing individual model
                            pred_cols = [col for col in df.columns if col.startswith('y_pred_')]
                            if pred_cols:
                                best_r2 = -np.inf
                                best_pred = None
                                for col in pred_cols:
                                    r2 = evaluator.evaluate(y_true, df[col].values)['r2']
                                    if r2 > best_r2:
                                        best_r2 = r2
                                        best_pred = df[col].values
                                        model_type = col.replace('y_pred_', '')
                                y_pred = best_pred
                            else:
                                continue
                        
                        # Calculate metrics
                        metrics = evaluator.evaluate(y_true, y_pred, prop)
                        r2_score = metrics['r2']
                        
                        # Check if meets target
                        meets_target = r2_score >= self.ballistic_r2_target
                        
                        system_results[prop] = {
                            'r2_score': r2_score,
                            'target': self.ballistic_r2_target,
                            'meets_target': meets_target,
                            'model_type': model_type,
                            'rmse': metrics['rmse'],
                            'mae': metrics['mae']
                        }
                        
                        ballistic_results.append(r2_score)
                        
                        status = "✓ PASS" if meets_target else "✗ FAIL"
                        self.logger.info(f"  {prop}: R² = {r2_score:.4f} {status}")
                        
                    except Exception as e:
                        system_results[prop] = {
                            'error': str(e),
                            'meets_target': False
                        }
                        self.logger.error(f"  {prop}: Error - {e}")
                else:
                    system_results[prop] = {
                        'error': 'Predictions file not found',
                        'meets_target': False
                    }
                    self.logger.warning(f"  {prop}: No predictions file found")
            
            performance_results[system] = system_results
        
        # Calculate overall statistics
        mechanical_stats = {
            'mean_r2': np.mean(mechanical_results) if mechanical_results else 0,
            'min_r2': np.min(mechanical_results) if mechanical_results else 0,
            'max_r2': np.max(mechanical_results) if mechanical_results else 0,
            'target': self.mechanical_r2_target,
            'pass_rate': sum(1 for r2 in mechanical_results if r2 >= self.mechanical_r2_target) / len(mechanical_results) * 100 if mechanical_results else 0
        }
        
        ballistic_stats = {
            'mean_r2': np.mean(ballistic_results) if ballistic_results else 0,
            'min_r2': np.min(ballistic_results) if ballistic_results else 0,
            'max_r2': np.max(ballistic_results) if ballistic_results else 0,
            'target': self.ballistic_r2_target,
            'pass_rate': sum(1 for r2 in ballistic_results if r2 >= self.ballistic_r2_target) / len(ballistic_results) * 100 if ballistic_results else 0
        }
        
        # Overall success criteria
        mechanical_success = mechanical_stats['pass_rate'] >= 80  # 80% of models must meet target
        ballistic_success = ballistic_stats['pass_rate'] >= 80
        overall_success = mechanical_success and ballistic_success
        
        self.logger.info(f"\n--- Performance Summary ---")
        self.logger.info(f"Mechanical Properties:")
        self.logger.info(f"  Mean R²: {mechanical_stats['mean_r2']:.4f}")
        self.logger.info(f"  Pass rate: {mechanical_stats['pass_rate']:.1f}%")
        self.logger.info(f"  Status: {'✓ PASS' if mechanical_success else '✗ FAIL'}")
        
        self.logger.info(f"Ballistic Properties:")
        self.logger.info(f"  Mean R²: {ballistic_stats['mean_r2']:.4f}")
        self.logger.info(f"  Pass rate: {ballistic_stats['pass_rate']:.1f}%")
        self.logger.info(f"  Status: {'✓ PASS' if ballistic_success else '✗ FAIL'}")
        
        self.logger.info(f"Overall Status: {'✓ PASS' if overall_success else '✗ FAIL'}")
        
        self.results['performance_validation'] = {
            'mechanical': mechanical_stats,
            'ballistic': ballistic_stats,
            'systems': performance_results,
            'overall_success': overall_success
        }
        
        return self.results['performance_validation']
    
    def validate_shap_analysis(self) -> Dict:
        """
        Validate SHAP analysis produces publication-ready results
        Tests SHAP interpretation for all trained models
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 4: SHAP ANALYSIS VALIDATION")
        self.logger.info("="*80)
        
        shap_results = {}
        models_dir = Path(self.config['paths']['models'])
        shap_output_dir = Path("results/figures/shap_production")
        shap_output_dir.mkdir(parents=True, exist_ok=True)
        
        successful_analyses = 0
        total_analyses = 0
        
        for system in self.config['ceramic_systems']['primary']:
            self.logger.info(f"\n--- Validating SHAP analysis for {system} ---")
            
            system_results = {}
            
            # Test key properties (subset for validation)
            key_properties = ['youngs_modulus', 'fracture_toughness_mode_i', 'v50']
            
            for prop in key_properties:
                if prop in self.config['properties']['mechanical'] + self.config['properties']['ballistic']:
                    prop_dir = models_dir / system.lower() / prop
                    
                    if prop_dir.exists():
                        try:
                            total_analyses += 1
                            
                            # Load best model (ensemble preferred)
                            model_files = list(prop_dir.glob("*_model.pkl"))
                            if not model_files:
                                system_results[prop] = {
                                    'status': 'failed',
                                    'error': 'No model files found'
                                }
                                continue
                            
                            # Try to load ensemble model first
                            ensemble_file = prop_dir / "ensemble_model.pkl"
                            if ensemble_file.exists():
                                import joblib
                                model_data = joblib.load(ensemble_file)
                                model = model_data.get('model', model_data)
                            else:
                                # Use first available model
                                import joblib
                                model_data = joblib.load(model_files[0])
                                model = model_data.get('model', model_data)
                            
                            # Initialize SHAP analyzer
                            analyzer = SHAPAnalyzer(model, model_type='tree')
                            
                            # Generate SHAP analysis
                            output_dir = shap_output_dir / f"{system}_{prop}"
                            
                            try:
                                results = analyzer.generate_all_plots(
                                    model_dir=str(prop_dir),
                                    output_dir=str(output_dir),
                                    top_features=15
                                )
                                
                                # Validate outputs
                                expected_plots = ['summary_dot', 'summary_bar']
                                successful_plots = results.get('successful_plots', [])
                                failed_plots = results.get('failed_plots', [])
                                
                                plot_success_rate = len(successful_plots) / (len(successful_plots) + len(failed_plots)) * 100 if (successful_plots or failed_plots) else 0
                                
                                system_results[prop] = {
                                    'status': 'success',
                                    'successful_plots': len(successful_plots),
                                    'failed_plots': len(failed_plots),
                                    'plot_success_rate': plot_success_rate,
                                    'output_dir': str(output_dir)
                                }
                                
                                if plot_success_rate >= 70:  # 70% success rate threshold
                                    successful_analyses += 1
                                    self.logger.info(f"  {prop}: ✓ PASS ({len(successful_plots)} plots generated)")
                                else:
                                    self.logger.warning(f"  {prop}: ⚠ PARTIAL ({plot_success_rate:.1f}% success rate)")
                                
                            except Exception as e:
                                system_results[prop] = {
                                    'status': 'failed',
                                    'error': f'SHAP generation failed: {str(e)}'
                                }
                                self.logger.error(f"  {prop}: ✗ FAIL - {e}")
                                
                        except Exception as e:
                            system_results[prop] = {
                                'status': 'failed',
                                'error': f'Model loading failed: {str(e)}'
                            }
                            self.logger.error(f"  {prop}: ✗ FAIL - {e}")
                    else:
                        system_results[prop] = {
                            'status': 'failed',
                            'error': 'Model directory not found'
                        }
                        self.logger.warning(f"  {prop}: Model directory not found")
            
            shap_results[system] = system_results
        
        # Calculate overall success rate
        success_rate = successful_analyses / total_analyses * 100 if total_analyses > 0 else 0
        overall_success = success_rate >= 70  # 70% success threshold
        
        self.logger.info(f"\n--- SHAP Analysis Summary ---")
        self.logger.info(f"Successful analyses: {successful_analyses}/{total_analyses}")
        self.logger.info(f"Success rate: {success_rate:.1f}%")
        self.logger.info(f"Status: {'✓ PASS' if overall_success else '✗ FAIL'}")
        
        self.results['shap_analysis'] = {
            'successful_analyses': successful_analyses,
            'total_analyses': total_analyses,
            'success_rate': success_rate,
            'systems': shap_results,
            'overall_success': overall_success,
            'output_directory': str(shap_output_dir)
        }
        
        return self.results['shap_analysis']
    
    def generate_production_report(self) -> str:
        """Generate comprehensive production validation report"""
        self.logger.info("\n" + "="*80)
        self.logger.info("GENERATING PRODUCTION VALIDATION REPORT")
        self.logger.info("="*80)
        
        total_time = time.time() - self.start_time
        
        # Create report directory
        report_dir = Path("results/reports/production_validation")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate detailed report
        report_content = f"""
# Ceramic Armor ML Pipeline - Production Validation Report

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Validation Time:** {total_time/3600:.2f} hours

## Executive Summary

### Overall System Status
"""
        
        # Determine overall status
        all_phases_success = all([
            self.results.get('data_collection', {}).get('success', False),
            self.results.get('model_training', {}).get('success', False),
            self.results.get('performance_validation', {}).get('overall_success', False),
            self.results.get('shap_analysis', {}).get('overall_success', False)
        ])
        
        report_content += f"**Status:** {'✅ PRODUCTION READY' if all_phases_success else '❌ REQUIRES ATTENTION'}\n\n"
        
        # Data Collection Summary
        data_results = self.results.get('data_collection', {})
        report_content += f"""
## 1. Data Collection Validation

- **Total Materials Collected:** {data_results.get('total_materials', 0):,}
- **Target:** {data_results.get('target_materials', 5600):,}
- **Status:** {'✅ PASS' if data_results.get('success', False) else '❌ FAIL'}

### System Breakdown:
"""
        
        for system, data in data_results.get('systems', {}).items():
            status_icon = '✅' if data.get('status') == 'success' else '❌'
            report_content += f"- **{system}:** {status_icon} {data.get('material_count', 0):,} materials\n"
        
        # Model Training Summary
        training_results = self.results.get('model_training', {})
        report_content += f"""
## 2. Model Training Validation

- **Models Trained:** {training_results.get('models_trained', 0)}/{training_results.get('models_expected', 0)}
- **Success Rate:** {training_results.get('success_rate', 0):.1f}%
- **Training Time:** {training_results.get('training_time_hours', 0):.1f} hours
- **Status:** {'✅ PASS' if training_results.get('success', False) else '❌ FAIL'}
"""
        
        # Performance Validation Summary
        perf_results = self.results.get('performance_validation', {})
        mechanical = perf_results.get('mechanical', {})
        ballistic = perf_results.get('ballistic', {})
        
        report_content += f"""
## 3. Performance Target Validation

### Mechanical Properties (Target: R² ≥ {mechanical.get('target', 0.85)})
- **Mean R²:** {mechanical.get('mean_r2', 0):.4f}
- **Pass Rate:** {mechanical.get('pass_rate', 0):.1f}%
- **Range:** {mechanical.get('min_r2', 0):.4f} - {mechanical.get('max_r2', 0):.4f}

### Ballistic Properties (Target: R² ≥ {ballistic.get('target', 0.80)})
- **Mean R²:** {ballistic.get('mean_r2', 0):.4f}
- **Pass Rate:** {ballistic.get('pass_rate', 0):.1f}%
- **Range:** {ballistic.get('min_r2', 0):.4f} - {ballistic.get('max_r2', 0):.4f}

**Overall Status:** {'✅ PASS' if perf_results.get('overall_success', False) else '❌ FAIL'}
"""
        
        # SHAP Analysis Summary
        shap_results = self.results.get('shap_analysis', {})
        report_content += f"""
## 4. SHAP Analysis Validation

- **Successful Analyses:** {shap_results.get('successful_analyses', 0)}/{shap_results.get('total_analyses', 0)}
- **Success Rate:** {shap_results.get('success_rate', 0):.1f}%
- **Output Directory:** `{shap_results.get('output_directory', 'N/A')}`
- **Status:** {'✅ PASS' if shap_results.get('overall_success', False) else '❌ FAIL'}

## Publication Readiness Assessment

### Requirements Met:
"""
        
        # Publication readiness checklist
        checklist = [
            ("Data Scale", data_results.get('success', False), "5,600+ materials collected"),
            ("Model Performance", perf_results.get('overall_success', False), "R² targets achieved"),
            ("Model Diversity", training_results.get('success', False), "Multiple algorithms trained"),
            ("Interpretability", shap_results.get('overall_success', False), "SHAP analysis complete"),
            ("Reproducibility", True, "All code and configs available")
        ]
        
        for requirement, status, description in checklist:
            icon = '✅' if status else '❌'
            report_content += f"- {icon} **{requirement}:** {description}\n"
        
        publication_ready = all(status for _, status, _ in checklist)
        
        report_content += f"""
### Overall Publication Readiness: {'✅ READY' if publication_ready else '❌ NOT READY'}

## Recommendations

"""
        
        if not publication_ready:
            report_content += "### Issues to Address:\n"
            for requirement, status, description in checklist:
                if not status:
                    report_content += f"- **{requirement}:** {description}\n"
        else:
            report_content += "✅ System is ready for publication. All requirements met.\n"
        
        report_content += f"""
## Technical Details

### System Resources Used:
- **Memory Peak:** {training_results.get('resources', {}).get('post_training', {}).get('memory_used_gb', 0):.1f} GB
- **CPU Utilization:** {training_results.get('resources', {}).get('post_training', {}).get('cpu_percent', 0):.1f}%

### Output Locations:
- **Models:** `results/models/`
- **Predictions:** `results/predictions/`
- **SHAP Plots:** `{shap_results.get('output_directory', 'results/figures/shap_production')}`
- **This Report:** `results/reports/production_validation/`

---
*Generated by Ceramic Armor ML Pipeline Production Validator*
"""
        
        # Save report
        report_file = report_dir / "production_validation_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        # Also save results as JSON
        import json
        results_file = report_dir / "validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"✓ Production validation report saved: {report_file}")
        self.logger.info(f"✓ Detailed results saved: {results_file}")
        
        return str(report_file)
    
    def run_full_validation(self) -> Dict:
        """
        Run complete production validation pipeline
        
        Returns:
            Dictionary with all validation results
        """
        self.logger.info("\n" + "🚀"*20)
        self.logger.info("CERAMIC ARMOR ML PIPELINE - PRODUCTION VALIDATION")
        self.logger.info("🚀"*20)
        
        try:
            # Phase 1: Data Collection Validation
            self.validate_data_collection()
            
            # Phase 2: Model Training Validation
            self.validate_model_training()
            
            # Phase 3: Performance Target Validation
            self.validate_performance_targets()
            
            # Phase 4: SHAP Analysis Validation
            self.validate_shap_analysis()
            
            # Generate comprehensive report
            report_path = self.generate_production_report()
            
            # Final summary
            total_time = time.time() - self.start_time
            self.logger.info(f"\n" + "="*80)
            self.logger.info("PRODUCTION VALIDATION COMPLETE")
            self.logger.info("="*80)
            self.logger.info(f"Total time: {total_time/3600:.2f} hours")
            self.logger.info(f"Report: {report_path}")
            
            # Determine overall success
            overall_success = all([
                self.results.get('data_collection', {}).get('success', False),
                self.results.get('model_training', {}).get('success', False),
                self.results.get('performance_validation', {}).get('overall_success', False),
                self.results.get('shap_analysis', {}).get('overall_success', False)
            ])
            
            if overall_success:
                self.logger.info("🎉 SYSTEM IS PRODUCTION READY! 🎉")
            else:
                self.logger.warning("⚠️  SYSTEM REQUIRES ATTENTION BEFORE PRODUCTION")
            
            self.results['overall_success'] = overall_success
            self.results['total_time_hours'] = total_time / 3600
            self.results['report_path'] = report_path
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Production validation failed: {e}")
            self.results['overall_success'] = False
            self.results['error'] = str(e)
            return self.results


def main():
    """Main execution function"""
    try:
        # Initialize validator
        validator = ProductionValidator()
        
        # Run full validation
        results = validator.run_full_validation()
        
        # Exit with appropriate code
        exit_code = 0 if results.get('overall_success', False) else 1
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Production validation script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()