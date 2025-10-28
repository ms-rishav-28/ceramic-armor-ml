#!/usr/bin/env python3
"""
Minimal Test Results Validator
Validates pipeline health and generates comprehensive test reports
"""

import sys
sys.path.append('.')

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time
from datetime import datetime

from src.utils.logger import get_logger
from src.evaluation.metrics import ModelEvaluator

logger = get_logger(__name__)


class MinimalTestValidator:
    """
    Validates minimal test pipeline results and generates comprehensive reports.
    
    Features:
    - Automated pass/fail determination
    - Pipeline health assessment
    - Performance benchmarking
    - Detailed error analysis
    - Recommendations for improvements
    """
    
    def __init__(self, test_dir: str = "data/test_pipeline"):
        """Initialize validator"""
        self.test_dir = Path(test_dir)
        self.evaluator = ModelEvaluator()
        self.validation_results = {}
        
        # Performance thresholds
        self.thresholds = {
            'min_r2': 0.5,          # Minimum R² for pass
            'target_r2': 0.7,       # Target R² for good performance
            'max_time_minutes': 30,  # Maximum allowed time
            'min_samples': 50,       # Minimum samples per system
            'min_features': 20       # Minimum features required
        }
        
        logger.info("Minimal Test Validator initialized")
    
    def validate_data_collection(self) -> Dict[str, Any]:
        """
        Validate data collection stage results.
        
        Returns:
            Dictionary of data collection validation results
        """
        logger.info("\n--- Validating Data Collection ---")
        
        results = {
            'stage': 'data_collection',
            'status': 'UNKNOWN',
            'systems_collected': [],
            'sample_counts': {},
            'data_quality': {},
            'issues': []
        }
        
        raw_dir = self.test_dir / "raw"
        
        if not raw_dir.exists():
            results['status'] = 'FAILED'
            results['issues'].append("Raw data directory not found")
            return results
        
        # Check for data files
        expected_systems = ['SiC', 'Al2O3', 'B4C']
        
        for system in expected_systems:
            data_file = raw_dir / f"{system.lower()}_test_raw.csv"
            
            if data_file.exists():
                try:
                    df = pd.read_csv(data_file)
                    results['systems_collected'].append(system)
                    results['sample_counts'][system] = len(df)
                    
                    # Basic data quality checks
                    quality_info = {
                        'total_samples': len(df),
                        'total_columns': len(df.columns),
                        'missing_data_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                        'has_material_id': 'material_id' in df.columns,
                        'has_formula': 'formula' in df.columns or 'formula_pretty' in df.columns
                    }
                    results['data_quality'][system] = quality_info
                    
                    # Check minimum sample requirement
                    if len(df) < self.thresholds['min_samples']:
                        results['issues'].append(f"{system}: Only {len(df)} samples (minimum {self.thresholds['min_samples']})")
                    
                    logger.info(f"✅ {system}: {len(df)} samples, {len(df.columns)} columns")
                    
                except Exception as e:
                    results['issues'].append(f"{system}: Error reading data file - {e}")
                    logger.error(f"❌ {system}: Error reading data - {e}")
            else:
                results['issues'].append(f"{system}: Data file not found")
                logger.warning(f"⚠️  {system}: Data file not found")
        
        # Determine overall status
        if len(results['systems_collected']) >= 2:  # At least 2 systems
            results['status'] = 'PASSED'
        elif len(results['systems_collected']) >= 1:
            results['status'] = 'PARTIAL'
        else:
            results['status'] = 'FAILED'
        
        logger.info(f"Data collection status: {results['status']}")
        return results
    
    def validate_preprocessing(self) -> Dict[str, Any]:
        """
        Validate preprocessing stage results.
        
        Returns:
            Dictionary of preprocessing validation results
        """
        logger.info("\n--- Validating Preprocessing ---")
        
        results = {
            'stage': 'preprocessing',
            'status': 'UNKNOWN',
            'systems_processed': [],
            'data_quality_improvement': {},
            'issues': []
        }
        
        processed_dir = self.test_dir / "processed"
        raw_dir = self.test_dir / "raw"
        
        if not processed_dir.exists():
            results['status'] = 'FAILED'
            results['issues'].append("Processed data directory not found")
            return results
        
        for system in ['SiC', 'Al2O3', 'B4C']:
            processed_file = processed_dir / f"{system.lower()}_test_processed.csv"
            raw_file = raw_dir / f"{system.lower()}_test_raw.csv"
            
            if processed_file.exists() and raw_file.exists():
                try:
                    df_processed = pd.read_csv(processed_file)
                    df_raw = pd.read_csv(raw_file)
                    
                    results['systems_processed'].append(system)
                    
                    # Calculate data quality improvements
                    raw_missing_pct = (df_raw.isnull().sum().sum() / (len(df_raw) * len(df_raw.columns))) * 100
                    processed_missing_pct = (df_processed.isnull().sum().sum() / (len(df_processed) * len(df_processed.columns))) * 100
                    
                    improvement_info = {
                        'samples_retained': len(df_processed) / len(df_raw),
                        'missing_data_reduction': raw_missing_pct - processed_missing_pct,
                        'columns_added': len(df_processed.columns) - len(df_raw.columns)
                    }
                    results['data_quality_improvement'][system] = improvement_info
                    
                    logger.info(f"✅ {system}: {len(df_processed)} samples processed")
                    
                except Exception as e:
                    results['issues'].append(f"{system}: Error validating preprocessing - {e}")
                    logger.error(f"❌ {system}: Preprocessing validation error - {e}")
            else:
                if not processed_file.exists():
                    results['issues'].append(f"{system}: Processed data file not found")
        
        # Determine status
        if len(results['systems_processed']) >= 2:
            results['status'] = 'PASSED'
        elif len(results['systems_processed']) >= 1:
            results['status'] = 'PARTIAL'
        else:
            results['status'] = 'FAILED'
        
        logger.info(f"Preprocessing status: {results['status']}")
        return results
    
    def validate_feature_engineering(self) -> Dict[str, Any]:
        """
        Validate feature engineering stage results.
        
        Returns:
            Dictionary of feature engineering validation results
        """
        logger.info("\n--- Validating Feature Engineering ---")
        
        results = {
            'stage': 'feature_engineering',
            'status': 'UNKNOWN',
            'systems_engineered': [],
            'feature_counts': {},
            'feature_types': {},
            'issues': []
        }
        
        features_dir = self.test_dir / "features"
        
        if not features_dir.exists():
            results['status'] = 'FAILED'
            results['issues'].append("Features data directory not found")
            return results
        
        for system in ['SiC', 'Al2O3', 'B4C']:
            features_file = features_dir / f"{system.lower()}_test_features.csv"
            
            if features_file.exists():
                try:
                    df = pd.read_csv(features_file)
                    results['systems_engineered'].append(system)
                    results['feature_counts'][system] = len(df.columns)
                    
                    # Analyze feature types
                    feature_types = {
                        'compositional': len([col for col in df.columns if 'comp_' in col]),
                        'derived': len([col for col in df.columns if any(term in col.lower() for term in ['pugh', 'cauchy', 'specific'])]),
                        'elastic': len([col for col in df.columns if 'elastic_' in col]),
                        'thermal': len([col for col in df.columns if 'thermal_' in col]),
                        'stability': len([col for col in df.columns if 'stability' in col.lower()])
                    }
                    results['feature_types'][system] = feature_types
                    
                    # Check minimum feature requirement
                    if len(df.columns) < self.thresholds['min_features']:
                        results['issues'].append(f"{system}: Only {len(df.columns)} features (minimum {self.thresholds['min_features']})")
                    
                    logger.info(f"✅ {system}: {len(df.columns)} features engineered")
                    
                except Exception as e:
                    results['issues'].append(f"{system}: Error validating features - {e}")
                    logger.error(f"❌ {system}: Feature validation error - {e}")
            else:
                results['issues'].append(f"{system}: Features file not found")
        
        # Determine status
        if len(results['systems_engineered']) >= 2:
            results['status'] = 'PASSED'
        elif len(results['systems_engineered']) >= 1:
            results['status'] = 'PARTIAL'
        else:
            results['status'] = 'FAILED'
        
        logger.info(f"Feature engineering status: {results['status']}")
        return results
    
    def validate_model_training(self) -> Dict[str, Any]:
        """
        Validate model training stage results.
        
        Returns:
            Dictionary of model training validation results
        """
        logger.info("\n--- Validating Model Training ---")
        
        results = {
            'stage': 'model_training',
            'status': 'UNKNOWN',
            'systems_trained': [],
            'models_per_system': {},
            'performance_metrics': {},
            'issues': []
        }
        
        models_dir = self.test_dir / "models"
        
        if not models_dir.exists():
            results['status'] = 'FAILED'
            results['issues'].append("Models directory not found")
            return results
        
        for system in ['SiC', 'Al2O3', 'B4C']:
            system_dir = models_dir / system.lower() / "youngs_modulus"
            
            if system_dir.exists():
                try:
                    # Check for model files
                    model_files = list(system_dir.glob("*_model.pkl"))
                    model_names = [f.stem.replace('_model', '') for f in model_files]
                    
                    if model_names:
                        results['systems_trained'].append(system)
                        results['models_per_system'][system] = model_names
                        
                        # Validate model performance
                        performance = self._evaluate_system_models(system, system_dir)
                        results['performance_metrics'][system] = performance
                        
                        logger.info(f"✅ {system}: {len(model_names)} models trained")
                    else:
                        results['issues'].append(f"{system}: No model files found")
                        
                except Exception as e:
                    results['issues'].append(f"{system}: Error validating models - {e}")
                    logger.error(f"❌ {system}: Model validation error - {e}")
            else:
                results['issues'].append(f"{system}: Model directory not found")
        
        # Determine status based on performance
        passed_systems = 0
        for system, metrics in results['performance_metrics'].items():
            best_r2 = max(model_metrics.get('r2', 0) for model_metrics in metrics.values())
            if best_r2 >= self.thresholds['min_r2']:
                passed_systems += 1
        
        if passed_systems >= 2:
            results['status'] = 'PASSED'
        elif passed_systems >= 1:
            results['status'] = 'PARTIAL'
        else:
            results['status'] = 'FAILED'
        
        logger.info(f"Model training status: {results['status']}")
        return results
    
    def _evaluate_system_models(self, system: str, model_dir: Path) -> Dict[str, Dict]:
        """
        Evaluate models for a specific system.
        
        Args:
            system: System name
            model_dir: Path to model directory
            
        Returns:
            Dictionary of model performance metrics
        """
        performance = {}
        
        # Load test data
        try:
            X_test = np.load(model_dir / "X_test.npy")
            y_test = np.load(model_dir / "y_test.npy")
            
            # Check for predictions file
            predictions_file = model_dir / "test_predictions.csv"
            if predictions_file.exists():
                pred_df = pd.read_csv(predictions_file)
                
                # Evaluate each model
                for col in pred_df.columns:
                    if col.startswith('y_pred_'):
                        model_name = col.replace('y_pred_', '')
                        y_pred = pred_df[col].values
                        
                        metrics = self.evaluator.evaluate(y_test, y_pred, 'youngs_modulus')
                        performance[model_name] = metrics
                        
                        logger.info(f"  {model_name}: R² = {metrics['r2']:.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluating {system} models: {e}")
        
        return performance
    
    def validate_overall_pipeline(self) -> Dict[str, Any]:
        """
        Validate overall pipeline health and performance.
        
        Returns:
            Dictionary of overall validation results
        """
        logger.info("\n--- Validating Overall Pipeline ---")
        
        # Check if test report exists
        report_file = self.test_dir / "minimal_test_report.txt"
        
        results = {
            'stage': 'overall',
            'status': 'UNKNOWN',
            'execution_time': None,
            'systems_completed': 0,
            'average_performance': 0.0,
            'time_target_met': False,
            'report_generated': report_file.exists(),
            'issues': []
        }
        
        if report_file.exists():
            try:
                # Parse report for key metrics
                with open(report_file, 'r') as f:
                    content = f.read()
                
                # Extract execution time
                if "Test Duration:" in content:
                    time_line = [line for line in content.split('\n') if 'Test Duration:' in line][0]
                    time_seconds = float(time_line.split(':')[1].strip().split()[0])
                    results['execution_time'] = time_seconds
                    results['time_target_met'] = time_seconds <= (self.thresholds['max_time_minutes'] * 60)
                
                # Count completed systems
                if "Systems Tested:" in content:
                    systems_line = [line for line in content.split('\n') if 'Systems Tested:' in line][0]
                    results['systems_completed'] = int(systems_line.split(':')[1].strip())
                
                logger.info(f"✅ Test report found and parsed")
                
            except Exception as e:
                results['issues'].append(f"Error parsing test report: {e}")
                logger.error(f"❌ Error parsing test report: {e}")
        else:
            results['issues'].append("Test report not found")
            logger.warning("⚠️  Test report not found")
        
        # Calculate average performance from model training results
        if 'model_training' in self.validation_results:
            training_results = self.validation_results['model_training']
            all_r2_scores = []
            
            for system, metrics in training_results.get('performance_metrics', {}).items():
                for model, model_metrics in metrics.items():
                    if 'r2' in model_metrics:
                        all_r2_scores.append(model_metrics['r2'])
            
            if all_r2_scores:
                results['average_performance'] = np.mean(all_r2_scores)
        
        # Determine overall status
        if (results['systems_completed'] >= 2 and 
            results['average_performance'] >= self.thresholds['min_r2'] and
            results['time_target_met']):
            results['status'] = 'PASSED'
        elif results['systems_completed'] >= 1:
            results['status'] = 'PARTIAL'
        else:
            results['status'] = 'FAILED'
        
        logger.info(f"Overall pipeline status: {results['status']}")
        return results
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate comprehensive validation report.
        
        Returns:
            Path to generated report file
        """
        logger.info("\n--- Generating Comprehensive Validation Report ---")
        
        report_file = self.test_dir / "validation_report.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(report_file, 'w') as f:
            f.write("CERAMIC ARMOR ML PIPELINE - VALIDATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {timestamp}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            
            overall_status = self.validation_results.get('overall', {}).get('status', 'UNKNOWN')
            f.write(f"Overall Status: {overall_status}\n")
            
            if 'overall' in self.validation_results:
                overall = self.validation_results['overall']
                f.write(f"Execution Time: {overall.get('execution_time', 'N/A')} seconds\n")
                f.write(f"Time Target Met: {overall.get('time_target_met', False)}\n")
                f.write(f"Systems Completed: {overall.get('systems_completed', 0)}\n")
                f.write(f"Average Performance: {overall.get('average_performance', 0):.4f}\n")
            f.write("\n")
            
            # Stage-by-Stage Results
            f.write("STAGE-BY-STAGE RESULTS\n")
            f.write("-" * 30 + "\n")
            
            stages = ['data_collection', 'preprocessing', 'feature_engineering', 'model_training']
            
            for stage in stages:
                if stage in self.validation_results:
                    stage_results = self.validation_results[stage]
                    f.write(f"\n{stage.upper().replace('_', ' ')}:\n")
                    f.write(f"  Status: {stage_results.get('status', 'UNKNOWN')}\n")
                    
                    if stage == 'data_collection':
                        f.write(f"  Systems Collected: {len(stage_results.get('systems_collected', []))}\n")
                        f.write(f"  Sample Counts: {stage_results.get('sample_counts', {})}\n")
                    
                    elif stage == 'feature_engineering':
                        f.write(f"  Systems Engineered: {len(stage_results.get('systems_engineered', []))}\n")
                        f.write(f"  Feature Counts: {stage_results.get('feature_counts', {})}\n")
                    
                    elif stage == 'model_training':
                        f.write(f"  Systems Trained: {len(stage_results.get('systems_trained', []))}\n")
                        f.write(f"  Models per System: {stage_results.get('models_per_system', {})}\n")
                        
                        # Performance details
                        for system, metrics in stage_results.get('performance_metrics', {}).items():
                            f.write(f"  {system} Performance:\n")
                            for model, model_metrics in metrics.items():
                                f.write(f"    {model}: R² = {model_metrics.get('r2', 'N/A'):.4f}\n")
                    
                    # Issues
                    if stage_results.get('issues'):
                        f.write(f"  Issues: {len(stage_results['issues'])}\n")
                        for issue in stage_results['issues']:
                            f.write(f"    - {issue}\n")
            
            # Recommendations
            f.write("\nRECOMMENDations\n")
            f.write("-" * 20 + "\n")
            
            if overall_status == 'PASSED':
                f.write("✅ Pipeline validation PASSED!\n")
                f.write("The minimal test pipeline is working correctly.\n")
                f.write("Ready for full-scale execution.\n")
            elif overall_status == 'PARTIAL':
                f.write("⚠️  Pipeline validation PARTIALLY PASSED.\n")
                f.write("Some components are working, but issues were found.\n")
                f.write("Recommendations:\n")
                f.write("- Check API connectivity for failed systems\n")
                f.write("- Verify data quality and preprocessing steps\n")
                f.write("- Consider adjusting model parameters\n")
            else:
                f.write("❌ Pipeline validation FAILED.\n")
                f.write("Critical issues prevent proper pipeline execution.\n")
                f.write("Recommendations:\n")
                f.write("- Verify API keys and connectivity\n")
                f.write("- Check all dependencies are installed\n")
                f.write("- Review error logs for specific issues\n")
                f.write("- Run setup validation script\n")
            
            f.write(f"\nValidation completed at: {timestamp}\n")
        
        logger.info(f"✅ Comprehensive validation report generated: {report_file}")
        return str(report_file)
    
    def run_complete_validation(self) -> Dict[str, Any]:
        """
        Run complete validation of minimal test results.
        
        Returns:
            Dictionary of complete validation results
        """
        logger.info("\n" + "🔍" * 20)
        logger.info("MINIMAL TEST VALIDATION")
        logger.info("🔍" * 20)
        
        start_time = time.time()
        
        # Run all validation stages
        self.validation_results['data_collection'] = self.validate_data_collection()
        self.validation_results['preprocessing'] = self.validate_preprocessing()
        self.validation_results['feature_engineering'] = self.validate_feature_engineering()
        self.validation_results['model_training'] = self.validate_model_training()
        self.validation_results['overall'] = self.validate_overall_pipeline()
        
        # Generate comprehensive report
        report_file = self.generate_comprehensive_report()
        
        # Summary
        validation_time = time.time() - start_time
        overall_status = self.validation_results['overall']['status']
        
        logger.info("\n" + "📊" * 20)
        logger.info("VALIDATION COMPLETE")
        logger.info("📊" * 20)
        logger.info(f"Validation time: {validation_time:.1f} seconds")
        logger.info(f"Overall status: {overall_status}")
        logger.info(f"Report saved: {report_file}")
        
        return {
            'validation_results': self.validation_results,
            'report_file': report_file,
            'overall_status': overall_status,
            'validation_time': validation_time
        }


def main():
    """Run minimal test validation"""
    try:
        validator = MinimalTestValidator()
        results = validator.run_complete_validation()
        
        # Exit code based on validation results
        status = results['overall_status']
        if status == 'PASSED':
            return 0
        elif status == 'PARTIAL':
            return 1
        else:
            return 2
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())