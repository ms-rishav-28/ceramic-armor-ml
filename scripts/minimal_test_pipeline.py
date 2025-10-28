#!/usr/bin/env python3
"""
Minimal Test Pipeline - Fast End-to-End Test
Validates the complete pipeline using 100 samples per ceramic system
Focuses on Young's modulus prediction for speed
Target completion time: under 30 minutes
"""

import sys
sys.path.append('.')

import time
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import existing pipeline components
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.data_collection.materials_project_collector import MaterialsProjectCollector
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.unit_standardizer import standardize
from src.preprocessing.outlier_detector import remove_iqr_outliers
from src.preprocessing.missing_value_handler import impute_knn
from src.feature_engineering.compositional_features import CompositionalFeatureCalculator
from src.feature_engineering.derived_properties import DerivedPropertiesCalculator
from src.feature_engineering.phase_stability import PhaseStabilityAnalyzer
from src.training.trainer import CeramicPropertyTrainer
from src.evaluation.metrics import ModelEvaluator

logger = get_logger(__name__)


class MinimalTestPipeline:
    """
    Minimal test pipeline for fast end-to-end validation.
    
    Features:
    - 100 samples per ceramic system
    - Young's modulus prediction only
    - Simplified preprocessing
    - Fast model training with reduced parameters
    - Comprehensive validation reporting
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize minimal test pipeline"""
        self.start_time = time.time()
        self.config = load_config(config_path)
        self.test_results = {}
        self.ceramic_systems = ['SiC', 'Al2O3', 'B4C']  # Reduced for speed
        self.target_property = 'youngs_modulus'  # Single property for speed
        self.max_samples_per_system = 100
        
        # Create test directories
        self.test_dir = Path("data/test_pipeline")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Minimal Test Pipeline initialized")
        logger.info(f"Target systems: {self.ceramic_systems}")
        logger.info(f"Target property: {self.target_property}")
        logger.info(f"Max samples per system: {self.max_samples_per_system}")
    
    def load_api_key(self) -> str:
        """Load Materials Project API key"""
        try:
            with open('config/api_keys.yaml', 'r') as f:
                api_keys = yaml.safe_load(f)
            
            mp_api_key = api_keys.get('materials_project')
            if not mp_api_key:
                raise ValueError("Materials Project API key not found in config/api_keys.yaml")
            
            logger.info(f"✅ API key loaded: {mp_api_key[:8]}...")
            return mp_api_key
            
        except FileNotFoundError:
            logger.error("❌ config/api_keys.yaml not found")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading API key: {e}")
            raise
    
    def collect_test_data(self) -> Dict[str, pd.DataFrame]:
        """
        Collect minimal test data using existing Materials Project collector.
        
        Returns:
            Dictionary mapping system names to DataFrames
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 1: DATA COLLECTION")
        logger.info("="*60)
        
        api_key = self.load_api_key()
        collector = MaterialsProjectCollector(api_key)
        
        collected_data = {}
        
        for system in self.ceramic_systems:
            logger.info(f"\n--- Collecting {system} data ---")
            
            try:
                # Use existing collector with limited samples
                df = collector.collect_ceramic_materials(
                    ceramic_systems=[system],
                    max_materials_per_system=self.max_samples_per_system,
                    save_intermediate=True,
                    output_dir=str(self.test_dir / "raw")
                )
                
                if len(df) > 0:
                    collected_data[system] = df
                    logger.info(f"✅ Collected {len(df)} samples for {system}")
                    
                    # Save for inspection
                    output_file = self.test_dir / "raw" / f"{system.lower()}_test_raw.csv"
                    df.to_csv(output_file, index=False)
                    
                else:
                    logger.warning(f"⚠️  No data collected for {system}")
                    
            except Exception as e:
                logger.error(f"❌ Data collection failed for {system}: {e}")
                # Continue with other systems
                continue
        
        total_samples = sum(len(df) for df in collected_data.values())
        logger.info(f"\n✅ Data collection complete: {total_samples} total samples")
        
        return collected_data
    
    def preprocess_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess data using existing preprocessing pipeline.
        
        Args:
            raw_data: Dictionary of raw DataFrames
            
        Returns:
            Dictionary of preprocessed DataFrames
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 2: DATA PREPROCESSING")
        logger.info("="*60)
        
        cleaner = DataCleaner()
        processed_data = {}
        
        for system, df in raw_data.items():
            logger.info(f"\n--- Processing {system} data ---")
            
            try:
                # Step 1: Unit standardization
                logger.info("Standardizing units...")
                df_std = standardize(df)
                
                # Step 2: Basic cleaning
                logger.info("Cleaning data...")
                df_clean = cleaner.clean_dataframe(df_std)
                
                # Step 3: Outlier removal (simplified)
                logger.info("Removing outliers...")
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
                # Remove non-feature columns
                exclude_cols = ['material_id', 'space_group']
                numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
                
                if numeric_cols:
                    df_clean = remove_iqr_outliers(df_clean, numeric_cols, k=2.0)  # More lenient
                
                # Step 4: Missing value imputation (simplified)
                logger.info("Imputing missing values...")
                df_final = impute_knn(df_clean, n_neighbors=3)  # Reduced neighbors for speed
                
                processed_data[system] = df_final
                logger.info(f"✅ Preprocessing complete for {system}: {len(df_final)} samples")
                
                # Save processed data
                output_file = self.test_dir / "processed" / f"{system.lower()}_test_processed.csv"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                df_final.to_csv(output_file, index=False)
                
            except Exception as e:
                logger.error(f"❌ Preprocessing failed for {system}: {e}")
                continue
        
        total_processed = sum(len(df) for df in processed_data.values())
        logger.info(f"\n✅ Preprocessing complete: {total_processed} samples processed")
        
        return processed_data
    
    def engineer_features(self, processed_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Engineer features using existing feature engineering modules.
        
        Args:
            processed_data: Dictionary of processed DataFrames
            
        Returns:
            Dictionary of feature-engineered DataFrames
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 3: FEATURE ENGINEERING")
        logger.info("="*60)
        
        # Initialize feature calculators
        comp_calc = CompositionalFeatureCalculator()
        derived_calc = DerivedPropertiesCalculator()
        stability_analyzer = PhaseStabilityAnalyzer()
        
        feature_data = {}
        
        for system, df in processed_data.items():
            logger.info(f"\n--- Engineering features for {system} ---")
            
            try:
                # Step 1: Compositional features
                logger.info("Adding compositional features...")
                if 'formula' in df.columns:
                    df_comp = comp_calc.augment_dataframe(df, formula_col='formula')
                else:
                    logger.warning("No formula column found, skipping compositional features")
                    df_comp = df.copy()
                
                # Step 2: Derived properties
                logger.info("Calculating derived properties...")
                df_derived = derived_calc.calculate_all_derived_properties(df_comp)
                
                # Step 3: Phase stability (simplified)
                logger.info("Analyzing phase stability...")
                df_final = stability_analyzer.analyze_dataframe(df_derived)
                
                feature_data[system] = df_final
                logger.info(f"✅ Feature engineering complete for {system}: {df_final.shape[1]} features")
                
                # Save feature data
                output_file = self.test_dir / "features" / f"{system.lower()}_test_features.csv"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                df_final.to_csv(output_file, index=False)
                
            except Exception as e:
                logger.error(f"❌ Feature engineering failed for {system}: {e}")
                continue
        
        logger.info(f"\n✅ Feature engineering complete for {len(feature_data)} systems")
        
        return feature_data
    
    def train_models(self, feature_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Train models using existing trainer with simplified parameters.
        
        Args:
            feature_data: Dictionary of feature-engineered DataFrames
            
        Returns:
            Dictionary of training results
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 4: MODEL TRAINING")
        logger.info("="*60)
        
        # Create simplified config for fast training
        test_config = self.config.copy()
        test_config['models']['xgboost']['n_estimators'] = 100  # Reduced from 1000
        test_config['models']['catboost']['iterations'] = 100   # Reduced from 1000
        test_config['models']['random_forest']['n_estimators'] = 100  # Reduced from 500
        test_config['paths']['models'] = str(self.test_dir / "models")
        
        trainer = CeramicPropertyTrainer(test_config)
        training_results = {}
        
        for system in feature_data.keys():
            logger.info(f"\n--- Training models for {system} ---")
            
            try:
                # Prepare data for training
                df = feature_data[system]
                
                # Check if target property exists
                if self.target_property not in df.columns:
                    logger.warning(f"Target property '{self.target_property}' not found in {system} data")
                    continue
                
                # Remove rows with missing target
                df_clean = df.dropna(subset=[self.target_property])
                
                if len(df_clean) < 10:
                    logger.warning(f"Insufficient data for {system}: only {len(df_clean)} samples")
                    continue
                
                # Save temporary data for trainer
                temp_dir = Path(test_config['paths']['features']) / system.lower()
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_file = temp_dir / f"{system.lower()}_features.csv"
                df_clean.to_csv(temp_file, index=False)
                
                # Train models for this system and property
                models = trainer.train_system_models(system, self.target_property)
                training_results[system] = models
                
                logger.info(f"✅ Training complete for {system}")
                
            except Exception as e:
                logger.error(f"❌ Training failed for {system}: {e}")
                continue
        
        logger.info(f"\n✅ Model training complete for {len(training_results)} systems")
        
        return training_results
    
    def validate_results(self, training_results: Dict[str, Dict]) -> Dict[str, any]:
        """
        Validate results and generate test completion report.
        
        Args:
            training_results: Dictionary of training results
            
        Returns:
            Dictionary of validation results
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 5: RESULT VALIDATION")
        logger.info("="*60)
        
        validation_results = {
            'systems_tested': list(training_results.keys()),
            'models_trained': {},
            'performance_metrics': {},
            'pipeline_health': 'UNKNOWN',
            'total_time': time.time() - self.start_time,
            'pass_fail_status': {}
        }
        
        evaluator = ModelEvaluator()
        
        for system, models in training_results.items():
            logger.info(f"\n--- Validating {system} results ---")
            
            system_metrics = {}
            
            for model_name, model in models.items():
                try:
                    # Load test data for evaluation
                    model_dir = Path(self.config['paths']['models']) / system.lower() / self.target_property
                    
                    if (model_dir / "X_test.npy").exists() and (model_dir / "y_test.npy").exists():
                        X_test = np.load(model_dir / "X_test.npy")
                        y_test = np.load(model_dir / "y_test.npy")
                        
                        # Make predictions
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        metrics = evaluator.evaluate(y_test, y_pred, self.target_property)
                        system_metrics[model_name] = metrics
                        
                        logger.info(f"  {model_name}: R² = {metrics['r2']:.4f}")
                        
                    else:
                        logger.warning(f"Test data not found for {system} {model_name}")
                        
                except Exception as e:
                    logger.error(f"Validation failed for {system} {model_name}: {e}")
            
            validation_results['models_trained'][system] = list(models.keys())
            validation_results['performance_metrics'][system] = system_metrics
            
            # Determine pass/fail for this system
            best_r2 = 0
            if system_metrics:
                best_r2 = max(metrics.get('r2', 0) for metrics in system_metrics.values())
            
            # Pass criteria: R² > 0.5 for minimal test (relaxed from production targets)
            validation_results['pass_fail_status'][system] = {
                'status': 'PASS' if best_r2 > 0.5 else 'FAIL',
                'best_r2': best_r2,
                'threshold': 0.5
            }
        
        # Overall pipeline health
        passed_systems = sum(1 for status in validation_results['pass_fail_status'].values() 
                           if status['status'] == 'PASS')
        total_systems = len(validation_results['pass_fail_status'])
        
        if passed_systems == total_systems and total_systems > 0:
            validation_results['pipeline_health'] = 'HEALTHY'
        elif passed_systems > 0:
            validation_results['pipeline_health'] = 'PARTIAL'
        else:
            validation_results['pipeline_health'] = 'FAILED'
        
        logger.info(f"\n✅ Validation complete: {passed_systems}/{total_systems} systems passed")
        
        return validation_results
    
    def generate_test_report(self, validation_results: Dict[str, any]) -> str:
        """
        Generate comprehensive test completion report.
        
        Args:
            validation_results: Dictionary of validation results
            
        Returns:
            Path to generated report file
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 6: GENERATING TEST REPORT")
        logger.info("="*60)
        
        report_file = self.test_dir / "minimal_test_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("CERAMIC ARMOR ML PIPELINE - MINIMAL TEST REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Test Summary
            f.write("TEST SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Test Duration: {validation_results['total_time']:.1f} seconds\n")
            f.write(f"Pipeline Health: {validation_results['pipeline_health']}\n")
            f.write(f"Systems Tested: {len(validation_results['systems_tested'])}\n")
            f.write(f"Target Property: {self.target_property}\n")
            f.write(f"Max Samples per System: {self.max_samples_per_system}\n\n")
            
            # System Results
            f.write("SYSTEM RESULTS\n")
            f.write("-" * 20 + "\n")
            for system in validation_results['systems_tested']:
                status_info = validation_results['pass_fail_status'][system]
                f.write(f"{system}:\n")
                f.write(f"  Status: {status_info['status']}\n")
                f.write(f"  Best R²: {status_info['best_r2']:.4f}\n")
                f.write(f"  Models: {', '.join(validation_results['models_trained'][system])}\n")
                
                # Model performance details
                if system in validation_results['performance_metrics']:
                    for model_name, metrics in validation_results['performance_metrics'][system].items():
                        f.write(f"    {model_name}: R² = {metrics.get('r2', 'N/A'):.4f}\n")
                f.write("\n")
            
            # Pass/Fail Summary
            f.write("PASS/FAIL SUMMARY\n")
            f.write("-" * 20 + "\n")
            passed = sum(1 for status in validation_results['pass_fail_status'].values() 
                        if status['status'] == 'PASS')
            total = len(validation_results['pass_fail_status'])
            f.write(f"Passed: {passed}/{total} systems\n")
            f.write(f"Success Rate: {passed/total*100:.1f}%\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            if validation_results['pipeline_health'] == 'HEALTHY':
                f.write("✅ Pipeline is working correctly!\n")
                f.write("Ready for full-scale execution.\n")
            elif validation_results['pipeline_health'] == 'PARTIAL':
                f.write("⚠️  Pipeline partially working.\n")
                f.write("Some systems failed - check data quality and API connectivity.\n")
            else:
                f.write("❌ Pipeline has issues.\n")
                f.write("Check API keys, dependencies, and data sources.\n")
            
            f.write(f"\nTest completed in {validation_results['total_time']:.1f} seconds\n")
        
        logger.info(f"✅ Test report generated: {report_file}")
        return str(report_file)
    
    def run_complete_test(self) -> Dict[str, any]:
        """
        Run the complete minimal test pipeline.
        
        Returns:
            Dictionary of test results
        """
        logger.info("\n" + "🎯" * 20)
        logger.info("CERAMIC ARMOR ML PIPELINE - MINIMAL TEST")
        logger.info("🎯" * 20)
        logger.info(f"Target: Complete test in under 30 minutes")
        logger.info(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Data Collection
            raw_data = self.collect_test_data()
            if not raw_data:
                raise ValueError("No data collected - check API connectivity")
            
            # Step 2: Preprocessing
            processed_data = self.preprocess_data(raw_data)
            if not processed_data:
                raise ValueError("Preprocessing failed for all systems")
            
            # Step 3: Feature Engineering
            feature_data = self.engineer_features(processed_data)
            if not feature_data:
                raise ValueError("Feature engineering failed for all systems")
            
            # Step 4: Model Training
            training_results = self.train_models(feature_data)
            if not training_results:
                raise ValueError("Model training failed for all systems")
            
            # Step 5: Validation
            validation_results = self.validate_results(training_results)
            
            # Step 6: Report Generation
            report_file = self.generate_test_report(validation_results)
            
            # Final summary
            total_time = time.time() - self.start_time
            logger.info("\n" + "🎉" * 20)
            logger.info("MINIMAL TEST COMPLETE!")
            logger.info("🎉" * 20)
            logger.info(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            logger.info(f"Pipeline health: {validation_results['pipeline_health']}")
            logger.info(f"Report saved: {report_file}")
            
            if total_time < 1800:  # 30 minutes
                logger.info("✅ Test completed within 30-minute target!")
            else:
                logger.warning("⚠️  Test exceeded 30-minute target")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Minimal test failed: {e}")
            return {
                'pipeline_health': 'FAILED',
                'error': str(e),
                'total_time': time.time() - self.start_time
            }


def main():
    """Run minimal test pipeline"""
    try:
        pipeline = MinimalTestPipeline()
        results = pipeline.run_complete_test()
        
        # Exit code based on results
        if results.get('pipeline_health') == 'HEALTHY':
            return 0
        elif results.get('pipeline_health') == 'PARTIAL':
            return 1
        else:
            return 2
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())