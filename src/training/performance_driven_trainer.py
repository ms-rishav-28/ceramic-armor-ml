"""
Performance-Driven Training System
Integrates performance target enforcement with model training
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.evaluation.performance_enforcer import PerformanceTargetEnforcer
from src.training.cross_validator import EnhancedCrossValidator
from src.training.hyperparameter_tuner import HyperparameterTuner
from src.models.xgboost_model import XGBoostModel
from src.models.catboost_model import CatBoostModel
from src.models.random_forest_model import RandomForestModel
from src.models.gradient_boosting_model import GradientBoostingModel
from src.models.ensemble_model import EnsembleModel
from src.evaluation.metrics import ModelEvaluator


class PerformanceDrivenTrainer:
    """
    Training system with automatic performance target enforcement
    
    Features:
    - Automatic performance validation for R² ≥ 0.85 (mechanical) and R² ≥ 0.80 (ballistic)
    - Automatic hyperparameter adjustment when targets not met
    - Stacking weight optimization for ensemble performance
    - 5-fold cross-validation and leave-one-ceramic-family-out validation
    - Prediction uncertainty estimation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize performance-driven trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.performance_enforcer = PerformanceTargetEnforcer(config)
        self.cross_validator = EnhancedCrossValidator(
            n_splits=config.get('cross_validation', {}).get('k_fold', {}).get('n_splits', 5),
            random_state=config.get('reproducibility', {}).get('seed', 42)
        )
        self.evaluator = ModelEvaluator()
        
        # Training state
        self.trained_models = {}
        self.performance_results = {}
        self.validation_results = {}
        self.scalers = {}
        
        # Model constructors
        self.model_constructors = {
            'xgboost': self._create_xgboost_model,
            'catboost': self._create_catboost_model,
            'random_forest': self._create_random_forest_model,
            'gradient_boosting': self._create_gradient_boosting_model,
            'ensemble': self._create_ensemble_model
        }
        
        logger.info("✓ Performance-Driven Trainer initialized")
        logger.info(f"✓ Performance targets: Mechanical R²≥{config['targets']['mechanical_r2']}, Ballistic R²≥{config['targets']['ballistic_r2']}")
    
    def train_with_performance_enforcement(self,
                                         X_train: np.ndarray,
                                         y_train: np.ndarray,
                                         X_val: np.ndarray,
                                         y_val: np.ndarray,
                                         X_test: np.ndarray,
                                         y_test: np.ndarray,
                                         property_name: str,
                                         property_type: str,
                                         feature_names: List[str],
                                         ceramic_system: str = "SiC") -> Dict[str, Any]:
        """
        Train models with automatic performance target enforcement
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
            property_name: Name of property being predicted
            property_type: 'mechanical' or 'ballistic'
            feature_names: List of feature names
            ceramic_system: Ceramic system name
        
        Returns:
            Dictionary with trained models and results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Training with performance enforcement: {property_name} ({property_type})")
        logger.info(f"System: {ceramic_system}")
        logger.info(f"{'='*80}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        scaler_key = f"{ceramic_system}_{property_name}"
        self.scalers[scaler_key] = scaler
        
        # Train individual models
        models = {}
        initial_performance = {}
        
        # 1. Train base models
        for model_type in ['xgboost', 'catboost', 'random_forest', 'gradient_boosting']:
            logger.info(f"\n[{model_type.upper()}] Training initial model...")
            
            # Create model with default parameters
            model = self.model_constructors[model_type](self.config['models'][model_type])
            model.feature_names = feature_names
            model.target_name = property_name
            
            # Train model
            model.train(X_train_scaled, y_train, X_val_scaled, y_val)
            models[model_type] = model
            
            # Evaluate initial performance
            y_pred = model.predict(X_test_scaled)
            metrics = self.evaluator.evaluate(y_test, y_pred, f"{property_name}_{model_type}")
            initial_performance[model_type] = metrics
            
            logger.info(f"  Initial {model_type} R²: {metrics['r2']:.4f}")
        
        # 2. Validate performance targets
        logger.info(f"\n[VALIDATION] Checking performance targets...")
        target_results = self.performance_enforcer.validate_performance_targets(
            models, X_test_scaled, y_test, property_name, property_type
        )
        
        # 3. Automatic hyperparameter adjustment for models not meeting targets
        adjusted_models = {}
        for model_type, meets_target in target_results.items():
            if not meets_target:
                logger.info(f"\n[ADJUSTMENT] {model_type} below target, adjusting hyperparameters...")
                
                adjusted_model, best_params = self.performance_enforcer.automatic_hyperparameter_adjustment(
                    model_constructor=lambda params: self.model_constructors[model_type](params),
                    X_train=X_train_scaled,
                    y_train=y_train,
                    X_val=X_val_scaled,
                    y_val=y_val,
                    property_type=property_type,
                    model_name=model_type,
                    max_iterations=3
                )
                
                if adjusted_model is not None:
                    adjusted_model.feature_names = feature_names
                    adjusted_model.target_name = property_name
                    adjusted_models[model_type] = adjusted_model
                    models[model_type] = adjusted_model  # Replace original
                    logger.info(f"✓ {model_type} hyperparameters adjusted")
                else:
                    logger.warning(f"⚠ {model_type} adjustment failed, keeping original")
                    adjusted_models[model_type] = models[model_type]
            else:
                logger.info(f"✓ {model_type} meets target, no adjustment needed")
                adjusted_models[model_type] = models[model_type]
        
        # 4. Create and optimize ensemble
        logger.info(f"\n[ENSEMBLE] Creating optimized ensemble...")
        
        # Optimize stacking weights
        optimal_weights = self.performance_enforcer.optimize_stacking_weights(
            base_models=adjusted_models,
            X_val=X_val_scaled,
            y_val=y_val,
            property_type=property_type
        )
        
        # Create ensemble with optimized weights
        ensemble_config = self.config['models']['ensemble'].copy()
        ensemble_config['weights'] = {
            model_type: weight for model_type, weight in 
            zip(['xgboost', 'catboost', 'random_forest', 'gradient_boosting'], optimal_weights)
        }
        
        ensemble_model = EnsembleModel(
            config=ensemble_config,
            model_configs=self.config['models'],
            method='voting',  # Use voting with optimized weights
            n_jobs=self.config.get('intel_optimization', {}).get('num_threads', 20)
        )
        ensemble_model.feature_names = feature_names
        ensemble_model.target_name = property_name
        ensemble_model.train(X_train_scaled, y_train, X_val_scaled, y_val)
        
        models['ensemble'] = ensemble_model
        
        # Evaluate ensemble
        y_pred_ensemble = ensemble_model.predict(X_test_scaled)
        ensemble_metrics = self.evaluator.evaluate(y_test, y_pred_ensemble, f"{property_name}_ensemble")
        logger.info(f"  Optimized ensemble R²: {ensemble_metrics['r2']:.4f}")
        
        # 5. Perform comprehensive cross-validation
        logger.info(f"\n[CROSS-VALIDATION] Performing 5-fold CV...")
        
        # Combine train and validation for CV
        X_combined = np.vstack([X_train_scaled, X_val_scaled])
        y_combined = np.concatenate([y_train, y_val])
        
        cv_results = {}
        for model_type in models.keys():
            if model_type == 'ensemble':
                continue  # Skip ensemble for individual CV
            
            cv_result = self.cross_validator.kfold_with_performance_targets(
                model_constructor=lambda params: self.model_constructors[model_type](params),
                X=X_combined,
                y=y_combined,
                model_params=self.config['models'][model_type],
                property_name=f"{property_name}_{model_type}",
                property_type=property_type,
                performance_targets=self.config['targets']
            )
            cv_results[model_type] = cv_result
        
        # 6. Uncertainty estimation
        logger.info(f"\n[UNCERTAINTY] Estimating prediction uncertainty...")
        
        # Ensemble variance uncertainty
        ensemble_pred, ensemble_uncertainty = self.performance_enforcer.estimate_prediction_uncertainty(
            models=adjusted_models,
            X=X_test_scaled,
            method='ensemble_variance'
        )
        
        # Random Forest uncertainty
        rf_pred, rf_uncertainty = None, None
        if 'random_forest' in models:
            try:
                rf_pred, rf_uncertainty = self.performance_enforcer.estimate_prediction_uncertainty(
                    models={'random_forest': models['random_forest']},
                    X=X_test_scaled,
                    method='random_forest_variance'
                )
            except Exception as e:
                logger.warning(f"Random Forest uncertainty estimation failed: {e}")
        
        # CatBoost uncertainty
        cat_pred, cat_uncertainty = None, None
        if 'catboost' in models:
            try:
                cat_pred, cat_uncertainty = self.performance_enforcer.estimate_prediction_uncertainty(
                    models={'catboost': models['catboost']},
                    X=X_test_scaled,
                    method='catboost_uncertainty'
                )
            except Exception as e:
                logger.warning(f"CatBoost uncertainty estimation failed: {e}")
        
        # 7. Final performance validation
        logger.info(f"\n[FINAL VALIDATION] Validating final performance...")
        final_target_results = self.performance_enforcer.validate_performance_targets(
            models, X_test_scaled, y_test, property_name, property_type
        )
        
        # Calculate final metrics for all models
        final_metrics = {}
        for model_type, model in models.items():
            y_pred = model.predict(X_test_scaled)
            metrics = self.evaluator.evaluate(y_test, y_pred, f"{property_name}_{model_type}_final")
            final_metrics[model_type] = metrics
        
        # 8. Compile comprehensive results
        training_results = {
            'property_name': property_name,
            'property_type': property_type,
            'ceramic_system': ceramic_system,
            'models': models,
            'scaler': scaler,
            'feature_names': feature_names,
            'performance': {
                'initial_performance': initial_performance,
                'final_metrics': final_metrics,
                'target_results': final_target_results,
                'targets_met': all(final_target_results.values()),
                'ensemble_weights': optimal_weights
            },
            'cross_validation': cv_results,
            'uncertainty': {
                'ensemble_uncertainty': {
                    'predictions': ensemble_pred,
                    'uncertainties': ensemble_uncertainty,
                    'mean_uncertainty': np.mean(ensemble_uncertainty),
                    'max_uncertainty': np.max(ensemble_uncertainty)
                },
                'random_forest_uncertainty': {
                    'predictions': rf_pred,
                    'uncertainties': rf_uncertainty,
                    'mean_uncertainty': np.mean(rf_uncertainty) if rf_uncertainty is not None else None
                } if rf_uncertainty is not None else None,
                'catboost_uncertainty': {
                    'predictions': cat_pred,
                    'uncertainties': cat_uncertainty,
                    'mean_uncertainty': np.mean(cat_uncertainty) if cat_uncertainty is not None else None
                } if cat_uncertainty is not None else None
            },
            'data_info': {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test),
                'n_features': X_train.shape[1]
            }
        }
        
        # Store results
        result_key = f"{ceramic_system}_{property_name}"
        self.trained_models[result_key] = training_results
        
        # Log final summary
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING COMPLETE: {property_name} ({property_type})")
        logger.info(f"{'='*80}")
        
        targets_met_count = sum(final_target_results.values())
        total_models = len(final_target_results)
        
        logger.info(f"Performance Summary:")
        logger.info(f"  Models meeting targets: {targets_met_count}/{total_models}")
        logger.info(f"  Best model R²: {max(metrics['r2'] for metrics in final_metrics.values()):.4f}")
        logger.info(f"  Ensemble R²: {final_metrics.get('ensemble', {}).get('r2', 0):.4f}")
        logger.info(f"  Mean uncertainty: {np.mean(ensemble_uncertainty):.4f}")
        
        return training_results
    
    def train_ceramic_system_with_loco(self,
                                     datasets_by_system: Dict[str, Dict[str, np.ndarray]],
                                     property_name: str,
                                     property_type: str,
                                     feature_names: List[str]) -> Dict:
        """
        Train models with leave-one-ceramic-out validation
        
        Args:
            datasets_by_system: {system: {'X': features, 'y': targets}}
            property_name: Property name
            property_type: 'mechanical' or 'ballistic'
            feature_names: Feature names
        
        Returns:
            LOCO training results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"LOCO Training: {property_name} ({property_type})")
        logger.info(f"Systems: {list(datasets_by_system.keys())}")
        logger.info(f"{'='*80}")
        
        # Perform LOCO validation for each model type
        loco_results = {}
        
        for model_type in ['xgboost', 'catboost', 'random_forest', 'gradient_boosting']:
            logger.info(f"\n[{model_type.upper()}] LOCO validation...")
            
            loco_result = self.cross_validator.leave_one_ceramic_out_with_targets(
                model_constructor=lambda params: self.model_constructors[model_type](params),
                datasets_by_system=datasets_by_system,
                model_params=self.config['models'][model_type],
                property_name=f"{property_name}_{model_type}",
                property_type=property_type,
                performance_targets=self.config['targets']
            )
            
            loco_results[model_type] = loco_result
        
        # Store LOCO results
        self.validation_results[f"{property_name}_loco"] = loco_results
        
        return loco_results
    
    def _create_xgboost_model(self, params: Dict) -> XGBoostModel:
        """Create XGBoost model with given parameters"""
        return XGBoostModel(params, n_jobs=self.config.get('intel_optimization', {}).get('num_threads', 20))
    
    def _create_catboost_model(self, params: Dict) -> CatBoostModel:
        """Create CatBoost model with given parameters"""
        return CatBoostModel(params, n_jobs=self.config.get('intel_optimization', {}).get('num_threads', 20))
    
    def _create_random_forest_model(self, params: Dict) -> RandomForestModel:
        """Create Random Forest model with given parameters"""
        return RandomForestModel(params, n_jobs=self.config.get('intel_optimization', {}).get('num_threads', 20))
    
    def _create_gradient_boosting_model(self, params: Dict) -> GradientBoostingModel:
        """Create Gradient Boosting model with given parameters"""
        return GradientBoostingModel(params, n_jobs=self.config.get('intel_optimization', {}).get('num_threads', 20))
    
    def _create_ensemble_model(self, params: Dict) -> EnsembleModel:
        """Create Ensemble model with given parameters"""
        return EnsembleModel(
            config=params,
            model_configs=self.config['models'],
            method='stacking',
            n_jobs=self.config.get('intel_optimization', {}).get('num_threads', 20)
        )
    
    def save_training_results(self, output_dir: str = "results/performance_driven_training"):
        """Save all training results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save trained models
        models_dir = output_path / "models"
        models_dir.mkdir(exist_ok=True)
        
        for result_key, training_result in self.trained_models.items():
            result_dir = models_dir / result_key
            result_dir.mkdir(exist_ok=True)
            
            # Save individual models
            for model_type, model in training_result['models'].items():
                model_path = result_dir / f"{model_type}_model.pkl"
                model.save_model(str(model_path))
            
            # Save scaler
            import pickle
            scaler_path = result_dir / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(training_result['scaler'], f)
            
            # Save feature names
            feature_path = result_dir / "feature_names.pkl"
            with open(feature_path, 'wb') as f:
                pickle.dump(training_result['feature_names'], f)
        
        # Save performance results
        performance_report = self.performance_enforcer.generate_performance_report(
            str(output_path / "performance_report.yaml")
        )
        
        # Save validation results
        with open(output_path / "validation_results.yaml", 'w') as f:
            yaml.dump(self.validation_results, f, default_flow_style=False)
        
        # Save cross-validation summary
        cv_summary = self.cross_validator.get_cv_summary()
        with open(output_path / "cross_validation_summary.yaml", 'w') as f:
            yaml.dump(cv_summary, f, default_flow_style=False)
        
        logger.info(f"✓ All training results saved to {output_dir}")
    
    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary"""
        summary = {
            'total_properties_trained': len(self.trained_models),
            'performance_targets': {
                'mechanical_r2': self.config['targets']['mechanical_r2'],
                'ballistic_r2': self.config['targets']['ballistic_r2']
            },
            'models_trained': {},
            'performance_summary': {},
            'validation_summary': {}
        }
        
        # Summarize model performance
        for result_key, training_result in self.trained_models.items():
            property_name = training_result['property_name']
            property_type = training_result['property_type']
            
            # Model performance
            final_metrics = training_result['performance']['final_metrics']
            targets_met = training_result['performance']['targets_met']
            
            summary['models_trained'][result_key] = {
                'property_type': property_type,
                'targets_met': targets_met,
                'best_r2': max(metrics['r2'] for metrics in final_metrics.values()),
                'ensemble_r2': final_metrics.get('ensemble', {}).get('r2', 0),
                'mean_uncertainty': training_result['uncertainty']['ensemble_uncertainty']['mean_uncertainty']
            }
        
        # Performance statistics
        all_r2_scores = [info['best_r2'] for info in summary['models_trained'].values()]
        targets_met_count = sum(1 for info in summary['models_trained'].values() if info['targets_met'])
        
        summary['performance_summary'] = {
            'mean_r2': np.mean(all_r2_scores) if all_r2_scores else 0,
            'std_r2': np.std(all_r2_scores) if all_r2_scores else 0,
            'min_r2': np.min(all_r2_scores) if all_r2_scores else 0,
            'max_r2': np.max(all_r2_scores) if all_r2_scores else 0,
            'targets_achievement_rate': targets_met_count / len(self.trained_models) if self.trained_models else 0
        }
        
        # Validation summary
        cv_summary = self.cross_validator.get_cv_summary()
        summary['validation_summary'] = cv_summary
        
        return summary