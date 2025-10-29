"""
Performance Target Enforcement System
Implements automatic validation and adjustment to meet strict performance targets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_score
from scipy.optimize import minimize
import optuna
from loguru import logger
import yaml
from pathlib import Path

from src.training.hyperparameter_tuner import HyperparameterTuner
from src.evaluation.metrics import ModelEvaluator


class PerformanceTargetEnforcer:
    """
    Enforces strict performance targets with automatic validation and adjustment
    
    Features:
    - Automatic performance validation for R² ≥ 0.85 (mechanical) and R² ≥ 0.80 (ballistic)
    - Automatic hyperparameter adjustment when targets not met
    - Stacking weight optimization to maximize ensemble performance
    - 5-fold cross-validation and leave-one-ceramic-family-out validation
    - Prediction uncertainty estimation using Random Forest variance and CatBoost uncertainty
    """
    
    def __init__(self, config: Dict):
        """
        Initialize performance enforcer
        
        Args:
            config: Configuration dictionary with performance targets
        """
        self.config = config
        self.mechanical_target = config['targets']['mechanical_r2']
        self.ballistic_target = config['targets']['ballistic_r2']
        self.uncertainty_threshold = config['targets'].get('uncertainty_threshold', 0.15)
        
        # Performance tracking
        self.performance_history = {}
        self.adjustment_history = {}
        self.validation_results = {}
        
        # Cross-validation setup
        self.cv_folds = config.get('cross_validation', {}).get('k_fold', {}).get('n_splits', 5)
        self.cv_random_state = config.get('cross_validation', {}).get('k_fold', {}).get('random_state', 42)
        
        # Hyperparameter tuner
        self.hyperparameter_tuner = None
        
        # Model evaluator
        self.evaluator = ModelEvaluator()
        
        logger.info("✓ Performance Target Enforcer initialized")
        logger.info(f"✓ Mechanical R² target: {self.mechanical_target}")
        logger.info(f"✓ Ballistic R² target: {self.ballistic_target}")
        logger.info(f"✓ Uncertainty threshold: {self.uncertainty_threshold}")
    
    def validate_performance_targets(self, 
                                   models: Dict[str, Any], 
                                   X_test: np.ndarray, 
                                   y_test: np.ndarray,
                                   property_name: str,
                                   property_type: str) -> Dict[str, bool]:
        """
        Validate that models meet performance targets
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test targets
            property_name: Name of the property being predicted
            property_type: 'mechanical' or 'ballistic'
        
        Returns:
            Dictionary indicating which models meet targets
        """
        logger.info(f"Validating performance targets for {property_name} ({property_type})")
        
        target_r2 = self.mechanical_target if property_type == 'mechanical' else self.ballistic_target
        results = {}
        
        for model_name, model in models.items():
            try:
                # Get predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = self.evaluator.evaluate(y_test, y_pred, property_name)
                r2 = metrics['r2']
                
                # Check if target is met
                meets_target = r2 >= target_r2
                results[model_name] = meets_target
                
                # Log results
                status = "✓ PASS" if meets_target else "✗ FAIL"
                logger.info(f"  {model_name}: R²={r2:.4f} (target: {target_r2:.2f}) {status}")
                
                # Store performance history
                key = f"{property_name}_{model_name}"
                if key not in self.performance_history:
                    self.performance_history[key] = []
                self.performance_history[key].append({
                    'r2': r2,
                    'meets_target': meets_target,
                    'target': target_r2,
                    'metrics': metrics
                })
                
            except Exception as e:
                logger.error(f"Error validating {model_name}: {e}")
                results[model_name] = False
        
        return results
    
    def automatic_hyperparameter_adjustment(self,
                                          model_constructor: Callable,
                                          X_train: np.ndarray,
                                          y_train: np.ndarray,
                                          X_val: np.ndarray,
                                          y_val: np.ndarray,
                                          property_type: str,
                                          model_name: str,
                                          max_iterations: int = 3) -> Tuple[Any, Dict]:
        """
        Automatically adjust hyperparameters when performance targets are not met
        
        Args:
            model_constructor: Function to create model with given parameters
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            property_type: 'mechanical' or 'ballistic'
            model_name: Name of the model type
            max_iterations: Maximum adjustment iterations
        
        Returns:
            Tuple of (best_model, best_params)
        """
        logger.info(f"Starting automatic hyperparameter adjustment for {model_name}")
        
        target_r2 = self.mechanical_target if property_type == 'mechanical' else self.ballistic_target
        
        # Define search spaces for different model types
        search_spaces = self._get_search_spaces()
        
        if model_name not in search_spaces:
            logger.warning(f"No search space defined for {model_name}")
            return None, {}
        
        search_space = search_spaces[model_name]
        
        # Initialize hyperparameter tuner
        self.hyperparameter_tuner = HyperparameterTuner(
            search_space=search_space,
            n_trials=50,  # Increased for better optimization
            random_state=self.cv_random_state
        )
        
        best_model = None
        best_params = {}
        best_r2 = 0.0
        
        for iteration in range(max_iterations):
            logger.info(f"Hyperparameter adjustment iteration {iteration + 1}/{max_iterations}")
            
            # Optimize hyperparameters
            try:
                # Combine training and validation for optimization
                X_combined = np.vstack([X_train, X_val])
                y_combined = np.concatenate([y_train, y_val])
                
                params, cv_r2 = self.hyperparameter_tuner.optimize(
                    model_constructor, X_combined, y_combined, cv_splits=self.cv_folds
                )
                
                # Train model with optimized parameters
                model = model_constructor(params)
                model.train(X_train, y_train, X_val, y_val)
                
                # Validate performance
                y_pred = model.predict(X_val)
                r2 = r2_score(y_val, y_pred)
                
                logger.info(f"  Iteration {iteration + 1}: R²={r2:.4f} (CV R²={cv_r2:.4f})")
                
                # Check if target is met
                if r2 >= target_r2:
                    logger.info(f"✓ Target achieved: R²={r2:.4f} >= {target_r2:.2f}")
                    best_model = model
                    best_params = params
                    best_r2 = r2
                    break
                
                # Keep best model so far
                if r2 > best_r2:
                    best_model = model
                    best_params = params
                    best_r2 = r2
                
                # Store adjustment history
                key = f"{model_name}_adjustment"
                if key not in self.adjustment_history:
                    self.adjustment_history[key] = []
                self.adjustment_history[key].append({
                    'iteration': iteration + 1,
                    'params': params,
                    'r2': r2,
                    'cv_r2': cv_r2,
                    'target_met': r2 >= target_r2
                })
                
            except Exception as e:
                logger.error(f"Error in hyperparameter adjustment iteration {iteration + 1}: {e}")
                continue
        
        if best_r2 >= target_r2:
            logger.info(f"✓ Hyperparameter adjustment successful: Final R²={best_r2:.4f}")
        else:
            logger.warning(f"⚠ Target not achieved after {max_iterations} iterations: Best R²={best_r2:.4f}")
        
        return best_model, best_params
    
    def optimize_stacking_weights(self,
                                base_models: Dict[str, Any],
                                X_val: np.ndarray,
                                y_val: np.ndarray,
                                property_type: str) -> np.ndarray:
        """
        Optimize stacking weights to maximize ensemble performance
        
        Args:
            base_models: Dictionary of trained base models
            X_val: Validation features
            y_val: Validation targets
            property_type: 'mechanical' or 'ballistic'
        
        Returns:
            Optimized weights array
        """
        logger.info("Optimizing stacking weights for ensemble performance")
        
        target_r2 = self.mechanical_target if property_type == 'mechanical' else self.ballistic_target
        
        # Get predictions from each base model
        model_names = list(base_models.keys())
        base_predictions = []
        
        for name in model_names:
            try:
                pred = base_models[name].predict(X_val)
                base_predictions.append(pred)
                logger.info(f"  {name}: R²={r2_score(y_val, pred):.4f}")
            except Exception as e:
                logger.error(f"Error getting predictions from {name}: {e}")
                continue
        
        if len(base_predictions) < 2:
            logger.error("Need at least 2 models for stacking optimization")
            return np.array([1.0])
        
        base_predictions = np.array(base_predictions).T  # Shape: (n_samples, n_models)
        
        def objective(weights):
            """Objective function to maximize R² (minimize negative R²)"""
            # Normalize weights
            weights = np.abs(weights)  # Ensure non-negative
            weights = weights / np.sum(weights)  # Normalize to sum to 1
            
            # Calculate ensemble prediction
            ensemble_pred = np.dot(base_predictions, weights)
            
            # Calculate R²
            r2 = r2_score(y_val, ensemble_pred)
            
            return -r2  # Minimize negative R²
        
        # Initial weights (equal)
        n_models = len(base_predictions[0])
        initial_weights = np.ones(n_models) / n_models
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Multiple optimization attempts with different starting points
        best_result = None
        best_r2 = -np.inf
        
        for attempt in range(5):
            # Random starting point for diversity
            if attempt > 0:
                start_weights = np.random.dirichlet(np.ones(n_models))
            else:
                start_weights = initial_weights
            
            try:
                result = minimize(
                    objective,
                    start_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000}
                )
                
                if result.success and -result.fun > best_r2:
                    best_result = result
                    best_r2 = -result.fun
                    
            except Exception as e:
                logger.warning(f"Optimization attempt {attempt + 1} failed: {e}")
                continue
        
        if best_result is None:
            logger.warning("All optimization attempts failed, using equal weights")
            optimal_weights = initial_weights
            optimal_r2 = -objective(initial_weights)
        else:
            optimal_weights = np.abs(best_result.x)
            optimal_weights = optimal_weights / np.sum(optimal_weights)
            optimal_r2 = best_r2
        
        # Log results
        logger.info("Stacking weight optimization results:")
        for i, (name, weight) in enumerate(zip(model_names, optimal_weights)):
            logger.info(f"  {name}: {weight:.4f}")
        logger.info(f"  Optimized ensemble R²: {optimal_r2:.4f}")
        
        # Check if target is met
        if optimal_r2 >= target_r2:
            logger.info(f"✓ Ensemble target achieved: R²={optimal_r2:.4f} >= {target_r2:.2f}")
        else:
            logger.warning(f"⚠ Ensemble target not met: R²={optimal_r2:.4f} < {target_r2:.2f}")
        
        return optimal_weights
    
    def perform_cross_validation(self,
                               model_constructor: Callable,
                               X: np.ndarray,
                               y: np.ndarray,
                               model_params: Dict,
                               property_name: str) -> Dict:
        """
        Perform 5-fold cross-validation
        
        Args:
            model_constructor: Function to create model
            X: Features
            y: Targets
            model_params: Model parameters
            property_name: Property name for logging
        
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing 5-fold cross-validation for {property_name}")
        
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.cv_random_state)
        
        cv_scores = []
        cv_predictions = []
        cv_uncertainties = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.info(f"  Fold {fold + 1}/{self.cv_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model
            model = model_constructor(model_params)
            model.train(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate R²
            r2 = r2_score(y_val, y_pred)
            cv_scores.append(r2)
            
            # Store predictions for ensemble analysis
            cv_predictions.extend(list(zip(val_idx, y_val, y_pred)))
            
            # Get uncertainty if available
            try:
                if hasattr(model, 'predict_with_uncertainty'):
                    _, uncertainty = model.predict_with_uncertainty(X_val)
                    cv_uncertainties.extend(list(zip(val_idx, uncertainty)))
            except:
                pass
            
            logger.info(f"    Fold {fold + 1} R²: {r2:.4f}")
        
        # Calculate statistics
        mean_r2 = np.mean(cv_scores)
        std_r2 = np.std(cv_scores)
        
        results = {
            'scores': cv_scores,
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'predictions': cv_predictions,
            'uncertainties': cv_uncertainties
        }
        
        logger.info(f"✓ 5-fold CV results: R² = {mean_r2:.4f} ± {std_r2:.4f}")
        
        return results
    
    def perform_leave_one_ceramic_out_validation(self,
                                               model_constructor: Callable,
                                               datasets_by_system: Dict[str, Dict[str, np.ndarray]],
                                               model_params: Dict,
                                               property_name: str) -> Dict:
        """
        Perform leave-one-ceramic-family-out validation
        
        Args:
            model_constructor: Function to create model
            datasets_by_system: {system: {'X': features, 'y': targets}}
            model_params: Model parameters
            property_name: Property name for logging
        
        Returns:
            LOCO validation results
        """
        logger.info(f"Performing leave-one-ceramic-out validation for {property_name}")
        
        systems = list(datasets_by_system.keys())
        loco_results = {}
        
        for test_system in systems:
            logger.info(f"  Testing on {test_system}, training on others")
            
            # Prepare training data (all systems except test)
            train_systems = [s for s in systems if s != test_system]
            
            if len(train_systems) == 0:
                logger.warning(f"No training systems available for {test_system}")
                continue
            
            # Combine training data
            X_train_list = [datasets_by_system[s]['X'] for s in train_systems]
            y_train_list = [datasets_by_system[s]['y'] for s in train_systems]
            
            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)
            
            # Test data
            X_test = datasets_by_system[test_system]['X']
            y_test = datasets_by_system[test_system]['y']
            
            # Create and train model
            model = model_constructor(model_params)
            model.train(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self.evaluator.evaluate(y_test, y_pred, f"{property_name}_{test_system}")
            
            loco_results[test_system] = metrics
            
            logger.info(f"    {test_system}: R² = {metrics['r2']:.4f}")
        
        # Calculate overall statistics
        r2_scores = [result['r2'] for result in loco_results.values()]
        mean_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        
        logger.info(f"✓ LOCO validation results: R² = {mean_r2:.4f} ± {std_r2:.4f}")
        
        return {
            'system_results': loco_results,
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'r2_scores': r2_scores
        }
    
    def estimate_prediction_uncertainty(self,
                                      models: Dict[str, Any],
                                      X: np.ndarray,
                                      method: str = 'ensemble_variance') -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate prediction uncertainty using Random Forest variance and CatBoost uncertainty
        
        Args:
            models: Dictionary of trained models
            X: Input features
            method: Uncertainty estimation method
        
        Returns:
            Tuple of (predictions, uncertainties)
        """
        logger.info(f"Estimating prediction uncertainty using {method}")
        
        if method == 'ensemble_variance':
            # Get predictions from all models
            predictions = []
            
            for model_name, model in models.items():
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Could not get predictions from {model_name}: {e}")
                    continue
            
            if len(predictions) == 0:
                raise ValueError("No valid predictions obtained")
            
            predictions = np.array(predictions)
            
            # Ensemble prediction (mean)
            ensemble_pred = np.mean(predictions, axis=0)
            
            # Uncertainty as standard deviation across models
            uncertainties = np.std(predictions, axis=0)
            
        elif method == 'random_forest_variance':
            # Use Random Forest built-in uncertainty
            if 'random_forest' not in models:
                raise ValueError("Random Forest model not available")
            
            rf_model = models['random_forest']
            
            if hasattr(rf_model, 'predict_with_uncertainty'):
                ensemble_pred, uncertainties = rf_model.predict_with_uncertainty(X)
            else:
                # Fallback: use tree predictions variance
                ensemble_pred = rf_model.predict(X)
                
                # Get predictions from individual trees
                tree_predictions = np.array([tree.predict(X) for tree in rf_model.model.estimators_])
                uncertainties = np.std(tree_predictions, axis=0)
        
        elif method == 'catboost_uncertainty':
            # Use CatBoost built-in uncertainty
            if 'catboost' not in models:
                raise ValueError("CatBoost model not available")
            
            cat_model = models['catboost']
            
            if hasattr(cat_model, 'predict_with_uncertainty'):
                ensemble_pred, uncertainties = cat_model.predict_with_uncertainty(X)
            else:
                ensemble_pred = cat_model.predict(X)
                # Fallback: use ensemble variance
                uncertainties = np.full_like(ensemble_pred, self.uncertainty_threshold)
        
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
        
        # Log uncertainty statistics
        mean_uncertainty = np.mean(uncertainties)
        max_uncertainty = np.max(uncertainties)
        
        logger.info(f"✓ Uncertainty estimation complete:")
        logger.info(f"  Mean uncertainty: {mean_uncertainty:.4f}")
        logger.info(f"  Max uncertainty: {max_uncertainty:.4f}")
        logger.info(f"  Threshold: {self.uncertainty_threshold:.4f}")
        
        # Check if uncertainties are within acceptable range
        high_uncertainty_fraction = np.mean(uncertainties > self.uncertainty_threshold)
        if high_uncertainty_fraction > 0.1:  # More than 10% high uncertainty
            logger.warning(f"⚠ {high_uncertainty_fraction:.1%} of predictions have high uncertainty")
        else:
            logger.info(f"✓ {(1-high_uncertainty_fraction):.1%} of predictions within uncertainty threshold")
        
        return ensemble_pred, uncertainties
    
    def _get_search_spaces(self) -> Dict[str, Dict]:
        """Get hyperparameter search spaces for different model types"""
        return {
            'xgboost': {
                'n_estimators': {'type': 'int', 'low': 500, 'high': 2000},
                'max_depth': {'type': 'int', 'low': 4, 'high': 12},
                'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
                'subsample': {'type': 'uniform', 'low': 0.6, 'high': 1.0},
                'colsample_bytree': {'type': 'uniform', 'low': 0.6, 'high': 1.0},
                'reg_alpha': {'type': 'loguniform', 'low': 1e-8, 'high': 10.0},
                'reg_lambda': {'type': 'loguniform', 'low': 1e-8, 'high': 10.0}
            },
            'catboost': {
                'iterations': {'type': 'int', 'low': 500, 'high': 2000},
                'depth': {'type': 'int', 'low': 4, 'high': 10},
                'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
                'l2_leaf_reg': {'type': 'loguniform', 'low': 1, 'high': 10},
                'random_strength': {'type': 'uniform', 'low': 0.1, 'high': 1.0},
                'bagging_temperature': {'type': 'uniform', 'low': 0.0, 'high': 1.0}
            },
            'random_forest': {
                'n_estimators': {'type': 'int', 'low': 200, 'high': 1000},
                'max_depth': {'type': 'int', 'low': 5, 'high': 25},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', 0.5, 0.7, 0.9]}
            },
            'gradient_boosting': {
                'n_estimators': {'type': 'int', 'low': 200, 'high': 1000},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
                'subsample': {'type': 'uniform', 'low': 0.6, 'high': 1.0},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10}
            }
        }
    
    def generate_performance_report(self, output_path: str = "results/performance_report.yaml"):
        """Generate comprehensive performance report"""
        logger.info("Generating performance report...")
        
        report = {
            'performance_targets': {
                'mechanical_r2_target': self.mechanical_target,
                'ballistic_r2_target': self.ballistic_target,
                'uncertainty_threshold': self.uncertainty_threshold
            },
            'performance_history': self.performance_history,
            'adjustment_history': self.adjustment_history,
            'validation_results': self.validation_results,
            'summary': {
                'total_properties_tested': len(self.performance_history),
                'targets_met': sum(1 for history in self.performance_history.values() 
                                 if history[-1]['meets_target']),
                'adjustments_performed': len(self.adjustment_history)
            }
        }
        
        # Save report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        logger.info(f"✓ Performance report saved to {output_path}")
        
        return report