"""
Gradient Boosting Model Implementation with Intel Extension Acceleration.

This module provides a complete implementation of Gradient Boosting Regressor
for ceramic armor property prediction with uncertainty quantification,
hyperparameter optimization, and Intel CPU acceleration.

Classes:
    GradientBoostingModel: Gradient Boosting Regressor with uncertainty estimation

Example:
    >>> config = {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05}
    >>> model = GradientBoostingModel(config, n_jobs=20)
    >>> model.train(X_train, y_train, X_val, y_val)
    >>> predictions, uncertainties = model.predict_with_uncertainty(X_test)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import validation_curve
from typing import Dict, Tuple, Optional, List, Union, Any
import logging
from pathlib import Path

from .base_model import BaseModel
from src.utils.intel_optimizer import intel_opt
from src.utils.logger import get_logger

logger = get_logger(__name__)

class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting Regressor with uncertainty quantification and CPU optimization.
    
    This class implements a complete Gradient Boosting Regressor for ceramic armor
    property prediction with Intel Extension acceleration, uncertainty estimation
    through staged predictions, and comprehensive error handling.
    
    Attributes:
        n_jobs (int): Number of CPU threads for consistency with other models
        staged_predictions (Optional[Dict]): Stored staged predictions for uncertainty
        validation_scores (List[float]): Validation scores during training
        
    Example:
        >>> config = {
        ...     'n_estimators': 500,
        ...     'max_depth': 6,
        ...     'learning_rate': 0.05,
        ...     'subsample': 0.8
        ... }
        >>> model = GradientBoostingModel(config, n_jobs=20)
        >>> model.train(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, config: Dict[str, Any], n_jobs: int = 20) -> None:
        """
        Initialize Gradient Boosting model with comprehensive validation.
        
        Args:
            config: Gradient Boosting hyperparameters dictionary containing:
                - n_estimators (int): Number of boosting stages (default: 500)
                - max_depth (int): Maximum depth of trees (default: 6)
                - learning_rate (float): Learning rate (default: 0.05)
                - subsample (float): Fraction of samples for fitting (default: 0.8)
                - min_samples_split (int): Min samples to split node (default: 5)
                - min_samples_leaf (int): Min samples in leaf (default: 2)
                - max_features (str): Features to consider for splits (default: 'sqrt')
            n_jobs: Number of CPU threads (kept for API consistency)
            
        Raises:
            TypeError: If config is not a dictionary
            ValueError: If config contains invalid parameter values
            
        Example:
            >>> config = {'n_estimators': 500, 'learning_rate': 0.05}
            >>> model = GradientBoostingModel(config, n_jobs=20)
        """
        try:
            # Validate inputs
            if not isinstance(config, dict):
                raise TypeError(f"config must be a dictionary, got {type(config)}")
            
            if not isinstance(n_jobs, int) or n_jobs < 1:
                raise ValueError(f"n_jobs must be a positive integer, got {n_jobs}")
            
            # Validate config parameters
            self._validate_config(config)
            
            super().__init__('gradient_boosting', config)
            self.n_jobs = n_jobs
            self.staged_predictions: Optional[Dict[str, List[np.ndarray]]] = None
            self.validation_scores: List[float] = []
            
            logger.info(f"Initializing GradientBoostingModel with {n_jobs} threads")
            self.build_model()
            
        except Exception as e:
            logger.error(f"Failed to initialize GradientBoostingModel: {e}")
            raise
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration parameters for Gradient Boosting model.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If any parameter is invalid
        """
        try:
            # Validate n_estimators
            n_estimators = config.get('n_estimators', 500)
            if not isinstance(n_estimators, int) or n_estimators < 1:
                raise ValueError(f"n_estimators must be positive integer, got {n_estimators}")
            
            # Validate learning_rate
            learning_rate = config.get('learning_rate', 0.05)
            if not isinstance(learning_rate, (int, float)) or learning_rate <= 0 or learning_rate > 1:
                raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
            
            # Validate max_depth
            max_depth = config.get('max_depth', 6)
            if not isinstance(max_depth, int) or max_depth < 1:
                raise ValueError(f"max_depth must be positive integer, got {max_depth}")
            
            # Validate subsample
            subsample = config.get('subsample', 0.8)
            if not isinstance(subsample, (int, float)) or subsample <= 0 or subsample > 1:
                raise ValueError(f"subsample must be in (0, 1], got {subsample}")
            
            logger.debug("Configuration validation passed")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

    def build_model(self) -> None:
        """
        Build Gradient Boosting model with Intel Extension acceleration.
        
        Creates a GradientBoostingRegressor with optimized parameters and
        Intel CPU acceleration when available.
        
        Raises:
            RuntimeError: If model building fails
            
        Example:
            >>> model = GradientBoostingModel(config)
            >>> # Model is automatically built during initialization
        """
        try:
            # Ensure Intel optimizations are applied
            if not intel_opt.optimization_applied:
                intel_opt.apply_optimizations()
            
            # Build model with validated parameters
            self.model = GradientBoostingRegressor(
                n_estimators=self.config.get('n_estimators', 500),
                max_depth=self.config.get('max_depth', 6),
                learning_rate=self.config.get('learning_rate', 0.05),
                subsample=self.config.get('subsample', 0.8),
                min_samples_split=self.config.get('min_samples_split', 5),
                min_samples_leaf=self.config.get('min_samples_leaf', 2),
                max_features=self.config.get('max_features', 'sqrt'),
                validation_fraction=self.config.get('validation_fraction', 0.1),
                n_iter_no_change=self.config.get('n_iter_no_change', 10),
                tol=float(self.config.get('tol', 1e-4)),
                random_state=42,
                verbose=0,
                warm_start=False
            )
            
            logger.info(f"✓ Gradient Boosting model built with Intel Extension acceleration "
                       f"and {self.config.get('n_estimators', 500)} estimators")
            
        except Exception as e:
            logger.error(f"Failed to build Gradient Boosting model: {e}")
            raise RuntimeError(f"Model building failed: {e}") from e
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              monitor_staged: bool = True) -> None:
        """
        Train Gradient Boosting model with comprehensive validation and monitoring.
        
        Trains the model on provided data with optional validation monitoring
        and staged prediction storage for uncertainty estimation.
        
        Args:
            X_train: Training feature matrix of shape (n_samples, n_features)
            y_train: Training target vector of shape (n_samples,)
            X_val: Optional validation features for monitoring performance
            y_val: Optional validation targets for monitoring performance
            monitor_staged: Whether to store staged predictions for uncertainty estimation
            
        Raises:
            ValueError: If input data is invalid or model not initialized
            RuntimeError: If training fails
            
        Example:
            >>> model.train(X_train, y_train, X_val, y_val)
            >>> # Model is now trained and ready for predictions
        """
        try:
            # Validate inputs
            if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
                raise ValueError("X_train and y_train must be numpy arrays")
            
            if X_train.shape[0] != y_train.shape[0]:
                raise ValueError(f"X_train and y_train must have same number of samples: "
                               f"{X_train.shape[0]} vs {y_train.shape[0]}")
            
            if X_train.shape[0] < 2:
                raise ValueError(f"Need at least 2 samples for training, got {X_train.shape[0]}")
            
            if X_val is not None and y_val is not None:
                if not isinstance(X_val, np.ndarray) or not isinstance(y_val, np.ndarray):
                    raise ValueError("X_val and y_val must be numpy arrays")
                
                if X_val.shape[0] != y_val.shape[0]:
                    raise ValueError(f"X_val and y_val must have same number of samples: "
                                   f"{X_val.shape[0]} vs {y_val.shape[0]}")
                
                if X_val.shape[1] != X_train.shape[1]:
                    raise ValueError(f"X_val must have same number of features as X_train: "
                                   f"{X_val.shape[1]} vs {X_train.shape[1]}")
            
            if self.model is None:
                raise ValueError("Model not initialized. Call build_model() first.")
            
            logger.info(f"Training Gradient Boosting on {X_train.shape[0]} samples, "
                       f"{X_train.shape[1]} features")
            
            # Fit the model
            self.model.fit(X_train, y_train)
            
            # Store staged predictions for uncertainty estimation if validation set provided
            if monitor_staged and X_val is not None and y_val is not None:
                try:
                    self.staged_predictions = {
                        'train': list(self.model.staged_predict(X_train)),
                        'val': list(self.model.staged_predict(X_val))
                    }
                    
                    # Calculate validation scores
                    self.validation_scores = []
                    y_val_var = np.var(y_val)
                    
                    if y_val_var > 0:  # Avoid division by zero
                        for pred in self.staged_predictions['val']:
                            mse = np.mean((y_val - pred) ** 2)
                            r2_score = 1 - mse / y_val_var
                            self.validation_scores.append(r2_score)
                        
                        if self.validation_scores:
                            best_iteration = np.argmax(self.validation_scores)
                            best_score = self.validation_scores[best_iteration]
                            logger.info(f"✓ Training complete (best iteration: {best_iteration + 1}, "
                                       f"val R²: {best_score:.4f})")
                        else:
                            logger.warning("No validation scores calculated")
                    else:
                        logger.warning("Validation target has zero variance, cannot calculate R²")
                        
                except Exception as e:
                    logger.warning(f"Failed to store staged predictions: {e}")
                    self.staged_predictions = None
                    self.validation_scores = []
            else:
                logger.info("✓ Training complete")
            
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Model training failed: {e}") from e
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained Gradient Boosting model.
        
        Args:
            X: Input feature matrix of shape (n_samples, n_features)
            
        Returns:
            Prediction array of shape (n_samples,)
            
        Raises:
            ValueError: If model not trained or input invalid
            RuntimeError: If prediction fails
            
        Example:
            >>> predictions = model.predict(X_test)
            >>> print(f"Predicted values: {predictions[:5]}")
        """
        try:
            if not self.is_trained:
                raise ValueError("Model not trained. Call train() first.")
            
            if not isinstance(X, np.ndarray):
                raise ValueError(f"X must be numpy array, got {type(X)}")
            
            if X.ndim != 2:
                raise ValueError(f"X must be 2D array, got shape {X.shape}")
            
            if self.model is None:
                raise ValueError("Model is None. Rebuild model.")
            
            predictions = self.model.predict(X)
            
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            
            logger.debug(f"Generated predictions for {X.shape[0]} samples")
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e
    
    def predict_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates (alias for predict_with_uncertainty).
        
        Args:
            X: Input feature matrix of shape (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, uncertainties)
            
        Example:
            >>> predictions, uncertainties = model.predict_uncertainty(X_test)
        """
        return self.predict_with_uncertainty(X)

    def predict_with_uncertainty(self, X: np.ndarray, n_estimators_range: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates using staged predictions variance.
        
        Uses the variance across different boosting stages to estimate prediction
        uncertainty. This provides a measure of model confidence in predictions.
        
        Args:
            X: Input feature matrix of shape (n_samples, n_features)
            n_estimators_range: Optional tuple (start, end) specifying which
                               estimator range to use for uncertainty calculation.
                               If None, uses last 20% of estimators.
        
        Returns:
            Tuple of (predictions, uncertainties) where:
            - predictions: Final model predictions of shape (n_samples,)
            - uncertainties: Uncertainty estimates of shape (n_samples,)
            
        Raises:
            ValueError: If model not trained or input invalid
            RuntimeError: If uncertainty calculation fails
            
        Example:
            >>> predictions, uncertainties = model.predict_with_uncertainty(X_test)
            >>> high_uncertainty_mask = uncertainties > np.percentile(uncertainties, 90)
            >>> print(f"High uncertainty predictions: {np.sum(high_uncertainty_mask)}")
        """
        try:
            if not self.is_trained:
                raise ValueError("Model not trained. Call train() first.")
            
            if not isinstance(X, np.ndarray):
                raise ValueError(f"X must be numpy array, got {type(X)}")
            
            if X.ndim != 2:
                raise ValueError(f"X must be 2D array, got shape {X.shape}")
            
            if self.model is None:
                raise ValueError("Model is None. Rebuild model.")
            
            # Get staged predictions for uncertainty estimation
            staged_preds = list(self.model.staged_predict(X))
            
            if not staged_preds:
                raise RuntimeError("No staged predictions available")
            
            if n_estimators_range is None:
                # Use last 20% of estimators for uncertainty estimation
                start_idx = max(0, int(0.8 * len(staged_preds)))
                end_idx = len(staged_preds)
            else:
                if not isinstance(n_estimators_range, tuple) or len(n_estimators_range) != 2:
                    raise ValueError("n_estimators_range must be tuple of (start, end)")
                
                start_idx, end_idx = n_estimators_range
                start_idx = max(0, min(start_idx, len(staged_preds)))
                end_idx = max(start_idx + 1, min(end_idx, len(staged_preds)))
            
            # Calculate uncertainty from variance in staged predictions
            uncertainty_preds = np.array(staged_preds[start_idx:end_idx])
            
            # Final prediction (last stage)
            predictions = staged_preds[-1]
            
            # Uncertainty as standard deviation across recent stages
            if uncertainty_preds.shape[0] > 1:
                uncertainties = np.std(uncertainty_preds, axis=0)
            else:
                # Fallback: use 5% of prediction magnitude as uncertainty
                uncertainties = np.abs(predictions) * 0.05
                logger.warning("Using fallback uncertainty estimation (5% of prediction magnitude)")
            
            logger.debug(f"Calculated uncertainties for {X.shape[0]} samples using "
                        f"{uncertainty_preds.shape[0]} estimator stages")
            
            return predictions, uncertainties
            
        except Exception as e:
            logger.error(f"Uncertainty prediction failed: {e}")
            raise RuntimeError(f"Uncertainty prediction failed: {e}") from e
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on impurity decrease.
        
        Returns feature importance scores calculated from the trained model,
        sorted in descending order of importance.
        
        Returns:
            DataFrame with columns ['feature', 'importance'] sorted by importance
            
        Raises:
            ValueError: If model not trained
            RuntimeError: If importance calculation fails
            
        Example:
            >>> importance_df = model.get_feature_importance()
            >>> print(f"Top 5 features: {importance_df.head()}")
        """
        try:
            if not self.is_trained:
                raise ValueError("Model not trained. Call train() first.")
            
            if self.model is None:
                raise ValueError("Model is None. Rebuild model.")
            
            if not hasattr(self.model, 'feature_importances_'):
                raise RuntimeError("Model does not have feature_importances_ attribute")
            
            importance = self.model.feature_importances_
            
            if importance is None or len(importance) == 0:
                raise RuntimeError("Feature importances are empty")
            
            # Generate feature names
            feature_names = [f"feature_{i}" for i in range(len(importance))]
            if self.feature_names is not None and len(self.feature_names) == len(importance):
                feature_names = self.feature_names
            elif self.feature_names is not None:
                logger.warning(f"Feature names length mismatch: {len(self.feature_names)} vs {len(importance)}")
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            logger.debug(f"Generated feature importance for {len(importance)} features")
            return importance_df
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            raise RuntimeError(f"Feature importance calculation failed: {e}") from e
    
    def get_learning_curve(self) -> Optional[Dict[str, List[float]]]:
        """
        Get training learning curve from staged predictions.
        
        Returns learning curve data including training and validation scores
        across boosting iterations if staged predictions were stored during training.
        
        Returns:
            Dictionary with keys ['train_scores', 'val_scores', 'n_estimators']
            or None if no staged predictions available
            
        Example:
            >>> curve_data = model.get_learning_curve()
            >>> if curve_data:
            ...     plt.plot(curve_data['n_estimators'], curve_data['val_scores'])
        """
        try:
            if not self.is_trained:
                logger.warning("Model not trained, no learning curve available")
                return None
            
            if not self.staged_predictions:
                logger.debug("No staged predictions stored, no learning curve available")
                return None
            
            train_preds = self.staged_predictions.get('train', [])
            val_preds = self.staged_predictions.get('val', [])
            
            if not train_preds:
                logger.warning("No training staged predictions available")
                return None
            
            curve_data = {
                'train_scores': [float(pred.mean()) for pred in train_preds],
                'val_scores': [float(pred.mean()) for pred in val_preds] if val_preds else [],
                'n_estimators': list(range(1, len(train_preds) + 1))
            }
            
            logger.debug(f"Generated learning curve with {len(train_preds)} points")
            return curve_data
            
        except Exception as e:
            logger.error(f"Learning curve generation failed: {e}")
            return None
    
    def optimize_n_estimators(self, X: np.ndarray, y: np.ndarray, 
                            n_estimators_range: Tuple[int, int] = (100, 1000),
                            cv: int = 5) -> int:
        """
        Optimize number of estimators using cross-validation.
        
        Performs hyperparameter optimization to find the optimal number of
        estimators using validation curves and cross-validation.
        
        Args:
            X: Training feature matrix of shape (n_samples, n_features)
            y: Training target vector of shape (n_samples,)
            n_estimators_range: Tuple (min, max) defining search range for n_estimators
            cv: Number of cross-validation folds
        
        Returns:
            Optimal number of estimators
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If optimization fails
            
        Example:
            >>> optimal_n = model.optimize_n_estimators(X_train, y_train, (100, 500), cv=3)
            >>> print(f"Optimal n_estimators: {optimal_n}")
        """
        try:
            # Validate inputs
            if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
                raise ValueError("X and y must be numpy arrays")
            
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"X and y must have same number of samples: "
                               f"{X.shape[0]} vs {y.shape[0]}")
            
            if not isinstance(n_estimators_range, tuple) or len(n_estimators_range) != 2:
                raise ValueError("n_estimators_range must be tuple of (min, max)")
            
            min_est, max_est = n_estimators_range
            if min_est >= max_est or min_est < 1:
                raise ValueError(f"Invalid n_estimators_range: {n_estimators_range}")
            
            if not isinstance(cv, int) or cv < 2:
                raise ValueError(f"cv must be integer >= 2, got {cv}")
            
            logger.info(f"Optimizing n_estimators using validation curve "
                       f"(range: {n_estimators_range}, CV folds: {cv})")
            
            # Create range of n_estimators to test
            n_estimators_values = np.linspace(min_est, max_est, 10, dtype=int)
            n_estimators_values = np.unique(n_estimators_values)  # Remove duplicates
            
            # Create a temporary model for validation
            temp_model = GradientBoostingRegressor(
                max_depth=self.config.get('max_depth', 6),
                learning_rate=self.config.get('learning_rate', 0.05),
                subsample=self.config.get('subsample', 0.8),
                min_samples_split=self.config.get('min_samples_split', 5),
                min_samples_leaf=self.config.get('min_samples_leaf', 2),
                max_features=self.config.get('max_features', 'sqrt'),
                random_state=42,
                verbose=0
            )
            
            # Calculate validation curve
            train_scores, val_scores = validation_curve(
                temp_model, X, y,
                param_name='n_estimators',
                param_range=n_estimators_values,
                cv=cv,
                scoring='r2',
                n_jobs=1  # GradientBoosting doesn't support parallel CV
            )
            
            # Find optimal n_estimators
            mean_val_scores = val_scores.mean(axis=1)
            std_val_scores = val_scores.std(axis=1)
            
            if len(mean_val_scores) == 0:
                raise RuntimeError("No validation scores calculated")
            
            optimal_idx = np.argmax(mean_val_scores)
            optimal_n_estimators = int(n_estimators_values[optimal_idx])
            optimal_score = mean_val_scores[optimal_idx]
            optimal_std = std_val_scores[optimal_idx]
            
            logger.info(f"Optimal n_estimators: {optimal_n_estimators} "
                       f"(CV R²: {optimal_score:.4f} ± {optimal_std:.4f})")
            
            return optimal_n_estimators
            
        except Exception as e:
            logger.error(f"n_estimators optimization failed: {e}")
            raise RuntimeError(f"n_estimators optimization failed: {e}") from e
