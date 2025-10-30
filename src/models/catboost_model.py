"""
CatBoost Model Implementation with Built-in Uncertainty Quantification.

This module provides a complete implementation of CatBoost Regressor
for ceramic armor property prediction with built-in uncertainty estimation,
categorical feature handling, and comprehensive error handling.

Classes:
    CatBoostModel: CatBoost Regressor with uncertainty estimation

Example:
    >>> config = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05}
    >>> model = CatBoostModel(config, n_jobs=20)
    >>> model.train(X_train, y_train, X_val, y_val)
    >>> predictions, uncertainties = model.predict_with_uncertainty(X_test)
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from typing import Dict, Tuple, Optional, List, Union, Any
import logging
from pathlib import Path

from .base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CatBoostModel(BaseModel):
    """
    CatBoost Regressor with built-in uncertainty quantification.
    
    This class implements a complete CatBoost Regressor for ceramic armor
    property prediction with built-in uncertainty estimation through virtual
    ensembles, categorical feature handling, and comprehensive error handling.
    
    Attributes:
        n_jobs (int): Number of CPU threads for parallel processing
        cat_features (Optional[List[int]]): Indices of categorical features
        best_iteration (Optional[int]): Best iteration from early stopping
        
    Example:
        >>> config = {
        ...     'iterations': 1000,
        ...     'depth': 8,
        ...     'learning_rate': 0.05,
        ...     'l2_leaf_reg': 3
        ... }
        >>> model = CatBoostModel(config, n_jobs=20)
        >>> model.train(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, config: Dict[str, Any], n_jobs: int = 20) -> None:
        """
        Initialize CatBoost model with comprehensive validation.
        
        Args:
            config: CatBoost hyperparameters dictionary containing:
                - iterations (int): Number of boosting iterations (default: 1000)
                - depth (int): Tree depth (default: 8)
                - learning_rate (float): Learning rate (default: 0.05)
                - l2_leaf_reg (float): L2 regularization coefficient (default: 3)
                - random_strength (float): Random strength for scoring (default: 0.5)
                - bagging_temperature (float): Bagging temperature (default: 0.2)
                - border_count (int): Number of splits for numerical features (default: 128)
            n_jobs: Number of CPU threads for parallel processing
            
        Raises:
            TypeError: If config is not a dictionary
            ValueError: If config contains invalid parameter values
            
        Example:
            >>> config = {'iterations': 1000, 'depth': 8}
            >>> model = CatBoostModel(config, n_jobs=20)
        """
        try:
            # Validate inputs
            if not isinstance(config, dict):
                raise TypeError(f"config must be a dictionary, got {type(config)}")
            
            if not isinstance(n_jobs, int) or n_jobs < 1:
                raise ValueError(f"n_jobs must be a positive integer, got {n_jobs}")
            
            # Validate config parameters
            self._validate_config(config)
            
            super().__init__('catboost', config)
            self.n_jobs = n_jobs
            self.cat_features: Optional[List[int]] = None
            self.best_iteration: Optional[int] = None
            
            logger.info(f"Initializing CatBoostModel with {n_jobs} threads")
            self.build_model()
            
        except Exception as e:
            logger.error(f"Failed to initialize CatBoostModel: {e}")
            raise
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration parameters for CatBoost model.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If any parameter is invalid
        """
        try:
            # Validate iterations
            iterations = config.get('iterations', 1000)
            if not isinstance(iterations, int) or iterations < 1:
                raise ValueError(f"iterations must be positive integer, got {iterations}")
            
            # Validate learning_rate
            learning_rate = config.get('learning_rate', 0.05)
            if not isinstance(learning_rate, (int, float)) or learning_rate <= 0 or learning_rate > 1:
                raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
            
            # Validate depth
            depth = config.get('depth', 8)
            if not isinstance(depth, int) or depth < 1 or depth > 16:
                raise ValueError(f"depth must be integer in [1, 16], got {depth}")
            
            # Validate l2_leaf_reg
            l2_leaf_reg = config.get('l2_leaf_reg', 3)
            if not isinstance(l2_leaf_reg, (int, float)) or l2_leaf_reg < 0:
                raise ValueError(f"l2_leaf_reg must be non-negative, got {l2_leaf_reg}")
            
            logger.debug("CatBoost configuration validation passed")
            
        except Exception as e:
            logger.error(f"CatBoost configuration validation failed: {e}")
            raise

    def build_model(self) -> None:
        """
        Build CatBoost model with strict compliance to specification.
        
        Creates a CatBoostRegressor with optimized parameters for ceramic
        armor property prediction, including Bayesian bootstrap for uncertainty.
        
        Raises:
            RuntimeError: If model building fails
            
        Example:
            >>> model = CatBoostModel(config)
            >>> # Model is automatically built during initialization
        """
        try:
            self.model = CatBoostRegressor(
                iterations=self.config.get('iterations', 1000),
                depth=self.config.get('depth', 8),
                learning_rate=self.config.get('learning_rate', 0.05),
                l2_leaf_reg=self.config.get('l2_leaf_reg', 3),
                random_strength=self.config.get('random_strength', 0.5),
                bagging_temperature=self.config.get('bagging_temperature', 0.2),
                border_count=self.config.get('border_count', 128),
                thread_count=self.n_jobs,
                task_type='CPU',
                bootstrap_type='Bayesian',  # Enables uncertainty estimation
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
                train_dir=None  # Prevent temporary file creation
            )
            
            logger.info(f"✓ CatBoost model built with {self.n_jobs} threads (strict compliance)")
            
        except Exception as e:
            logger.error(f"Failed to build CatBoost model: {e}")
            raise RuntimeError(f"Model building failed: {e}") from e
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              cat_features: Optional[List[int]] = None) -> None:
        """
        Train CatBoost model with comprehensive validation and monitoring.
        
        Trains the model on provided data with optional validation monitoring,
        early stopping, and categorical feature handling.
        
        Args:
            X_train: Training feature matrix of shape (n_samples, n_features)
            y_train: Training target vector of shape (n_samples,)
            X_val: Optional validation features for early stopping
            y_val: Optional validation targets for early stopping
            cat_features: Optional list of categorical feature indices
            
        Raises:
            ValueError: If input data is invalid or model not initialized
            RuntimeError: If training fails
            
        Example:
            >>> model.train(X_train, y_train, X_val, y_val, cat_features=[0, 2])
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
            
            if cat_features is not None:
                if not isinstance(cat_features, list):
                    raise ValueError("cat_features must be a list of integers")
                
                max_feature_idx = X_train.shape[1] - 1
                invalid_features = [f for f in cat_features if f < 0 or f > max_feature_idx]
                if invalid_features:
                    raise ValueError(f"Invalid categorical feature indices: {invalid_features}")
            
            if self.model is None:
                raise ValueError("Model not initialized. Call build_model() first.")
            
            self.cat_features = cat_features
            
            logger.info(f"Training CatBoost on {X_train.shape[0]} samples, "
                       f"{X_train.shape[1]} features")
            
            if cat_features:
                logger.info(f"Using {len(cat_features)} categorical features: {cat_features}")
            
            # Create Pool objects for efficient training
            train_pool = Pool(X_train, y_train, cat_features=cat_features)
            
            if X_val is not None and y_val is not None:
                val_pool = Pool(X_val, y_val, cat_features=cat_features)
                
                self.model.fit(
                    train_pool,
                    eval_set=val_pool,
                    early_stopping_rounds=50,
                    verbose=False,
                    use_best_model=True
                )
                
                self.best_iteration = getattr(self.model, 'best_iteration_', None)
                if self.best_iteration is not None:
                    logger.info(f"✓ Training complete (best iteration: {self.best_iteration})")
                else:
                    logger.info("✓ Training complete (no early stopping)")
            else:
                self.model.fit(train_pool, verbose=False)
                logger.info("✓ Training complete")
            
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"CatBoost training failed: {e}")
            raise RuntimeError(f"Model training failed: {e}") from e
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained CatBoost model.
        
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
            logger.error(f"CatBoost prediction failed: {e}")
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

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates using CatBoost virtual ensembles.
        
        Uses CatBoost's built-in virtual ensemble functionality to provide
        uncertainty estimates alongside predictions. Falls back to heuristic
        uncertainty if virtual ensembles are not available.
        
        Args:
            X: Input feature matrix of shape (n_samples, n_features)
        
        Returns:
            Tuple of (predictions, uncertainties) where:
            - predictions: Model predictions of shape (n_samples,)
            - uncertainties: Uncertainty estimates of shape (n_samples,)
            
        Raises:
            ValueError: If model not trained or input invalid
            RuntimeError: If uncertainty prediction fails
            
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
            
            # Standard predictions
            predictions = self.model.predict(X)
            
            # Virtual ensemble predictions for uncertainty
            try:
                pool = Pool(X, cat_features=self.cat_features)
                
                # Try to use virtual ensembles for uncertainty estimation
                uncertainty = self.model.virtual_ensembles_predict(
                    pool,
                    prediction_type='TotalUncertainty',
                    virtual_ensembles_count=10
                )
                
                logger.debug(f"Generated uncertainty estimates using virtual ensembles "
                           f"for {X.shape[0]} samples")
                
            except Exception as ve_error:
                logger.warning(f"Virtual ensembles failed ({ve_error}), using fallback uncertainty")
                
                # Fallback: use 5% of prediction magnitude as uncertainty
                uncertainty = np.abs(predictions) * 0.05
                
                # Add some noise based on model complexity
                if hasattr(self.model, 'tree_count_'):
                    complexity_factor = min(0.1, self.model.tree_count_ / 10000)
                    uncertainty += np.abs(predictions) * complexity_factor
            
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            
            if not isinstance(uncertainty, np.ndarray):
                uncertainty = np.array(uncertainty)
            
            # Ensure uncertainty is non-negative
            uncertainty = np.abs(uncertainty)
            
            return predictions, uncertainty
            
        except Exception as e:
            logger.error(f"CatBoost uncertainty prediction failed: {e}")
            raise RuntimeError(f"Uncertainty prediction failed: {e}") from e
    
    def get_feature_importance(self, importance_type: str = 'FeatureImportance') -> pd.DataFrame:
        """
        Get feature importance using CatBoost's built-in importance calculation.
        
        Returns feature importance scores calculated from the trained model
        using the specified importance type, sorted in descending order.
        
        Args:
            importance_type: Type of importance calculation:
                - 'FeatureImportance': Standard feature importance (default)
                - 'PredictionValuesChange': Importance based on prediction changes
                - 'LossFunctionChange': Importance based on loss function changes
        
        Returns:
            DataFrame with columns ['feature', 'importance'] sorted by importance
            
        Raises:
            ValueError: If model not trained or invalid importance type
            RuntimeError: If importance calculation fails
            
        Example:
            >>> importance_df = model.get_feature_importance('PredictionValuesChange')
            >>> print(f"Top 5 features: {importance_df.head()}")
        """
        try:
            if not self.is_trained:
                raise ValueError("Model not trained. Call train() first.")
            
            if self.model is None:
                raise ValueError("Model is None. Rebuild model.")
            
            # Validate importance type
            valid_types = ['FeatureImportance', 'PredictionValuesChange', 'LossFunctionChange']
            if importance_type not in valid_types:
                raise ValueError(f"importance_type must be one of {valid_types}, got {importance_type}")
            
            # Get feature importance
            importance = self.model.get_feature_importance(type=importance_type)
            
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
            
            logger.debug(f"Generated {importance_type} feature importance for {len(importance)} features")
            return importance_df
            
        except Exception as e:
            logger.error(f"CatBoost feature importance calculation failed: {e}")
            raise RuntimeError(f"Feature importance calculation failed: {e}") from e
    
    def get_best_iteration(self) -> Optional[int]:
        """
        Get the best iteration from early stopping.
        
        Returns:
            Best iteration number or None if no early stopping was used
            
        Example:
            >>> best_iter = model.get_best_iteration()
            >>> if best_iter:
            ...     print(f"Best iteration: {best_iter}")
        """
        return self.best_iteration
    
    def get_evaluation_results(self) -> Optional[Dict[str, List[float]]]:
        """
        Get evaluation results from training if validation was used.
        
        Returns:
            Dictionary with evaluation metrics or None if no validation
            
        Example:
            >>> eval_results = model.get_evaluation_results()
            >>> if eval_results:
            ...     print(f"Validation RMSE: {eval_results['validation']['RMSE'][-1]}")
        """
        try:
            if not self.is_trained or self.model is None:
                return None
            
            if hasattr(self.model, 'get_evals_result'):
                return self.model.get_evals_result()
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Could not retrieve evaluation results: {e}")
            return None