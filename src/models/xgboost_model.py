"""
XGBoost Model Implementation
Intel MKL accelerated with strict compliance to specification
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Tuple, Optional
from loguru import logger
from .base_model import BaseModel
from src.utils.intel_optimizer import intel_opt

class XGBoostModel(BaseModel):
    """XGBoost Regressor for ceramic property prediction"""
    
    def __init__(self, config: Dict, n_jobs: int = 20):
        """
        Initialize XGBoost model
        
        Args:
            config: XGBoost hyperparameters
            n_jobs: Number of CPU threads
        """
        super().__init__('xgboost', config)
        self.n_jobs = n_jobs
        self.build_model()
    
    def build_model(self):
        """Build XGBoost model with Intel MKL acceleration and strict compliance"""
        # Ensure Intel optimizations are applied
        if not intel_opt.optimization_applied:
            intel_opt.apply_optimizations()
        
        self.model = xgb.XGBRegressor(
            objective=self.config.get('objective', 'reg:squarederror'),
            n_estimators=self.config.get('n_estimators', 1000),
            max_depth=self.config.get('max_depth', 8),
            learning_rate=self.config.get('learning_rate', 0.05),
            subsample=self.config.get('subsample', 0.8),
            colsample_bytree=self.config.get('colsample_bytree', 0.8),
            colsample_bylevel=self.config.get('colsample_bylevel', 0.8),
            min_child_weight=self.config.get('min_child_weight', 3),
            gamma=self.config.get('gamma', 0.1),
            reg_alpha=self.config.get('reg_alpha', 0.01),
            reg_lambda=self.config.get('reg_lambda', 1.0),
            n_jobs=self.n_jobs,
            tree_method='hist',  # Fast histogram-based algorithm
            predictor='cpu_predictor',
            random_state=42,
            verbosity=0
        )
        logger.info(f"✓ XGBoost model built with Intel MKL acceleration and {self.n_jobs} threads")  
  
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              early_stopping_rounds: int = 50):
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (for early stopping)
            y_val: Validation targets
            early_stopping_rounds: Rounds for early stopping
        """
        logger.info(f"Training XGBoost on {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            try:
                # Try new XGBoost API
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
                if hasattr(self.model, 'best_iteration'):
                    logger.info(f"✓ Training complete (best iteration: {self.model.best_iteration})")
                else:
                    logger.info("✓ Training complete")
            except TypeError:
                # Fallback for older XGBoost versions
                self.model.fit(X_train, y_train)
                logger.info("✓ Training complete (no early stopping)")
        else:
            self.model.fit(X_train, y_train)
            logger.info("✓ Training complete")
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
        
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def predict_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates using tree ensemble variance
        
        Args:
            X: Input features
        
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Standard predictions
        predictions = self.model.predict(X)
        
        # For uncertainty estimation, use the variance across trees
        try:
            # Get predictions from subsets of trees to estimate variance
            dmatrix = xgb.DMatrix(X)
            
            # Get total number of trees
            n_trees = self.model.n_estimators
            if hasattr(self.model, 'best_iteration') and self.model.best_iteration:
                n_trees = self.model.best_iteration
            
            # Collect predictions from different tree subsets
            tree_predictions = []
            step_size = max(1, n_trees // 10)  # Use 10 subsets
            
            for i in range(step_size, n_trees + 1, step_size):
                pred = self.model.predict(X, ntree_limit=i)
                tree_predictions.append(pred)
            
            if len(tree_predictions) > 1:
                tree_predictions = np.array(tree_predictions)
                uncertainties = np.std(tree_predictions, axis=0)
            else:
                # Fallback: use prediction magnitude
                uncertainties = np.abs(predictions) * 0.05
                
        except Exception as e:
            logger.warning(f"Tree variance uncertainty estimation failed: {e}")
            # Fallback uncertainty estimation
            uncertainties = np.abs(predictions) * 0.05  # 5% uncertainty estimate
        
        return predictions, uncertainties
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates (alias for predict_uncertainty)
        
        Args:
            X: Input features
        
        Returns:
            Tuple of (predictions, uncertainties)
        """
        return self.predict_uncertainty(X)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            importance_type: 'gain', 'weight', or 'cover'
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Convert to DataFrame
        importance_df = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_learning_curve(self) -> Dict:
        """Get training and validation learning curves"""
        if not hasattr(self.model, 'evals_result_'):
            return None
        
        return self.model.evals_result()