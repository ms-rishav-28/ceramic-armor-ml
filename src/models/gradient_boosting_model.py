"""
Gradient Boosting Model Implementation
Includes uncertainty quantification and hyperparameter optimization
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import validation_curve
from typing import Dict, Tuple, Optional
from loguru import logger
from .base_model import BaseModel

class GradientBoostingModel(BaseModel):
    """Gradient Boosting Regressor with uncertainty quantification and CPU optimization"""
    
    def __init__(self, config: Dict, n_jobs: int = 20):
        """
        Initialize Gradient Boosting model
        
        Args:
            config: Gradient Boosting hyperparameters
            n_jobs: Number of CPU threads (not directly used by GradientBoosting but kept for consistency)
        """
        super().__init__('gradient_boosting', config)
        self.n_jobs = n_jobs
        self.staged_predictions = None  # For uncertainty estimation
        self.build_model()
    
    def build_model(self):
        """Build Gradient Boosting model with CPU-friendly settings"""
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
            tol=self.config.get('tol', 1e-4),
            random_state=42,
            verbose=0,
            warm_start=False
        )
        logger.info(f"✓ Gradient Boosting model built with {self.config.get('n_estimators', 500)} estimators")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              monitor_staged: bool = True):
        """
        Train Gradient Boosting model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (for monitoring)
            y_val: Validation targets
            monitor_staged: Whether to store staged predictions for uncertainty
        """
        logger.info(f"Training Gradient Boosting on {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        # Fit the model
        self.model.fit(X_train, y_train)
        
        # Store staged predictions for uncertainty estimation if validation set provided
        if monitor_staged and X_val is not None:
            self.staged_predictions = {
                'train': list(self.model.staged_predict(X_train)),
                'val': list(self.model.staged_predict(X_val))
            }
            
            # Calculate validation scores
            val_scores = []
            for pred in self.staged_predictions['val']:
                score = 1 - np.mean((y_val - pred) ** 2) / np.var(y_val)  # R² score
                val_scores.append(score)
            
            best_iteration = np.argmax(val_scores)
            logger.info(f"✓ Training complete (best iteration: {best_iteration + 1}, val R²: {val_scores[best_iteration]:.4f})")
        else:
            logger.info("✓ Training complete")
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X: np.ndarray, n_estimators_range: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates using staged predictions
        
        Args:
            X: Input features
            n_estimators_range: Range of estimators to use for uncertainty (start, end)
        
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Get staged predictions for uncertainty estimation
        staged_preds = list(self.model.staged_predict(X))
        
        if n_estimators_range is None:
            # Use last 20% of estimators for uncertainty estimation
            start_idx = max(0, int(0.8 * len(staged_preds)))
            end_idx = len(staged_preds)
        else:
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
            uncertainties = np.zeros_like(predictions)
        
        return predictions, uncertainties
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance based on impurity decrease"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        importance = self.model.feature_importances_
        
        feature_names = [f"f{i}" for i in range(len(importance))]
        if self.feature_names is not None:
            feature_names = self.feature_names
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_learning_curve(self) -> Optional[Dict]:
        """Get training learning curve from staged predictions"""
        if not self.is_trained or not self.staged_predictions:
            return None
        
        return {
            'train_scores': [pred.mean() for pred in self.staged_predictions.get('train', [])],
            'val_scores': [pred.mean() for pred in self.staged_predictions.get('val', [])],
            'n_estimators': list(range(1, len(self.staged_predictions.get('train', [])) + 1))
        }
    
    def optimize_n_estimators(self, X: np.ndarray, y: np.ndarray, 
                            n_estimators_range: Tuple[int, int] = (100, 1000),
                            cv: int = 5) -> int:
        """
        Optimize number of estimators using validation curve
        
        Args:
            X: Training features
            y: Training targets
            n_estimators_range: Range of n_estimators to test
            cv: Number of cross-validation folds
        
        Returns:
            Optimal number of estimators
        """
        logger.info("Optimizing n_estimators using validation curve")
        
        # Create range of n_estimators to test
        n_estimators_values = np.linspace(
            n_estimators_range[0], 
            n_estimators_range[1], 
            10, 
            dtype=int
        )
        
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
        optimal_idx = np.argmax(mean_val_scores)
        optimal_n_estimators = n_estimators_values[optimal_idx]
        
        logger.info(f"Optimal n_estimators: {optimal_n_estimators} (CV R²: {mean_val_scores[optimal_idx]:.4f})")
        
        return optimal_n_estimators
