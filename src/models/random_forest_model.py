"""
Random Forest Model Implementation
Intel Extension accelerated with natural uncertainty quantification via tree variance
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Tuple, Optional
from loguru import logger
from .base_model import BaseModel
from src.utils.intel_optimizer import intel_opt

class RandomForestModel(BaseModel):
    """Random Forest with uncertainty quantification"""
    
    def __init__(self, config: Dict, n_jobs: int = 20):
        """
        Initialize Random Forest model
        
        Args:
            config: RF hyperparameters
            n_jobs: Number of CPU threads
        """
        super().__init__('random_forest', config)
        self.n_jobs = n_jobs
        self.build_model()
    
    def build_model(self):
        """Build Random Forest model with Intel Extension acceleration"""
        # Ensure Intel optimizations are applied
        if not intel_opt.optimization_applied:
            intel_opt.apply_optimizations()
        
        self.model = RandomForestRegressor(
            n_estimators=self.config.get('n_estimators', 500),
            max_depth=self.config.get('max_depth', 15),
            min_samples_split=self.config.get('min_samples_split', 5),
            min_samples_leaf=self.config.get('min_samples_leaf', 2),
            max_features=self.config.get('max_features', 'sqrt'),
            n_jobs=self.n_jobs,
            oob_score=True,
            random_state=42,
            verbose=0
        )
        logger.info(f"✓ Random Forest model built with Intel Extension acceleration and {self.n_jobs} threads")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None):
        """Train Random Forest model"""
        logger.info(f"Training Random Forest on {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        self.model.fit(X_train, y_train)
        
        if hasattr(self.model, 'oob_score_'):
            logger.info(f"✓ Training complete (OOB R²: {self.model.oob_score_:.4f})")
        else:
            logger.info("✓ Training complete")
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates from tree variance
        
        Args:
            X: Input features
        
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Get predictions from all trees
        all_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        
        # Mean prediction
        predictions = all_predictions.mean(axis=0)
        
        # Standard deviation as uncertainty
        uncertainties = all_predictions.std(axis=0)
        
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