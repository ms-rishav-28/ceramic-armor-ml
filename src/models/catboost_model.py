"""
CatBoost Model Implementation
Includes built-in uncertainty quantification
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from typing import Dict, Tuple, Optional
from loguru import logger
from .base_model import BaseModel

class CatBoostModel(BaseModel):
    """CatBoost Regressor with uncertainty estimates"""
    
    def __init__(self, config: Dict, n_jobs: int = 20):
        """
        Initialize CatBoost model
        
        Args:
            config: CatBoost hyperparameters
            n_jobs: Number of CPU threads
        """
        super().__init__('catboost', config)
        self.n_jobs = n_jobs
        self.build_model()
    
    def build_model(self):
        """Build CatBoost model"""
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
            bootstrap_type='Bayesian',
            random_seed=42,
            verbose=False,
            allow_writing_files=False
        )
        logger.info(f"✓ CatBoost model built with {self.n_jobs} threads")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              cat_features: Optional[list] = None):
        """
        Train CatBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            cat_features: Indices of categorical features
        """
        logger.info(f"Training CatBoost on {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        # Create Pool objects for efficient training
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        
        if X_val is not None and y_val is not None:
            val_pool = Pool(X_val, y_val, cat_features=cat_features)
            self.model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=50,
                verbose=False
            )
            logger.info(f"✓ Training complete (best iteration: {self.model.best_iteration_})")
        else:
            self.model.fit(train_pool, verbose=False)
            logger.info("✓ Training complete")
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates using virtual ensembles
        
        Args:
            X: Input features
        
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Standard predictions
        predictions = self.model.predict(X)
        
        # Virtual ensemble predictions for uncertainty
        try:
            pool = Pool(X)
            uncertainty = self.model.virtual_ensembles_predict(
                pool,
                prediction_type='TotalUncertainty',
                virtual_ensembles_count=10
            )
        except:
            # Fallback: use standard deviation across trees
            uncertainty = np.zeros_like(predictions)
            logger.warning("Virtual ensembles not available, returning zero uncertainty")
        
        return predictions, uncertainty
    
    def get_feature_importance(self, importance_type: str = 'FeatureImportance') -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            importance_type: 'FeatureImportance', 'PredictionValuesChange', or 'LossFunctionChange'
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        importance = self.model.get_feature_importance(type=importance_type)
        
        feature_names = [f"f{i}" for i in range(len(importance))]
        if self.feature_names is not None:
            feature_names = self.feature_names
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df