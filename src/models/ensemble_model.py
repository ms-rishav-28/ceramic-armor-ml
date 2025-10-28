"""
Stacking Ensemble Model
Combines XGBoost, CatBoost, Random Forest, Gradient Boosting
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional
from loguru import logger

from .xgboost_model import XGBoostModel
from .catboost_model import CatBoostModel
from .random_forest_model import RandomForestModel
from .gradient_boosting_model import GradientBoostingModel
from .base_model import BaseModel

class EnsembleModel(BaseModel):
    """
    Stacking ensemble combining multiple tree-based models
    
    Architecture:
    Level 0: XGBoost, CatBoost, Random Forest, Gradient Boosting
    Level 1: Ridge regression meta-learner
    """
    
    def __init__(self, config: Dict, model_configs: Dict, 
                 method: str = 'stacking', n_jobs: int = 20):
        """
        Initialize ensemble model
        
        Args:
            config: Ensemble configuration
            model_configs: Configuration for each base model
            method: 'stacking' or 'voting'
            n_jobs: Number of CPU threads
        """
        super().__init__('ensemble', config)
        self.model_configs = model_configs
        self.method = method
        self.n_jobs = n_jobs
        self.base_models = {}
        self.build_model()
    
    def build_model(self):
        """Build ensemble model"""
        logger.info(f"Building {self.method} ensemble...")
        
        # Create base models
        base_estimators = []
        
        # XGBoost
        xgb_model = XGBoostModel(self.model_configs['xgboost'], self.n_jobs)
        base_estimators.append(('xgboost', xgb_model.model))
        self.base_models['xgboost'] = xgb_model
        
        # CatBoost
        cat_model = CatBoostModel(self.model_configs['catboost'], self.n_jobs)
        base_estimators.append(('catboost', cat_model.model))
        self.base_models['catboost'] = cat_model
        
        # Random Forest
        rf_model = RandomForestModel(self.model_configs['random_forest'], self.n_jobs)
        base_estimators.append(('random_forest', rf_model.model))
        self.base_models['random_forest'] = rf_model
        
        # Gradient Boosting
        gb_model = GradientBoostingModel(self.model_configs['gradient_boosting'])
        base_estimators.append(('gradient_boosting', gb_model.model))
        self.base_models['gradient_boosting'] = gb_model
        
        # Create ensemble
        if self.method == 'stacking':
            # Stacking with Ridge meta-learner
            meta_learner = Ridge(alpha=1.0)
            self.model = StackingRegressor(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=5,
                n_jobs=self.n_jobs,
                passthrough=False  # Don't pass original features to meta-learner
            )
            logger.info("✓ Stacking ensemble created with Ridge meta-learner")
            
        elif self.method == 'voting':
            # Weighted voting
            weights = [
                self.config.get('weights', {}).get('xgboost', 0.40),
                self.config.get('weights', {}).get('catboost', 0.35),
                self.config.get('weights', {}).get('random_forest', 0.15),
                self.config.get('weights', {}).get('gradient_boosting', 0.10)
            ]
            self.model = VotingRegressor(
                estimators=base_estimators,
                weights=weights,
                n_jobs=self.n_jobs
            )
            logger.info(f"✓ Voting ensemble created with weights: {weights}")
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None):
        """Train ensemble model"""
        logger.info(f"Training {self.method} ensemble on {X_train.shape[0]} samples")
        
        self.model.fit(X_train, y_train)
        logger.info("✓ Ensemble training complete")
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def predict_with_individual_models(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model
        
        Args:
            X: Input features
        
        Returns:
            Dictionary mapping model name to predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        individual_preds = {}
        for name, model_obj in self.base_models.items():
            individual_preds[name] = model_obj.predict(X)
        
        return individual_preds
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get aggregated feature importance from base models
        
        Returns:
            DataFrame with average importance across models
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        importance_dfs = []
        
        # Get importance from each base model
        for name, model_obj in self.base_models.items():
            try:
                imp_df = model_obj.get_feature_importance()
                imp_df.columns = ['feature', f'importance_{name}']
                importance_dfs.append(imp_df)
            except:
                logger.warning(f"Could not get importance from {name}")
        
        # Merge and average
        if importance_dfs:
            merged = importance_dfs[0]
            for df in importance_dfs[1:]:
                merged = merged.merge(df, on='feature', how='outer')
            
            # Calculate average importance
            importance_cols = [col for col in merged.columns if col.startswith('importance_')]
            merged['importance'] = merged[importance_cols].mean(axis=1)
            
            result = merged[['feature', 'importance']].sort_values('importance', ascending=False)
            return result
        
        return pd.DataFrame()
    
    def get_base_model_scores(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate individual base model performance
        
        Args:
            X: Test features
            y: Test targets
        
        Returns:
            Dictionary mapping model name to R² score
        """
        from sklearn.metrics import r2_score
        
        scores = {}
        for name, model_obj in self.base_models.items():
            y_pred = model_obj.predict(X)
            scores[name] = r2_score(y, y_pred)
        
        return scores