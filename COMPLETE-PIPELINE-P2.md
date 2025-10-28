# COMPLETE ML PIPELINE - PART 2
## Core Model Implementations & Training System

---

## FILE 6: src/feature_engineering/phase_stability.py

```python
"""
Phase Stability Classification using DFT Hull Distance
CRITICAL: Distinguishes single-phase from multi-phase systems
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from mp_api.client import MPRester
from loguru import logger

class PhaseStabilityAnalyzer:
    """
    Analyze phase stability using DFT-calculated energy above convex hull.
    
    Classification Scheme:
    - ΔE_hull < 0.05 eV/atom → Stable single-phase
    - 0.05 ≤ ΔE_hull < 0.10 eV/atom → Metastable
    - ΔE_hull ≥ 0.10 eV/atom → Unstable (multi-phase expected)
    """
    
    def __init__(self, api_key: str = None, 
                 stable_threshold: float = 0.05,
                 metastable_threshold: float = 0.10):
        """
        Initialize phase stability analyzer
        
        Args:
            api_key: Materials Project API key
            stable_threshold: Threshold for stable classification (eV/atom)
            metastable_threshold: Threshold for metastable classification (eV/atom)
        """
        self.stable_threshold = stable_threshold
        self.metastable_threshold = metastable_threshold
        
        if api_key:
            self.mpr = MPRester(api_key)
            logger.info("✓ Materials Project API initialized for phase stability")
        else:
            self.mpr = None
            logger.warning("No API key provided - using existing hull distance values")
    
    def classify_stability(self, energy_above_hull: float) -> str:
        """
        Classify phase stability based on hull distance
        
        Args:
            energy_above_hull: Energy above hull (eV/atom)
        
        Returns:
            Classification: 'stable', 'metastable', or 'unstable'
        """
        if pd.isna(energy_above_hull):
            return 'unknown'
        
        if energy_above_hull < self.stable_threshold:
            return 'stable'
        elif energy_above_hull < self.metastable_threshold:
            return 'metastable'
        else:
            return 'unstable'
    
    def get_hull_distance(self, material_id: str) -> float:
        """
        Query Materials Project for hull distance
        
        Args:
            material_id: Materials Project ID (e.g., 'mp-149')
        
        Returns:
            Energy above hull (eV/atom)
        """
        if not self.mpr:
            raise ValueError("Materials Project API not initialized")
        
        try:
            docs = self.mpr.materials.summary.search(
                material_ids=[material_id],
                fields=["energy_above_hull"]
            )
            if docs:
                return docs[0].energy_above_hull
            return np.nan
        except Exception as e:
            logger.error(f"Error fetching hull distance for {material_id}: {e}")
            return np.nan
    
    def analyze_dataframe(self, df: pd.DataFrame, 
                         material_id_col: str = 'material_id',
                         hull_distance_col: str = 'energy_above_hull') -> pd.DataFrame:
        """
        Analyze phase stability for entire dataset
        
        Args:
            df: DataFrame with materials data
            material_id_col: Column name for material IDs
            hull_distance_col: Column name for hull distance (if present)
        
        Returns:
            DataFrame with phase stability classifications
        """
        logger.info(f"Analyzing phase stability for {len(df)} materials...")
        
        df_stability = df.copy()
        
        # Get hull distances if not present
        if hull_distance_col not in df_stability.columns and self.mpr:
            logger.info("Fetching hull distances from Materials Project...")
            hull_distances = []
            for mat_id in df_stability[material_id_col]:
                hull_dist = self.get_hull_distance(mat_id)
                hull_distances.append(hull_dist)
            df_stability[hull_distance_col] = hull_distances
        
        # Classify stability
        df_stability['phase_stability'] = df_stability[hull_distance_col].apply(
            self.classify_stability
        )
        
        # Add binary flags for modeling
        df_stability['is_stable'] = (df_stability['phase_stability'] == 'stable').astype(int)
        df_stability['is_single_phase'] = (
            df_stability['phase_stability'].isin(['stable', 'metastable'])
        ).astype(int)
        
        # Statistics
        stability_counts = df_stability['phase_stability'].value_counts()
        logger.info("Phase stability distribution:")
        for status, count in stability_counts.items():
            pct = (count / len(df_stability)) * 100
            logger.info(f"  {status}: {count} ({pct:.1f}%)")
        
        return df_stability
    
    def get_single_phase_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract only stable/metastable (single-phase) materials
        
        Args:
            df: DataFrame with phase_stability column
        
        Returns:
            Filtered DataFrame with only single-phase materials
        """
        if 'phase_stability' not in df.columns:
            raise ValueError("DataFrame must have 'phase_stability' column")
        
        single_phase = df[df['phase_stability'].isin(['stable', 'metastable'])].copy()
        logger.info(f"Extracted {len(single_phase)} single-phase materials from {len(df)} total")
        
        return single_phase
```

---

## FILE 7: src/models/base_model.py

```python
"""
Abstract Base Class for All Models
Ensures consistent interface across XGBoost, CatBoost, RF, etc.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import joblib
from pathlib import Path
from loguru import logger

class BaseModel(ABC):
    """Abstract base class for all ceramic property prediction models"""
    
    def __init__(self, model_name: str, config: Dict):
        """
        Initialize base model
        
        Args:
            model_name: Name of the model (e.g., 'xgboost', 'catboost')
            config: Model configuration dictionary
        """
        self.model_name = model_name
        self.config = config
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.target_name = None
    
    @abstractmethod
    def build_model(self):
        """Build the model with specified configuration"""
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        pass
    
    def save_model(self, filepath: str):
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model not trained - cannot save")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'config': self.config,
            'feature_names': self.feature_names,
            'target_name': self.target_name
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"✓ Model saved: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from disk
        
        Args:
            filepath: Path to saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']
        self.target_name = model_data['target_name']
        self.is_trained = True
        
        logger.info(f"✓ Model loaded: {filepath}")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.model_name}', trained={self.is_trained})"
```

---

## FILE 8: src/models/xgboost_model.py

```python
"""
XGBoost Model Implementation
Optimized for CPU with histogram-based tree method
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Tuple, Optional
from loguru import logger
from .base_model import BaseModel

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
        """Build XGBoost model with CPU optimizations"""
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
        logger.info(f"✓ XGBoost model built with {self.n_jobs} threads")
    
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
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
            logger.info(f"✓ Training complete (best iteration: {self.model.best_iteration})")
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
```

---

## FILE 9: src/models/catboost_model.py

```python
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
```

---

## FILE 10: src/models/random_forest_model.py

```python
"""
Random Forest Model Implementation
Provides natural uncertainty quantification via tree variance
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Tuple, Optional
from loguru import logger
from .base_model import BaseModel

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
        """Build Random Forest model"""
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
        logger.info(f"✓ Random Forest model built with {self.n_jobs} threads")
    
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
```

---

## FILE 11: src/models/ensemble_model.py

```python
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
```

This completes the core model implementations. Would you like me to continue with:

1. **Training Pipeline** (with transfer learning for WC/TiC)
2. **SHAP Interpretation Module** 
3. **Complete Execution Scripts**
4. **Evaluation & Metrics**
5. **Visualization System**

Let me know which components you'd like me to implement next!
