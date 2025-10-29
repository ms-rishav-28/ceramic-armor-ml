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
        self.name = model_name  # Add name attribute for test compatibility
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
    
    def predict_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates (default implementation)
        
        Args:
            X: Input features
        
        Returns:
            Tuple of (predictions, uncertainties)
        """
        predictions = self.predict(X)
        # Default uncertainty as 5% of prediction magnitude
        uncertainties = np.abs(predictions) * 0.05
        return predictions, uncertainties    
 
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