"""
Complete Evaluation Metrics Suite
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from typing import Dict
from loguru import logger

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                property_name: str = "") -> Dict:
        """
        Calculate all evaluation metrics
        
        Returns:
            Dictionary with R², RMSE, MAE, MAPE, Pearson r
        """
        # R² score
        r2 = r2_score(y_true, y_pred)
        
        # RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAE
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE (handling division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            mape = np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        
        metrics = {
            'property': property_name,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'n_samples': len(y_true)
        }
        
        return metrics

class PerformanceChecker:
    """Check if models meet R² targets"""
    
    def __init__(self, config: Dict):
        self.mechanical_target = config['targets']['mechanical_r2']
        self.ballistic_target = config['targets']['ballistic_r2']
    
    def check_target(self, r2: float, property_type: str) -> bool:
        """Check if R² meets target"""
        target = self.mechanical_target if property_type == 'mechanical' else self.ballistic_target
        
        meets_target = r2 >= target
        status = "✓ PASS" if meets_target else "✗ FAIL"
        
        logger.info(f"{status}: R² = {r2:.4f} (target: {target:.2f})")
        
        return meets_target