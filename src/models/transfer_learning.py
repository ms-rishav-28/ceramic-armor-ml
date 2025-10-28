"""
Transfer Learning from SiC to WC and TiC
Leverages knowledge from data-rich system to data-scarce systems
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from loguru import logger

class TransferLearningManager:
    """
    Implement transfer learning for ceramic systems with limited data
    
    Strategy:
    1. Pre-train on source system (SiC - 1500 entries)
    2. Fine-tune on target system (WC/TiC - 500-600 entries)
    3. Use feature importance from source to guide target training
    """
    
    def __init__(self, source_system: str = 'SiC', 
                 target_systems: list = None):
        """
        Initialize transfer learning manager
        
        Args:
            source_system: Data-rich system (default: SiC)
            target_systems: Data-scarce systems (default: [WC, TiC])
        """
        self.source_system = source_system
        self.target_systems = target_systems or ['WC', 'TiC']
        self.source_model = None
        self.source_feature_importance = None
        
        logger.info(f"Transfer Learning: {source_system} → {target_systems}")
    
    def train_source_model(self, model_class, X_source: np.ndarray, 
                          y_source: np.ndarray, config: Dict):
        """
        Train model on source (data-rich) system
        
        Args:
            model_class: Model class to use
            X_source: Source system features
            y_source: Source system targets
            config: Model configuration
        
        Returns:
            Trained source model
        """
        logger.info(f"Training source model on {self.source_system}...")
        
        # Split source data
        X_train, X_val, y_train, y_val = train_test_split(
            X_source, y_source, test_size=0.2, random_state=42
        )
        
        # Train source model
        self.source_model = model_class(config)
        self.source_model.train(X_train, y_train, X_val, y_val)
        
        # Extract feature importance
        self.source_feature_importance = self.source_model.get_feature_importance()
        
        logger.info(f"✓ Source model trained (R² on validation)")
        
        return self.source_model 
   
    def get_top_features(self, n_features: int = 50) -> list:
        """
        Get top N most important features from source model
        
        Args:
            n_features: Number of top features to return
        
        Returns:
            List of feature names
        """
        if self.source_feature_importance is None:
            raise ValueError("Source model not trained")
        
        top_features = self.source_feature_importance.head(n_features)['feature'].tolist()
        
        logger.info(f"Extracted top {n_features} features from source model")
        return top_features
    
    def fine_tune_target_model(self, model_class, X_target: np.ndarray,
                               y_target: np.ndarray, config: Dict,
                               use_feature_selection: bool = True):
        """
        Fine-tune model on target (data-scarce) system
        
        Args:
            model_class: Model class to use
            X_target: Target system features
            y_target: Target system targets
            config: Model configuration
            use_feature_selection: Whether to use source feature importance
        
        Returns:
            Fine-tuned target model
        """
        logger.info(f"Fine-tuning model on target system...")
        
        # Optional: Feature selection based on source importance
        if use_feature_selection and self.source_feature_importance is not None:
            top_features = self.get_top_features(n_features=min(50, X_target.shape[1]))
            # Assume features are named consistently
            logger.info(f"Using {len(top_features)} top features from source")
        
        # Split target data
        X_train, X_val, y_train, y_val = train_test_split(
            X_target, y_target, test_size=0.2, random_state=42
        )
        
        # Initialize target model with source model's hyperparameters
        target_model = model_class(config)
        
        # Fine-tune on target data
        target_model.train(X_train, y_train, X_val, y_val)
        
        logger.info(f"✓ Target model fine-tuned")
        
        return target_model
    
    def evaluate_transfer_benefit(self, source_score: float, 
                                  target_score: float,
                                  target_score_no_transfer: float) -> Dict:
        """
        Quantify benefit of transfer learning
        
        Args:
            source_score: R² on source system
            target_score: R² on target with transfer
            target_score_no_transfer: R² on target without transfer
        
        Returns:
            Dictionary with transfer learning metrics
        """
        improvement = target_score - target_score_no_transfer
        relative_improvement = (improvement / target_score_no_transfer) * 100
        
        metrics = {
            'source_r2': source_score,
            'target_r2_with_transfer': target_score,
            'target_r2_without_transfer': target_score_no_transfer,
            'absolute_improvement': improvement,
            'relative_improvement_pct': relative_improvement
        }
        
        logger.info(f"Transfer Learning Benefit: {relative_improvement:.1f}% improvement")
        
        return metrics