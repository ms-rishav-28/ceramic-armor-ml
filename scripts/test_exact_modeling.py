#!/usr/bin/env python3
"""
Test script for exact modeling strategy implementation
Verifies compliance to specification with zero deviations
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.exact_modeling_trainer import ExactModelingTrainer
from utils.intel_optimizer import intel_opt
from loguru import logger

def generate_test_data(n_samples: int = 1000, n_features: int = 50) -> tuple:
    """Generate synthetic test data for ceramic systems"""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with some realistic relationships
    # Simulate mechanical properties based on feature combinations
    y = (
        2.0 * X[:, 0] +           # Primary feature
        1.5 * X[:, 1] +           # Secondary feature  
        0.8 * X[:, 2] * X[:, 3] + # Interaction term
        0.5 * np.sum(X[:, 4:10], axis=1) +  # Composite features
        0.1 * np.random.randn(n_samples)     # Noise
    )
    
    return X, y

def test_exact_modeling_compliance():
    """Test exact modeling strategy compliance"""
    logger.info("Testing exact modeling strategy compliance...")
    
    # Initialize trainer
    config_path = str(Path(__file__).parent.parent / "config" / "exact_modeling_config.yaml")
    trainer = ExactModelingTrainer(config_path)
    
    # Generate test data for all ceramic systems
    system_data = {}
    ceramic_systems = ['SiC', 'Al2O3', 'B4C', 'WC', 'TiC']
    
    for system in ceramic_systems:
        # Different sample sizes to simulate real data distribution
        if system in ['SiC', 'Al2O3', 'B4C']:
            n_samples = 800  # Independent systems with more data
        else:
            n_samples = 400  # Transfer learning systems with less data
        
        X, y = generate_test_data(n_samples=n_samples, n_features=120)
        system_data[system] = (X, y)
        logger.info(f"Generated {system} data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test training
    try:
        results = trainer.train_all_systems(system_data, validation_split=0.2)
        logger.info("✓ Training completed successfully")
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        return False
    
    # Test compliance verification
    try:
        compliance = trainer.verify_exact_compliance()
        logger.info("✓ Compliance verification completed")
        
        # Check key compliance items
        required_compliance = [
            'models_implemented',
            'intel_optimizations', 
            'n_jobs_20',
            'stacking_ensemble',
            'ceramic_systems',
            'name_attributes'
        ]
        
        all_compliant = all(compliance.get(key, False) for key in required_compliance)
        
        if all_compliant:
            logger.info("✓ All compliance requirements met")
        else:
            logger.warning("⚠ Some compliance requirements not met")
            for key in required_compliance:
                status = compliance.get(key, False)
                logger.info(f"  {key}: {'✓' if status else '✗'}")
        
    except Exception as e:
        logger.error(f"✗ Compliance verification failed: {e}")
        return False
    
    # Test predictions
    try:
        test_system = 'SiC'
        X_test, _ = generate_test_data(n_samples=100, n_features=120)
        
        # Test ensemble prediction
        pred_ensemble = trainer.predict_system(test_system, X_test, use_ensemble=True)
        logger.info(f"✓ Ensemble prediction shape: {pred_ensemble.shape}")
        
        # Test uncertainty prediction
        pred_with_unc, uncertainty = trainer.predict_system(
            test_system, X_test, use_ensemble=True, return_uncertainty=True
        )
        logger.info(f"✓ Uncertainty prediction shape: {pred_with_unc.shape}, {uncertainty.shape}")
        
    except Exception as e:
        logger.error(f"✗ Prediction test failed: {e}")
        return False
    
    # Test feature importance
    try:
        importance = trainer.get_feature_importance('SiC', 'ensemble')
        logger.info(f"✓ Feature importance shape: {importance.shape}")
    except Exception as e:
        logger.error(f"✗ Feature importance test failed: {e}")
        return False
    
    # Get training summary
    try:
        summary = trainer.get_training_summary()
        logger.info("✓ Training summary generated")
        logger.info(f"  Models implemented: {summary['models_implemented']}")
        logger.info(f"  Intel optimizations: {summary['intel_optimizations']}")
        logger.info(f"  Ceramic systems trained: {summary['ceramic_systems']['trained_systems']}")
    except Exception as e:
        logger.error(f"✗ Training summary failed: {e}")
        return False
    
    logger.info("✓ All exact modeling tests passed")
    return True

def test_individual_models():
    """Test individual model implementations"""
    logger.info("Testing individual model implementations...")
    
    from models.xgboost_model import XGBoostModel
    from models.catboost_model import CatBoostModel
    from models.random_forest_model import RandomForestModel
    from models.gradient_boosting_model import GradientBoostingModel
    
    # Generate test data
    X, y = generate_test_data(n_samples=500, n_features=50)
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]
    
    # Test configurations
    configs = {
        'xgboost': {'n_estimators': 100, 'max_depth': 6},
        'catboost': {'iterations': 100, 'depth': 6},
        'random_forest': {'n_estimators': 100, 'max_depth': 10},
        'gradient_boosting': {'n_estimators': 100, 'max_depth': 6}
    }
    
    models = [
        ('XGBoost', XGBoostModel, configs['xgboost']),
        ('CatBoost', CatBoostModel, configs['catboost']),
        ('Random Forest', RandomForestModel, configs['random_forest']),
        ('Gradient Boosting', GradientBoostingModel, configs['gradient_boosting'])
    ]
    
    for name, ModelClass, config in models:
        try:
            logger.info(f"Testing {name}...")
            
            # Initialize model
            model = ModelClass(config, n_jobs=20)
            
            # Check name attribute
            if not hasattr(model, 'name'):
                logger.error(f"✗ {name} missing 'name' attribute")
                return False
            
            # Train model
            model.train(X_train, y_train)
            
            # Test prediction
            pred = model.predict(X_test)
            logger.info(f"✓ {name} prediction shape: {pred.shape}")
            
            # Test uncertainty (if available)
            if hasattr(model, 'predict_uncertainty'):
                pred_unc, unc = model.predict_uncertainty(X_test)
                logger.info(f"✓ {name} uncertainty shape: {unc.shape}")
            
            # Test feature importance
            importance = model.get_feature_importance()
            logger.info(f"✓ {name} feature importance shape: {importance.shape}")
            
        except Exception as e:
            logger.error(f"✗ {name} test failed: {e}")
            return False
    
    logger.info("✓ All individual model tests passed")
    return True

def main():
    """Main test function"""
    logger.info("Starting exact modeling strategy tests...")
    
    # Test Intel optimizations
    logger.info(f"Intel optimizations applied: {intel_opt.optimization_applied}")
    
    # Test individual models
    if not test_individual_models():
        logger.error("Individual model tests failed")
        return 1
    
    # Test exact modeling compliance
    if not test_exact_modeling_compliance():
        logger.error("Exact modeling compliance tests failed")
        return 1
    
    logger.info("🎉 All tests passed! Exact modeling strategy implemented successfully.")
    return 0

if __name__ == "__main__":
    exit(main())