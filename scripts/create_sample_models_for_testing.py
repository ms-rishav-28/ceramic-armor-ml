#!/usr/bin/env python3
"""
Create Sample Models for Interpretability Testing
Creates mock model files and training data to test the interpretability analysis
"""

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from loguru import logger


class MockModel:
    """Mock model class with required attributes for SHAP analysis"""
    
    def __init__(self, name: str = "MockModel"):
        self.name = name
        self.feature_importances_ = None
        self.n_features_in_ = None
        
    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        # Create mock feature importances
        self.feature_importances_ = np.random.dirichlet(np.ones(X.shape[1]))
        return self
    
    def predict(self, X):
        # Simple linear combination with noise
        if self.feature_importances_ is not None:
            return np.dot(X, self.feature_importances_) + np.random.normal(0, 0.1, X.shape[0])
        else:
            return np.random.normal(0, 1, X.shape[0])


def create_sample_training_data(n_samples: int = 200, n_features: int = 50) -> tuple:
    """Create sample training data with realistic feature names"""
    
    # Create realistic feature names for ceramic materials
    feature_names = [
        'vickers_hardness', 'specific_hardness', 'fracture_toughness_mode_i',
        'youngs_modulus', 'bulk_modulus', 'shear_modulus', 'pugh_ratio',
        'density', 'thermal_conductivity', 'thermal_expansion',
        'ballistic_efficiency', 'penetration_resistance', 'v50',
        'compressive_strength', 'tensile_strength', 'elastic_anisotropy',
        'debye_temperature', 'heat_capacity', 'thermal_shock_resistance',
        'brittleness_index', 'cauchy_pressure', 'poisson_ratio'
    ]
    
    # Add more generic features to reach n_features
    while len(feature_names) < n_features:
        feature_names.append(f'feature_{len(feature_names)}')
    
    feature_names = feature_names[:n_features]
    
    # Generate synthetic data with some correlations
    np.random.seed(42)
    
    # Create base features
    X = np.random.randn(n_samples, n_features)
    
    # Add some realistic correlations
    # Hardness and elastic modulus correlation
    if 'vickers_hardness' in feature_names and 'youngs_modulus' in feature_names:
        hardness_idx = feature_names.index('vickers_hardness')
        modulus_idx = feature_names.index('youngs_modulus')
        X[:, modulus_idx] = 0.7 * X[:, hardness_idx] + 0.3 * X[:, modulus_idx]
    
    # Density and specific properties correlation
    if 'density' in feature_names and 'specific_hardness' in feature_names:
        density_idx = feature_names.index('density')
        spec_hard_idx = feature_names.index('specific_hardness')
        X[:, spec_hard_idx] = X[:, hardness_idx] - 0.5 * X[:, density_idx] if 'vickers_hardness' in feature_names else X[:, spec_hard_idx]
    
    # Create target variable with realistic relationships
    # Combine multiple features with different weights
    weights = np.random.dirichlet(np.ones(n_features))
    y = np.dot(X, weights) + np.random.normal(0, 0.2, n_samples)
    
    return X, y, feature_names


def create_sample_model_directory(system: str, property_name: str, models_dir: Path):
    """Create a sample model directory with all required files"""
    
    model_dir = models_dir / system.lower() / property_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating sample model directory: {model_dir}")
    
    # Generate sample training data
    X, y, feature_names = create_sample_training_data(n_samples=300, n_features=45)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train a real RandomForest model for better SHAP compatibility
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=4)
    rf_model.fit(X_train, y_train)
    rf_model.name = "RandomForest"  # Add required name attribute
    
    # Save model
    model_data = {
        'model': rf_model,
        'trained_model': rf_model,
        'model_type': 'random_forest',
        'feature_names': feature_names,
        'training_info': {
            'n_samples': len(X_train),
            'n_features': len(feature_names),
            'system': system,
            'property': property_name
        }
    }
    
    joblib.dump(model_data, model_dir / "random_forest_model.pkl")
    
    # Also save as ensemble model (preferred by SHAP analyzer)
    joblib.dump(model_data, model_dir / "ensemble_model.pkl")
    
    # Save training data for SHAP analysis
    np.save(model_dir / "X_test.npy", X_test)
    np.save(model_dir / "y_test.npy", y_test)
    
    # Save feature names using pickle (required format)
    with open(model_dir / "feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    
    # Save readable feature names
    with open(model_dir / "feature_names.txt", "w") as f:
        for i, name in enumerate(feature_names):
            f.write(f"{i}: {name}\n")
    
    # Save training data info
    training_info = {
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'system': system,
        'property': property_name,
        'data_shape': {
            'X_train': X_train.shape,
            'X_test': X_test.shape,
            'y_train': y_train.shape,
            'y_test': y_test.shape
        }
    }
    
    with open(model_dir / "training_info.json", "w") as f:
        import json
        json.dump(training_info, f, indent=2, default=str)
    
    logger.info(f"✓ Created sample model for {system} - {property_name}")
    logger.info(f"  Model file: {model_dir / 'random_forest_model.pkl'}")
    logger.info(f"  Training data: {X_train.shape}, Test data: {X_test.shape}")
    logger.info(f"  Features: {len(feature_names)}")


def main():
    """Create sample models for testing interpretability analysis"""
    
    logger.info("Creating sample models for interpretability testing...")
    
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Define sample system-property combinations
    sample_combinations = [
        ("SiC", "youngs_modulus"),
        ("SiC", "vickers_hardness"),
        ("SiC", "fracture_toughness_mode_i"),
        ("Al2O3", "youngs_modulus"),
        ("Al2O3", "vickers_hardness"),
        ("B4C", "youngs_modulus"),
        ("B4C", "ballistic_efficiency")
    ]
    
    logger.info(f"Creating {len(sample_combinations)} sample model directories...")
    
    for system, property_name in sample_combinations:
        try:
            create_sample_model_directory(system, property_name, models_dir)
        except Exception as e:
            logger.error(f"Failed to create model for {system} - {property_name}: {e}")
    
    logger.info("✅ Sample model creation complete!")
    logger.info(f"Models created in: {models_dir.absolute()}")
    
    # Verify created models
    model_count = len(list(models_dir.rglob("*.pkl")))
    logger.info(f"Total model files created: {model_count}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)