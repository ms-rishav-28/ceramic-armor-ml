"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock


@pytest.fixture(scope="session")
def test_config():
    """Test configuration dictionary."""
    return {
        'ceramic_systems': {
            'primary': ['SiC', 'Al2O3', 'B4C', 'WC', 'TiC']
        },
        'properties': {
            'mechanical': ['vickers_hardness', 'fracture_toughness', 'youngs_modulus'],
            'ballistic': ['v50_ballistic_limit', 'dop_depth']
        },
        'training': {
            'test_size': 0.2,
            'validation_size': 0.15,
            'random_state': 42
        },
        'models': {
            'xgboost': {
                'n_estimators': 10,  # Small for testing
                'max_depth': 3,
                'learning_rate': 0.1
            },
            'random_forest': {
                'n_estimators': 10,
                'max_depth': 3
            }
        }
    }


@pytest.fixture
def sample_ceramic_dataset():
    """Generate a comprehensive sample ceramic dataset."""
    np.random.seed(42)
    n_samples = 200
    
    # Ceramic systems
    systems = ['SiC', 'Al2O3', 'B4C', 'WC', 'TiC']
    formulas = np.random.choice(systems, n_samples)
    
    # Base properties with realistic correlations
    density = np.random.normal(3.5, 0.8, n_samples)
    density = np.clip(density, 2.0, 6.0)  # Realistic range
    
    # Young's modulus correlated with density
    youngs_modulus = 300 + 50 * density + np.random.normal(0, 30, n_samples)
    youngs_modulus = np.clip(youngs_modulus, 200, 600)
    
    # Hardness correlated with both density and modulus
    vickers_hardness = 10 + 3 * density + 0.02 * youngs_modulus + np.random.normal(0, 3, n_samples)
    vickers_hardness = np.clip(vickers_hardness, 5, 40)
    
    # Fracture toughness (inversely related to hardness for ceramics)
    fracture_toughness = 8 - 0.1 * vickers_hardness + np.random.normal(0, 0.8, n_samples)
    fracture_toughness = np.clip(fracture_toughness, 2, 8)
    
    # Thermal conductivity
    thermal_conductivity = np.random.lognormal(4, 0.5, n_samples)
    thermal_conductivity = np.clip(thermal_conductivity, 10, 300)
    
    # Ballistic properties (correlated with mechanical properties)
    v50_ballistic_limit = 200 + 5 * vickers_hardness + 2 * fracture_toughness + np.random.normal(0, 20, n_samples)
    v50_ballistic_limit = np.clip(v50_ballistic_limit, 150, 400)
    
    # Create dataset
    data = pd.DataFrame({
        'formula': formulas,
        'density': density,
        'youngs_modulus': youngs_modulus,
        'vickers_hardness': vickers_hardness,
        'fracture_toughness': fracture_toughness,
        'thermal_conductivity': thermal_conductivity,
        'v50_ballistic_limit': v50_ballistic_limit,
        'source': np.random.choice(['MP', 'AFLOW', 'JARVIS'], n_samples),
        'material_id': [f'test_mat_{i:03d}' for i in range(n_samples)]
    })
    
    return data


@pytest.fixture
def temp_workspace():
    """Create temporary workspace directory structure."""
    temp_dir = tempfile.mkdtemp()
    workspace = Path(temp_dir)
    
    # Create directory structure
    dirs_to_create = [
        'data/raw/materials_project',
        'data/raw/aflow',
        'data/raw/jarvis',
        'data/raw/nist',
        'data/processed',
        'data/features',
        'results/models',
        'results/predictions',
        'results/figures',
        'config'
    ]
    
    for dir_path in dirs_to_create:
        (workspace / dir_path).mkdir(parents=True, exist_ok=True)
    
    yield workspace
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_model_factory():
    """Factory for creating mock models."""
    def create_mock_model(model_type='simple'):
        model = Mock()
        
        if model_type == 'simple':
            # Simple linear model for testing
            model.coef_ = np.random.random(10)  # Assume 10 features
            model.predict = lambda X: X @ model.coef_ if X.shape[1] == len(model.coef_) else np.random.random(X.shape[0])
        elif model_type == 'random':
            # Random predictions
            model.predict = lambda X: np.random.random(X.shape[0])
        
        model.train = Mock()
        model.fit = Mock()
        model.build_model = Mock()
        model.save_model = Mock()
        model.load_model = Mock()
        
        return model
    
    return create_mock_model


@pytest.fixture
def sample_feature_importance():
    """Generate sample feature importance data for testing."""
    features = [
        'vickers_hardness', 'fracture_toughness', 'density', 'youngs_modulus',
        'thermal_conductivity', 'pugh_ratio', 'elastic_anisotropy',
        'comp_atomic_mass_mean', 'comp_en_mean', 'comp_mixing_entropy',
        'hp_term', 'tortuosity_proxy', 'specific_hardness'
    ]
    
    # Generate importance values with some realistic patterns
    importance = np.random.exponential(0.1, len(features))
    importance = importance / importance.sum()  # Normalize
    
    df = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return df


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Custom pytest collection hook
def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark integration tests
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests (those that might take longer)
        if any(keyword in item.nodeid for keyword in ["integration", "end_to_end", "pipeline"]):
            item.add_marker(pytest.mark.slow)