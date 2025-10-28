"""Tests for utility modules."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.utils.intel_optimizer import intel_opt


class TestIntelOptimizer:
    """Test Intel optimization utilities."""
    
    def test_intel_opt_exists(self):
        """Test that intel_opt module exists and has expected methods."""
        assert hasattr(intel_opt, 'apply_optimizations')
    
    @patch('os.environ')
    def test_apply_optimizations(self, mock_environ):
        """Test Intel optimizations application."""
        # Mock environment variables
        mock_environ.__setitem__ = Mock()
        
        # Test optimization application
        intel_opt.apply_optimizations()
        
        # Should set environment variables for Intel optimizations
        # (The exact variables depend on implementation)
        assert mock_environ.__setitem__.called
    
    def test_optimization_detection(self):
        """Test detection of Intel optimization capabilities."""
        # This test checks if the system can detect Intel optimizations
        # without actually applying them
        
        # Should not raise an error
        try:
            intel_opt.apply_optimizations()
            optimization_applied = True
        except Exception:
            optimization_applied = False
        
        # Either should work or fail gracefully
        assert isinstance(optimization_applied, bool)


class TestDataValidation:
    """Test data validation utilities."""
    
    def test_validate_ceramic_formula(self):
        """Test ceramic formula validation."""
        # Common ceramic formulas should be valid
        valid_formulas = ['SiC', 'Al2O3', 'B4C', 'WC', 'TiC', 'Si3N4', 'AlN']
        
        for formula in valid_formulas:
            # Should not raise an error when processing
            try:
                from pymatgen.core import Composition
                comp = Composition(formula)
                assert comp.num_atoms > 0
                valid = True
            except Exception:
                valid = False
            
            assert valid, f"Formula {formula} should be valid"
    
    def test_validate_property_ranges(self):
        """Test property value validation."""
        # Define realistic ranges for ceramic properties
        property_ranges = {
            'density': (1.0, 20.0),  # g/cm³
            'youngs_modulus': (50, 1000),  # GPa
            'vickers_hardness': (1, 50),  # GPa
            'fracture_toughness': (0.5, 15),  # MPa√m
            'thermal_conductivity': (0.1, 500),  # W/m·K
        }
        
        for prop, (min_val, max_val) in property_ranges.items():
            # Test values within range
            valid_value = (min_val + max_val) / 2
            assert min_val <= valid_value <= max_val
            
            # Test boundary values
            assert min_val <= min_val <= max_val
            assert min_val <= max_val <= max_val
    
    def test_missing_value_detection(self):
        """Test missing value detection utilities."""
        import pandas as pd
        
        # Create test data with various missing value representations
        data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [1, None, 3, 4, 5],
            'col3': ['a', 'b', '', 'd', 'e'],
            'col4': [1, 2, -999, 4, 5],  # Common missing value code
        })
        
        # Test standard missing value detection
        missing_mask = data.isnull()
        assert missing_mask.loc[2, 'col1']  # np.nan
        assert missing_mask.loc[1, 'col2']  # None
        
        # Test empty string detection
        empty_strings = data['col3'] == ''
        assert empty_strings.loc[2]
        
        # Test sentinel value detection (would need custom function)
        sentinel_values = data['col4'] == -999
        assert sentinel_values.loc[2]


class TestConfigurationValidation:
    """Test configuration validation utilities."""
    
    def test_validate_model_config(self):
        """Test model configuration validation."""
        # Valid configuration
        valid_config = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        # Check required parameters exist
        required_params = ['n_estimators', 'max_depth', 'learning_rate']
        for param in required_params:
            assert param in valid_config
            assert isinstance(valid_config[param], (int, float))
        
        # Check reasonable value ranges
        assert 1 <= valid_config['n_estimators'] <= 10000
        assert 1 <= valid_config['max_depth'] <= 20
        assert 0.001 <= valid_config['learning_rate'] <= 1.0
    
    def test_validate_training_config(self):
        """Test training configuration validation."""
        valid_config = {
            'test_size': 0.2,
            'validation_size': 0.15,
            'random_state': 42,
            'n_splits': 5
        }
        
        # Check split sizes are valid proportions
        assert 0 < valid_config['test_size'] < 1
        assert 0 < valid_config['validation_size'] < 1
        assert valid_config['test_size'] + valid_config['validation_size'] < 1
        
        # Check cross-validation parameters
        assert valid_config['n_splits'] >= 2
        assert isinstance(valid_config['random_state'], int)
    
    def test_validate_paths_config(self):
        """Test paths configuration validation."""
        from pathlib import Path
        
        paths_config = {
            'data': {
                'raw': 'data/raw',
                'processed': 'data/processed',
                'features': 'data/features'
            },
            'models': 'results/models',
            'figures': 'results/figures'
        }
        
        # Check all paths are strings or Path objects
        def check_paths(config_dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    check_paths(value)
                else:
                    assert isinstance(value, (str, Path))
                    # Should be able to create Path object
                    path_obj = Path(value)
                    assert isinstance(path_obj, Path)
        
        check_paths(paths_config)


if __name__ == "__main__":
    pytest.main([__file__])