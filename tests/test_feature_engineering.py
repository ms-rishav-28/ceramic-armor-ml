"""Tests for feature engineering modules."""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering.compositional_features import CompositionalFeatureCalculator
from src.feature_engineering.microstructure_features import MicrostructureFeatureCalculator


class TestCompositionalFeatureCalculator:
    """Test compositional feature calculation."""
    
    def test_init(self):
        calc = CompositionalFeatureCalculator()
        assert calc is not None
    
    def test_from_formula_sic(self):
        """Test compositional features for SiC."""
        calc = CompositionalFeatureCalculator()
        features = calc.from_formula("SiC")
        
        # Check expected features exist
        expected_features = [
            "comp_atomic_mass_mean",
            "comp_atomic_radius_mean", 
            "comp_en_mean",
            "comp_mixing_entropy"
        ]
        
        for feature in expected_features:
            assert feature in features
            assert not np.isnan(features[feature])
    
    def test_from_formula_al2o3(self):
        """Test compositional features for Al2O3."""
        calc = CompositionalFeatureCalculator()
        features = calc.from_formula("Al2O3")
        
        # Mixing entropy should be positive for multi-element compound
        assert features["comp_mixing_entropy"] > 0
        
        # Mean atomic mass should be reasonable
        assert 10 < features["comp_atomic_mass_mean"] < 30
    
    def test_augment_dataframe(self):
        """Test dataframe augmentation."""
        df = pd.DataFrame({
            "formula": ["SiC", "Al2O3", "B4C"],
            "density": [3.2, 3.9, 2.5]
        })
        
        calc = CompositionalFeatureCalculator()
        result = calc.augment_dataframe(df, formula_col="formula")
        
        # Should have more columns
        assert result.shape[1] > df.shape[1]
        assert result.shape[0] == df.shape[0]
        
        # Original columns should be preserved
        assert "formula" in result.columns
        assert "density" in result.columns


class TestMicrostructureFeatureCalculator:
    """Test microstructure feature calculation."""
    
    def test_init(self):
        calc = MicrostructureFeatureCalculator()
        assert calc is not None
    
    def test_safe_div(self):
        """Test safe division utility."""
        calc = MicrostructureFeatureCalculator()
        
        # Normal division
        result = calc._safe_div(10, 2)
        assert result == 5.0
        
        # Division by zero
        result = calc._safe_div(10, 0)
        assert result == 0.0
        
        # Division by very small number
        result = calc._safe_div(10, 1e-12)
        assert result == 0.0
    
    def test_add_features_with_grain_size(self):
        """Test feature addition with grain size data."""
        df = pd.DataFrame({
            "grain_size": [5.0, 10.0, 2.0],
            "density": [3.2, 3.9, 2.5],
            "compressive_strength": [2000, 1800, 2200],
            "porosity": [0.02, 0.05, 0.01]
        })
        
        calc = MicrostructureFeatureCalculator()
        result = calc.add_features(df)
        
        # Should have new features
        expected_features = ["hp_term", "tortuosity_proxy", "comp_strength_porosity_adj"]
        for feature in expected_features:
            assert feature in result.columns
        
        # Hall-Petch term should be inversely related to grain size
        assert result["hp_term"].iloc[2] > result["hp_term"].iloc[1]  # smaller grain size -> larger HP term
    
    def test_add_features_missing_data(self):
        """Test feature addition with missing microstructure data."""
        df = pd.DataFrame({
            "density": [3.2, 3.9, 2.5],
            "hardness": [25, 30, 20]
        })
        
        calc = MicrostructureFeatureCalculator()
        result = calc.add_features(df)
        
        # Should still add features (with default values)
        assert "hp_term" in result.columns
        assert "tortuosity_proxy" in result.columns
        
        # Default values should be 0
        assert all(result["hp_term"] == 0.0)
        assert all(result["tortuosity_proxy"] == 0.0)


if __name__ == "__main__":
    pytest.main([__file__])