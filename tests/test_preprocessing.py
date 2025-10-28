"""Tests for preprocessing modules."""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing.unit_standardizer import standardize, PRESSURE_TO_GPA
from src.preprocessing.outlier_detector import remove_iqr_outliers
from src.preprocessing.missing_value_handler import impute_knn, impute_median


class TestUnitStandardizer:
    """Test unit standardization."""
    
    def test_pressure_conversion_factors(self):
        """Test pressure conversion factors are correct."""
        assert PRESSURE_TO_GPA["Pa"] == 1e-9
        assert PRESSURE_TO_GPA["MPa"] == 1e-3
        assert PRESSURE_TO_GPA["GPa"] == 1.0
    
    def test_standardize_basic(self):
        """Test basic standardization functionality."""
        df = pd.DataFrame({
            "youngs_modulus": [400, 350, 420],
            "density": [3.2, 3.9, 2.3]
        })
        
        result = standardize(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)


class TestOutlierDetector:
    """Test outlier detection."""
    
    def test_remove_iqr_outliers(self):
        """Test IQR outlier removal."""
        # Create data with clear outliers
        df = pd.DataFrame({
            "density": [3.2, 3.1, 3.3, 10.0, 3.0],  # 10.0 is outlier
            "hardness": [25, 24, 26, 23, 100]       # 100 is outlier
        })
        
        result = remove_iqr_outliers(df, ["density", "hardness"], k=1.5)
        
        # Should remove rows with outliers
        assert len(result) < len(df)
        assert result["density"].max() < 10.0
        assert result["hardness"].max() < 100
    
    def test_remove_iqr_outliers_no_outliers(self):
        """Test IQR with no outliers."""
        df = pd.DataFrame({
            "density": [3.2, 3.1, 3.3, 3.0, 3.4],
            "hardness": [25, 24, 26, 23, 27]
        })
        
        result = remove_iqr_outliers(df, ["density", "hardness"], k=1.5)
        
        # Should keep all data
        assert len(result) == len(df)


class TestMissingValueHandler:
    """Test missing value imputation."""
    
    def test_impute_median(self):
        """Test median imputation."""
        df = pd.DataFrame({
            "density": [3.2, np.nan, 3.3, 3.0],
            "hardness": [25, 24, np.nan, 23]
        })
        
        result = impute_median(df)
        
        # Should have no missing values
        assert not result.isnull().any().any()
        
        # Check median imputation
        expected_density_median = pd.Series([3.2, 3.3, 3.0]).median()
        assert abs(result.loc[1, "density"] - expected_density_median) < 1e-6
    
    def test_impute_knn(self):
        """Test KNN imputation."""
        df = pd.DataFrame({
            "density": [3.2, np.nan, 3.3, 3.0, 3.1],
            "hardness": [25, 24, np.nan, 23, 26],
            "modulus": [400, 380, 420, 390, 410]
        })
        
        result = impute_knn(df, n_neighbors=3)
        
        # Should have no missing values
        assert not result.isnull().any().any()
        
        # Imputed values should be reasonable
        assert 3.0 <= result.loc[1, "density"] <= 3.3
        assert 23 <= result.loc[2, "hardness"] <= 26


if __name__ == "__main__":
    pytest.main([__file__])