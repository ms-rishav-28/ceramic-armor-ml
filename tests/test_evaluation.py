"""Tests for evaluation modules."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.evaluation.metrics import ModelEvaluator
from src.evaluation.error_analyzer import ErrorAnalyzer


@pytest.fixture
def sample_predictions():
    """Generate sample prediction data for testing."""
    np.random.seed(42)
    y_true = np.random.normal(100, 20, 100)  # True values around 100
    y_pred = y_true + np.random.normal(0, 5, 100)  # Predictions with some error
    return y_true, y_pred


class TestModelEvaluator:
    """Test model evaluation metrics."""
    
    def test_init(self):
        evaluator = ModelEvaluator()
        assert evaluator is not None
    
    def test_evaluate_basic_metrics(self, sample_predictions):
        y_true, y_pred = sample_predictions
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred, "test_property")
        
        # Check all expected metrics are present
        expected_metrics = ['r2', 'rmse', 'mae', 'mape']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        # Check reasonable values
        assert 0.5 <= metrics['r2'] <= 1.0  # Should be good correlation
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        assert metrics['mape'] >= 0
    
    def test_evaluate_perfect_predictions(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])  # Perfect predictions
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred, "perfect_test")
        
        assert metrics['r2'] == 1.0
        assert metrics['rmse'] == 0.0
        assert metrics['mae'] == 0.0
        assert metrics['mape'] == 0.0
    
    def test_evaluate_with_zeros(self):
        """Test evaluation when true values contain zeros."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0.1, 1.1, 1.9, 3.1, 3.9])
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred, "zero_test")
        
        # Should handle zeros gracefully
        assert 'r2' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        # MAPE might be inf due to division by zero, but should not crash


class TestErrorAnalyzer:
    """Test error analysis functionality."""
    
    def test_summarize_errors(self, sample_predictions):
        y_true, y_pred = sample_predictions
        
        errors_df, stats = ErrorAnalyzer.summarize_errors(y_true, y_pred)
        
        # Check errors dataframe
        assert isinstance(errors_df, pd.DataFrame)
        assert len(errors_df) == len(y_true)
        assert 'y_true' in errors_df.columns
        assert 'y_pred' in errors_df.columns
        assert 'error' in errors_df.columns
        assert 'abs_error' in errors_df.columns
        
        # Check statistics
        assert isinstance(stats, pd.DataFrame)
        assert 'abs_error' in stats.columns
        assert 'mean' in stats.index
        assert 'std' in stats.index
        assert 'min' in stats.index
        assert 'max' in stats.index
    
    def test_by_category(self):
        # Create sample error data with categories
        errors_df = pd.DataFrame({
            'y_true': [100, 200, 150, 300, 250],
            'y_pred': [95, 210, 145, 290, 260],
            'error': [-5, 10, -5, -10, 10],
            'abs_error': [5, 10, 5, 10, 10]
        })
        
        features_df = pd.DataFrame({
            'ceramic_system': ['SiC', 'Al2O3', 'SiC', 'B4C', 'Al2O3']
        })
        
        result = ErrorAnalyzer.by_category(errors_df, features_df, 'ceramic_system')
        
        assert isinstance(result, pd.DataFrame)
        assert 'ceramic_system' in result.columns
        assert 'mean' in result.columns
        assert 'median' in result.columns
        assert 'count' in result.columns
        
        # Should have one row per ceramic system
        systems = result['ceramic_system'].unique()
        assert 'SiC' in systems
        assert 'Al2O3' in systems
        assert 'B4C' in systems
    
    def test_by_category_missing_column(self):
        errors_df = pd.DataFrame({
            'y_true': [100, 200],
            'y_pred': [95, 210],
            'abs_error': [5, 10]
        })
        
        features_df = pd.DataFrame({
            'other_column': ['A', 'B']
        })
        
        # Should return empty dataframe when category column doesn't exist
        result = ErrorAnalyzer.by_category(errors_df, features_df, 'missing_column')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_error_distribution_analysis(self, sample_predictions):
        """Test error distribution analysis."""
        y_true, y_pred = sample_predictions
        errors_df, _ = ErrorAnalyzer.summarize_errors(y_true, y_pred)
        
        # Check error distribution properties
        errors = errors_df['error']
        abs_errors = errors_df['abs_error']
        
        # Errors should be roughly centered around 0
        assert abs(errors.mean()) < errors.std()  # Mean should be small relative to std
        
        # Absolute errors should be non-negative
        assert all(abs_errors >= 0)
        
        # Check for outliers (errors > 3 standard deviations)
        error_threshold = 3 * errors.std()
        outliers = errors_df[abs(errors_df['error']) > error_threshold]
        outlier_fraction = len(outliers) / len(errors_df)
        
        # Should have few outliers (< 5% for normal distribution)
        assert outlier_fraction < 0.05


if __name__ == "__main__":
    pytest.main([__file__])