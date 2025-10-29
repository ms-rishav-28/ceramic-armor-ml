"""Tests for training modules."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from sklearn.datasets import make_regression

from src.training.cross_validator import CrossValidator
from src.training.hyperparameter_tuner import HyperparameterTuner


@pytest.fixture
def sample_data():
    """Generate sample regression data for testing."""
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    return X, y


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.train = Mock()
    # Make predict return array with same length as input
    model.predict = Mock(side_effect=lambda X: np.random.random(len(X)))
    model.build_model = Mock()
    return model


class TestCrossValidator:
    """Test cross-validation functionality."""
    
    def test_init(self):
        cv = CrossValidator(n_splits=3, random_state=42)
        assert cv.kf.n_splits == 3
        assert cv.kf.random_state == 42
    
    def test_kfold(self, sample_data, mock_model):
        X, y = sample_data
        
        cv = CrossValidator(n_splits=3, random_state=42)
        results = cv.kfold(mock_model, X, y)
        
        assert 'scores' in results
        assert 'mean_r2' in results
        assert 'std_r2' in results
        assert len(results['scores']) == 3
        assert isinstance(results['mean_r2'], float)
        assert isinstance(results['std_r2'], float)
        
        # Verify model was trained for each fold
        assert mock_model.train.call_count == 3
        assert mock_model.predict.call_count == 3
    
    def test_leave_one_ceramic_out(self):
        # Create mock datasets for different ceramic systems
        datasets_by_system = {
            'SiC': {
                'X': np.random.random((50, 10)),
                'y': np.random.random(50)
            },
            'Al2O3': {
                'X': np.random.random((40, 10)),
                'y': np.random.random(40)
            },
            'B4C': {
                'X': np.random.random((30, 10)),
                'y': np.random.random(30)
            }
        }
        
        def model_factory():
            model = Mock()
            model.train = Mock()
            model.predict = Mock(side_effect=lambda X: np.random.random(len(X)))
            return model
        
        cv = CrossValidator()
        results = cv.leave_one_ceramic_out(model_factory, datasets_by_system)
        
        assert len(results) == 3
        assert 'SiC' in results
        assert 'Al2O3' in results
        assert 'B4C' in results
        
        for system, r2 in results.items():
            assert isinstance(r2, float)
            assert -10 <= r2 <= 1  # R² can be very negative with random data


class TestHyperparameterTuner:
    """Test hyperparameter optimization."""
    
    def test_init(self):
        search_space = {
            'n_estimators': {'type': 'int', 'low': 10, 'high': 100},
            'learning_rate': {'type': 'uniform', 'low': 0.01, 'high': 0.3}
        }
        
        tuner = HyperparameterTuner(search_space, n_trials=5)
        assert tuner.search_space == search_space
        assert tuner.n_trials == 5
    
    def test_suggest_parameters(self):
        search_space = {
            'n_estimators': {'type': 'int', 'low': 10, 'high': 100},
            'learning_rate': {'type': 'uniform', 'low': 0.01, 'high': 0.3},
            'max_depth': {'type': 'categorical', 'choices': [3, 5, 7]},
            'reg_alpha': {'type': 'loguniform', 'low': 1e-8, 'high': 1.0}
        }
        
        tuner = HyperparameterTuner(search_space, n_trials=5)
        
        # Mock optuna trial
        mock_trial = Mock()
        mock_trial.suggest_int.return_value = 50
        mock_trial.suggest_float.return_value = 0.1
        mock_trial.suggest_categorical.return_value = 5
        
        params = tuner._suggest(mock_trial, search_space)
        
        assert 'n_estimators' in params
        assert 'learning_rate' in params
        assert 'max_depth' in params
        assert 'reg_alpha' in params
        
        # Verify correct suggestion methods were called
        mock_trial.suggest_int.assert_called()
        mock_trial.suggest_float.assert_called()
        mock_trial.suggest_categorical.assert_called()
    
    @patch('optuna.create_study')
    def test_optimize(self, mock_create_study, sample_data):
        X, y = sample_data
        
        # Mock optuna study
        mock_study = Mock()
        mock_study.best_params = {'n_estimators': 50, 'learning_rate': 0.1}
        mock_study.best_value = -0.85  # Negative because Optuna minimizes negative R²
        mock_create_study.return_value = mock_study
        
        search_space = {
            'n_estimators': {'type': 'int', 'low': 10, 'high': 100},
            'learning_rate': {'type': 'uniform', 'low': 0.01, 'high': 0.3}
        }
        
        def model_factory(params):
            model = Mock()
            model.train = Mock()
            model.predict = Mock(return_value=np.random.random(20))
            return model
        
        tuner = HyperparameterTuner(search_space, n_trials=5)
        best_params, best_score = tuner.optimize(model_factory, X, y)
        
        assert best_params == {'n_estimators': 50, 'learning_rate': 0.1}
        assert best_score == 0.85
        
        # Verify study was created and optimized
        mock_create_study.assert_called_once()
        mock_study.optimize.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])