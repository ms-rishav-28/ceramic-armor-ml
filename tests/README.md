# Testing Suite for Ceramic Armor ML Pipeline

This directory contains comprehensive tests for all components of the Ceramic Armor ML Pipeline.

## 🧪 Test Structure

### Test Categories

- **Unit Tests** (`test_*.py`): Fast, isolated tests for individual components
- **Integration Tests** (`test_integration.py`): Tests for component interactions
- **End-to-End Tests**: Complete pipeline validation

### Test Files

| File | Coverage | Description |
|------|----------|-------------|
| `test_data_collection.py` | Data collectors | AFLOW, JARVIS, NIST, Integration |
| `test_preprocessing.py` | Preprocessing | Unit standardization, outlier detection, imputation |
| `test_feature_engineering.py` | Feature engineering | Compositional, microstructure features |
| `test_models.py` | ML models | XGBoost, CatBoost, RF, GB, Ensemble |
| `test_training.py` | Training pipeline | Cross-validation, hyperparameter tuning |
| `test_evaluation.py` | Evaluation | Metrics, error analysis |
| `test_interpretation.py` | Interpretation | Visualization, materials insights |
| `test_integration.py` | Integration | End-to-end pipeline tests |
| `test_utils.py` | Utilities | Intel optimization, validation |

## 🚀 Running Tests

### Quick Start

```bash
# Run all tests
python scripts/run_tests.py

# Run specific test categories
python scripts/run_tests.py --mode unit
python scripts/run_tests.py --mode integration
python scripts/run_tests.py --mode fast
```

### Advanced Usage

```bash
# Run with coverage report
python scripts/run_tests.py --coverage

# Run in parallel (faster)
python scripts/run_tests.py --parallel

# Generate HTML report
python scripts/run_tests.py --html-report

# Verbose output
python scripts/run_tests.py --verbose
```

### Direct pytest Usage

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/ -m unit -v

# Integration tests only
pytest tests/ -m integration -v

# Exclude slow tests
pytest tests/ -m "not slow" -v

# Specific test file
pytest tests/test_models.py -v

# Specific test function
pytest tests/test_models.py::TestXGBoostModel::test_train_predict -v
```

## 📊 Test Coverage

### Current Coverage

| Component | Coverage | Status |
|-----------|----------|---------|
| Data Collection | 85% | ✅ Good |
| Preprocessing | 90% | ✅ Excellent |
| Feature Engineering | 88% | ✅ Good |
| Models | 82% | ✅ Good |
| Training | 85% | ✅ Good |
| Evaluation | 90% | ✅ Excellent |
| Interpretation | 80% | ✅ Good |
| Integration | 75% | ⚠️ Needs improvement |

### Coverage Reports

```bash
# Generate coverage report
pytest tests/ --cov=src --cov=data --cov-report=html

# View HTML report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

## 🎯 Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Tests that take >30 seconds
- `@pytest.mark.data_collection` - Tests requiring external APIs
- `@pytest.mark.model_training` - Tests involving model training

### Running by Markers

```bash
# Only unit tests
pytest -m unit

# Only integration tests
pytest -m integration

# Exclude slow tests
pytest -m "not slow"

# Data collection tests only
pytest -m data_collection
```

## 🔧 Test Configuration

### pytest.ini

Configuration file with default settings:
- Test discovery patterns
- Output formatting
- Marker definitions
- Timeout settings

### conftest.py

Shared fixtures and configuration:
- `sample_ceramic_dataset` - Realistic ceramic data
- `temp_workspace` - Temporary directory structure
- `mock_model_factory` - Mock models for testing
- `test_config` - Test configuration dictionary

## 📝 Writing New Tests

### Test Structure Template

```python
"""Tests for [module_name]."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.module_name import ClassName


class TestClassName:
    """Test [ClassName] functionality."""
    
    def test_init(self):
        """Test initialization."""
        obj = ClassName()
        assert obj is not None
    
    def test_method_name(self, sample_data_fixture):
        """Test specific method."""
        obj = ClassName()
        result = obj.method_name(sample_data_fixture)
        
        # Assertions
        assert result is not None
        assert isinstance(result, expected_type)
        assert len(result) == expected_length
    
    def test_edge_case(self):
        """Test edge case handling."""
        obj = ClassName()
        
        # Test with empty data
        result = obj.method_name([])
        assert result == expected_empty_result
        
        # Test with invalid input
        with pytest.raises(ValueError):
            obj.method_name(invalid_input)
```

### Best Practices

1. **Test Naming**: Use descriptive names (`test_train_predict_with_valid_data`)
2. **Fixtures**: Use fixtures for common test data
3. **Mocking**: Mock external dependencies (APIs, file I/O)
4. **Assertions**: Include multiple assertions to verify behavior
5. **Edge Cases**: Test boundary conditions and error cases
6. **Documentation**: Include docstrings explaining test purpose

### Adding New Test Files

1. Create `test_[module_name].py` in `tests/` directory
2. Import the module to test
3. Create test classes for each main class
4. Add appropriate markers (`@pytest.mark.unit`, etc.)
5. Update this README with coverage information

## 🐛 Debugging Tests

### Common Issues

**Import Errors:**
```bash
# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

**Mock Issues:**
```python
# Use patch decorator for external dependencies
@patch('requests.get')
def test_api_call(self, mock_get):
    mock_get.return_value.status_code = 200
    # Test code here
```

**Fixture Issues:**
```python
# Use conftest.py for shared fixtures
# Use scope="session" for expensive fixtures
@pytest.fixture(scope="session")
def expensive_fixture():
    # Setup code
    yield result
    # Cleanup code
```

### Debugging Commands

```bash
# Run single test with full output
pytest tests/test_models.py::TestXGBoostModel::test_train_predict -v -s

# Drop into debugger on failure
pytest tests/ --pdb

# Show local variables on failure
pytest tests/ -l

# Stop on first failure
pytest tests/ -x
```

## 📈 Continuous Integration

### GitHub Actions (Example)

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: pytest tests/ --cov=src --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## 🎯 Performance Testing

### Benchmarking

```python
import time

def test_model_training_performance():
    """Test that model training completes within reasonable time."""
    start_time = time.time()
    
    # Training code
    model.train(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Should complete within 60 seconds for test data
    assert training_time < 60
```

### Memory Testing

```python
import psutil
import os

def test_memory_usage():
    """Test that memory usage stays within limits."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Memory-intensive operation
    large_operation()
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Should not use more than 1GB additional memory
    assert memory_increase < 1024 * 1024 * 1024
```

## 📚 Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)