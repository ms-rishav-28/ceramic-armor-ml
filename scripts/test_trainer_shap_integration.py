#!/usr/bin/env python3
"""
Integration test for trainer-SHAP analyzer compatibility
Tests that data saved by trainer can be loaded by SHAP analyzer
"""

import numpy as np
import pickle
import tempfile
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from interpretation.shap_analyzer import SHAPAnalyzer
from loguru import logger

def test_trainer_shap_integration():
    """Test that SHAP analyzer can load data saved by trainer"""
    
    logger.info("Testing trainer-SHAP integration...")
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir) / 'sic' / 'youngs_modulus'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Simulate data saved by trainer
        n_samples, n_features = 100, 50
        X_test = np.random.randn(n_samples, n_features)
        y_test = np.random.randn(n_samples)
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Save in trainer format
        np.save(model_dir / "X_test.npy", X_test)
        np.save(model_dir / "y_test.npy", y_test)
        
        # Save feature names using pickle (new format)
        with open(model_dir / "feature_names.pkl", "wb") as f:
            pickle.dump(feature_names, f)
        
        # Save readable version
        with open(model_dir / "feature_names.txt", "w") as f:
            for i, name in enumerate(feature_names):
                f.write(f"{i}: {name}\n")
        
        logger.info("✓ Test data saved in trainer format")
        
        # Test SHAP analyzer can load the data
        try:
            analyzer = SHAPAnalyzer(None, 'tree')  # Mock model for testing
            X_loaded, y_loaded, features_loaded = analyzer.load_training_data(str(model_dir))
            
            # Verify data integrity
            assert np.array_equal(X_loaded, X_test), "X_test data mismatch"
            assert np.array_equal(y_loaded, y_test), "y_test data mismatch"
            assert features_loaded == feature_names, "Feature names mismatch"
            assert len(features_loaded) == X_loaded.shape[1], "Feature count mismatch"
            
            logger.info("✓ SHAP analyzer successfully loaded trainer data")
            logger.info(f"  Loaded {X_loaded.shape[0]} samples, {X_loaded.shape[1]} features")
            logger.info(f"  Feature names: {len(features_loaded)} items")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ SHAP analyzer failed to load trainer data: {e}")
            return False

def test_missing_files_handling():
    """Test SHAP analyzer handles missing files gracefully"""
    
    logger.info("Testing missing files handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir) / 'empty_dir'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        analyzer = SHAPAnalyzer(None, 'tree')
        
        try:
            analyzer.load_training_data(str(model_dir))
            logger.error("✗ Should have raised FileNotFoundError")
            return False
        except FileNotFoundError as e:
            logger.info(f"✓ Correctly raised FileNotFoundError: {e}")
            return True
        except Exception as e:
            logger.error(f"✗ Unexpected error: {e}")
            return False

def test_data_validation():
    """Test SHAP analyzer validates data consistency"""
    
    logger.info("Testing data validation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir) / 'invalid_data'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mismatched data
        X_test = np.random.randn(100, 50)  # 50 features
        y_test = np.random.randn(80)       # 80 samples (mismatch)
        feature_names = [f'feature_{i}' for i in range(30)]  # 30 names (mismatch)
        
        np.save(model_dir / "X_test.npy", X_test)
        np.save(model_dir / "y_test.npy", y_test)
        
        with open(model_dir / "feature_names.pkl", "wb") as f:
            pickle.dump(feature_names, f)
        
        analyzer = SHAPAnalyzer(None, 'tree')
        
        try:
            analyzer.load_training_data(str(model_dir))
            logger.error("✗ Should have raised ValueError for mismatched data")
            return False
        except ValueError as e:
            logger.info(f"✓ Correctly raised ValueError: {e}")
            return True
        except Exception as e:
            logger.error(f"✗ Unexpected error: {e}")
            return False

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("TRAINER-SHAP INTEGRATION TEST")
    logger.info("="*60)
    
    tests = [
        ("Basic Integration", test_trainer_shap_integration),
        ("Missing Files Handling", test_missing_files_handling),
        ("Data Validation", test_data_validation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} ERROR: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {passed}/{total} tests passed")
    logger.info(f"{'='*60}")
    
    if passed == total:
        logger.info("🎉 All integration tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed!")
        sys.exit(1)