"""
Test Script for Performance Target Enforcement System
Validates all components of task 4 implementation
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import sys
sys.path.append('.')

from src.evaluation.performance_enforcer import PerformanceTargetEnforcer
from src.training.performance_driven_trainer import PerformanceDrivenTrainer
from src.training.cross_validator import EnhancedCrossValidator
from src.models.xgboost_model import XGBoostModel
from src.models.catboost_model import CatBoostModel
from src.models.random_forest_model import RandomForestModel
from src.models.gradient_boosting_model import GradientBoostingModel
from src.models.ensemble_model import EnsembleModel
from src.utils.config_loader import load_project_config
from loguru import logger


def generate_synthetic_ceramic_data(n_samples: int = 1000, n_features: int = 50, 
                                  ceramic_system: str = "SiC", 
                                  property_type: str = "mechanical") -> tuple:
    """
    Generate synthetic ceramic materials data for testing
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        ceramic_system: Ceramic system name
        property_type: 'mechanical' or 'ballistic'
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    np.random.seed(42)
    
    # Generate features with realistic ranges for ceramic materials
    X = np.random.randn(n_samples, n_features)
    
    # Add some structure to make it learnable
    # Simulate relationships between features and target
    if property_type == "mechanical":
        # Mechanical properties typically correlate with density, hardness, etc.
        y = (2.5 * X[:, 0] +      # Density effect
             1.8 * X[:, 1] +      # Hardness effect  
             -0.5 * X[:, 2] +     # Porosity effect (negative)
             0.3 * X[:, 3] * X[:, 4] +  # Interaction term
             np.random.normal(0, 0.5, n_samples))  # Noise
        
        # Scale to realistic mechanical property range (e.g., Young's modulus in GPa)
        y = 200 + 150 * (y - y.min()) / (y.max() - y.min())
        
    else:  # ballistic
        # Ballistic properties have different relationships
        y = (1.5 * X[:, 0] +      # Density effect
             2.2 * X[:, 1] +      # Hardness effect
             1.0 * X[:, 2] +      # Toughness effect
             -0.8 * X[:, 3] +     # Brittleness effect (negative)
             0.4 * X[:, 4] * X[:, 5] +  # Interaction
             np.random.normal(0, 0.8, n_samples))  # More noise for ballistic
        
        # Scale to realistic ballistic property range
        y = 50 + 100 * (y - y.min()) / (y.max() - y.min())
    
    # Generate feature names
    feature_names = [
        'density', 'hardness', 'porosity', 'grain_size', 'elastic_modulus',
        'thermal_conductivity', 'fracture_toughness', 'compressive_strength',
        'specific_hardness', 'brittleness_index', 'ballistic_efficiency'
    ] + [f'feature_{i}' for i in range(11, n_features)]
    
    return X, y, feature_names


def test_performance_target_validation():
    """Test performance target validation functionality"""
    logger.info("\n" + "="*80)
    logger.info("TESTING: Performance Target Validation")
    logger.info("="*80)
    
    # Load configuration
    config = load_project_config()
    
    # Initialize performance enforcer
    enforcer = PerformanceTargetEnforcer(config)
    
    # Generate test data
    X_test, y_test, feature_names = generate_synthetic_ceramic_data(
        n_samples=200, property_type="mechanical"
    )
    
    # Create and train a simple model
    xgb_model = XGBoostModel(config['models']['xgboost'])
    xgb_model.feature_names = feature_names
    xgb_model.train(X_test[:150], y_test[:150])
    
    # Test performance validation
    models = {'xgboost': xgb_model}
    results = enforcer.validate_performance_targets(
        models=models,
        X_test=X_test[150:],
        y_test=y_test[150:],
        property_name="youngs_modulus",
        property_type="mechanical"
    )
    
    logger.info(f"✓ Performance validation results: {results}")
    assert isinstance(results, dict)
    assert 'xgboost' in results
    
    return True


def test_hyperparameter_adjustment():
    """Test automatic hyperparameter adjustment"""
    logger.info("\n" + "="*80)
    logger.info("TESTING: Automatic Hyperparameter Adjustment")
    logger.info("="*80)
    
    # Load configuration
    config = load_project_config()
    
    # Initialize performance enforcer
    enforcer = PerformanceTargetEnforcer(config)
    
    # Generate test data
    X_train, y_train, feature_names = generate_synthetic_ceramic_data(
        n_samples=500, property_type="mechanical"
    )
    X_val, y_val, _ = generate_synthetic_ceramic_data(
        n_samples=100, property_type="mechanical"
    )
    
    # Model constructor
    def create_xgb_model(params):
        return XGBoostModel(params)
    
    # Test hyperparameter adjustment
    adjusted_model, best_params = enforcer.automatic_hyperparameter_adjustment(
        model_constructor=create_xgb_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        property_type="mechanical",
        model_name="xgboost",
        max_iterations=2  # Reduced for testing
    )
    
    logger.info(f"✓ Hyperparameter adjustment complete")
    logger.info(f"✓ Best parameters: {best_params}")
    assert adjusted_model is not None
    assert isinstance(best_params, dict)
    
    return True


def test_stacking_weight_optimization():
    """Test stacking weight optimization"""
    logger.info("\n" + "="*80)
    logger.info("TESTING: Stacking Weight Optimization")
    logger.info("="*80)
    
    # Load configuration
    config = load_project_config()
    
    # Initialize performance enforcer
    enforcer = PerformanceTargetEnforcer(config)
    
    # Generate test data
    X_train, y_train, feature_names = generate_synthetic_ceramic_data(
        n_samples=400, property_type="mechanical"
    )
    X_val, y_val, _ = generate_synthetic_ceramic_data(
        n_samples=100, property_type="mechanical"
    )
    
    # Train multiple models
    models = {}
    
    # XGBoost
    xgb_model = XGBoostModel(config['models']['xgboost'])
    xgb_model.feature_names = feature_names
    xgb_model.train(X_train, y_train)
    models['xgboost'] = xgb_model
    
    # Random Forest
    rf_model = RandomForestModel(config['models']['random_forest'])
    rf_model.feature_names = feature_names
    rf_model.train(X_train, y_train)
    models['random_forest'] = rf_model
    
    # Test weight optimization
    optimal_weights = enforcer.optimize_stacking_weights(
        base_models=models,
        X_val=X_val,
        y_val=y_val,
        property_type="mechanical"
    )
    
    logger.info(f"✓ Stacking weight optimization complete")
    logger.info(f"✓ Optimal weights: {optimal_weights}")
    assert len(optimal_weights) == len(models)
    assert np.abs(np.sum(optimal_weights) - 1.0) < 1e-6  # Weights should sum to 1
    
    return True


def test_cross_validation():
    """Test 5-fold cross-validation with performance targets"""
    logger.info("\n" + "="*80)
    logger.info("TESTING: 5-Fold Cross-Validation")
    logger.info("="*80)
    
    # Load configuration
    config = load_project_config()
    
    # Initialize cross-validator
    cv = EnhancedCrossValidator(n_splits=3)  # Reduced for testing
    
    # Generate test data
    X, y, feature_names = generate_synthetic_ceramic_data(
        n_samples=300, property_type="mechanical"
    )
    
    # Model constructor
    def create_xgb_model(params):
        model = XGBoostModel(params)
        model.feature_names = feature_names
        return model
    
    # Test cross-validation
    cv_results = cv.kfold_with_performance_targets(
        model_constructor=create_xgb_model,
        X=X,
        y=y,
        model_params=config['models']['xgboost'],
        property_name="youngs_modulus",
        property_type="mechanical",
        performance_targets=config['targets']
    )
    
    logger.info(f"✓ Cross-validation complete")
    logger.info(f"✓ Mean R²: {cv_results['statistics']['mean_r2']:.4f}")
    assert 'statistics' in cv_results
    assert 'fold_results' in cv_results
    
    return True


def test_leave_one_ceramic_out():
    """Test leave-one-ceramic-out validation"""
    logger.info("\n" + "="*80)
    logger.info("TESTING: Leave-One-Ceramic-Out Validation")
    logger.info("="*80)
    
    # Load configuration
    config = load_project_config()
    
    # Initialize cross-validator
    cv = EnhancedCrossValidator()
    
    # Generate datasets for multiple ceramic systems
    systems = ['SiC', 'Al2O3', 'B4C']
    datasets_by_system = {}
    
    for system in systems:
        X, y, feature_names = generate_synthetic_ceramic_data(
            n_samples=150, ceramic_system=system, property_type="mechanical"
        )
        datasets_by_system[system] = {'X': X, 'y': y}
    
    # Model constructor
    def create_xgb_model(params):
        model = XGBoostModel(params)
        model.feature_names = feature_names
        return model
    
    # Test LOCO validation
    loco_results = cv.leave_one_ceramic_out_with_targets(
        model_constructor=create_xgb_model,
        datasets_by_system=datasets_by_system,
        model_params=config['models']['xgboost'],
        property_name="youngs_modulus",
        property_type="mechanical",
        performance_targets=config['targets']
    )
    
    logger.info(f"✓ LOCO validation complete")
    logger.info(f"✓ Mean R²: {loco_results['statistics']['mean_r2']:.4f}")
    assert 'statistics' in loco_results
    assert 'system_results' in loco_results
    
    return True


def test_uncertainty_estimation():
    """Test prediction uncertainty estimation"""
    logger.info("\n" + "="*80)
    logger.info("TESTING: Prediction Uncertainty Estimation")
    logger.info("="*80)
    
    # Load configuration
    config = load_project_config()
    
    # Initialize performance enforcer
    enforcer = PerformanceTargetEnforcer(config)
    
    # Generate test data
    X_train, y_train, feature_names = generate_synthetic_ceramic_data(
        n_samples=400, property_type="mechanical"
    )
    X_test, y_test, _ = generate_synthetic_ceramic_data(
        n_samples=100, property_type="mechanical"
    )
    
    # Train multiple models
    models = {}
    
    # XGBoost
    xgb_model = XGBoostModel(config['models']['xgboost'])
    xgb_model.feature_names = feature_names
    xgb_model.train(X_train, y_train)
    models['xgboost'] = xgb_model
    
    # Random Forest
    rf_model = RandomForestModel(config['models']['random_forest'])
    rf_model.feature_names = feature_names
    rf_model.train(X_train, y_train)
    models['random_forest'] = rf_model
    
    # Test ensemble variance uncertainty
    pred_ensemble, unc_ensemble = enforcer.estimate_prediction_uncertainty(
        models=models,
        X=X_test,
        method='ensemble_variance'
    )
    
    logger.info(f"✓ Ensemble uncertainty estimation complete")
    logger.info(f"✓ Mean uncertainty: {np.mean(unc_ensemble):.4f}")
    
    # Test Random Forest uncertainty
    pred_rf, unc_rf = enforcer.estimate_prediction_uncertainty(
        models={'random_forest': models['random_forest']},
        X=X_test,
        method='random_forest_variance'
    )
    
    logger.info(f"✓ Random Forest uncertainty estimation complete")
    logger.info(f"✓ Mean RF uncertainty: {np.mean(unc_rf):.4f}")
    
    assert len(pred_ensemble) == len(X_test)
    assert len(unc_ensemble) == len(X_test)
    assert len(pred_rf) == len(X_test)
    assert len(unc_rf) == len(X_test)
    
    return True


def test_performance_driven_trainer():
    """Test the complete performance-driven training system"""
    logger.info("\n" + "="*80)
    logger.info("TESTING: Performance-Driven Training System")
    logger.info("="*80)
    
    # Load configuration
    config = load_project_config()
    
    # Initialize trainer
    trainer = PerformanceDrivenTrainer(config)
    
    # Generate test data
    X, y, feature_names = generate_synthetic_ceramic_data(
        n_samples=600, property_type="mechanical"
    )
    
    # Split data
    n_train = 300
    n_val = 150
    n_test = 150
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]
    
    # Test complete training with performance enforcement
    training_results = trainer.train_with_performance_enforcement(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        property_name="youngs_modulus",
        property_type="mechanical",
        feature_names=feature_names,
        ceramic_system="SiC"
    )
    
    logger.info(f"✓ Performance-driven training complete")
    logger.info(f"✓ Models trained: {list(training_results['models'].keys())}")
    logger.info(f"✓ Targets met: {training_results['performance']['targets_met']}")
    
    assert 'models' in training_results
    assert 'performance' in training_results
    assert 'cross_validation' in training_results
    assert 'uncertainty' in training_results
    
    return True


def run_all_tests():
    """Run all performance enforcement tests"""
    logger.info("\n" + "="*100)
    logger.info("PERFORMANCE TARGET ENFORCEMENT SYSTEM - COMPREHENSIVE TESTING")
    logger.info("="*100)
    
    tests = [
        ("Performance Target Validation", test_performance_target_validation),
        ("Hyperparameter Adjustment", test_hyperparameter_adjustment),
        ("Stacking Weight Optimization", test_stacking_weight_optimization),
        ("5-Fold Cross-Validation", test_cross_validation),
        ("Leave-One-Ceramic-Out", test_leave_one_ceramic_out),
        ("Uncertainty Estimation", test_uncertainty_estimation),
        ("Performance-Driven Trainer", test_performance_driven_trainer)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*20} {test_name} {'='*20}")
            success = test_func()
            results[test_name] = "PASS" if success else "FAIL"
            logger.info(f"✓ {test_name}: PASS")
        except Exception as e:
            results[test_name] = f"FAIL: {str(e)}"
            logger.error(f"✗ {test_name}: FAIL - {str(e)}")
    
    # Summary
    logger.info("\n" + "="*100)
    logger.info("TEST SUMMARY")
    logger.info("="*100)
    
    passed = sum(1 for result in results.values() if result == "PASS")
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        logger.info(f"{status} {test_name}: {result}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Performance Target Enforcement System Ready!")
    else:
        logger.warning("⚠️  Some tests failed - Review implementation")
    
    return results


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Run tests
    test_results = run_all_tests()
    
    # Save results
    results_path = Path("results/performance_enforcement_test_results.yaml")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        yaml.dump(test_results, f, default_flow_style=False)
    
    logger.info(f"\n✓ Test results saved to {results_path}")