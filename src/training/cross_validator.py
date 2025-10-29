# src/training/cross_validator.py
from typing import Dict, Callable, List, Tuple, Optional, Any
import numpy as np
from sklearn.model_selection import KFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from loguru import logger
import pandas as pd

from src.evaluation.metrics import ModelEvaluator


class EnhancedCrossValidator:
    """
    Enhanced cross-validation system with performance target enforcement
    
    Features:
    - 5-fold cross-validation with detailed metrics
    - Leave-one-ceramic-family-out validation
    - Performance target validation during CV
    - Uncertainty estimation across folds
    - Comprehensive result tracking
    """
    
    def __init__(self, n_splits: int = 5, random_state: int = 42, shuffle: bool = True):
        """
        Initialize enhanced cross-validator
        
        Args:
            n_splits: Number of CV folds
            random_state: Random seed for reproducibility
            shuffle: Whether to shuffle data before splitting
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        
        # Cross-validation objects
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        # Evaluator for comprehensive metrics
        self.evaluator = ModelEvaluator()
        
        # Results storage
        self.cv_results = {}
        self.loco_results = {}
        
        logger.info(f"✓ Enhanced CrossValidator initialized with {n_splits}-fold CV")
    
    def kfold_with_performance_targets(self, 
                                     model_constructor: Callable,
                                     X: np.ndarray, 
                                     y: np.ndarray,
                                     model_params: Dict,
                                     property_name: str,
                                     property_type: str,
                                     performance_targets: Dict) -> Dict:
        """
        Perform K-fold cross-validation with performance target validation
        
        Args:
            model_constructor: Function to create model instance
            X: Features
            y: Targets
            model_params: Model parameters
            property_name: Name of property being predicted
            property_type: 'mechanical' or 'ballistic'
            performance_targets: Dictionary with target R² values
        
        Returns:
            Comprehensive CV results
        """
        logger.info(f"Performing {self.n_splits}-fold CV for {property_name} ({property_type})")
        
        target_r2 = performance_targets.get(f'{property_type}_r2', 0.80)
        
        # Storage for results
        fold_results = []
        fold_predictions = []
        fold_uncertainties = []
        
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(X)):
            logger.info(f"  Processing fold {fold + 1}/{self.n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model
            model = model_constructor(model_params)
            
            try:
                # Train model
                if hasattr(model, 'train'):
                    model.train(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val)
                
                # Calculate comprehensive metrics
                metrics = self.evaluator.evaluate(y_val, y_pred, f"{property_name}_fold_{fold+1}")
                
                # Check performance target
                meets_target = metrics['r2'] >= target_r2
                
                # Get uncertainty if available
                uncertainty = None
                if hasattr(model, 'predict_with_uncertainty'):
                    try:
                        _, uncertainty = model.predict_with_uncertainty(X_val)
                        mean_uncertainty = np.mean(uncertainty)
                    except:
                        mean_uncertainty = None
                else:
                    mean_uncertainty = None
                
                # Store fold results
                fold_result = {
                    'fold': fold + 1,
                    'train_size': len(X_train),
                    'val_size': len(X_val),
                    'metrics': metrics,
                    'meets_target': meets_target,
                    'target_r2': target_r2,
                    'mean_uncertainty': mean_uncertainty
                }
                fold_results.append(fold_result)
                
                # Store predictions for ensemble analysis
                for i, (true_val, pred_val) in enumerate(zip(y_val, y_pred)):
                    fold_predictions.append({
                        'fold': fold + 1,
                        'sample_idx': val_idx[i],
                        'y_true': true_val,
                        'y_pred': pred_val
                    })
                
                # Store uncertainties if available
                if uncertainty is not None:
                    for i, unc_val in enumerate(uncertainty):
                        fold_uncertainties.append({
                            'fold': fold + 1,
                            'sample_idx': val_idx[i],
                            'uncertainty': unc_val
                        })
                
                # Log fold results
                status = "✓ PASS" if meets_target else "✗ FAIL"
                logger.info(f"    Fold {fold + 1}: R²={metrics['r2']:.4f} (target: {target_r2:.2f}) {status}")
                
            except Exception as e:
                logger.error(f"Error in fold {fold + 1}: {e}")
                # Add failed fold result
                fold_results.append({
                    'fold': fold + 1,
                    'train_size': len(X_train),
                    'val_size': len(X_val),
                    'metrics': {'r2': 0.0, 'rmse': np.inf, 'mae': np.inf},
                    'meets_target': False,
                    'target_r2': target_r2,
                    'mean_uncertainty': None,
                    'error': str(e)
                })
        
        # Calculate overall statistics
        valid_r2_scores = [result['metrics']['r2'] for result in fold_results 
                          if 'error' not in result]
        
        if len(valid_r2_scores) > 0:
            mean_r2 = np.mean(valid_r2_scores)
            std_r2 = np.std(valid_r2_scores)
            min_r2 = np.min(valid_r2_scores)
            max_r2 = np.max(valid_r2_scores)
            
            # Calculate target achievement rate
            targets_met = sum(1 for result in fold_results if result['meets_target'])
            target_rate = targets_met / len(fold_results)
        else:
            mean_r2 = std_r2 = min_r2 = max_r2 = 0.0
            target_rate = 0.0
        
        # Compile comprehensive results
        cv_results = {
            'property_name': property_name,
            'property_type': property_type,
            'n_folds': self.n_splits,
            'target_r2': target_r2,
            'statistics': {
                'mean_r2': mean_r2,
                'std_r2': std_r2,
                'min_r2': min_r2,
                'max_r2': max_r2,
                'target_achievement_rate': target_rate,
                'folds_meeting_target': targets_met,
                'total_folds': len(fold_results)
            },
            'fold_results': fold_results,
            'predictions': fold_predictions,
            'uncertainties': fold_uncertainties
        }
        
        # Log summary
        logger.info(f"✓ {self.n_splits}-fold CV complete:")
        logger.info(f"  Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
        logger.info(f"  Target achievement: {targets_met}/{len(fold_results)} folds ({target_rate:.1%})")
        
        # Store results
        self.cv_results[property_name] = cv_results
        
        return cv_results
    
    def leave_one_ceramic_out_with_targets(self,
                                         model_constructor: Callable,
                                         datasets_by_system: Dict[str, Dict[str, np.ndarray]],
                                         model_params: Dict,
                                         property_name: str,
                                         property_type: str,
                                         performance_targets: Dict) -> Dict:
        """
        Perform leave-one-ceramic-family-out validation with performance targets
        
        Args:
            model_constructor: Function to create model instance
            datasets_by_system: {system: {'X': features, 'y': targets}}
            model_params: Model parameters
            property_name: Property name
            property_type: 'mechanical' or 'ballistic'
            performance_targets: Dictionary with target R² values
        
        Returns:
            LOCO validation results
        """
        logger.info(f"Performing leave-one-ceramic-out validation for {property_name}")
        
        target_r2 = performance_targets.get(f'{property_type}_r2', 0.80)
        systems = list(datasets_by_system.keys())
        
        system_results = {}
        all_predictions = []
        
        for test_system in systems:
            logger.info(f"  Testing on {test_system}, training on others")
            
            # Prepare training data (all systems except test)
            train_systems = [s for s in systems if s != test_system]
            
            if len(train_systems) == 0:
                logger.warning(f"No training systems available for {test_system}")
                continue
            
            try:
                # Combine training data
                X_train_list = [datasets_by_system[s]['X'] for s in train_systems]
                y_train_list = [datasets_by_system[s]['y'] for s in train_systems]
                
                X_train = np.vstack(X_train_list)
                y_train = np.concatenate(y_train_list)
                
                # Test data
                X_test = datasets_by_system[test_system]['X']
                y_test = datasets_by_system[test_system]['y']
                
                # Create and train model
                model = model_constructor(model_params)
                
                if hasattr(model, 'train'):
                    model.train(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate comprehensive metrics
                metrics = self.evaluator.evaluate(y_test, y_pred, f"{property_name}_{test_system}")
                
                # Check performance target
                meets_target = metrics['r2'] >= target_r2
                
                # Get uncertainty if available
                mean_uncertainty = None
                if hasattr(model, 'predict_with_uncertainty'):
                    try:
                        _, uncertainty = model.predict_with_uncertainty(X_test)
                        mean_uncertainty = np.mean(uncertainty)
                    except:
                        pass
                
                # Store system results
                system_result = {
                    'test_system': test_system,
                    'train_systems': train_systems,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'metrics': metrics,
                    'meets_target': meets_target,
                    'target_r2': target_r2,
                    'mean_uncertainty': mean_uncertainty
                }
                system_results[test_system] = system_result
                
                # Store predictions
                for i, (true_val, pred_val) in enumerate(zip(y_test, y_pred)):
                    all_predictions.append({
                        'test_system': test_system,
                        'sample_idx': i,
                        'y_true': true_val,
                        'y_pred': pred_val
                    })
                
                # Log results
                status = "✓ PASS" if meets_target else "✗ FAIL"
                logger.info(f"    {test_system}: R²={metrics['r2']:.4f} (target: {target_r2:.2f}) {status}")
                
            except Exception as e:
                logger.error(f"Error processing {test_system}: {e}")
                system_results[test_system] = {
                    'test_system': test_system,
                    'error': str(e),
                    'meets_target': False,
                    'target_r2': target_r2
                }
        
        # Calculate overall statistics
        valid_results = [result for result in system_results.values() 
                        if 'error' not in result]
        
        if len(valid_results) > 0:
            r2_scores = [result['metrics']['r2'] for result in valid_results]
            mean_r2 = np.mean(r2_scores)
            std_r2 = np.std(r2_scores)
            min_r2 = np.min(r2_scores)
            max_r2 = np.max(r2_scores)
            
            # Calculate target achievement rate
            targets_met = sum(1 for result in valid_results if result['meets_target'])
            target_rate = targets_met / len(valid_results)
        else:
            r2_scores = []
            mean_r2 = std_r2 = min_r2 = max_r2 = 0.0
            target_rate = 0.0
        
        # Compile comprehensive results
        loco_results = {
            'property_name': property_name,
            'property_type': property_type,
            'target_r2': target_r2,
            'statistics': {
                'mean_r2': mean_r2,
                'std_r2': std_r2,
                'min_r2': min_r2,
                'max_r2': max_r2,
                'target_achievement_rate': target_rate,
                'systems_meeting_target': targets_met,
                'total_systems': len(system_results),
                'r2_scores': r2_scores
            },
            'system_results': system_results,
            'predictions': all_predictions
        }
        
        # Log summary
        logger.info(f"✓ LOCO validation complete:")
        logger.info(f"  Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
        logger.info(f"  Target achievement: {targets_met}/{len(valid_results)} systems ({target_rate:.1%})")
        
        # Store results
        self.loco_results[property_name] = loco_results
        
        return loco_results
    
    def get_cv_summary(self) -> Dict:
        """Get summary of all cross-validation results"""
        summary = {
            'kfold_results': {},
            'loco_results': {},
            'overall_statistics': {}
        }
        
        # Summarize K-fold results
        if self.cv_results:
            kfold_r2_scores = []
            kfold_target_rates = []
            
            for prop_name, results in self.cv_results.items():
                summary['kfold_results'][prop_name] = results['statistics']
                kfold_r2_scores.append(results['statistics']['mean_r2'])
                kfold_target_rates.append(results['statistics']['target_achievement_rate'])
            
            summary['overall_statistics']['kfold'] = {
                'mean_r2_across_properties': np.mean(kfold_r2_scores),
                'std_r2_across_properties': np.std(kfold_r2_scores),
                'mean_target_rate': np.mean(kfold_target_rates)
            }
        
        # Summarize LOCO results
        if self.loco_results:
            loco_r2_scores = []
            loco_target_rates = []
            
            for prop_name, results in self.loco_results.items():
                summary['loco_results'][prop_name] = results['statistics']
                loco_r2_scores.append(results['statistics']['mean_r2'])
                loco_target_rates.append(results['statistics']['target_achievement_rate'])
            
            summary['overall_statistics']['loco'] = {
                'mean_r2_across_properties': np.mean(loco_r2_scores),
                'std_r2_across_properties': np.std(loco_r2_scores),
                'mean_target_rate': np.mean(loco_target_rates)
            }
        
        return summary


# Backward compatibility
class CrossValidator(EnhancedCrossValidator):
    """Backward compatibility wrapper"""
    
    def kfold(self, model, X, y) -> Dict:
        """Legacy K-fold method for backward compatibility"""
        scores = []
        for tr, te in self.kf.split(X):
            model_local = model
            if hasattr(model_local, "build_model"):
                model_local.build_model()
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]
            model_local.train(Xtr, ytr)
            pred = model_local.predict(Xte)
            scores.append(r2_score(yte, pred))
        logger.info(f"KFold R²: mean={np.mean(scores):.4f} std={np.std(scores):.4f}")
        return {"scores": scores, "mean_r2": float(np.mean(scores)), "std_r2": float(np.std(scores))}

    def leave_one_ceramic_out(self, model_factory, datasets_by_system: Dict[str, Dict[str, np.ndarray]]):
        """Legacy LOCO method for backward compatibility"""
        systems = list(datasets_by_system.keys())
        results = {}
        for test_sys in systems:
            train_sys = [s for s in systems if s != test_sys]
            Xtr = np.vstack([datasets_by_system[s]['X'] for s in train_sys])
            ytr = np.concatenate([datasets_by_system[s]['y'] for s in train_sys])
            Xte = datasets_by_system[test_sys]['X']
            yte = datasets_by_system[test_sys]['y']
            model = model_factory()
            model.train(Xtr, ytr)
            pred = model.predict(Xte)
            r2 = r2_score(yte, pred)
            results[test_sys] = float(r2)
            logger.info(f"LOCO {test_sys}: R²={r2:.4f}")
        return results
