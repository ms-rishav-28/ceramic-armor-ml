"""
SHAP (SHapley Additive exPlanations) Analysis
Provides interpretable explanations for model predictions
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple
from loguru import logger

class SHAPAnalyzer:
    """
    SHAP-based model interpretation for ceramic property prediction
    
    Generates:
    1. Feature importance (global)
    2. Dependence plots (feature interactions)
    3. Force plots (individual predictions)
    4. Waterfall plots (prediction breakdown)
    """
    
    def __init__(self, model, model_type: str = 'tree'):
        """
        Initialize SHAP analyzer
        
        Args:
            model: Trained model
            model_type: 'tree' for tree-based models, 'linear' for linear models
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        
        logger.info(f"SHAP Analyzer initialized for {model_type} model")
    
    def load_training_data(self, model_dir: str) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Load training data saved by trainer for SHAP analysis
        
        Args:
            model_dir: Directory containing saved training data
            
        Returns:
            Tuple of (X_test, y_test, feature_names)
            
        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If data formats are inconsistent
        """
        model_path = Path(model_dir)
        
        # Validate required files exist
        required_files = {
            'X_test.npy': model_path / 'X_test.npy',
            'y_test.npy': model_path / 'y_test.npy', 
            'feature_names.pkl': model_path / 'feature_names.pkl'
        }
        
        missing_files = []
        for name, path in required_files.items():
            if not path.exists():
                missing_files.append(str(path))
        
        if missing_files:
            error_msg = f"Missing required files for SHAP analysis: {missing_files}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            # Load test data
            X_test = np.load(required_files['X_test.npy'])
            y_test = np.load(required_files['y_test.npy'])
            
            # Load feature names using pickle for consistency
            with open(required_files['feature_names.pkl'], 'rb') as f:
                feature_names = pickle.load(f)
            
            # Validation checks
            if X_test.shape[1] != len(feature_names):
                raise ValueError(
                    f"Feature count mismatch: X_test has {X_test.shape[1]} features "
                    f"but feature_names has {len(feature_names)} names"
                )
            
            if X_test.shape[0] != y_test.shape[0]:
                raise ValueError(
                    f"Sample count mismatch: X_test has {X_test.shape[0]} samples "
                    f"but y_test has {y_test.shape[0]} samples"
                )
            
            logger.info(f"✓ Loaded training data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
            
            return X_test, y_test, feature_names
            
        except Exception as e:
            error_msg = f"Error loading training data from {model_dir}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def create_explainer(self, X_background: np.ndarray = None,
                        feature_names: list = None):
        """
        Create SHAP explainer
        
        Args:
            X_background: Background dataset for TreeExplainer (optional)
            feature_names: List of feature names
        """
        self.feature_names = feature_names
        
        if self.model_type == 'tree':
            # TreeExplainer for XGBoost, RF, CatBoost, GB
            if X_background is not None:
                # Use subset of data for faster computation
                background = shap.sample(X_background, min(100, len(X_background)))
                self.explainer = shap.TreeExplainer(self.model, background)
            else:
                self.explainer = shap.TreeExplainer(self.model)
            
            logger.info("✓ TreeExplainer created")
            
        elif self.model_type == 'linear':
            # LinearExplainer for linear models
            self.explainer = shap.LinearExplainer(self.model, X_background)
            logger.info("✓ LinearExplainer created")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")    

    def calculate_shap_values(self, X: np.ndarray, 
                            n_samples: Optional[int] = None) -> np.ndarray:
        """
        Calculate SHAP values for dataset
        
        Args:
            X: Feature matrix
            n_samples: Number of samples to calculate (None = all)
        
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        try:
            # Subsample if requested
            if n_samples is not None and n_samples < len(X):
                indices = np.random.choice(len(X), n_samples, replace=False)
                X_sample = X[indices]
                logger.info(f"Subsampled {n_samples} from {len(X)} total samples")
            else:
                X_sample = X
            
            logger.info(f"Calculating SHAP values for {len(X_sample)} samples...")
            
            # Add progress indication for large datasets
            if len(X_sample) > 100:
                logger.info("This may take several minutes for large datasets...")
            
            self.shap_values = self.explainer.shap_values(X_sample)
            
            logger.info("✓ SHAP values calculated successfully")
            
            return self.shap_values
            
        except Exception as e:
            error_msg = f"Failed to calculate SHAP values: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def plot_summary(self, save_path: str = None, 
                    plot_type: str = 'dot', max_display: int = 20):
        """
        Create SHAP summary plot (feature importance)
        
        Args:
            save_path: Path to save figure
            plot_type: 'dot', 'bar', or 'violin'
            max_display: Maximum number of features to display
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated")
        
        try:
            logger.info(f"Creating SHAP summary plot ({plot_type})...")
            plt.figure(figsize=(10, 8))
            
            shap.summary_plot(
                self.shap_values,
                features=self.X_sample if hasattr(self, 'X_sample') else None,
                feature_names=self.feature_names,
                plot_type=plot_type,
                max_display=max_display,
                show=False
            )
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✓ Summary plot saved: {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create summary plot ({plot_type}): {str(e)}")
            plt.close()  # Ensure figure is closed even on error
            if save_path:
                logger.warning(f"Skipping summary plot: {save_path}")
            raise
    
    
    def plot_dependence(self, feature_name: str, 
                       interaction_feature: str = None,
                       save_path: str = None):
        """
        Create SHAP dependence plot for specific feature
        
        Args:
            feature_name: Feature to plot
            interaction_feature: Feature to color by (auto-detected if None)
            save_path: Path to save figure
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated")
        
        try:
            logger.info(f"Creating dependence plot for feature: {feature_name}")
            plt.figure(figsize=(10, 6))
            
            shap.dependence_plot(
                feature_name,
                self.shap_values,
                features=self.X_sample if hasattr(self, 'X_sample') else None,
                feature_names=self.feature_names,
                interaction_index=interaction_feature,
                show=False
            )
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✓ Dependence plot saved: {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create dependence plot for {feature_name}: {str(e)}")
            plt.close()  # Ensure figure is closed even on error
            if save_path:
                logger.warning(f"Skipping dependence plot: {save_path}")
            # Don't re-raise - continue with other plots
    
    def plot_waterfall(self, sample_idx: int, save_path: str = None):
        """
        Create waterfall plot for individual prediction
        
        Args:
            sample_idx: Index of sample to explain
            save_path: Path to save figure
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated")
        
        try:
            logger.info(f"Creating waterfall plot for sample {sample_idx}")
            plt.figure(figsize=(10, 6))
            
            shap.waterfall_plot(
                shap.Explanation(
                    values=self.shap_values[sample_idx],
                    base_values=self.explainer.expected_value,
                    data=self.X_sample[sample_idx] if hasattr(self, 'X_sample') else None,
                    feature_names=self.feature_names
                ),
                show=False
            )
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✓ Waterfall plot saved: {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create waterfall plot for sample {sample_idx}: {str(e)}")
            plt.close()  # Ensure figure is closed even on error
            if save_path:
                logger.warning(f"Skipping waterfall plot: {save_path}")
            # Don't re-raise - continue with other plots
    
    def get_feature_importance(self, importance_type: str = 'mean_abs') -> pd.DataFrame:
        """
        Calculate feature importance from SHAP values
        
        Args:
            importance_type: 'mean_abs' or 'mean'
        
        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated")
        
        if importance_type == 'mean_abs':
            importance = np.abs(self.shap_values).mean(axis=0)
        elif importance_type == 'mean':
            importance = self.shap_values.mean(axis=0)
        else:
            raise ValueError(f"Unknown importance type: {importance_type}")
        
        feature_names = self.feature_names or [f"Feature {i}" for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def generate_all_plots(self, X: np.ndarray = None, output_dir: str = None,
                          model_dir: str = None, top_features: int = 10):
        """
        Generate complete SHAP analysis with all plot types
        
        Args:
            X: Feature matrix (optional if model_dir provided)
            output_dir: Directory to save plots
            model_dir: Directory containing saved training data (alternative to X)
            top_features: Number of top features for dependence plots
        """
        logger.info("Starting complete SHAP analysis...")
        
        # Load data from model directory if provided
        if model_dir is not None:
            try:
                X_test, y_test, feature_names = self.load_training_data(model_dir)
                X = X_test
                self.feature_names = feature_names
                logger.info("✓ Using training data from model directory")
            except (FileNotFoundError, ValueError) as e:
                if X is None:
                    raise ValueError(f"Cannot load data from model_dir and no X provided: {e}")
                logger.warning(f"Failed to load from model_dir, using provided X: {e}")
        
        if X is None:
            raise ValueError("Either X or model_dir must be provided")
        
        # Calculate SHAP values
        logger.info("[1/4] Calculating SHAP values...")
        self.calculate_shap_values(X, n_samples=min(1000, len(X)))
        self.X_sample = X[:len(self.shap_values)]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Track successful plots
        successful_plots = []
        failed_plots = []
        
        # 2. Summary plots
        logger.info("[2/4] Creating summary plots...")
        try:
            self.plot_summary(
                save_path=str(output_path / 'shap_summary_dot.png'),
                plot_type='dot'
            )
            successful_plots.append('summary_dot')
        except Exception as e:
            failed_plots.append(f'summary_dot: {str(e)}')
        
        try:
            self.plot_summary(
                save_path=str(output_path / 'shap_summary_bar.png'),
                plot_type='bar'
            )
            successful_plots.append('summary_bar')
        except Exception as e:
            failed_plots.append(f'summary_bar: {str(e)}')
        
        # 3. Dependence plots for top features
        logger.info(f"[3/4] Creating dependence plots for top {top_features} features...")
        try:
            importance_df = self.get_feature_importance()
            top_feature_names = importance_df.head(top_features)['feature'].tolist()
            
            for i, feature in enumerate(top_feature_names, 1):
                try:
                    logger.info(f"  Creating dependence plot {i}/{len(top_feature_names)}: {feature}")
                    safe_name = feature.replace('/', '_').replace(' ', '_')
                    self.plot_dependence(
                        feature,
                        save_path=str(output_path / f'shap_dependence_{safe_name}.png')
                    )
                    successful_plots.append(f'dependence_{safe_name}')
                except Exception as e:
                    failed_plots.append(f'dependence_{safe_name}: {str(e)}')
                    continue  # Continue with next feature
                    
        except Exception as e:
            logger.error(f"Failed to create dependence plots: {str(e)}")
            failed_plots.append(f'dependence_plots: {str(e)}')
        
        # 4. Waterfall plots for representative samples
        logger.info("[4/4] Creating waterfall plots...")
        sample_indices = [0, len(self.shap_values)//2, len(self.shap_values)-1]
        for idx in sample_indices:
            try:
                self.plot_waterfall(
                    idx,
                    save_path=str(output_path / f'shap_waterfall_sample_{idx}.png')
                )
                successful_plots.append(f'waterfall_{idx}')
            except Exception as e:
                failed_plots.append(f'waterfall_{idx}: {str(e)}')
                continue  # Continue with next sample
        
        # Summary report
        logger.info(f"✓ SHAP analysis complete!")
        logger.info(f"  Successful plots: {len(successful_plots)}")
        logger.info(f"  Failed plots: {len(failed_plots)}")
        
        if failed_plots:
            logger.warning("Failed plots:")
            for failure in failed_plots:
                logger.warning(f"  - {failure}")
        
        logger.info(f"Results saved to: {output_dir}")
        
        return {
            'successful_plots': successful_plots,
            'failed_plots': failed_plots,
            'output_dir': str(output_path)
        }