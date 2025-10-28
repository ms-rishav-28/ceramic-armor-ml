"""
Main Training Orchestrator
Coordinates training across all ceramic systems
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from loguru import logger

from src.models.xgboost_model import XGBoostModel
from src.models.catboost_model import CatBoostModel
from src.models.random_forest_model import RandomForestModel
from src.models.ensemble_model import EnsembleModel
from src.models.transfer_learning import TransferLearningManager
from src.evaluation.metrics import ModelEvaluator

class CeramicPropertyTrainer:
    """
    Complete training pipeline for ceramic armor property prediction
    
    Workflow:
    1. Load and prepare data for each ceramic system
    2. Train individual models (XGBoost, CatBoost, RF, GB)
    3. Create ensemble models
    4. Apply transfer learning for WC/TiC
    5. Evaluate and save models
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer
        
        Args:
            config: Master configuration dictionary
        """
        self.config = config
        self.ceramic_systems = config['ceramic_systems']['primary']
        self.models = {}
        self.scalers = {}
        self.evaluator = ModelEvaluator()
        
        logger.info(f"Trainer initialized for systems: {self.ceramic_systems}")
    
    def load_system_data(self, system: str, property_name: str) -> tuple:
        """
        Load data for specific ceramic system and property
        
        Args:
            system: Ceramic system (SiC, Al2O3, etc.)
            property_name: Target property
        
        Returns:
            Tuple of (X, y, feature_names)
        """
        data_path = Path(self.config['paths']['features']) / system.lower() / f"{system.lower()}_features.csv"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Remove rows with missing target
        df_clean = df.dropna(subset=[property_name])
        
        # Define feature columns
        exclude_cols = ['material_id', 'formula', 'ceramic_system', 'phase_stability'] + \
                      self.config['properties']['mechanical'] + \
                      self.config['properties']['ballistic']
        
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        X = df_clean[feature_cols].fillna(0).values
        y = df_clean[property_name].values
        
        logger.info(f"Loaded {system} data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, feature_cols
    
    def train_system_models(self, system: str, property_name: str) -> Dict:
        """
        Train all models for specific system and property
        
        Args:
            system: Ceramic system
            property_name: Target property
        
        Returns:
            Dictionary of trained models
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Training models for {system} - {property_name}")
        logger.info(f"{'='*80}")
        
        # Load data
        X, y, feature_names = self.load_system_data(system, property_name)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['training']['test_size'], random_state=42
        )
        
        # Further split training for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=self.config['training']['validation_size'], 
            random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Persist for SHAP and reproducibility
        import numpy as np
        import pickle
        from pathlib import Path
        model_dir = Path(self.config['paths']['models']) / system.lower() / property_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save test data and feature names in consistent format
        np.save(model_dir / "X_test.npy", X_test_scaled)
        np.save(model_dir / "y_test.npy", y_test)
        
        # Save feature names as Python list using pickle for consistency
        with open(model_dir / "feature_names.pkl", "wb") as f:
            pickle.dump(feature_names, f)
        
        # Also save as text file for human readability
        with open(model_dir / "feature_names.txt", "w") as f:
            for i, name in enumerate(feature_names):
                f.write(f"{i}: {name}\n")
        
        self.scalers[f"{system}_{property_name}"] = scaler
        
        # Dictionary to store models
        models = {}
        
        # 1. Train XGBoost
        logger.info("\n[1/4] Training XGBoost...")
        xgb_model = XGBoostModel(self.config['models']['xgboost'], n_jobs=20)
        xgb_model.feature_names = feature_names
        xgb_model.target_name = property_name
        xgb_model.train(X_train_scaled, y_train, X_val_scaled, y_val)
        models['xgboost'] = xgb_model
        
        # Evaluate
        y_pred = xgb_model.predict(X_test_scaled)
        xgb_metrics = self.evaluator.evaluate(y_test, y_pred, property_name)
        logger.info(f"XGBoost Test R²: {xgb_metrics['r2']:.4f}")
        
        # 2. Train CatBoost
        logger.info("\n[2/4] Training CatBoost...")
        cat_model = CatBoostModel(self.config['models']['catboost'], n_jobs=20)
        cat_model.feature_names = feature_names
        cat_model.target_name = property_name
        cat_model.train(X_train_scaled, y_train, X_val_scaled, y_val)
        models['catboost'] = cat_model
        
        y_pred = cat_model.predict(X_test_scaled)
        cat_metrics = self.evaluator.evaluate(y_test, y_pred, property_name)
        logger.info(f"CatBoost Test R²: {cat_metrics['r2']:.4f}")
        
        # 3. Train Random Forest
        logger.info("\n[3/4] Training Random Forest...")
        rf_model = RandomForestModel(self.config['models']['random_forest'], n_jobs=20)
        rf_model.feature_names = feature_names
        rf_model.target_name = property_name
        rf_model.train(X_train_scaled, y_train)
        models['random_forest'] = rf_model
        
        y_pred = rf_model.predict(X_test_scaled)
        rf_metrics = self.evaluator.evaluate(y_test, y_pred, property_name)
        logger.info(f"Random Forest Test R²: {rf_metrics['r2']:.4f}")
        
        # 4. Train Ensemble
        logger.info("\n[4/4] Training Ensemble...")
        ensemble_model = EnsembleModel(
            self.config['models']['ensemble'],
            self.config['models'],
            method='stacking',
            n_jobs=20
        )
        ensemble_model.feature_names = feature_names
        ensemble_model.target_name = property_name
        ensemble_model.train(X_train_scaled, y_train)
        models['ensemble'] = ensemble_model
        
        y_pred = ensemble_model.predict(X_test_scaled)
        ensemble_metrics = self.evaluator.evaluate(y_test, y_pred, property_name)
        logger.info(f"Ensemble Test R²: {ensemble_metrics['r2']:.4f}")
        
        # Save models
        model_dir = Path(self.config['paths']['models']) / system.lower() / property_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in models.items():
            model.save_model(str(model_dir / f"{name}_model.pkl"))
        
        # Save test data for later analysis
        test_data = pd.DataFrame({
            'y_true': y_test,
            'y_pred_xgboost': models['xgboost'].predict(X_test_scaled),
            'y_pred_catboost': models['catboost'].predict(X_test_scaled),
            'y_pred_rf': models['random_forest'].predict(X_test_scaled),
            'y_pred_ensemble': models['ensemble'].predict(X_test_scaled)
        })
        test_data.to_csv(model_dir / 'test_predictions.csv', index=False)
        
        logger.info(f"\n✓ All models trained and saved for {system} - {property_name}")
        
        return models
    
    def train_all_properties(self, system: str) -> Dict:
        """
        Train models for all properties of a ceramic system
        
        Args:
            system: Ceramic system
        
        Returns:
            Dictionary of all models
        """
        all_models = {}
        
        # Train mechanical property models
        for prop in self.config['properties']['mechanical']:
            try:
                models = self.train_system_models(system, prop)
                all_models[prop] = models
            except Exception as e:
                logger.error(f"Error training {system} - {prop}: {e}")
        
        # Train ballistic property models
        for prop in self.config['properties']['ballistic']:
            try:
                models = self.train_system_models(system, prop)
                all_models[prop] = models
            except Exception as e:
                logger.error(f"Error training {system} - {prop}: {e}")
        
        return all_models
    
    def train_all_systems(self):
        """Train models for all ceramic systems"""
        logger.info("\n" + "="*80)
        logger.info("STARTING COMPLETE TRAINING PIPELINE")
        logger.info("="*80)
        
        for system in self.ceramic_systems:
            logger.info(f"\n{'#'*80}")
            logger.info(f"# CERAMIC SYSTEM: {system}")
            logger.info(f"{'#'*80}")
            
            system_models = self.train_all_properties(system)
            self.models[system] = system_models
        
        logger.info("\n" + "="*80)
        logger.info("✓ TRAINING PIPELINE COMPLETE")
        logger.info("="*80)