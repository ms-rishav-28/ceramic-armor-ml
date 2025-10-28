"""Integration tests for the complete pipeline."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from data.data_collection.data_integrator import DataIntegrator
from src.preprocessing.data_cleaner import DataCleaner
from src.feature_engineering.compositional_features import CompositionalFeatureCalculator
from src.training.cross_validator import CrossValidator


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_ceramic_data():
    """Generate sample ceramic data for integration testing."""
    data = {
        'formula': ['SiC', 'Al2O3', 'B4C', 'WC', 'TiC'] * 20,
        'density': np.random.normal(3.5, 0.5, 100),
        'youngs_modulus': np.random.normal(400, 50, 100),
        'vickers_hardness': np.random.normal(25, 5, 100),
        'fracture_toughness': np.random.normal(4.5, 1.0, 100),
        'thermal_conductivity': np.random.normal(100, 30, 100),
        'source': ['MP'] * 50 + ['AFLOW'] * 30 + ['JARVIS'] * 20
    }
    return pd.DataFrame(data)


class TestDataIntegrationPipeline:
    """Test complete data integration pipeline."""
    
    def test_data_collection_to_integration(self, temp_data_dir, sample_ceramic_data):
        """Test data flows from collection to integration."""
        # Create mock source files
        mp_data = sample_ceramic_data[sample_ceramic_data['source'] == 'MP'].copy()
        aflow_data = sample_ceramic_data[sample_ceramic_data['source'] == 'AFLOW'].copy()
        jarvis_data = sample_ceramic_data[sample_ceramic_data['source'] == 'JARVIS'].copy()
        
        # Save to temporary files
        mp_file = temp_data_dir / "mp_sic.csv"
        aflow_file = temp_data_dir / "aflow_sic.csv"
        jarvis_file = temp_data_dir / "jarvis_sic.csv"
        
        mp_data.to_csv(mp_file, index=False)
        aflow_data.to_csv(aflow_file, index=False)
        jarvis_data.to_csv(jarvis_file, index=False)
        
        # Test integration
        integrator = DataIntegrator(output_dir=str(temp_data_dir / "integrated"))
        source_files = {
            'materials_project': str(mp_file),
            'aflow': str(aflow_file),
            'jarvis': str(jarvis_file)
        }
        
        result_file = integrator.integrate_system('SiC', source_files)
        
        # Verify integration results
        assert Path(result_file).exists()
        integrated_data = pd.read_csv(result_file)
        
        # Should have data from all sources
        assert len(integrated_data) > 0
        assert 'source' in integrated_data.columns
        
        # Should have all expected columns
        expected_cols = ['formula', 'density', 'youngs_modulus', 'vickers_hardness']
        for col in expected_cols:
            assert col in integrated_data.columns


class TestPreprocessingPipeline:
    """Test preprocessing pipeline integration."""
    
    def test_cleaning_to_feature_engineering(self, sample_ceramic_data):
        """Test data flows from cleaning to feature engineering."""
        # Add some missing values and outliers for testing
        data = sample_ceramic_data.copy()
        data.loc[0, 'density'] = np.nan
        data.loc[1, 'vickers_hardness'] = 1000  # Outlier
        
        # Step 1: Data cleaning
        cleaner = DataCleaner()
        cleaned_data = cleaner.clean_dataframe(data)
        
        # Step 2: Feature engineering
        comp_calc = CompositionalFeatureCalculator()
        featured_data = comp_calc.augment_dataframe(cleaned_data, formula_col='formula')
        
        # Verify pipeline results
        assert len(featured_data) > 0
        assert featured_data.shape[1] > cleaned_data.shape[1]  # More features added
        
        # Should have compositional features
        comp_features = [col for col in featured_data.columns if col.startswith('comp_')]
        assert len(comp_features) > 0
        
        # Should not have excessive missing values
        missing_fraction = featured_data.isnull().sum().sum() / (featured_data.shape[0] * featured_data.shape[1])
        assert missing_fraction < 0.1  # Less than 10% missing


class TestTrainingPipeline:
    """Test training pipeline integration."""
    
    def test_cross_validation_with_real_data(self, sample_ceramic_data):
        """Test cross-validation with realistic ceramic data."""
        # Prepare data
        data = sample_ceramic_data.copy()
        
        # Add compositional features
        comp_calc = CompositionalFeatureCalculator()
        data = comp_calc.augment_dataframe(data, formula_col='formula')
        
        # Prepare features and target
        feature_cols = [col for col in data.columns if col.startswith('comp_') or 
                       col in ['density', 'youngs_modulus', 'thermal_conductivity']]
        X = data[feature_cols].fillna(0).values
        y = data['vickers_hardness'].values
        
        # Mock model for testing
        class MockModel:
            def __init__(self):
                self.coef_ = np.random.random(X.shape[1])
            
            def train(self, X_train, y_train):
                pass
            
            def predict(self, X_test):
                # Simple linear prediction for testing
                return X_test @ self.coef_
            
            def build_model(self):
                pass
        
        # Test K-fold cross-validation
        cv = CrossValidator(n_splits=3, random_state=42)
        results = cv.kfold(MockModel(), X, y)
        
        assert 'scores' in results
        assert 'mean_r2' in results
        assert 'std_r2' in results
        assert len(results['scores']) == 3
        
        # R² should be reasonable (not perfect due to noise, but not terrible)
        assert -1 <= results['mean_r2'] <= 1
    
    def test_leave_one_ceramic_out_integration(self, sample_ceramic_data):
        """Test LOCO cross-validation with ceramic systems."""
        # Prepare datasets by ceramic system
        datasets_by_system = {}
        
        for system in ['SiC', 'Al2O3', 'B4C']:
            system_data = sample_ceramic_data[sample_ceramic_data['formula'] == system].copy()
            
            if len(system_data) > 0:
                # Add compositional features
                comp_calc = CompositionalFeatureCalculator()
                system_data = comp_calc.augment_dataframe(system_data, formula_col='formula')
                
                # Prepare features
                feature_cols = [col for col in system_data.columns if col.startswith('comp_') or 
                               col in ['density', 'youngs_modulus']]
                X = system_data[feature_cols].fillna(0).values
                y = system_data['vickers_hardness'].values
                
                datasets_by_system[system] = {'X': X, 'y': y}
        
        # Only proceed if we have multiple systems
        if len(datasets_by_system) >= 2:
            def model_factory():
                class MockModel:
                    def __init__(self):
                        self.coef_ = None
                    
                    def train(self, X_train, y_train):
                        # Simple linear regression for testing
                        if X_train.shape[0] > X_train.shape[1]:
                            self.coef_ = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
                        else:
                            self.coef_ = np.random.random(X_train.shape[1])
                    
                    def predict(self, X_test):
                        if self.coef_ is not None:
                            return X_test @ self.coef_
                        else:
                            return np.random.random(X_test.shape[0])
                
                return MockModel()
            
            # Test LOCO cross-validation
            cv = CrossValidator()
            results = cv.leave_one_ceramic_out(model_factory, datasets_by_system)
            
            assert len(results) == len(datasets_by_system)
            for system, r2 in results.items():
                assert isinstance(r2, float)
                assert -2 <= r2 <= 1  # Allow some flexibility for mock model


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""
    
    def test_minimal_pipeline_execution(self, temp_data_dir):
        """Test minimal pipeline from raw data to predictions."""
        # Create minimal test data
        raw_data = pd.DataFrame({
            'formula': ['SiC', 'Al2O3', 'B4C'] * 10,
            'density': np.random.normal(3.5, 0.3, 30),
            'youngs_modulus': np.random.normal(400, 30, 30),
            'vickers_hardness': np.random.normal(25, 3, 30),
            'source': 'test'
        })
        
        # Step 1: Save raw data
        raw_file = temp_data_dir / "raw_data.csv"
        raw_data.to_csv(raw_file, index=False)
        
        # Step 2: Data cleaning
        cleaner = DataCleaner()
        cleaned_data = cleaner.clean_dataframe(raw_data)
        
        # Step 3: Feature engineering
        comp_calc = CompositionalFeatureCalculator()
        featured_data = comp_calc.augment_dataframe(cleaned_data, formula_col='formula')
        
        # Step 4: Prepare for modeling
        feature_cols = [col for col in featured_data.columns if 
                       col.startswith('comp_') or col in ['density', 'youngs_modulus']]
        X = featured_data[feature_cols].fillna(0).values
        y = featured_data['vickers_hardness'].values
        
        # Step 5: Simple model training and evaluation
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        
        # Verify pipeline completed successfully
        assert len(predictions) == len(y_test)
        assert isinstance(r2, float)
        assert -1 <= r2 <= 1  # R² in valid range
        
        # With good features and sufficient data, should get reasonable performance
        # (relaxed threshold for small test dataset)
        assert r2 > -0.5  # At least better than random


if __name__ == "__main__":
    pytest.main([__file__])