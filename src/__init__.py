"""
Ceramic Armor ML Pipeline - Main Package

A comprehensive machine learning pipeline for predicting mechanical and ballistic 
properties of ceramic armor materials.
"""

__version__ = "1.0.0"
__author__ = "Ceramic Armor ML Team"

# Core utilities
from .utils import (
    get_logger,
    load_config,
    load_project_config,
    safe_save_data,
    safe_load_data,
    validate_data_schema,
    create_directories
)

# Data collection
from .data_collection import MaterialsProjectCollector

# Preprocessing
from .preprocessing import (
    DataCleaner,
    impute_median,
    impute_mean,
    impute_knn,
    remove_iqr_outliers,
    remove_zscore_outliers,
    isolation_forest_filter,
    standardize
)

# Feature engineering
from .feature_engineering import (
    CompositionalFeatureCalculator,
    DerivedPropertiesCalculator,
    MicrostructureFeatureCalculator,
    PhaseStabilityAnalyzer
)

# Models
from .models import (
    BaseModel,
    XGBoostModel,
    CatBoostModel,
    RandomForestModel,
    GradientBoostingModel,
    EnsembleModel,
    TransferLearningManager
)

# Training
from .training import (
    CeramicPropertyTrainer,
    CrossValidator,
    HyperparameterTuner
)

# Evaluation
from .evaluation import (
    ModelEvaluator,
    PerformanceChecker,
    ErrorAnalyzer
)

# Interpretation
from .interpretation import (
    SHAPAnalyzer,
    interpret_feature_ranking,
    parity_plot,
    residual_plot,
    corr_heatmap
)

__all__ = [
    # Utilities
    'get_logger',
    'load_config', 
    'load_project_config',
    'safe_save_data',
    'safe_load_data',
    'validate_data_schema',
    'create_directories',
    
    # Data collection
    'MaterialsProjectCollector',
    
    # Preprocessing
    'DataCleaner',
    'MissingValueHandler',
    'OutlierDetector',
    'UnitStandardizer',
    
    # Feature engineering
    'CompositionalFeatures',
    'DerivedProperties',
    'MicrostructureFeatures',
    'PhaseStability',
    
    # Models
    'BaseModel',
    'XGBoostModel',
    'CatBoostModel',
    'RandomForestModel',
    'GradientBoostingModel',
    'EnsembleModel',
    'TransferLearning',
    
    # Training
    'Trainer',
    'CrossValidator',
    'HyperparameterTuner',
    
    # Evaluation
    'Metrics',
    'ErrorAnalyzer',
    
    # Interpretation
    'SHAPAnalyzer',
    'MaterialsInsights',
    'Visualization'
]