"""
Machine learning models module for ceramic armor property prediction.
"""

try:
    from .base_model import BaseModel
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import BaseModel: {e}")
    BaseModel = None

try:
    from .xgboost_model import XGBoostModel
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import XGBoostModel: {e}")
    XGBoostModel = None

try:
    from .catboost_model import CatBoostModel
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import CatBoostModel: {e}")
    CatBoostModel = None

try:
    from .random_forest_model import RandomForestModel
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import RandomForestModel: {e}")
    RandomForestModel = None

try:
    from .gradient_boosting_model import GradientBoostingModel
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import GradientBoostingModel: {e}")
    GradientBoostingModel = None

try:
    from .ensemble_model import EnsembleModel
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import EnsembleModel: {e}")
    EnsembleModel = None

try:
    from .transfer_learning import TransferLearningManager
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import TransferLearningManager: {e}")
    TransferLearningManager = None

try:
    from .ceramic_system_manager import CeramicSystemManager
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import CeramicSystemManager: {e}")
    CeramicSystemManager = None

__all__ = [
    'BaseModel',
    'XGBoostModel',
    'CatBoostModel',
    'RandomForestModel',
    'GradientBoostingModel',
    'EnsembleModel',
    'TransferLearningManager',
    'CeramicSystemManager'
]