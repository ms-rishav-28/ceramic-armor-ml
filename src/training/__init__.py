"""
Training module for model training, cross-validation, and hyperparameter tuning.
"""

try:
    from .trainer import CeramicPropertyTrainer
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import CeramicPropertyTrainer: {e}")
    CeramicPropertyTrainer = None

try:
    from .cross_validator import CrossValidator
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import CrossValidator: {e}")
    CrossValidator = None

try:
    from .hyperparameter_tuner import HyperparameterTuner
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import HyperparameterTuner: {e}")
    HyperparameterTuner = None

try:
    from .exact_modeling_trainer import ExactModelingTrainer
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import ExactModelingTrainer: {e}")
    ExactModelingTrainer = None

__all__ = [
    'CeramicPropertyTrainer',
    'CrossValidator',
    'HyperparameterTuner',
    'ExactModelingTrainer'
]