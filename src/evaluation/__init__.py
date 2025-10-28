"""
Evaluation module for model performance assessment and error analysis.
"""

try:
    from .metrics import (
        ModelEvaluator,
        PerformanceChecker
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import metrics classes: {e}")
    ModelEvaluator = None
    PerformanceChecker = None

try:
    from .error_analyzer import ErrorAnalyzer
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import ErrorAnalyzer: {e}")
    ErrorAnalyzer = None

__all__ = [
    'ModelEvaluator',
    'PerformanceChecker',
    'ErrorAnalyzer'
]