"""
Interpretation module for model explainability and materials science insights.
"""

try:
    from .shap_analyzer import SHAPAnalyzer
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import SHAPAnalyzer: {e}")
    SHAPAnalyzer = None

try:
    from .materials_insights import interpret_feature_ranking
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import materials insights functions: {e}")
    interpret_feature_ranking = None

try:
    from .visualization import (
        parity_plot,
        residual_plot,
        corr_heatmap,
        feature_importance_plot
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import visualization functions: {e}")
    parity_plot = None
    residual_plot = None
    corr_heatmap = None
    feature_importance_plot = None

__all__ = [
    'SHAPAnalyzer',
    'interpret_feature_ranking',
    'parity_plot',
    'residual_plot',
    'corr_heatmap',
    'feature_importance_plot'
]