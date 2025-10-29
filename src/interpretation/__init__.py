"""
Interpretation module for model explainability and materials science insights.
Provides comprehensive interpretability analysis for ceramic armor ML pipeline.
"""

try:
    from .shap_analyzer import SHAPAnalyzer
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import SHAPAnalyzer: {e}")
    SHAPAnalyzer = None

try:
    from .comprehensive_interpretability import ComprehensiveInterpretabilityAnalyzer
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import ComprehensiveInterpretabilityAnalyzer: {e}")
    ComprehensiveInterpretabilityAnalyzer = None

try:
    from .materials_insights import (
        interpret_feature_ranking,
        generate_comprehensive_materials_insights,
        get_system_specific_insights,
        get_property_specific_insights
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import materials insights functions: {e}")
    interpret_feature_ranking = None
    generate_comprehensive_materials_insights = None
    get_system_specific_insights = None
    get_property_specific_insights = None

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
    'ComprehensiveInterpretabilityAnalyzer',
    'interpret_feature_ranking',
    'generate_comprehensive_materials_insights',
    'get_system_specific_insights',
    'get_property_specific_insights',
    'parity_plot',
    'residual_plot',
    'corr_heatmap',
    'feature_importance_plot'
]