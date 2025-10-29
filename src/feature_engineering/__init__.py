"""
Feature engineering module for creating derived material properties.
"""

try:
    from .compositional_features import CompositionalFeatureCalculator
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import CompositionalFeatureCalculator: {e}")
    CompositionalFeatureCalculator = None

try:
    from .derived_properties import DerivedPropertiesCalculator
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import DerivedPropertiesCalculator: {e}")
    DerivedPropertiesCalculator = None

try:
    from .microstructure_features import MicrostructureFeatureCalculator
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import MicrostructureFeatureCalculator: {e}")
    MicrostructureFeatureCalculator = None

# Temporarily disable PhaseStabilityAnalyzer due to pydantic compatibility issues with mp-api
# try:
#     from .phase_stability import PhaseStabilityAnalyzer
# except ImportError as e:
#     import warnings
#     warnings.warn(f"Could not import PhaseStabilityAnalyzer: {e}")
PhaseStabilityAnalyzer = None

try:
    from .comprehensive_feature_generator import ComprehensiveFeatureGenerator
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import ComprehensiveFeatureGenerator: {e}")
    ComprehensiveFeatureGenerator = None

__all__ = [
    'CompositionalFeatureCalculator',
    'DerivedPropertiesCalculator',
    'MicrostructureFeatureCalculator',
    'PhaseStabilityAnalyzer',
    'ComprehensiveFeatureGenerator'
]