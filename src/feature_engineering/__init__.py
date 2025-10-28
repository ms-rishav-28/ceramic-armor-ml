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

try:
    from .phase_stability import PhaseStabilityAnalyzer
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import PhaseStabilityAnalyzer: {e}")
    PhaseStabilityAnalyzer = None

__all__ = [
    'CompositionalFeatureCalculator',
    'DerivedPropertiesCalculator',
    'MicrostructureFeatureCalculator',
    'PhaseStabilityAnalyzer'
]