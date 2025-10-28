"""
Data collection module for gathering materials data from external APIs.

Provides collectors for Materials Project and other materials databases.
"""

try:
    from .materials_project_collector import (
        MaterialsProjectCollector,
        MaterialRecord
    )
    _materials_project_available = True
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(f"Could not import Materials Project components: {e}")
    MaterialsProjectCollector = None
    MaterialRecord = None
    _materials_project_available = False

__all__ = [
    'MaterialsProjectCollector',
    'MaterialRecord'
]