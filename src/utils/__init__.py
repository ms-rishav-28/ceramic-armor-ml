"""
Utilities module for the ceramic armor ML pipeline.

Provides logging, configuration management, data utilities, and Intel optimizations.
"""

from .logger import (
    Logger_System,
    get_logger,
    setup_file_handler
)

from .config_loader import (
    Config_System,
    load_config,
    validate_config,
    merge_configs,
    load_project_config
)

from .data_utils import (
    DataUtils,
    safe_save_data,
    safe_load_data,
    validate_data_schema,
    create_directories
)

# Import intel_optimizer components
try:
    from .intel_optimizer import (
        IntelOptimizer,
        intel_opt
    )
    _intel_available = True
except ImportError:
    _intel_available = False

__all__ = [
    # Logger
    'Logger_System',
    'get_logger',
    'setup_file_handler',
    
    # Config
    'Config_System',
    'load_config',
    'validate_config',
    'merge_configs',
    'load_project_config',
    
    # Data utilities
    'DataUtils',
    'safe_save_data',
    'safe_load_data',
    'validate_data_schema',
    'create_directories'
]

# Add Intel optimizer to __all__ if available
if _intel_available:
    __all__.extend([
        'IntelOptimizer',
        'intel_opt'
    ])