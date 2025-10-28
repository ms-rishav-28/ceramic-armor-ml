"""
Configuration management system with YAML validation and environment variable support.

This module provides robust configuration loading with hierarchical config support,
environment variable substitution, and comprehensive validation.
"""

import yaml
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from copy import deepcopy
import logging

from .logger import get_logger

logger = get_logger(__name__)


class Config_System:
    """
    Robust configuration management system with YAML validation and environment support.
    
    Features:
    - Hierarchical config loading (config.yaml + model_params.yaml)
    - Environment variable substitution
    - Default value fallbacks
    - Type validation and schema checking
    - Windows-compatible path handling
    """
    
    _config_cache: Dict[str, Dict[str, Any]] = {}
    _default_config: Dict[str, Any] = {
        'data': {
            'raw_dir': 'data/raw',
            'processed_dir': 'data/processed',
            'features_dir': 'data/features',
            'splits_dir': 'data/splits'
        },
        'results': {
            'models_dir': 'results/models',
            'metrics_dir': 'results/metrics',
            'figures_dir': 'results/figures',
            'predictions_dir': 'results/predictions',
            'reports_dir': 'results/reports'
        },
        'logging': {
            'level': 'INFO',
            'log_dir': 'logs'
        },
        'api': {
            'materials_project': {
                'base_url': 'https://api.materialsproject.org',
                'timeout': 30,
                'max_retries': 5,
                'retry_delay': 1
            }
        },
        'processing': {
            'batch_size': 100,
            'n_jobs': -1,
            'random_state': 42
        }
    }
    
    @classmethod
    def load_config(cls, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from YAML file with validation and environment substitution.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Loaded and validated configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If configuration validation fails
        """
        config_path = Path(config_path)
        cache_key = str(config_path.absolute())
        
        # Return cached config if available
        if cache_key in cls._config_cache:
            logger.debug(f"Using cached config for {config_path}")
            return cls._config_cache[cache_key]
        
        logger.info(f"Loading configuration from {config_path}")
        
        if not config_path.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return deepcopy(cls._default_config)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
                
            if raw_config is None:
                logger.warning(f"Empty config file {config_path}, using defaults")
                return deepcopy(cls._default_config)
            
            # Substitute environment variables
            config = cls._substitute_env_vars(raw_config)
            
            # Merge with defaults
            merged_config = cls._merge_configs(cls._default_config, config)
            
            # Validate configuration
            if cls.validate_config(merged_config):
                cls._config_cache[cache_key] = merged_config
                logger.info(f"Successfully loaded config from {config_path}")
                return merged_config
            else:
                logger.error(f"Config validation failed for {config_path}")
                raise ValueError(f"Invalid configuration in {config_path}")
                
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {config_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise
    
    @classmethod
    def load_hierarchical_config(cls, config_files: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Load and merge multiple configuration files hierarchically.
        
        Args:
            config_files: List of config file paths (later files override earlier ones)
            
        Returns:
            Merged configuration dictionary
        """
        logger.info(f"Loading hierarchical config from {len(config_files)} files")
        
        merged_config = deepcopy(cls._default_config)
        
        for config_file in config_files:
            try:
                config = cls.load_config(config_file)
                merged_config = cls._merge_configs(merged_config, config)
                logger.debug(f"Merged config from {config_file}")
            except FileNotFoundError:
                logger.warning(f"Config file {config_file} not found, skipping")
                continue
            except Exception as e:
                logger.error(f"Error loading {config_file}: {e}")
                raise
        
        return merged_config
    
    @classmethod
    def _substitute_env_vars(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Substitute environment variables in configuration values.
        
        Supports patterns like ${VAR_NAME} and ${VAR_NAME:default_value}
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with environment variables substituted
        """
        def substitute_value(value: Any) -> Any:
            if isinstance(value, str):
                # Pattern: ${VAR_NAME} or ${VAR_NAME:default}
                pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
                
                def replace_env_var(match):
                    var_name = match.group(1)
                    default_value = match.group(2) if match.group(2) is not None else ''
                    return os.getenv(var_name, default_value)
                
                return re.sub(pattern, replace_env_var, value)
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            else:
                return value
        
        return substitute_value(config)
    
    @classmethod
    def _merge_configs(cls, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration (lower priority)
            override_config: Override configuration (higher priority)
            
        Returns:
            Merged configuration dictionary
        """
        merged = deepcopy(base_config)
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = cls._merge_configs(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        
        return merged
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """
        Validate configuration structure and required fields.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required top-level sections
            required_sections = ['data', 'results', 'logging']
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing required config section: {section}")
                    return False
            
            # Validate data section
            data_config = config.get('data', {})
            required_data_dirs = ['raw_dir', 'processed_dir', 'features_dir', 'splits_dir']
            for dir_key in required_data_dirs:
                if dir_key not in data_config:
                    logger.error(f"Missing required data directory config: {dir_key}")
                    return False
            
            # Validate results section
            results_config = config.get('results', {})
            required_results_dirs = ['models_dir', 'metrics_dir', 'figures_dir']
            for dir_key in required_results_dirs:
                if dir_key not in results_config:
                    logger.error(f"Missing required results directory config: {dir_key}")
                    return False
            
            # Validate logging section
            logging_config = config.get('logging', {})
            if 'level' not in logging_config:
                logger.error("Missing logging level configuration")
                return False
            
            valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
            if logging_config['level'].upper() not in valid_log_levels:
                logger.error(f"Invalid logging level: {logging_config['level']}")
                return False
            
            logger.debug("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get a copy of the default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return deepcopy(cls._default_config)
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the configuration cache."""
        cls._config_cache.clear()
        logger.debug("Configuration cache cleared")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to load a configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Loaded configuration dictionary
    """
    return Config_System.load_config(config_path)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Convenience function to validate a configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    return Config_System.validate_config(config)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge (later ones override earlier)
        
    Returns:
        Merged configuration dictionary
    """
    if not configs:
        return {}
    
    result = deepcopy(configs[0])
    for config in configs[1:]:
        result = Config_System._merge_configs(result, config)
    
    return result


def load_project_config() -> Dict[str, Any]:
    """
    Load the standard project configuration files.
    
    Loads config.yaml and model_params.yaml from the config directory.
    
    Returns:
        Merged project configuration
    """
    config_dir = Path('config')
    config_files = [
        config_dir / 'config.yaml',
        config_dir / 'model_params.yaml'
    ]
    
    return Config_System.load_hierarchical_config(config_files)