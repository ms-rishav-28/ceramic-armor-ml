"""
Data utilities with safe file I/O, schema validation, and progress tracking.

This module provides comprehensive data handling operations with robust error handling,
automatic directory creation, and memory-efficient processing capabilities.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import shutil
import hashlib
from datetime import datetime
import psutil
import gc
from tqdm import tqdm

from .logger import get_logger

logger = get_logger(__name__)


class DataUtils:
    """
    Comprehensive data utilities with error handling and validation.
    
    Features:
    - Safe file I/O with automatic directory creation
    - Data validation and schema checking
    - Progress tracking for long operations
    - Memory-efficient data processing
    - Windows-compatible path handling
    """
    
    @staticmethod
    def safe_save_data(data: Any, filepath: Union[str, Path], 
                      create_backup: bool = True, 
                      validate_after_save: bool = True) -> bool:
        """
        Safely save data to file with backup and validation options.
        
        Args:
            data: Data to save (supports pandas DataFrame, numpy array, dict, list)
            filepath: Target file path
            create_backup: Whether to create backup of existing file
            validate_after_save: Whether to validate saved data
            
        Returns:
            True if save successful, False otherwise
        """
        filepath = Path(filepath)
        
        try:
            # Create directory if it doesn't exist
            DataUtils.create_directories([filepath.parent])
            
            # Create backup if requested and file exists
            if create_backup and filepath.exists():
                backup_path = filepath.with_suffix(f"{filepath.suffix}.backup")
                shutil.copy2(filepath, backup_path)
                logger.debug(f"Created backup: {backup_path}")
            
            # Determine file format and save
            if filepath.suffix.lower() == '.csv':
                if isinstance(data, pd.DataFrame):
                    data.to_csv(filepath, index=False)
                else:
                    raise ValueError(f"CSV format requires pandas DataFrame, got {type(data)}")
                    
            elif filepath.suffix.lower() == '.json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                    
            elif filepath.suffix.lower() == '.pkl':
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
                    
            elif filepath.suffix.lower() == '.npy':
                if isinstance(data, np.ndarray):
                    np.save(filepath, data)
                else:
                    raise ValueError(f"NPY format requires numpy array, got {type(data)}")
                    
            elif filepath.suffix.lower() == '.npz':
                if isinstance(data, dict):
                    np.savez(filepath, **data)
                else:
                    raise ValueError(f"NPZ format requires dictionary, got {type(data)}")
                    
            else:
                # Default to pickle for unknown formats
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
                logger.warning(f"Unknown format {filepath.suffix}, using pickle")
            
            # Validate saved data if requested
            if validate_after_save:
                if not DataUtils._validate_saved_file(filepath, data):
                    logger.error(f"Validation failed for saved file: {filepath}")
                    return False
            
            logger.info(f"Successfully saved data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {e}")
            return False
    
    @staticmethod
    def safe_load_data(filepath: Union[str, Path], 
                      expected_type: Optional[type] = None,
                      validate_schema: bool = True) -> Optional[Any]:
        """
        Safely load data from file with type checking and validation.
        
        Args:
            filepath: Path to file to load
            expected_type: Expected data type (for validation)
            validate_schema: Whether to validate data schema
            
        Returns:
            Loaded data or None if loading failed
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return None
        
        try:
            # Load based on file extension
            if filepath.suffix.lower() == '.csv':
                data = pd.read_csv(filepath)
                
            elif filepath.suffix.lower() == '.json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
            elif filepath.suffix.lower() == '.pkl':
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    
            elif filepath.suffix.lower() == '.npy':
                data = np.load(filepath)
                
            elif filepath.suffix.lower() == '.npz':
                data = dict(np.load(filepath))
                
            else:
                # Try pickle as default
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                logger.warning(f"Unknown format {filepath.suffix}, tried pickle")
            
            # Type validation
            if expected_type and not isinstance(data, expected_type):
                logger.warning(f"Expected {expected_type}, got {type(data)} from {filepath}")
            
            # Schema validation for DataFrames
            if validate_schema and isinstance(data, pd.DataFrame):
                if not DataUtils._validate_dataframe_schema(data):
                    logger.warning(f"Schema validation failed for {filepath}")
            
            logger.debug(f"Successfully loaded data from {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}")
            return None
    
    @staticmethod
    def validate_data_schema(data: pd.DataFrame, 
                           expected_columns: List[str],
                           required_columns: Optional[List[str]] = None) -> bool:
        """
        Validate DataFrame schema against expected columns.
        
        Args:
            data: DataFrame to validate
            expected_columns: List of expected column names
            required_columns: List of required column names (subset of expected)
            
        Returns:
            True if schema is valid, False otherwise
        """
        try:
            if not isinstance(data, pd.DataFrame):
                logger.error(f"Expected DataFrame, got {type(data)}")
                return False
            
            # Check for required columns
            if required_columns:
                missing_required = set(required_columns) - set(data.columns)
                if missing_required:
                    logger.error(f"Missing required columns: {missing_required}")
                    return False
            
            # Check for unexpected columns
            unexpected_columns = set(data.columns) - set(expected_columns)
            if unexpected_columns:
                logger.warning(f"Unexpected columns found: {unexpected_columns}")
            
            # Check for missing expected columns
            missing_expected = set(expected_columns) - set(data.columns)
            if missing_expected:
                logger.warning(f"Missing expected columns: {missing_expected}")
            
            # Basic data quality checks
            if data.empty:
                logger.warning("DataFrame is empty")
                return False
            
            logger.debug(f"Schema validation passed for DataFrame with {len(data)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return False
    
    @staticmethod
    def create_directories(paths: List[Union[str, Path]]) -> None:
        """
        Create directories with error handling and logging.
        
        Args:
            paths: List of directory paths to create
        """
        for path in paths:
            try:
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {path}")
            except Exception as e:
                logger.error(f"Error creating directory {path}: {e}")
                raise
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                'percent': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / 1024 / 1024
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {}
    
    @staticmethod
    def process_in_batches(data: Union[pd.DataFrame, List[Any]], 
                          batch_size: int,
                          process_func: Callable,
                          description: str = "Processing",
                          **kwargs) -> List[Any]:
        """
        Process data in batches with progress tracking and memory management.
        
        Args:
            data: Data to process (DataFrame or list)
            batch_size: Size of each batch
            process_func: Function to apply to each batch
            description: Description for progress bar
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of results from processing each batch
        """
        try:
            if isinstance(data, pd.DataFrame):
                total_batches = len(data) // batch_size + (1 if len(data) % batch_size else 0)
                data_iter = (data.iloc[i:i+batch_size] for i in range(0, len(data), batch_size))
            else:
                total_batches = len(data) // batch_size + (1 if len(data) % batch_size else 0)
                data_iter = (data[i:i+batch_size] for i in range(0, len(data), batch_size))
            
            results = []
            
            with tqdm(total=total_batches, desc=description) as pbar:
                for batch in data_iter:
                    try:
                        # Process batch
                        result = process_func(batch, **kwargs)
                        results.append(result)
                        
                        # Update progress
                        pbar.update(1)
                        
                        # Memory management
                        if len(results) % 10 == 0:  # Every 10 batches
                            gc.collect()
                            memory_info = DataUtils.get_memory_usage()
                            if memory_info.get('percent', 0) > 80:
                                logger.warning(f"High memory usage: {memory_info['percent']:.1f}%")
                        
                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")
                        continue
            
            logger.info(f"Processed {len(results)} batches successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return []
    
    @staticmethod
    def calculate_file_hash(filepath: Union[str, Path]) -> Optional[str]:
        """
        Calculate MD5 hash of a file for integrity checking.
        
        Args:
            filepath: Path to file
            
        Returns:
            MD5 hash string or None if error
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                return None
            
            hash_md5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            
            return hash_md5.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating hash for {filepath}: {e}")
            return None
    
    @staticmethod
    def _validate_saved_file(filepath: Path, original_data: Any) -> bool:
        """
        Validate that saved file can be loaded and matches original data.
        
        Args:
            filepath: Path to saved file
            original_data: Original data that was saved
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Try to load the saved file
            loaded_data = DataUtils.safe_load_data(filepath, validate_schema=False)
            
            if loaded_data is None:
                return False
            
            # Basic type check
            if type(loaded_data) != type(original_data):
                logger.warning(f"Type mismatch: saved {type(loaded_data)}, original {type(original_data)}")
                return False
            
            # For DataFrames, check shape and columns
            if isinstance(original_data, pd.DataFrame):
                if loaded_data.shape != original_data.shape:
                    logger.error(f"Shape mismatch: saved {loaded_data.shape}, original {original_data.shape}")
                    return False
                
                if not loaded_data.columns.equals(original_data.columns):
                    logger.error("Column names don't match")
                    return False
            
            # For numpy arrays, check shape
            elif isinstance(original_data, np.ndarray):
                if loaded_data.shape != original_data.shape:
                    logger.error(f"Array shape mismatch: saved {loaded_data.shape}, original {original_data.shape}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    @staticmethod
    def _validate_dataframe_schema(df: pd.DataFrame) -> bool:
        """
        Basic DataFrame schema validation.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if basic validation passes
        """
        try:
            # Check for empty DataFrame
            if df.empty:
                logger.warning("DataFrame is empty")
                return False
            
            # Check for duplicate columns
            if df.columns.duplicated().any():
                logger.error("DataFrame has duplicate column names")
                return False
            
            # Check for all-null columns
            null_columns = df.columns[df.isnull().all()].tolist()
            if null_columns:
                logger.warning(f"Columns with all null values: {null_columns}")
            
            return True
            
        except Exception as e:
            logger.error(f"DataFrame validation error: {e}")
            return False


# Convenience functions
def safe_save_data(data: Any, filepath: Union[str, Path], **kwargs) -> bool:
    """Convenience function for safe data saving."""
    return DataUtils.safe_save_data(data, filepath, **kwargs)


def safe_load_data(filepath: Union[str, Path], **kwargs) -> Optional[Any]:
    """Convenience function for safe data loading."""
    return DataUtils.safe_load_data(filepath, **kwargs)


def validate_data_schema(data: pd.DataFrame, expected_columns: List[str], **kwargs) -> bool:
    """Convenience function for data schema validation."""
    return DataUtils.validate_data_schema(data, expected_columns, **kwargs)


def create_directories(paths: List[Union[str, Path]]) -> None:
    """Convenience function for directory creation."""
    return DataUtils.create_directories(paths)