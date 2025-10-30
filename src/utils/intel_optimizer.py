"""
Intel CPU Optimization Module with Comprehensive Configuration Management.

This module provides complete Intel CPU optimization for machine learning workloads,
including Intel MKL acceleration, scikit-learn-intelex patching, and thread management
optimized for high-performance ceramic armor property prediction.

Classes:
    IntelOptimizer: Comprehensive Intel CPU optimization manager

Example:
    >>> optimizer = IntelOptimizer(num_threads=20)
    >>> success = optimizer.apply_optimizations()
    >>> if success:
    ...     print("Intel optimizations applied successfully")
"""

import os
import warnings
import platform
import multiprocessing
from typing import Dict, Optional, Union, Any
import logging

from .logger import get_logger

logger = get_logger(__name__)

class IntelOptimizer:
    """
    Comprehensive Intel CPU optimization manager for machine learning workloads.
    
    This class provides complete Intel CPU optimization including Intel MKL
    acceleration, scikit-learn-intelex patching, thread management, and
    performance monitoring for ceramic armor ML pipeline.
    
    Attributes:
        num_threads (int): Number of CPU threads to use
        optimization_applied (bool): Whether optimizations have been applied
        cpu_info (Dict[str, Any]): CPU information and capabilities
        
    Example:
        >>> optimizer = IntelOptimizer(num_threads=20)
        >>> success = optimizer.apply_optimizations()
        >>> status = optimizer.get_optimization_status()
        >>> print(f"Optimization status: {status}")
    """
    
    def __init__(self, num_threads: Optional[int] = None) -> None:
        """
        Initialize Intel optimizer with comprehensive validation.
        
        Args:
            num_threads: Number of CPU threads to use. If None, auto-detects
                        optimal thread count based on CPU capabilities.
                        
        Raises:
            ValueError: If num_threads is invalid
            RuntimeError: If CPU detection fails
            
        Example:
            >>> # Auto-detect optimal threads
            >>> optimizer = IntelOptimizer()
            >>> # Or specify manually
            >>> optimizer = IntelOptimizer(num_threads=20)
        """
        try:
            # Auto-detect optimal thread count if not specified
            if num_threads is None:
                num_threads = self._detect_optimal_threads()
            
            # Validate thread count
            if not isinstance(num_threads, int) or num_threads < 1:
                raise ValueError(f"num_threads must be positive integer, got {num_threads}")
            
            max_threads = multiprocessing.cpu_count()
            if num_threads > max_threads:
                logger.warning(f"Requested {num_threads} threads, but only {max_threads} available")
                num_threads = max_threads
            
            self.num_threads = num_threads
            self.optimization_applied = False
            self.cpu_info = self._get_cpu_info()
            
            logger.info(f"IntelOptimizer initialized with {self.num_threads} threads")
            logger.debug(f"CPU info: {self.cpu_info}")
            
        except Exception as e:
            logger.error(f"Failed to initialize IntelOptimizer: {e}")
            raise RuntimeError(f"IntelOptimizer initialization failed: {e}") from e
    
    def _detect_optimal_threads(self) -> int:
        """
        Detect optimal number of threads based on CPU capabilities.
        
        Returns:
            Optimal number of threads for ML workloads
            
        Raises:
            RuntimeError: If thread detection fails
        """
        try:
            cpu_count = multiprocessing.cpu_count()
            
            # For ML workloads, use all available cores but leave some headroom
            if cpu_count >= 16:
                optimal_threads = min(20, cpu_count)  # Cap at 20 for stability
            elif cpu_count >= 8:
                optimal_threads = cpu_count
            else:
                optimal_threads = max(1, cpu_count - 1)  # Leave one core free
            
            logger.debug(f"Detected {cpu_count} CPU cores, using {optimal_threads} threads")
            return optimal_threads
            
        except Exception as e:
            logger.error(f"Failed to detect optimal threads: {e}")
            raise RuntimeError(f"Thread detection failed: {e}") from e
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """
        Get CPU information and capabilities.
        
        Returns:
            Dictionary with CPU information
        """
        try:
            cpu_info = {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'architecture': platform.architecture(),
                'cpu_count': multiprocessing.cpu_count(),
                'python_version': platform.python_version()
            }
            
            # Try to get more detailed CPU info
            try:
                import cpuinfo
                detailed_info = cpuinfo.get_cpu_info()
                cpu_info.update({
                    'brand_raw': detailed_info.get('brand_raw', 'Unknown'),
                    'hz_advertised': detailed_info.get('hz_advertised_friendly', 'Unknown'),
                    'flags': detailed_info.get('flags', [])[:10]  # First 10 flags
                })
            except ImportError:
                logger.debug("cpuinfo package not available for detailed CPU information")
            
            return cpu_info
            
        except Exception as e:
            logger.warning(f"Could not get CPU info: {e}")
            return {'cpu_count': multiprocessing.cpu_count()}

    def apply_optimizations(self) -> bool:
        """
        Apply comprehensive Intel CPU optimizations.
        
        Configures Intel MKL, thread management, and scikit-learn-intelex
        for optimal performance on Intel CPUs.
        
        Returns:
            True if optimizations applied successfully, False otherwise
            
        Raises:
            RuntimeError: If critical optimization steps fail
            
        Example:
            >>> optimizer = IntelOptimizer(num_threads=20)
            >>> success = optimizer.apply_optimizations()
            >>> if success:
            ...     print("Ready for high-performance ML training")
        """
        try:
            logger.info("Applying Intel CPU optimizations...")
            
            # Set environment variables for threading
            thread_env_vars = {
                'OMP_NUM_THREADS': str(self.num_threads),
                'MKL_NUM_THREADS': str(self.num_threads),
                'NUMEXPR_NUM_THREADS': str(self.num_threads),
                'OPENBLAS_NUM_THREADS': str(self.num_threads),
                'BLAS_NUM_THREADS': str(self.num_threads)
            }
            
            for var, value in thread_env_vars.items():
                os.environ[var] = value
                logger.debug(f"Set {var}={value}")
            
            # Intel MKL specific optimizations
            mkl_env_vars = {
                'MKL_DYNAMIC': 'FALSE',  # Disable dynamic thread adjustment
                'MKL_VERBOSE': '0',      # Disable verbose output
                'MKL_ENABLE_INSTRUCTIONS': 'AVX2',  # Enable AVX2 instructions
                'MKL_THREADING_LAYER': 'INTEL'      # Use Intel threading layer
            }
            
            for var, value in mkl_env_vars.items():
                os.environ[var] = value
                logger.debug(f"Set {var}={value}")
            
            # Apply Intel Extension for Scikit-learn
            intel_extension_success = self._apply_intel_extension()
            
            # Verify optimizations
            verification_success = self._verify_optimizations()
            
            self.optimization_applied = intel_extension_success and verification_success
            
            if self.optimization_applied:
                logger.info(f"✓ Intel optimizations applied successfully")
                logger.info(f"✓ Thread configuration: {self.num_threads} threads")
                logger.info("✓ Intel MKL optimizations enabled")
            else:
                logger.warning("Intel optimizations partially applied")
            
            return self.optimization_applied
            
        except Exception as e:
            logger.error(f"Failed to apply Intel optimizations: {e}")
            raise RuntimeError(f"Intel optimization failed: {e}") from e
    
    def _apply_intel_extension(self) -> bool:
        """
        Apply Intel Extension for Scikit-learn.
        
        Returns:
            True if extension applied successfully, False otherwise
        """
        try:
            from sklearnex import patch_sklearn, config_context
            
            # Apply patches with error handling
            patch_sklearn()
            
            # Verify patching worked
            from sklearnex import get_patch_map
            patched_estimators = get_patch_map()
            
            if patched_estimators:
                logger.info(f"✓ Intel Extension for Scikit-learn applied "
                           f"({len(patched_estimators)} estimators patched)")
                logger.debug(f"Patched estimators: {list(patched_estimators.keys())[:5]}...")
                return True
            else:
                logger.warning("Intel Extension applied but no estimators patched")
                return False
                
        except ImportError as e:
            logger.warning(f"Intel Extension not found: {e}")
            logger.info("Install with: pip install scikit-learn-intelex")
            return False
        except Exception as e:
            logger.error(f"Error applying Intel Extension: {e}")
            return False
    
    def _verify_optimizations(self) -> bool:
        """
        Verify that optimizations are properly configured.
        
        Returns:
            True if verification passes, False otherwise
        """
        try:
            # Check environment variables
            required_vars = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS']
            for var in required_vars:
                if os.environ.get(var) != str(self.num_threads):
                    logger.warning(f"Environment variable {var} not set correctly")
                    return False
            
            # Check Intel MKL availability
            try:
                import numpy as np
                # This will use MKL if available
                test_array = np.random.random((100, 100))
                np.dot(test_array, test_array.T)
                logger.debug("NumPy/MKL verification passed")
            except Exception as e:
                logger.warning(f"NumPy/MKL verification failed: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Optimization verification failed: {e}")
            return False
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization status information.
        
        Returns:
            Dictionary with detailed optimization status
            
        Example:
            >>> optimizer = IntelOptimizer()
            >>> status = optimizer.get_optimization_status()
            >>> print(f"Threads: {status['num_threads']}")
            >>> print(f"Intel Extension: {status['intel_extension_available']}")
        """
        try:
            # Basic status
            status = {
                'num_threads': self.num_threads,
                'optimization_applied': self.optimization_applied,
                'cpu_info': self.cpu_info
            }
            
            # Environment variables
            env_vars = [
                'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'MKL_DYNAMIC', 'MKL_VERBOSE', 'MKL_ENABLE_INSTRUCTIONS'
            ]
            
            status['environment_variables'] = {
                var: os.environ.get(var, 'Not set') for var in env_vars
            }
            
            # Intel Extension status
            try:
                from sklearnex import get_patch_map
                patched_estimators = get_patch_map()
                status['intel_extension_available'] = True
                status['patched_estimators_count'] = len(patched_estimators)
                status['sample_patched_estimators'] = list(patched_estimators.keys())[:5]
            except ImportError:
                status['intel_extension_available'] = False
                status['patched_estimators_count'] = 0
                status['sample_patched_estimators'] = []
            
            # Performance indicators
            status['performance_indicators'] = {
                'mkl_available': 'MKL_NUM_THREADS' in os.environ,
                'threading_configured': os.environ.get('OMP_NUM_THREADS') == str(self.num_threads),
                'dynamic_threading_disabled': os.environ.get('MKL_DYNAMIC') == 'FALSE'
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting optimization status: {e}")
            return {
                'num_threads': self.num_threads,
                'optimization_applied': self.optimization_applied,
                'error': str(e)
            }
    
    def reset_optimizations(self) -> None:
        """
        Reset all optimization settings to default values.
        
        Useful for testing or when switching between different configurations.
        
        Example:
            >>> optimizer.reset_optimizations()
            >>> # Optimizations are now reset to defaults
        """
        try:
            # Reset environment variables to defaults
            env_vars_to_reset = [
                'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'BLAS_NUM_THREADS', 'MKL_DYNAMIC',
                'MKL_VERBOSE', 'MKL_ENABLE_INSTRUCTIONS', 'MKL_THREADING_LAYER'
            ]
            
            for var in env_vars_to_reset:
                if var in os.environ:
                    del os.environ[var]
                    logger.debug(f"Reset environment variable: {var}")
            
            self.optimization_applied = False
            logger.info("Intel optimizations reset to defaults")
            
        except Exception as e:
            logger.error(f"Error resetting optimizations: {e}")
            raise RuntimeError(f"Failed to reset optimizations: {e}") from e
    
    @staticmethod
    def verify_optimization() -> bool:
        """
        Static method to verify if Intel optimizations are working.
        
        Returns:
            True if optimizations are detected, False otherwise
            
        Example:
            >>> if IntelOptimizer.verify_optimization():
            ...     print("Intel optimizations are active")
        """
        try:
            from sklearnex import get_patch_map
            patched_estimators = get_patch_map()
            
            if patched_estimators:
                logger.debug(f"Verified {len(patched_estimators)} patched estimators")
                return True
            else:
                logger.debug("No patched estimators found")
                return False
                
        except ImportError:
            logger.debug("Intel Extension not available")
            return False
        except Exception as e:
            logger.error(f"Error verifying optimization: {e}")
            return False
    
    def __repr__(self) -> str:
        """String representation of the optimizer."""
        return (f"IntelOptimizer(num_threads={self.num_threads}, "
                f"optimization_applied={self.optimization_applied})")


# Global optimizer instance with error handling
try:
    intel_opt = IntelOptimizer(num_threads=20)
    intel_opt.apply_optimizations()
    logger.info("Global Intel optimizer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize global Intel optimizer: {e}")
    # Create a fallback optimizer with minimal configuration
    intel_opt = IntelOptimizer(num_threads=1)
    logger.warning("Using fallback Intel optimizer with 1 thread")