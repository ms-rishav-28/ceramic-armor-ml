"""
Intel CPU Optimization Module
Configures Intel MKL and scikit-learn-intelex for i7-12700K
"""

import os
import warnings
from loguru import logger

class IntelOptimizer:
    """Configure Intel optimizations for maximum CPU performance"""
    
    def __init__(self, num_threads=20):
        """
        Initialize Intel optimizations
        
        Args:
            num_threads: Number of threads (20 for i7-12700K)
        """
        self.num_threads = num_threads
        self.optimization_applied = False
    
    def apply_optimizations(self):
        """Apply all Intel optimizations"""
        logger.info("Applying Intel CPU optimizations...")
        
        # Set environment variables for threading
        os.environ['OMP_NUM_THREADS'] = str(self.num_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.num_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.num_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.num_threads)
        
        # Intel MKL optimizations
        os.environ['MKL_DYNAMIC'] = 'FALSE'
        os.environ['MKL_VERBOSE'] = '0'
        
        # Patch scikit-learn with Intel Extension
        try:
            from sklearnex import patch_sklearn, config_context
            patch_sklearn()
            logger.info("✓ Intel Extension for Scikit-learn applied")
            self.optimization_applied = True
        except ImportError:
            logger.warning("Intel Extension not found. Install: pip install scikit-learn-intelex")
            self.optimization_applied = False
        
        logger.info(f"✓ Thread configuration: {self.num_threads} threads")
        logger.info("✓ Intel MKL optimizations enabled")
        
        return self.optimization_applied
    
    def get_optimization_status(self):
        """Get current optimization status"""
        status = {
            'num_threads': self.num_threads,
            'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
            'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS'),
            'optimization_applied': self.optimization_applied
        }
        return status
    
    @staticmethod
    def verify_optimization():
        """Verify optimizations are working"""
        try:
            from sklearnex import get_patch_map
            patched_estimators = get_patch_map()
            logger.info(f"Patched estimators: {list(patched_estimators.keys())[:5]}...")
            return True
        except:
            return False


# Global optimizer instance
intel_opt = IntelOptimizer(num_threads=20)
intel_opt.apply_optimizations()