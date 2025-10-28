#!/usr/bin/env python3
"""
Minimal Test Runner - Complete Test Execution and Validation
Runs the minimal test pipeline and validates results
"""

import sys
sys.path.append('.')

import subprocess
import time
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_minimal_test_pipeline():
    """Run the minimal test pipeline"""
    logger.info("🚀 Starting Minimal Test Pipeline...")
    
    try:
        # Run the minimal test pipeline
        result = subprocess.run([
            sys.executable, 
            "scripts/minimal_test_pipeline.py"
        ], capture_output=True, text=True, timeout=2400)  # 40 minute timeout
        
        if result.returncode == 0:
            logger.info("✅ Minimal test pipeline completed successfully")
            return True
        else:
            logger.error(f"❌ Minimal test pipeline failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Minimal test pipeline timed out (40 minutes)")
        return False
    except Exception as e:
        logger.error(f"❌ Error running minimal test pipeline: {e}")
        return False


def run_validation():
    """Run the validation script"""
    logger.info("🔍 Starting Test Validation...")
    
    try:
        # Run the validation script
        result = subprocess.run([
            sys.executable, 
            "scripts/validate_minimal_test.py"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            logger.info("✅ Validation completed - PASSED")
            return 'PASSED'
        elif result.returncode == 1:
            logger.warning("⚠️  Validation completed - PARTIAL")
            return 'PARTIAL'
        else:
            logger.error("❌ Validation completed - FAILED")
            return 'FAILED'
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Validation timed out")
        return 'TIMEOUT'
    except Exception as e:
        logger.error(f"❌ Error running validation: {e}")
        return 'ERROR'


def main():
    """Run complete minimal test with validation"""
    logger.info("\n" + "🎯" * 30)
    logger.info("CERAMIC ARMOR ML PIPELINE - MINIMAL TEST RUNNER")
    logger.info("🎯" * 30)
    logger.info("This will run the complete minimal test pipeline and validate results")
    
    start_time = time.time()
    
    # Step 1: Run minimal test pipeline
    logger.info("\n" + "="*60)
    logger.info("STEP 1: RUNNING MINIMAL TEST PIPELINE")
    logger.info("="*60)
    
    pipeline_success = run_minimal_test_pipeline()
    
    if not pipeline_success:
        logger.error("❌ Pipeline execution failed - skipping validation")
        return 3
    
    # Step 2: Run validation
    logger.info("\n" + "="*60)
    logger.info("STEP 2: VALIDATING RESULTS")
    logger.info("="*60)
    
    validation_result = run_validation()
    
    # Final summary
    total_time = time.time() - start_time
    
    logger.info("\n" + "🏁" * 30)
    logger.info("MINIMAL TEST RUNNER COMPLETE")
    logger.info("🏁" * 30)
    logger.info(f"Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"Pipeline execution: {'SUCCESS' if pipeline_success else 'FAILED'}")
    logger.info(f"Validation result: {validation_result}")
    
    # Check if results are available
    test_dir = Path("data/test_pipeline")
    if (test_dir / "minimal_test_report.txt").exists():
        logger.info(f"📄 Test report: {test_dir / 'minimal_test_report.txt'}")
    
    if (test_dir / "validation_report.txt").exists():
        logger.info(f"📄 Validation report: {test_dir / 'validation_report.txt'}")
    
    # Time target check
    if total_time <= 1800:  # 30 minutes
        logger.info("✅ Completed within 30-minute target!")
    else:
        logger.warning(f"⚠️  Exceeded 30-minute target by {(total_time-1800)/60:.1f} minutes")
    
    # Return appropriate exit code
    if validation_result == 'PASSED':
        logger.info("🎉 MINIMAL TEST PASSED - Pipeline is healthy!")
        return 0
    elif validation_result == 'PARTIAL':
        logger.warning("⚠️  MINIMAL TEST PARTIAL - Some issues found")
        return 1
    else:
        logger.error("❌ MINIMAL TEST FAILED - Pipeline has issues")
        return 2


if __name__ == "__main__":
    sys.exit(main())