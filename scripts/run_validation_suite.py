#!/usr/bin/env python3
"""
Master Validation Suite for Ceramic Armor ML Pipeline.

Runs all validation scripts in sequence and provides comprehensive system validation.
This script orchestrates the complete validation workflow for the existing system.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5 - Complete validation suite
"""

import sys
import subprocess
import time
from pathlib import Path
from loguru import logger


def run_script(script_path: Path, script_name: str) -> bool:
    """Run a validation script and return success status."""
    logger.info(f"🚀 Running {script_name}...")
    logger.info("=" * 60)
    
    try:
        # Run the script
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=False, text=True, timeout=1800)  # 30 minute timeout
        
        if result.returncode == 0:
            logger.info(f"✅ {script_name} completed successfully")
            return True
        else:
            logger.error(f"❌ {script_name} failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {script_name} timed out after 30 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ {script_name} failed with error: {e}")
        return False


def main():
    """Run complete validation suite."""
    logger.info("🚀 Ceramic Armor ML Pipeline - Complete Validation Suite")
    logger.info("=" * 70)
    logger.info("This suite will validate all aspects of the existing system:")
    logger.info("1. Setup validation (dependencies, imports, API connectivity)")
    logger.info("2. Data collector testing (API functionality, schema validation)")
    logger.info("3. Data quality inspection (raw → processed → features)")
    logger.info("4. Training monitoring test (performance tracking)")
    logger.info("=" * 70)
    
    project_root = Path(__file__).parent.parent
    scripts_dir = project_root / 'scripts'
    
    # Define validation scripts in order
    validation_scripts = [
        (scripts_dir / '00_validate_setup.py', 'Setup Validation'),
        (scripts_dir / '01_test_data_collectors.py', 'Data Collector Testing'),
        (scripts_dir / '02_inspect_data_quality.py', 'Data Quality Inspection'),
        (scripts_dir / '03_monitor_training.py', 'Training Monitoring Test')
    ]
    
    results = []
    start_time = time.time()
    
    for script_path, script_name in validation_scripts:
        if not script_path.exists():
            logger.error(f"❌ Script not found: {script_path}")
            results.append((script_name, False))
            continue
        
        logger.info(f"\n{'='*20} {script_name} {'='*20}")
        success = run_script(script_path, script_name)
        results.append((script_name, success))
        
        if not success:
            logger.warning(f"⚠️  {script_name} failed - continuing with remaining validations")
        
        # Brief pause between scripts
        time.sleep(2)
    
    total_time = time.time() - start_time
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION SUITE SUMMARY")
    logger.info("=" * 70)
    
    passed = 0
    critical_passed = 0
    
    for script_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{script_name:<30} {status}")
        
        if success:
            passed += 1
            # Setup and data quality are critical
            if "Setup" in script_name or "Data Quality" in script_name:
                critical_passed += 1
    
    logger.info(f"\nOverall Results:")
    logger.info(f"  • Total validations: {len(results)}")
    logger.info(f"  • Passed: {passed}")
    logger.info(f"  • Failed: {len(results) - passed}")
    logger.info(f"  • Critical validations passed: {critical_passed}/2")
    logger.info(f"  • Total time: {total_time/60:.1f} minutes")
    
    # Determine overall status
    if passed == len(results):
        logger.info("\n🎉 ALL VALIDATIONS PASSED!")
        logger.info("The Ceramic Armor ML Pipeline is fully validated and ready for production use.")
        logger.info("\nNext steps:")
        logger.info("1. Run minimal test pipeline: python scripts/05_minimal_test.py")
        logger.info("2. Run full pipeline: python scripts/run_full_pipeline.py")
        logger.info("3. Check results in results/ directory")
        return_code = 0
        
    elif critical_passed >= 2:
        logger.info("\n✅ CRITICAL VALIDATIONS PASSED!")
        logger.info("Core system components are working. Some optional features may need attention.")
        logger.info("\nYou can proceed with:")
        logger.info("1. Minimal testing to verify core functionality")
        logger.info("2. Address non-critical issues as needed")
        return_code = 0
        
    else:
        logger.error("\n💥 CRITICAL VALIDATIONS FAILED!")
        logger.error("Core system issues must be resolved before proceeding.")
        logger.error("\nRequired actions:")
        logger.error("1. Fix setup issues (dependencies, imports, configuration)")
        logger.error("2. Resolve data quality problems")
        logger.error("3. Re-run validation suite")
        return_code = 1
    
    # Report locations
    logger.info(f"\n📊 Detailed reports available in:")
    logger.info(f"  • logs/setup_validation_report.yaml")
    logger.info(f"  • logs/data_collector_test_report.yaml")
    logger.info(f"  • logs/data_quality_report.yaml")
    logger.info(f"  • logs/training_monitoring_report.yaml")
    logger.info(f"  • results/figures/ (visualizations)")
    
    return return_code


if __name__ == "__main__":
    sys.exit(main())