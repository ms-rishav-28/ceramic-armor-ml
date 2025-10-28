#!/usr/bin/env python3
"""
Test JARVIS integration specifically.
"""

import sys
sys.path.append('.')

from loguru import logger
from data.data_collection.jarvis_collector import JARVISCollector


def test_jarvis_integration():
    """Test JARVIS data collection."""
    logger.info("🧪 Testing JARVIS Integration")
    logger.info("=" * 50)
    
    try:
        # Test import
        logger.info("Testing jarvis-tools import...")
        from jarvis.db.figshare import data as jdata
        logger.info("✅ jarvis-tools imported successfully")
        
        # Initialize collector
        collector = JARVISCollector()
        logger.info("✅ JARVIS collector initialized")
        
        # Test data loading (this will download on first run)
        logger.info("Loading JARVIS dataset (may take 5-10 minutes on first run)...")
        logger.info("This downloads ~500MB of data and caches it locally")
        
        # Test with SiC (should have good coverage)
        logger.info("Collecting SiC data from JARVIS...")
        sic_data = collector.collect("SiC")
        
        if len(sic_data) > 0:
            logger.info(f"✅ Successfully collected {len(sic_data)} SiC records from JARVIS")
            
            # Show sample data
            logger.info("\n📊 Sample JARVIS data:")
            logger.info(f"Columns: {list(sic_data.columns)}")
            
            sample = sic_data.iloc[0]
            logger.info(f"Sample material: {sample.get('material_id', 'N/A')}")
            logger.info(f"Formula: {sample.get('formula', 'N/A')}")
            logger.info(f"Formation energy: {sample.get('formation_energy', 'N/A')} eV/atom")
            logger.info(f"Band gap: {sample.get('band_gap', 'N/A')} eV")
            logger.info(f"Density: {sample.get('density', 'N/A')} g/cm³")
            
            # Check data quality
            non_null_props = []
            for prop in ['formation_energy', 'band_gap', 'bulk_modulus', 'density']:
                if prop in sic_data.columns:
                    non_null_count = sic_data[prop].notna().sum()
                    if non_null_count > 0:
                        non_null_props.append(f"{prop}({non_null_count})")
            
            logger.info(f"✅ Available properties: {', '.join(non_null_props)}")
            
            return True
        else:
            logger.error("❌ No SiC data found in JARVIS")
            return False
            
    except ImportError:
        logger.error("❌ jarvis-tools not installed")
        logger.error("Install with: pip install jarvis-tools")
        return False
    except Exception as e:
        logger.error(f"❌ JARVIS test failed: {e}")
        return False


def test_all_ceramic_systems():
    """Test JARVIS collection for all ceramic systems."""
    logger.info("\n🔍 Testing all ceramic systems...")
    
    systems = ["SiC", "Al2O3", "B4C", "WC", "TiC"]
    collector = JARVISCollector()
    
    results = {}
    
    for system in systems:
        try:
            logger.info(f"Testing {system}...")
            data = collector.collect(system)
            results[system] = len(data)
            logger.info(f"✅ {system}: {len(data)} records")
        except Exception as e:
            logger.error(f"❌ {system} failed: {e}")
            results[system] = 0
    
    # Summary
    logger.info("\n📊 JARVIS Collection Summary:")
    total_records = 0
    for system, count in results.items():
        logger.info(f"  {system:<8}: {count:>4} records")
        total_records += count
    
    logger.info(f"  {'Total':<8}: {total_records:>4} records")
    
    return total_records > 0


def main():
    """Run JARVIS integration tests."""
    logger.info("🚀 JARVIS Integration Test")
    
    tests = [
        ("Basic Integration", test_jarvis_integration),
        ("All Ceramic Systems", test_all_ceramic_systems),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("JARVIS TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name:<25} {status}")
    
    if passed == len(results):
        logger.info("\n🎉 JARVIS integration working perfectly!")
        logger.info("Ready for full pipeline execution.")
        return 0
    else:
        logger.error(f"\n💥 {len(results) - passed} tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())