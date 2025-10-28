#!/usr/bin/env python3
"""
Quick start test - validates your setup and runs a small data collection test.
"""

import sys
sys.path.append('.')

import yaml
import pandas as pd
from pathlib import Path
from loguru import logger
from src.data_collection.materials_project_collector import MaterialsProjectCollector


def test_materials_project_collection():
    """Test Materials Project data collection with your API key."""
    logger.info("🚀 Quick Start Test - Materials Project Collection")
    logger.info("=" * 60)
    
    # Load API key
    try:
        with open('config/api_keys.yaml', 'r') as f:
            api_keys = yaml.safe_load(f)
        
        mp_api_key = api_keys.get('materials_project')
        if not mp_api_key:
            logger.error("❌ Materials Project API key not found")
            return False
        
        logger.info(f"✅ Using API key: {mp_api_key[:8]}...")
        
    except Exception as e:
        logger.error(f"❌ Could not load API keys: {e}")
        return False
    
    # Test data collection
    try:
        logger.info("\n🔍 Testing Materials Project collector...")
        collector = MaterialsProjectCollector(mp_api_key)
        
        # Test with SiC (should have many results)
        logger.info("Collecting SiC data (limited to 10 records for testing)...")
        
        # Temporarily modify the collector to limit results for testing
        original_limit = getattr(collector, 'limit', 1000)
        collector.limit = 10  # Limit for quick test
        
        result_file = collector.collect_ceramic_data("SiC")
        
        if result_file and Path(result_file).exists():
            df = pd.read_csv(result_file)
            logger.info(f"✅ Successfully collected {len(df)} SiC records")
            logger.info(f"✅ Data saved to: {result_file}")
            
            # Show sample data
            logger.info("\n📊 Sample data:")
            logger.info(f"Columns: {list(df.columns)}")
            
            if len(df) > 0:
                sample = df.iloc[0]
                logger.info(f"Sample material: {sample.get('material_id', 'N/A')}")
                logger.info(f"Formula: {sample.get('formula_pretty', 'N/A')}")
                logger.info(f"Density: {sample.get('density', 'N/A')} g/cm³")
                logger.info(f"Formation energy: {sample.get('formation_energy_per_atom', 'N/A')} eV/atom")
            
            # Check for key properties
            key_properties = ['density', 'formation_energy_per_atom', 'band_gap', 'total_magnetization']
            available_props = [prop for prop in key_properties if prop in df.columns and not df[prop].isna().all()]
            logger.info(f"✅ Available key properties: {available_props}")
            
            return True
        else:
            logger.error("❌ Data collection failed - no file created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Data collection test failed: {e}")
        return False


def test_quick_feature_engineering():
    """Test basic feature engineering on collected data."""
    logger.info("\n🔧 Testing feature engineering...")
    
    try:
        # Look for the test data file
        test_file = Path("data/raw/materials_project/sic_raw.csv")
        
        if not test_file.exists():
            logger.warning("⚠️  No test data file found, skipping feature engineering test")
            return True
        
        df = pd.read_csv(test_file)
        logger.info(f"✅ Loaded {len(df)} records for feature engineering test")
        
        # Test compositional features
        from src.feature_engineering.compositional_features import CompositionalFeatureCalculator
        
        comp_calc = CompositionalFeatureCalculator()
        
        if 'formula_pretty' in df.columns:
            # Test on first few records
            test_df = df.head(3).copy()
            test_df = comp_calc.augment_dataframe(test_df, formula_col='formula_pretty')
            
            # Count new features
            original_cols = len(df.columns)
            new_cols = len(test_df.columns)
            added_features = new_cols - original_cols
            
            logger.info(f"✅ Feature engineering successful: added {added_features} compositional features")
            
            # Show some new features
            comp_features = [col for col in test_df.columns if col.startswith('comp_')]
            logger.info(f"Sample compositional features: {comp_features[:5]}")
            
            return True
        else:
            logger.warning("⚠️  No formula column found, skipping compositional features test")
            return True
            
    except Exception as e:
        logger.error(f"❌ Feature engineering test failed: {e}")
        return False


def main():
    """Run quick start tests."""
    logger.info("🎯 CERAMIC ARMOR ML PIPELINE - QUICK START TEST")
    logger.info("Testing your setup with real data collection...")
    logger.info("=" * 60)
    
    tests = [
        ("Materials Project Collection", test_materials_project_collection),
        ("Feature Engineering", test_quick_feature_engineering),
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
    logger.info("\n" + "=" * 60)
    logger.info("QUICK START TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name:<30} {status}")
        if success:
            passed += 1
    
    logger.info(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("\n🎉 QUICK START SUCCESSFUL!")
        logger.info("Your setup is working perfectly. Ready for full pipeline execution!")
        logger.info("\nNext steps:")
        logger.info("1. Run full pipeline: python scripts/run_full_pipeline.py")
        logger.info("2. Or test all APIs: python scripts/test_api_connectivity.py")
        return 0
    else:
        logger.error("\n💥 Some tests failed. Check the errors above.")
        logger.info("Try running: python scripts/test_api_connectivity.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())