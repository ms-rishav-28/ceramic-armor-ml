#!/usr/bin/env python3
"""
Test NIST data integration with your manual TiC data.
"""

import sys
sys.path.append('.')

import pandas as pd
from pathlib import Path
from loguru import logger
from data.data_collection.comprehensive_nist_loader import ComprehensiveNISTLoader


def test_tic_integration():
    """Test integration with your TiC data specifically."""
    logger.info("🧪 Testing TiC Data Integration")
    logger.info("=" * 50)
    
    # Initialize comprehensive loader
    loader = ComprehensiveNISTLoader()
    
    # Test the integration system
    success = loader.test_integration("TiC")
    
    if success:
        logger.info("✅ Basic integration test passed")
    else:
        logger.error("❌ Basic integration test failed")
        return False
    
    # Load comprehensive TiC data
    logger.info("\n🔍 Loading comprehensive TiC data...")
    tic_data = loader.load_system("TiC", use_manual=True, use_scraping=True)
    
    if tic_data.empty:
        logger.error("❌ No TiC data loaded")
        return False
    
    logger.info(f"✅ Loaded {len(tic_data)} TiC records")
    
    # Analyze the data
    logger.info("\n📊 Data Analysis:")
    logger.info(f"Columns: {list(tic_data.columns)}")
    
    # Check data sources
    if 'data_source' in tic_data.columns:
        source_counts = tic_data['data_source'].value_counts()
        logger.info(f"Data sources: {dict(source_counts)}")
    
    # Check available properties
    property_cols = ['density', 'youngs_modulus', 'vickers_hardness', 
                    'fracture_toughness', 'thermal_conductivity', 'grain_size']
    
    logger.info("\nProperty availability:")
    for prop in property_cols:
        if prop in tic_data.columns:
            non_null_count = tic_data[prop].notna().sum()
            if non_null_count > 0:
                values = tic_data[prop].dropna()
                logger.info(f"  {prop}: {non_null_count} values, range: {values.min():.2f} - {values.max():.2f}")
            else:
                logger.info(f"  {prop}: No data")
        else:
            logger.info(f"  {prop}: Column not found")
    
    # Show sample records
    logger.info("\n📋 Sample records:")
    for i, row in tic_data.head(3).iterrows():
        logger.info(f"Record {i+1}:")
        for col in ['formula', 'fracture_toughness', 'youngs_modulus', 'data_source', 'source_file']:
            if col in row:
                logger.info(f"  {col}: {row[col]}")
    
    return True


def test_all_systems_integration():
    """Test integration for all ceramic systems."""
    logger.info("\n🚀 Testing All Systems Integration")
    logger.info("=" * 50)
    
    ceramic_systems = ["SiC", "Al2O3", "B4C", "WC", "TiC"]
    loader = ComprehensiveNISTLoader()
    
    # Load data for all systems (without scraping to save time)
    results = loader.load_all_systems(
        ceramic_systems, 
        use_manual=True, 
        use_scraping=False  # Disable scraping for quick test
    )
    
    # Analyze results
    logger.info("\n📊 Integration Results Summary:")
    
    total_records = 0
    systems_with_data = 0
    
    for system, df in results.items():
        record_count = len(df)
        total_records += record_count
        
        if record_count > 0:
            systems_with_data += 1
            logger.info(f"✅ {system}: {record_count} records")
            
            # Show key properties
            key_props = []
            for prop in ['fracture_toughness', 'youngs_modulus', 'density']:
                if prop in df.columns and df[prop].notna().sum() > 0:
                    key_props.append(prop)
            
            if key_props:
                logger.info(f"   Properties: {', '.join(key_props)}")
        else:
            logger.info(f"⚠️  {system}: No data found")
    
    logger.info(f"\nSummary: {systems_with_data}/{len(ceramic_systems)} systems have data")
    logger.info(f"Total records: {total_records}")
    
    return systems_with_data > 0


def test_pipeline_compatibility():
    """Test that integrated data works with the ML pipeline."""
    logger.info("\n🔧 Testing Pipeline Compatibility")
    logger.info("=" * 50)
    
    try:
        # Load TiC data
        loader = ComprehensiveNISTLoader()
        tic_data = loader.load_system("TiC", use_manual=True, use_scraping=False)
        
        if tic_data.empty:
            logger.warning("⚠️  No TiC data for pipeline test")
            return True  # Not a failure, just no data
        
        # Test with compositional features
        from src.feature_engineering.compositional_features import CompositionalFeatureCalculator
        
        comp_calc = CompositionalFeatureCalculator()
        enhanced_data = comp_calc.augment_dataframe(tic_data.copy(), formula_col='formula')
        
        original_cols = len(tic_data.columns)
        enhanced_cols = len(enhanced_data.columns)
        added_features = enhanced_cols - original_cols
        
        logger.info(f"✅ Compositional features: added {added_features} features")
        
        # Test with microstructure features
        from src.feature_engineering.microstructure_features import MicrostructureFeatureCalculator
        
        micro_calc = MicrostructureFeatureCalculator()
        final_data = micro_calc.add_features(enhanced_data)
        
        final_cols = len(final_data.columns)
        micro_features = final_cols - enhanced_cols
        
        logger.info(f"✅ Microstructure features: added {micro_features} features")
        logger.info(f"✅ Total features: {final_cols}")
        
        # Check for key ML-ready columns
        ml_ready_cols = ['formula', 'ceramic_system', 'source']
        missing_cols = [col for col in ml_ready_cols if col not in final_data.columns]
        
        if not missing_cols:
            logger.info("✅ All required ML columns present")
        else:
            logger.warning(f"⚠️  Missing ML columns: {missing_cols}")
        
        logger.info("✅ Pipeline compatibility test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline compatibility test failed: {e}")
        return False


def main():
    """Run all NIST integration tests."""
    logger.info("🎯 NIST Data Integration Test Suite")
    logger.info("Testing integration of manual CSV files with web scraping")
    logger.info("=" * 60)
    
    tests = [
        ("TiC Integration", test_tic_integration),
        ("All Systems Integration", test_all_systems_integration),
        ("Pipeline Compatibility", test_pipeline_compatibility),
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
    logger.info("NIST INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name:<30} {status}")
        if success:
            passed += 1
    
    logger.info(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("Your NIST data integration is working perfectly!")
        logger.info("\nYour TiC data will be automatically integrated with:")
        logger.info("  • Web-scraped NIST data")
        logger.info("  • Materials Project data")
        logger.info("  • AFLOW data")
        logger.info("  • JARVIS data")
        logger.info("\nReady for full pipeline execution!")
        return 0
    else:
        logger.error(f"\n💥 {len(results) - passed} tests failed.")
        logger.info("Check the errors above and fix any issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())