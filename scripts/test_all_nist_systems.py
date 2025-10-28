#!/usr/bin/env python3
"""
Test unified NIST data integration for all ceramic systems.
"""

import sys
sys.path.append('.')

import pandas as pd
from pathlib import Path
from loguru import logger
from data.data_collection.comprehensive_nist_loader import ComprehensiveNISTLoader


def test_individual_systems():
    """Test each ceramic system individually."""
    logger.info("🧪 Testing Individual Ceramic Systems")
    logger.info("=" * 50)
    
    ceramic_systems = ['Al2O3', 'SiC', 'B4C', 'WC', 'TiC']
    loader = ComprehensiveNISTLoader()
    
    results = {}
    
    for system in ceramic_systems:
        logger.info(f"\n--- Testing {system} ---")
        
        try:
            # Load data for this system
            df = loader.load_system(system, use_manual=True, use_scraping=False)
            results[system] = df
            
            if not df.empty:
                logger.info(f"✅ {system}: {len(df)} records loaded")
                
                # Show available properties
                property_cols = ['density', 'youngs_modulus', 'fracture_toughness', 
                               'vickers_hardness', 'bulk_modulus', 'shear_modulus']
                
                available_props = []
                for prop in property_cols:
                    if prop in df.columns:
                        non_null_count = df[prop].notna().sum()
                        if non_null_count > 0:
                            values = df[prop].dropna()
                            available_props.append(f"{prop}({non_null_count}): {values.min():.1f}-{values.max():.1f}")
                
                if available_props:
                    logger.info(f"   Properties: {', '.join(available_props[:3])}")  # Show first 3
                
                # Show sample record
                if len(df) > 0:
                    sample = df.iloc[0]
                    logger.info(f"   Sample: {sample.get('formula', 'N/A')} at {sample.get('temperature', 20)}°C")
                
            else:
                logger.warning(f"⚠️  {system}: No data found")
                
        except Exception as e:
            logger.error(f"❌ {system}: Failed - {e}")
            results[system] = pd.DataFrame()
    
    return results


def test_data_quality(results):
    """Test data quality across all systems."""
    logger.info("\n🔍 Testing Data Quality")
    logger.info("=" * 50)
    
    quality_issues = []
    
    for system, df in results.items():
        if df.empty:
            continue
            
        logger.info(f"\n--- {system} Quality Check ---")
        
        # Check required columns
        required_cols = ['formula', 'ceramic_system', 'source']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            issue = f"{system}: Missing required columns: {missing_cols}"
            quality_issues.append(issue)
            logger.warning(f"⚠️  {issue}")
        else:
            logger.info("✅ Required columns present")
        
        # Check data ranges
        range_checks = {
            'density': (1.0, 25.0),
            'youngs_modulus': (10, 1000),
            'fracture_toughness': (0.5, 20),
            'temperature': (-273, 3000)
        }
        
        for prop, (min_val, max_val) in range_checks.items():
            if prop in df.columns:
                values = df[prop].dropna()
                if len(values) > 0:
                    out_of_range = values[(values < min_val) | (values > max_val)]
                    if len(out_of_range) > 0:
                        issue = f"{system}: {prop} out of range: {len(out_of_range)} values"
                        quality_issues.append(issue)
                        logger.warning(f"⚠️  {issue}")
                    else:
                        logger.info(f"✅ {prop} values in valid range")
        
        # Check for duplicate records
        if len(df) > 1:
            # Check for exact duplicates
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                issue = f"{system}: {duplicates} duplicate records"
                quality_issues.append(issue)
                logger.warning(f"⚠️  {issue}")
            else:
                logger.info("✅ No duplicate records")
    
    if quality_issues:
        logger.warning(f"\n⚠️  Found {len(quality_issues)} quality issues")
        for issue in quality_issues:
            logger.warning(f"   - {issue}")
    else:
        logger.info("\n✅ All quality checks passed!")
    
    return len(quality_issues) == 0


def test_pipeline_compatibility(results):
    """Test compatibility with ML pipeline components."""
    logger.info("\n🔧 Testing Pipeline Compatibility")
    logger.info("=" * 50)
    
    compatibility_issues = []
    
    try:
        # Test compositional features
        from src.feature_engineering.compositional_features import CompositionalFeatureCalculator
        comp_calc = CompositionalFeatureCalculator()
        
        for system, df in results.items():
            if df.empty:
                continue
                
            logger.info(f"\n--- {system} Pipeline Test ---")
            
            try:
                # Test compositional features
                if 'formula' in df.columns:
                    enhanced_df = comp_calc.augment_dataframe(df.copy(), formula_col='formula')
                    
                    original_cols = len(df.columns)
                    enhanced_cols = len(enhanced_df.columns)
                    added_features = enhanced_cols - original_cols
                    
                    logger.info(f"✅ Compositional features: +{added_features} features")
                else:
                    issue = f"{system}: No formula column for compositional features"
                    compatibility_issues.append(issue)
                    logger.warning(f"⚠️  {issue}")
                
                # Test microstructure features
                from src.feature_engineering.microstructure_features import MicrostructureFeatureCalculator
                micro_calc = MicrostructureFeatureCalculator()
                
                final_df = micro_calc.add_features(enhanced_df if 'enhanced_df' in locals() else df.copy())
                
                if 'enhanced_df' in locals():
                    micro_features = len(final_df.columns) - len(enhanced_df.columns)
                    logger.info(f"✅ Microstructure features: +{micro_features} features")
                    logger.info(f"✅ Total features: {len(final_df.columns)}")
                
            except Exception as e:
                issue = f"{system}: Pipeline compatibility failed - {e}"
                compatibility_issues.append(issue)
                logger.error(f"❌ {issue}")
    
    except ImportError as e:
        logger.error(f"❌ Could not import pipeline components: {e}")
        return False
    
    if compatibility_issues:
        logger.warning(f"\n⚠️  Found {len(compatibility_issues)} compatibility issues")
        for issue in compatibility_issues:
            logger.warning(f"   - {issue}")
    else:
        logger.info("\n✅ All pipeline compatibility tests passed!")
    
    return len(compatibility_issues) == 0


def generate_unified_summary(results):
    """Generate a unified summary of all NIST data."""
    logger.info("\n📊 Unified NIST Data Summary")
    logger.info("=" * 50)
    
    total_records = 0
    total_properties = set()
    
    summary_data = []
    
    for system, df in results.items():
        record_count = len(df)
        total_records += record_count
        
        if not df.empty:
            # Count properties
            property_cols = ['density', 'youngs_modulus', 'fracture_toughness', 
                           'vickers_hardness', 'bulk_modulus', 'shear_modulus',
                           'compressive_strength', 'flexural_strength', 'poissons_ratio']
            
            system_props = []
            for prop in property_cols:
                if prop in df.columns and df[prop].notna().sum() > 0:
                    system_props.append(prop)
                    total_properties.add(prop)
            
            # Temperature range
            temp_range = "20°C"
            if 'temperature' in df.columns:
                temps = df['temperature'].dropna()
                if len(temps) > 1:
                    temp_range = f"{temps.min():.0f}-{temps.max():.0f}°C"
            
            summary_data.append({
                'System': system,
                'Records': record_count,
                'Properties': len(system_props),
                'Temperature Range': temp_range,
                'Key Properties': ', '.join(system_props[:3])  # Show first 3
            })
        else:
            summary_data.append({
                'System': system,
                'Records': 0,
                'Properties': 0,
                'Temperature Range': 'N/A',
                'Key Properties': 'None'
            })
    
    # Create summary table
    summary_df = pd.DataFrame(summary_data)
    
    logger.info("\nSystem Summary:")
    for _, row in summary_df.iterrows():
        status = "✅" if row['Records'] > 0 else "❌"
        logger.info(f"{row['System']:<8} {row['Records']:>4} records {status} ({row['Properties']} props, {row['Temperature Range']})")
    
    logger.info(f"\nOverall Summary:")
    logger.info(f"  Total systems: {len(results)}")
    logger.info(f"  Systems with data: {sum(1 for df in results.values() if not df.empty)}")
    logger.info(f"  Total records: {total_records}")
    logger.info(f"  Unique properties: {len(total_properties)}")
    logger.info(f"  Properties: {', '.join(sorted(total_properties))}")
    
    # Save summary
    summary_file = Path("data/raw/nist/unified_nist_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"\n📄 Summary saved: {summary_file}")
    
    return summary_df


def main():
    """Run comprehensive NIST data testing for all systems."""
    logger.info("🎯 Unified NIST Data Integration Test")
    logger.info("Testing all ceramic systems: Al2O3, SiC, B4C, WC, TiC")
    logger.info("=" * 60)
    
    # Test individual systems
    results = test_individual_systems()
    
    # Test data quality
    quality_ok = test_data_quality(results)
    
    # Test pipeline compatibility
    compatibility_ok = test_pipeline_compatibility(results)
    
    # Generate unified summary
    summary_df = generate_unified_summary(results)
    
    # Final assessment
    logger.info("\n" + "=" * 60)
    logger.info("FINAL ASSESSMENT")
    logger.info("=" * 60)
    
    systems_with_data = sum(1 for df in results.values() if not df.empty)
    total_records = sum(len(df) for df in results.values())
    
    logger.info(f"Systems with data: {systems_with_data}/5")
    logger.info(f"Total records: {total_records}")
    logger.info(f"Data quality: {'✅ PASSED' if quality_ok else '⚠️  ISSUES FOUND'}")
    logger.info(f"Pipeline compatibility: {'✅ PASSED' if compatibility_ok else '⚠️  ISSUES FOUND'}")
    
    if systems_with_data >= 3 and total_records >= 10:
        logger.info("\n🎉 UNIFIED NIST INTEGRATION SUCCESSFUL!")
        logger.info("Your ceramic data is ready for the ML pipeline!")
        logger.info("\nNext steps:")
        logger.info("1. Run full pipeline: python scripts/run_full_pipeline.py")
        logger.info("2. Your data will be automatically integrated with:")
        logger.info("   • Materials Project data")
        logger.info("   • AFLOW data")
        logger.info("   • JARVIS data")
        logger.info("   • Web-scraped NIST data")
        return 0
    else:
        logger.error("\n💥 INTEGRATION NEEDS ATTENTION")
        logger.error("Some systems may need manual data preparation")
        logger.info("Check the issues above and fix any problems")
        return 1


if __name__ == "__main__":
    sys.exit(main())