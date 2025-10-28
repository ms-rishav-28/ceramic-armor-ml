#!/usr/bin/env python3
"""
Test script for NIST web scraping functionality.
Run this to test the NIST scraper before running the full pipeline.
"""

import sys
sys.path.append('.')

import yaml
from pathlib import Path
from loguru import logger
from data.data_collection.nist_web_scraper import AdvancedNISTScraper, EnhancedNISTLoader


def test_single_system(ceramic_system: str = "SiC"):
    """Test scraping for a single ceramic system."""
    logger.info(f"🧪 Testing NIST scraping for {ceramic_system}")
    
    try:
        # Initialize scraper
        scraper = AdvancedNISTScraper()
        
        # Test scraping
        result_df = scraper.scrape_ceramic_system(ceramic_system)
        
        if not result_df.empty:
            logger.info(f"✅ Successfully scraped {len(result_df)} records")
            logger.info(f"Columns: {list(result_df.columns)}")
            
            # Show sample data
            if len(result_df) > 0:
                logger.info("Sample data:")
                print(result_df.head())
                
                # Show property statistics
                numeric_cols = result_df.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 0:
                    logger.info("Property statistics:")
                    print(result_df[numeric_cols].describe())
            
            return True
        else:
            logger.warning(f"⚠️  No data found for {ceramic_system}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}")
        return False


def test_enhanced_loader():
    """Test the enhanced NIST loader."""
    logger.info("🧪 Testing Enhanced NIST Loader")
    
    try:
        loader = EnhancedNISTLoader()
        
        # Test with different ceramic systems
        test_systems = ["SiC", "Al2O3", "B4C"]
        results = {}
        
        for system in test_systems:
            logger.info(f"Testing {system}...")
            df = loader.load_system(system, use_scraping=True)
            results[system] = df
            
            if not df.empty:
                logger.info(f"✅ {system}: {len(df)} records")
            else:
                logger.warning(f"⚠️  {system}: No data")
        
        # Generate summary
        total_records = sum(len(df) for df in results.values())
        successful_systems = sum(1 for df in results.values() if not df.empty)
        
        logger.info(f"\n📊 Summary:")
        logger.info(f"  Systems tested: {len(test_systems)}")
        logger.info(f"  Successful: {successful_systems}")
        logger.info(f"  Total records: {total_records}")
        
        return successful_systems > 0
        
    except Exception as e:
        logger.error(f"❌ Enhanced loader test failed: {e}")
        return False


def test_configuration():
    """Test that the scraping configuration is valid."""
    logger.info("🧪 Testing NIST scraping configuration")
    
    try:
        config_path = "config/nist_scraping_config.yaml"
        
        if not Path(config_path).exists():
            logger.error(f"❌ Configuration file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = [
            'ceramic_search_terms',
            'property_patterns', 
            'column_mappings',
            'scraping_settings'
        ]
        
        for section in required_sections:
            if section not in config:
                logger.error(f"❌ Missing configuration section: {section}")
                return False
            logger.info(f"✅ Found section: {section}")
        
        # Check ceramic systems
        ceramic_systems = list(config['ceramic_search_terms'].keys())
        logger.info(f"✅ Configured ceramic systems: {ceramic_systems}")
        
        # Check property patterns
        properties = list(config['property_patterns'].keys())
        logger.info(f"✅ Property extraction patterns: {properties}")
        
        logger.info("✅ Configuration is valid")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


def test_web_connectivity():
    """Test basic web connectivity to NIST sites."""
    logger.info("🧪 Testing web connectivity to NIST")
    
    import requests
    
    test_urls = [
        "https://webbook.nist.gov",
        "https://www.nist.gov",
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    connectivity_ok = True
    
    for url in test_urls:
        try:
            response = session.get(url, timeout=10)
            if response.status_code == 200:
                logger.info(f"✅ {url} - OK")
            else:
                logger.warning(f"⚠️  {url} - Status: {response.status_code}")
                connectivity_ok = False
        except Exception as e:
            logger.error(f"❌ {url} - Failed: {e}")
            connectivity_ok = False
    
    return connectivity_ok


def main():
    """Run all NIST scraping tests."""
    logger.info("🚀 NIST Web Scraping Test Suite")
    logger.info("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Web Connectivity", test_web_connectivity),
        ("Single System Scraping", lambda: test_single_system("SiC")),
        ("Enhanced Loader", test_enhanced_loader),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running: {test_name}")
        logger.info("-" * 30)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name:<25} {status}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! NIST scraping is ready.")
        return 0
    else:
        logger.error("💥 Some tests failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())