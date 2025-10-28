#!/usr/bin/env python3
"""
Test API connectivity and validate API keys.
"""

import sys
sys.path.append('.')

import yaml
import requests
from pathlib import Path
from loguru import logger


def test_materials_project_api():
    """Test Materials Project API connectivity."""
    logger.info("🧪 Testing Materials Project API")
    
    # Load API key
    api_keys_path = 'config/api_keys.yaml'
    if not Path(api_keys_path).exists():
        logger.error(f"❌ API keys file not found: {api_keys_path}")
        return False
    
    with open(api_keys_path, 'r') as f:
        api_keys = yaml.safe_load(f)
    
    mp_api_key = api_keys.get('materials_project')
    if not mp_api_key:
        logger.error("❌ Materials Project API key not found in config")
        return False
    
    logger.info(f"✅ Found API key: {mp_api_key[:8]}...")
    
    # Test API connectivity
    try:
        # Test with a simple query for SiC
        url = "https://api.materialsproject.org/materials/summary"
        headers = {"X-API-KEY": mp_api_key}
        params = {
            "formula": "SiC",
            "_limit": 5,
            "_fields": "material_id,formula_pretty,density,formation_energy_per_atom"
        }
        
        logger.info("Making test API request...")
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                logger.info(f"✅ API test successful! Found {len(data['data'])} SiC materials")
                
                # Show sample data
                sample = data['data'][0]
                logger.info(f"Sample material: {sample.get('material_id')} - {sample.get('formula_pretty')}")
                logger.info(f"Density: {sample.get('density', 'N/A')} g/cm³")
                logger.info(f"Formation energy: {sample.get('formation_energy_per_atom', 'N/A')} eV/atom")
                
                return True
            else:
                logger.error("❌ API returned empty data")
                return False
        
        elif response.status_code == 401:
            logger.error("❌ API key authentication failed - check your API key")
            return False
        
        elif response.status_code == 429:
            logger.error("❌ API rate limit exceeded - try again later")
            return False
        
        else:
            logger.error(f"❌ API request failed with status {response.status_code}")
            logger.error(f"Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error("❌ API request timed out")
        return False
    except requests.exceptions.ConnectionError:
        logger.error("❌ Connection error - check internet connectivity")
        return False
    except Exception as e:
        logger.error(f"❌ API test failed: {e}")
        return False


def test_aflow_api():
    """Test AFLOW API connectivity."""
    logger.info("🧪 Testing AFLOW API")
    
    try:
        # Test AFLOW AFLUX API
        url = "https://aflowlib.duke.edu/search/API/"
        params = {
            "species": "Si,C",
            "nspecies": "2",
            "format": "json",
            "paging": "0"
        }
        
        logger.info("Making AFLOW API test request...")
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            try:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    logger.info(f"✅ AFLOW API test successful! Found {len(data)} SiC entries")
                    return True
                else:
                    logger.warning("⚠️  AFLOW API returned empty or invalid data")
                    return False
            except:
                # AFLOW might return non-JSON data sometimes
                if len(response.text) > 100:
                    logger.info("✅ AFLOW API responding (non-JSON format)")
                    return True
                else:
                    logger.error("❌ AFLOW API returned invalid data")
                    return False
        else:
            logger.error(f"❌ AFLOW API request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ AFLOW API test failed: {e}")
        return False


def test_jarvis_connectivity():
    """Test JARVIS data connectivity."""
    logger.info("🧪 Testing JARVIS connectivity")
    
    try:
        # Test JARVIS import
        from jarvis.db.figshare import data as jdata
        logger.info("✅ JARVIS-tools import successful")
        
        # Test data loading (this might take a moment)
        logger.info("Testing JARVIS data loading...")
        try:
            # Load a small sample
            records = jdata("dft_3d")
            if records and len(records) > 0:
                logger.info(f"✅ JARVIS data loaded successfully! {len(records)} total records")
                
                # Find SiC examples
                sic_count = sum(1 for r in records[:1000] if 'Si' in str(r.get('formula', '')) and 'C' in str(r.get('formula', '')))
                logger.info(f"Found ~{sic_count} SiC-related entries in first 1000 records")
                
                return True
            else:
                logger.error("❌ JARVIS data loading returned empty results")
                return False
                
        except Exception as e:
            logger.error(f"❌ JARVIS data loading failed: {e}")
            return False
            
    except ImportError:
        logger.error("❌ JARVIS-tools not installed. Install with: pip install jarvis-tools")
        return False
    except Exception as e:
        logger.error(f"❌ JARVIS test failed: {e}")
        return False


def test_nist_connectivity():
    """Test NIST website connectivity."""
    logger.info("🧪 Testing NIST connectivity")
    
    test_urls = [
        "https://webbook.nist.gov/",
        "https://www.nist.gov/mml/acmd"
    ]
    
    success_count = 0
    
    for url in test_urls:
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                logger.info(f"✅ {url} - accessible")
                success_count += 1
            else:
                logger.warning(f"⚠️  {url} - status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ {url} - failed: {e}")
    
    if success_count > 0:
        logger.info(f"✅ NIST connectivity: {success_count}/{len(test_urls)} sites accessible")
        return True
    else:
        logger.error("❌ No NIST sites accessible")
        return False


def main():
    """Run all API connectivity tests."""
    logger.info("🚀 API Connectivity Test Suite")
    logger.info("=" * 50)
    
    tests = [
        ("Materials Project API", test_materials_project_api),
        ("AFLOW API", test_aflow_api),
        ("JARVIS Connectivity", test_jarvis_connectivity),
        ("NIST Connectivity", test_nist_connectivity),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 {test_name}")
        logger.info("-" * 30)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("API CONNECTIVITY SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    critical_passed = 0
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name:<25} {status}")
        
        if success:
            passed += 1
            if test_name in ["Materials Project API", "AFLOW API", "JARVIS Connectivity"]:
                critical_passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
    logger.info(f"Critical APIs: {critical_passed}/3 passed")
    
    if critical_passed >= 2:
        logger.info("🎉 Sufficient APIs available! Pipeline ready to run.")
        return 0
    else:
        logger.error("💥 Too many critical APIs failed. Check connectivity and API keys.")
        return 1


if __name__ == "__main__":
    sys.exit(main())