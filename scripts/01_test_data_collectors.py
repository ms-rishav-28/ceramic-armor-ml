#!/usr/bin/env python3
"""
Data Collector Testing Script for Ceramic Armor ML Pipeline.

Tests existing collectors, validates output schemas, checks for missing columns,
and measures API response times with existing implementations.

Requirements: 3.2 - Data collector testing for existing system
"""

import sys
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import yaml
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.data_collection.materials_project_collector import MaterialsProjectCollector
    from src.utils.logger import get_logger
    from src.utils.config_loader import load_config
    from src.utils.data_utils import validate_data_schema
except ImportError as e:
    logger.error(f"Failed to import project modules: {e}")
    sys.exit(1)


class DataCollectorTester:
    """Comprehensive testing for data collectors."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config = self._load_config()
        self.test_results = {}
        self.performance_metrics = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load project configuration."""
        try:
            config_path = self.project_root / 'config' / 'config.yaml'
            return load_config(str(config_path))
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def test_materials_project_collector(self) -> bool:
        """Test Materials Project collector implementation."""
        logger.info("🧪 Testing Materials Project Collector")
        
        try:
            # Initialize collector
            collector = MaterialsProjectCollector()
            
            # Test with small sample of SiC materials
            test_materials = ["SiC"]
            test_limit = 10  # Small sample for testing
            
            logger.info(f"  Testing collection of {test_limit} {test_materials[0]} materials...")
            
            start_time = time.time()
            
            # Collect test data
            data = collector.collect_data(
                ceramic_systems=test_materials,
                limit_per_system=test_limit
            )
            
            collection_time = time.time() - start_time
            
            if data is None or len(data) == 0:
                logger.error("  ❌ No data collected")
                return False
            
            logger.info(f"  ✅ Collected {len(data)} materials in {collection_time:.2f}s")
            
            # Validate data schema
            expected_columns = [
                'material_id', 'formula', 'crystal_system', 'space_group',
                'density', 'formation_energy_per_atom', 'energy_above_hull',
                'band_gap', 'elastic_tensor', 'bulk_modulus', 'shear_modulus'
            ]
            
            missing_columns = []
            for col in expected_columns:
                if col not in data.columns:
                    missing_columns.append(col)
            
            if missing_columns:
                logger.error(f"  ❌ Missing columns: {missing_columns}")
                self.test_results['materials_project_missing_columns'] = missing_columns
                return False
            
            logger.info("  ✅ All expected columns present")
            
            # Check data quality
            null_counts = data.isnull().sum()
            high_null_columns = null_counts[null_counts > len(data) * 0.5].index.tolist()
            
            if high_null_columns:
                logger.warning(f"  ⚠️  High null counts in: {high_null_columns}")
                self.test_results['materials_project_high_nulls'] = high_null_columns
            
            # Performance metrics
            avg_time_per_material = collection_time / len(data)
            self.performance_metrics['materials_project'] = {
                'total_time': collection_time,
                'materials_collected': len(data),
                'avg_time_per_material': avg_time_per_material,
                'materials_per_second': len(data) / collection_time
            }
            
            logger.info(f"  📊 Performance: {avg_time_per_material:.2f}s per material")
            
            # Test specific properties
            self._test_elastic_properties(data)
            self._test_formation_energies(data)
            
            # Save test data for inspection
            test_data_path = self.project_root / 'data' / 'raw' / 'test_materials_project.csv'
            data.to_csv(test_data_path, index=False)
            logger.info(f"  💾 Test data saved: {test_data_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Materials Project collector test failed: {e}")
            return False
    
    def _test_elastic_properties(self, data: pd.DataFrame) -> None:
        """Test elastic properties data quality."""
        logger.info("    Testing elastic properties...")
        
        if 'bulk_modulus' in data.columns:
            bulk_modulus = data['bulk_modulus'].dropna()
            if len(bulk_modulus) > 0:
                # Check reasonable ranges for SiC (should be ~200-400 GPa)
                reasonable_range = (50, 1000)  # GPa, broad range
                out_of_range = bulk_modulus[(bulk_modulus < reasonable_range[0]) | 
                                          (bulk_modulus > reasonable_range[1])]
                
                if len(out_of_range) > 0:
                    logger.warning(f"    ⚠️  {len(out_of_range)} bulk modulus values out of range")
                else:
                    logger.info(f"    ✅ Bulk modulus values in reasonable range")
                    logger.info(f"    📊 Bulk modulus: {bulk_modulus.mean():.1f} ± {bulk_modulus.std():.1f} GPa")
        
        if 'shear_modulus' in data.columns:
            shear_modulus = data['shear_modulus'].dropna()
            if len(shear_modulus) > 0:
                logger.info(f"    📊 Shear modulus: {shear_modulus.mean():.1f} ± {shear_modulus.std():.1f} GPa")
    
    def _test_formation_energies(self, data: pd.DataFrame) -> None:
        """Test formation energy data quality."""
        logger.info("    Testing formation energies...")
        
        if 'formation_energy_per_atom' in data.columns:
            formation_energy = data['formation_energy_per_atom'].dropna()
            if len(formation_energy) > 0:
                # Formation energies should be negative for stable compounds
                positive_energies = formation_energy[formation_energy > 0]
                
                if len(positive_energies) > len(formation_energy) * 0.1:
                    logger.warning(f"    ⚠️  {len(positive_energies)} materials with positive formation energy")
                
                logger.info(f"    📊 Formation energy: {formation_energy.mean():.3f} ± {formation_energy.std():.3f} eV/atom")
        
        if 'energy_above_hull' in data.columns:
            hull_energy = data['energy_above_hull'].dropna()
            if len(hull_energy) > 0:
                stable_materials = hull_energy[hull_energy < 0.05]  # eV/atom
                logger.info(f"    📊 Stable materials (E_hull < 0.05): {len(stable_materials)}/{len(hull_energy)}")
    
    def test_aflow_collector(self) -> bool:
        """Test AFLOW collector if available."""
        logger.info("🧪 Testing AFLOW Collector")
        
        try:
            # Try to import AFLOW collector
            from src.data_collection.aflow_collector import AFLOWCollector
            
            collector = AFLOWCollector()
            
            # Test with small sample
            start_time = time.time()
            data = collector.collect_data(ceramic_systems=["SiC"], limit_per_system=5)
            collection_time = time.time() - start_time
            
            if data is None or len(data) == 0:
                logger.warning("  ⚠️  No AFLOW data collected")
                return False
            
            logger.info(f"  ✅ Collected {len(data)} materials in {collection_time:.2f}s")
            
            # Performance metrics
            self.performance_metrics['aflow'] = {
                'total_time': collection_time,
                'materials_collected': len(data),
                'avg_time_per_material': collection_time / len(data)
            }
            
            return True
            
        except ImportError:
            logger.info("  ℹ️  AFLOW collector not implemented yet")
            return True  # Not a failure if not implemented
        except Exception as e:
            logger.error(f"  ❌ AFLOW collector test failed: {e}")
            return False
    
    def test_jarvis_collector(self) -> bool:
        """Test JARVIS collector if available."""
        logger.info("🧪 Testing JARVIS Collector")
        
        try:
            # Try to import JARVIS collector
            from src.data_collection.jarvis_collector import JARVISCollector
            
            collector = JARVISCollector()
            
            # Test with small sample
            start_time = time.time()
            data = collector.collect_data(ceramic_systems=["SiC"], limit_per_system=5)
            collection_time = time.time() - start_time
            
            if data is None or len(data) == 0:
                logger.warning("  ⚠️  No JARVIS data collected")
                return False
            
            logger.info(f"  ✅ Collected {len(data)} materials in {collection_time:.2f}s")
            
            # Performance metrics
            self.performance_metrics['jarvis'] = {
                'total_time': collection_time,
                'materials_collected': len(data),
                'avg_time_per_material': collection_time / len(data)
            }
            
            return True
            
        except ImportError:
            logger.info("  ℹ️  JARVIS collector not implemented yet")
            return True  # Not a failure if not implemented
        except Exception as e:
            logger.error(f"  ❌ JARVIS collector test failed: {e}")
            return False
    
    def test_nist_collector(self) -> bool:
        """Test NIST collector if available."""
        logger.info("🧪 Testing NIST Collector")
        
        try:
            # Try to import NIST collector
            from src.data_collection.nist_collector import NISTCollector
            
            collector = NISTCollector()
            
            # Test with small sample
            start_time = time.time()
            data = collector.collect_data(ceramic_systems=["SiC"], limit_per_system=5)
            collection_time = time.time() - start_time
            
            if data is None or len(data) == 0:
                logger.warning("  ⚠️  No NIST data collected")
                return False
            
            logger.info(f"  ✅ Collected {len(data)} materials in {collection_time:.2f}s")
            
            # Performance metrics
            self.performance_metrics['nist'] = {
                'total_time': collection_time,
                'materials_collected': len(data),
                'avg_time_per_material': collection_time / len(data)
            }
            
            return True
            
        except ImportError:
            logger.info("  ℹ️  NIST collector not implemented yet")
            return True  # Not a failure if not implemented
        except Exception as e:
            logger.error(f"  ❌ NIST collector test failed: {e}")
            return False
    
    def validate_preprocessing_compatibility(self) -> bool:
        """Check that collector outputs match preprocessing expectations."""
        logger.info("🔧 Validating Preprocessing Compatibility")
        
        # Load test data from Materials Project
        test_data_path = self.project_root / 'data' / 'raw' / 'test_materials_project.csv'
        
        if not test_data_path.exists():
            logger.error("  ❌ No test data available for validation")
            return False
        
        try:
            data = pd.read_csv(test_data_path)
            
            # Check compatibility with preprocessing modules
            from src.preprocessing.data_cleaner import DataCleaner
            from src.preprocessing.unit_standardizer import UnitStandardizer
            
            cleaner = DataCleaner()
            standardizer = UnitStandardizer()
            
            # Test data cleaning
            logger.info("  Testing data cleaning compatibility...")
            cleaned_data = cleaner.clean(data)
            
            if cleaned_data is None or len(cleaned_data) == 0:
                logger.error("  ❌ Data cleaning failed")
                return False
            
            logger.info(f"  ✅ Data cleaning: {len(data)} → {len(cleaned_data)} materials")
            
            # Test unit standardization
            logger.info("  Testing unit standardization compatibility...")
            standardized_data = standardizer.standardize(cleaned_data)
            
            if standardized_data is None:
                logger.error("  ❌ Unit standardization failed")
                return False
            
            logger.info("  ✅ Unit standardization successful")
            
            # Check for missing columns expected by preprocessing
            expected_preprocessing_columns = [
                'density', 'formation_energy_per_atom', 'bulk_modulus', 'shear_modulus'
            ]
            
            missing_for_preprocessing = []
            for col in expected_preprocessing_columns:
                if col not in standardized_data.columns:
                    missing_for_preprocessing.append(col)
            
            if missing_for_preprocessing:
                logger.warning(f"  ⚠️  Missing columns for preprocessing: {missing_for_preprocessing}")
                self.test_results['missing_for_preprocessing'] = missing_for_preprocessing
            else:
                logger.info("  ✅ All preprocessing columns available")
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Preprocessing compatibility test failed: {e}")
            return False
    
    def measure_api_response_times(self) -> Dict[str, float]:
        """Measure API response times for performance monitoring."""
        logger.info("⏱️  Measuring API Response Times")
        
        response_times = {}
        
        # Test Materials Project API response times
        try:
            from src.data_collection.materials_project_collector import MaterialsProjectCollector
            
            collector = MaterialsProjectCollector()
            
            # Multiple small requests to measure consistency
            times = []
            for i in range(5):
                start_time = time.time()
                data = collector.collect_data(ceramic_systems=["SiC"], limit_per_system=2)
                request_time = time.time() - start_time
                times.append(request_time)
                time.sleep(1)  # Respect rate limits
            
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            
            response_times['materials_project'] = {
                'avg_response_time': avg_time,
                'std_response_time': std_time,
                'min_time': min(times),
                'max_time': max(times)
            }
            
            logger.info(f"  📊 Materials Project: {avg_time:.2f} ± {std_time:.2f}s")
            
        except Exception as e:
            logger.error(f"  ❌ Materials Project timing failed: {e}")
        
        return response_times
    
    def generate_collector_report(self) -> Dict[str, Any]:
        """Generate comprehensive data collector test report."""
        logger.info("📊 Generating Data Collector Report")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'api_response_times': self.measure_api_response_times(),
            'recommendations': []
        }
        
        # Add recommendations based on results
        if 'materials_project' in self.performance_metrics:
            mp_perf = self.performance_metrics['materials_project']
            if mp_perf['avg_time_per_material'] > 2.0:
                report['recommendations'].append(
                    "Materials Project API is slow - consider implementing caching"
                )
        
        if 'materials_project_missing_columns' in self.test_results:
            report['recommendations'].append(
                "Materials Project collector missing expected columns - update implementation"
            )
        
        # Save report
        report_path = self.project_root / 'logs' / 'data_collector_test_report.yaml'
        try:
            with open(report_path, 'w') as f:
                yaml.dump(report, f, default_flow_style=False, indent=2)
            logger.info(f"✅ Report saved: {report_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")
        
        return report
    
    def run_tests(self) -> bool:
        """Run all data collector tests."""
        logger.info("🚀 Data Collector Testing Suite")
        logger.info("=" * 60)
        
        test_functions = [
            ("Materials Project Collector", self.test_materials_project_collector),
            ("AFLOW Collector", self.test_aflow_collector),
            ("JARVIS Collector", self.test_jarvis_collector),
            ("NIST Collector", self.test_nist_collector),
            ("Preprocessing Compatibility", self.validate_preprocessing_compatibility)
        ]
        
        results = []
        
        for test_name, test_func in test_functions:
            logger.info(f"\n📋 {test_name}")
            logger.info("-" * 40)
            
            try:
                success = test_func()
                results.append((test_name, success))
            except Exception as e:
                logger.error(f"❌ {test_name} crashed: {e}")
                results.append((test_name, False))
        
        # Generate report
        report = self.generate_collector_report()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("DATA COLLECTOR TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = 0
        critical_passed = 0
        
        for test_name, success in results:
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{test_name:<30} {status}")
            
            if success:
                passed += 1
                if "Materials Project" in test_name or "Preprocessing" in test_name:
                    critical_passed += 1
        
        logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
        logger.info(f"Critical tests: {critical_passed}/2 passed")
        
        if report.get('recommendations'):
            logger.info(f"\n💡 Recommendations:")
            for rec in report['recommendations']:
                logger.info(f"  • {rec}")
        
        if critical_passed >= 2:
            logger.info("\n🎉 Data collectors ready for pipeline!")
            logger.info("Next step: Run data quality inspection")
            return True
        else:
            logger.error("\n💥 Critical data collector tests failed!")
            logger.error("Fix issues before proceeding with pipeline.")
            return False


def main():
    """Main entry point."""
    tester = DataCollectorTester()
    success = tester.run_tests()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())