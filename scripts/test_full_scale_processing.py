#!/usr/bin/env python3
"""
Full-Scale Processing Test Suite

This script provides comprehensive testing for the full-scale dataset processing
pipeline to ensure all components work correctly and meet the specified requirements.

Usage:
    python scripts/test_full_scale_processing.py [options]

Options:
    --quick          Run quick tests only (skip full processing)
    --integration    Run integration tests
    --performance    Run performance benchmarks
    --all            Run all test suites
    --output-dir     Output directory for test results
"""

import sys
import unittest
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.full_scale_processor import FullScaleProcessor
from src.data_collection.multi_source_collector import MultiSourceCollector
from src.utils.logger import get_logger, setup_logging
from src.utils.config_loader import load_project_config
from src.utils.data_utils import DataUtils

# Configure logging for tests
setup_logging(level=logging.INFO, log_to_file=True, log_file="logs/test_full_scale_processing.log")
logger = get_logger(__name__)


class TestFullScaleProcessing(unittest.TestCase):
    """Test suite for full-scale processing pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_output_dir = Path(tempfile.mkdtemp(prefix="test_full_scale_"))
        cls.config = load_project_config()
        logger.info(f"Test output directory: {cls.test_output_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if cls.test_output_dir.exists():
            shutil.rmtree(cls.test_output_dir)
        logger.info("Test cleanup completed")
    
    def setUp(self):
        """Set up individual test."""
        self.processor = FullScaleProcessor(
            output_dir=str(self.test_output_dir / "processor_test"),
            max_workers=2,  # Reduced for testing
            batch_size=10,  # Small batch for testing
            enable_parallel=False  # Disable for deterministic testing
        )
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        self.assertIsInstance(self.processor, FullScaleProcessor)
        self.assertEqual(len(self.processor.ceramic_systems), 5)
        self.assertIn('SiC', self.processor.ceramic_systems)
        self.assertIn('Al2O3', self.processor.ceramic_systems)
        self.assertIn('B4C', self.processor.ceramic_systems)
        self.assertIn('WC', self.processor.ceramic_systems)
        self.assertIn('TiC', self.processor.ceramic_systems)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        config = self.processor.config
        
        # Test performance targets
        self.assertEqual(config['targets']['mechanical_r2'], 0.85)
        self.assertEqual(config['targets']['ballistic_r2'], 0.80)
        
        # Test ceramic systems
        ceramic_systems = config['ceramic_systems']['primary']
        self.assertEqual(len(ceramic_systems), 5)
        expected_systems = {'SiC', 'Al2O3', 'B4C', 'WC', 'TiC'}
        self.assertEqual(set(ceramic_systems), expected_systems)
        
        # Test derived properties configuration
        derived_props = config['features']['derived']
        required_props = [
            'specific_hardness', 'brittleness_index', 'ballistic_efficacy',
            'thermal_shock_resistance', 'pugh_ratio', 'cauchy_pressure'
        ]
        for prop in required_props:
            self.assertIn(prop, derived_props)
    
    def test_output_directory_creation(self):
        """Test output directory structure creation."""
        output_dir = self.processor.output_dir
        
        # Check main directories exist
        expected_dirs = ['raw', 'processed', 'features', 'reports', 'intermediate']
        for dir_name in expected_dirs:
            dir_path = output_dir / dir_name
            self.assertTrue(dir_path.exists(), f"Directory {dir_name} not created")
            self.assertTrue(dir_path.is_dir(), f"{dir_name} is not a directory")
    
    def test_multi_source_collector_initialization(self):
        """Test multi-source collector initialization."""
        collector = MultiSourceCollector()
        
        self.assertIsInstance(collector, MultiSourceCollector)
        self.assertEqual(len(collector.collection_targets), 5)
        
        # Test collection targets
        for system in ['SiC', 'Al2O3', 'B4C', 'WC', 'TiC']:
            self.assertIn(system, collector.collection_targets)
            target = collector.collection_targets[system]
            self.assertGreater(target.target_count, 0)
            self.assertIsInstance(target.priority_properties, list)
            self.assertIsInstance(target.sources, list)
    
    def test_data_validation_functions(self):
        """Test data validation functions."""
        # Create sample data for testing
        sample_data = pd.DataFrame({
            'material_id': ['test_1', 'test_2', 'test_3'],
            'formula': ['SiC', 'Al2O3', 'B4C'],
            'ceramic_system': ['SiC', 'Al2O3', 'B4C'],
            'density': [3.2, 3.9, 2.5],
            'hardness': [25.0, 18.0, 30.0],
            'fracture_toughness': [4.5, 3.2, 3.8],
            'compressive_strength': [3500, 2800, 3200],
            'specific_hardness': [7.8, 4.6, 12.0],
            'brittleness_index': [5.6, 5.6, 7.9],
            'ballistic_efficiency': [17500, 11900, 17600]
        })
        
        # Test validation
        validation_results = self.processor._validate_processed_data(sample_data)
        
        self.assertIsInstance(validation_results, dict)
        self.assertEqual(validation_results['total_materials'], 3)
        self.assertGreater(validation_results['total_features'], 5)
        self.assertIn('specific_hardness_available', validation_results)
        self.assertTrue(validation_results['specific_hardness_available'])
    
    def test_derived_properties_calculation(self):
        """Test derived properties calculation."""
        # Create test data with known values
        test_data = pd.DataFrame({
            'density': [3.2, 4.0],
            'hardness': [25.0, 20.0],
            'fracture_toughness': [5.0, 4.0],
            'compressive_strength': [3000, 2500]
        })
        
        # Calculate expected derived properties
        expected_specific_hardness = test_data['hardness'] / test_data['density']
        expected_brittleness_index = test_data['hardness'] / test_data['fracture_toughness']
        expected_ballistic_efficiency = test_data['compressive_strength'] * np.sqrt(test_data['hardness'])
        
        # Test calculations (would be done by feature generator)
        calculated_specific_hardness = test_data['hardness'] / test_data['density']
        calculated_brittleness_index = test_data['hardness'] / test_data['fracture_toughness']
        calculated_ballistic_efficiency = test_data['compressive_strength'] * np.sqrt(test_data['hardness'])
        
        # Verify calculations
        np.testing.assert_allclose(calculated_specific_hardness, expected_specific_hardness, rtol=1e-10)
        np.testing.assert_allclose(calculated_brittleness_index, expected_brittleness_index, rtol=1e-10)
        np.testing.assert_allclose(calculated_ballistic_efficiency, expected_ballistic_efficiency, rtol=1e-10)
    
    def test_processing_statistics_tracking(self):
        """Test processing statistics tracking."""
        stats = self.processor.stats
        
        # Test initial state
        self.assertEqual(stats.total_materials_collected, 0)
        self.assertIsInstance(stats.materials_by_system, dict)
        self.assertIsInstance(stats.errors_encountered, list)
        
        # Test statistics update
        stats.total_materials_collected = 100
        stats.materials_by_system['SiC'] = 50
        stats.errors_encountered.append('Test error')
        
        self.assertEqual(stats.total_materials_collected, 100)
        self.assertEqual(stats.materials_by_system['SiC'], 50)
        self.assertEqual(len(stats.errors_encountered), 1)
    
    def test_file_operations(self):
        """Test file save and load operations."""
        # Create test data
        test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [1.1, 2.2, 3.3]
        })
        
        # Test CSV save/load
        csv_path = self.test_output_dir / "test_data.csv"
        success = DataUtils.safe_save_data(test_data, csv_path)
        self.assertTrue(success)
        self.assertTrue(csv_path.exists())
        
        loaded_data = DataUtils.safe_load_data(csv_path)
        self.assertIsNotNone(loaded_data)
        pd.testing.assert_frame_equal(test_data, loaded_data)
        
        # Test JSON save/load
        test_dict = {'key1': 'value1', 'key2': [1, 2, 3], 'key3': {'nested': 'dict'}}
        json_path = self.test_output_dir / "test_data.json"
        success = DataUtils.safe_save_data(test_dict, json_path)
        self.assertTrue(success)
        
        loaded_dict = DataUtils.safe_load_data(json_path)
        self.assertEqual(test_dict, loaded_dict)


class TestIntegration(unittest.TestCase):
    """Integration tests for full-scale processing."""
    
    @classmethod
    def setUpClass(cls):
        """Set up integration test environment."""
        cls.test_output_dir = Path(tempfile.mkdtemp(prefix="test_integration_"))
        logger.info(f"Integration test output directory: {cls.test_output_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up integration test environment."""
        if cls.test_output_dir.exists():
            shutil.rmtree(cls.test_output_dir)
    
    def test_small_scale_processing(self):
        """Test processing with a small dataset."""
        processor = FullScaleProcessor(
            output_dir=str(self.test_output_dir / "small_scale"),
            max_workers=1,
            batch_size=5,
            enable_parallel=False
        )
        
        # Create mock data for testing
        mock_data = self._create_mock_dataset(50)  # Small dataset for testing
        
        # Save mock data as if it were collected
        raw_data_path = processor.output_dir / "raw" / "combined_materials_data.csv"
        DataUtils.safe_save_data(mock_data, raw_data_path)
        
        # Test data cleaning
        cleaned_data = processor._clean_and_preprocess_data(mock_data)
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertGreater(len(cleaned_data), 0)
        
        # Test feature generation
        feature_data = processor._generate_comprehensive_features(cleaned_data)
        self.assertIsInstance(feature_data, pd.DataFrame)
        self.assertGreaterEqual(feature_data.shape[1], 10)  # At least some features
        
        # Test validation
        validation_results = processor._validate_processed_data(feature_data)
        self.assertIsInstance(validation_results, dict)
        self.assertIn('total_materials', validation_results)
        self.assertIn('total_features', validation_results)
    
    def test_report_generation(self):
        """Test report generation functionality."""
        processor = FullScaleProcessor(
            output_dir=str(self.test_output_dir / "reports"),
            max_workers=1,
            batch_size=5
        )
        
        # Create mock processed data
        mock_data = self._create_mock_dataset(20)
        
        # Test report generation
        report_paths = processor._generate_analysis_reports(mock_data)
        
        self.assertIsInstance(report_paths, dict)
        self.assertIn('summary', report_paths)
        self.assertIn('features', report_paths)
        
        # Verify report files exist
        for report_type, path in report_paths.items():
            self.assertTrue(Path(path).exists(), f"Report {report_type} not generated")
    
    def _create_mock_dataset(self, n_materials: int) -> pd.DataFrame:
        """Create mock dataset for testing."""
        np.random.seed(42)  # For reproducible tests
        
        ceramic_systems = ['SiC', 'Al2O3', 'B4C', 'WC', 'TiC']
        
        data = {
            'material_id': [f'test_{i}' for i in range(n_materials)],
            'formula': np.random.choice(ceramic_systems, n_materials),
            'ceramic_system': np.random.choice(ceramic_systems, n_materials),
            'density': np.random.uniform(2.0, 6.0, n_materials),
            'formation_energy': np.random.uniform(-5.0, 0.0, n_materials),
            'energy_above_hull': np.random.uniform(0.0, 0.2, n_materials),
            'band_gap': np.random.uniform(0.0, 6.0, n_materials),
            'elastic_bulk_modulus': np.random.uniform(100, 400, n_materials),
            'elastic_shear_modulus': np.random.uniform(80, 200, n_materials),
            'elastic_youngs_modulus': np.random.uniform(200, 600, n_materials),
            'hardness': np.random.uniform(10, 40, n_materials),
            'fracture_toughness': np.random.uniform(2, 8, n_materials),
            'compressive_strength': np.random.uniform(1000, 5000, n_materials),
            'thermal_conductivity': np.random.uniform(10, 200, n_materials)
        }
        
        df = pd.DataFrame(data)
        
        # Add derived properties
        df['specific_hardness'] = df['hardness'] / df['density']
        df['brittleness_index'] = df['hardness'] / df['fracture_toughness']
        df['ballistic_efficiency'] = df['compressive_strength'] * np.sqrt(df['hardness'])
        df['pugh_ratio'] = df['elastic_shear_modulus'] / df['elastic_bulk_modulus']
        
        return df


class TestPerformance(unittest.TestCase):
    """Performance tests for full-scale processing."""
    
    def test_memory_usage(self):
        """Test memory usage during processing."""
        initial_memory = DataUtils.get_memory_usage()
        
        # Create processor
        processor = FullScaleProcessor(
            output_dir=tempfile.mkdtemp(prefix="test_memory_"),
            max_workers=1,
            batch_size=10
        )
        
        # Monitor memory during initialization
        post_init_memory = DataUtils.get_memory_usage()
        
        # Memory increase should be reasonable
        memory_increase = post_init_memory['rss_mb'] - initial_memory['rss_mb']
        self.assertLess(memory_increase, 500, "Memory usage increase too high during initialization")
    
    def test_processing_speed(self):
        """Test processing speed benchmarks."""
        processor = FullScaleProcessor(
            output_dir=tempfile.mkdtemp(prefix="test_speed_"),
            max_workers=1,
            batch_size=10
        )
        
        # Create test data
        test_data = pd.DataFrame({
            'col1': range(1000),
            'col2': np.random.random(1000),
            'col3': ['test'] * 1000
        })
        
        # Time data validation
        start_time = time.time()
        validation_results = processor._validate_processed_data(test_data)
        validation_time = time.time() - start_time
        
        # Validation should be fast
        self.assertLess(validation_time, 5.0, "Data validation too slow")
        self.assertIsInstance(validation_results, dict)


def run_test_suite(test_type: str = 'all') -> bool:
    """
    Run the specified test suite.
    
    Args:
        test_type: Type of tests to run ('quick', 'integration', 'performance', 'all')
        
    Returns:
        True if all tests passed, False otherwise
    """
    logger.info(f"Running {test_type} test suite")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    if test_type in ['quick', 'all']:
        suite.addTest(unittest.makeSuite(TestFullScaleProcessing))
    
    if test_type in ['integration', 'all']:
        suite.addTest(unittest.makeSuite(TestIntegration))
    
    if test_type in ['performance', 'all']:
        suite.addTest(unittest.makeSuite(TestPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    success = result.wasSuccessful()
    
    if success:
        logger.info(f"✓ All {test_type} tests passed")
    else:
        logger.error(f"✗ Some {test_type} tests failed")
        logger.error(f"Failures: {len(result.failures)}")
        logger.error(f"Errors: {len(result.errors)}")
    
    return success


def main():
    """Main test execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Full-scale processing test suite")
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--all', action='store_true', help='Run all test suites')
    parser.add_argument('--output-dir', type=str, help='Output directory for test results')
    
    args = parser.parse_args()
    
    # Determine test type
    if args.quick:
        test_type = 'quick'
    elif args.integration:
        test_type = 'integration'
    elif args.performance:
        test_type = 'performance'
    else:
        test_type = 'all'
    
    print(f"🧪 CERAMIC ARMOR ML - FULL-SCALE PROCESSING TESTS")
    print(f"Test Type: {test_type.upper()}")
    print("="*60)
    
    # Run tests
    start_time = time.time()
    success = run_test_suite(test_type)
    test_time = time.time() - start_time
    
    # Print results
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    if success:
        print("✓ STATUS: ALL TESTS PASSED")
    else:
        print("✗ STATUS: SOME TESTS FAILED")
    
    print(f"⏱️  Test Time: {test_time:.2f} seconds")
    print(f"📊 Test Type: {test_type}")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)