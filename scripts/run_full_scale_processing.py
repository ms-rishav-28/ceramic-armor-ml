#!/usr/bin/env python3
"""
Full-Scale Dataset Processing Execution Script

This script demonstrates the complete execution of the full-scale dataset
processing pipeline for 5,600+ ceramic materials with complete reproducibility.

Usage:
    python scripts/run_full_scale_processing.py [options]

Options:
    --force-recollect    Force re-collection of data from sources
    --output-dir DIR     Output directory (default: data/processed/full_scale)
    --max-workers N      Maximum number of parallel workers (default: 4)
    --batch-size N       Batch size for processing (default: 100)
    --no-reports         Skip report generation
    --validate-only      Only validate existing data
    --config PATH        Path to configuration file
"""

import sys
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.full_scale_processor import FullScaleProcessor
from src.data_collection.multi_source_collector import MultiSourceCollector
from src.utils.logger import get_logger, setup_logging
from src.utils.config_loader import load_project_config
from src.utils.data_utils import DataUtils

# Configure logging
setup_logging(level=logging.INFO, log_to_file=True, log_file="logs/full_scale_processing.log")
logger = get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Execute full-scale dataset processing for ceramic materials",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--force-recollect',
        action='store_true',
        help='Force re-collection of data from all sources'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/full_scale',
        help='Output directory for processed data (default: data/processed/full_scale)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers (default: 4)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for processing (default: 100)'
    )
    
    parser.add_argument(
        '--no-reports',
        action='store_true',
        help='Skip generation of analysis reports'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing processed data'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def validate_system_requirements() -> bool:
    """
    Validate system requirements for full-scale processing.
    
    Returns:
        True if system meets requirements, False otherwise
    """
    logger.info("Validating system requirements")
    
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error(f"Python 3.8+ required, found {sys.version}")
            return False
        
        # Check available memory
        memory_info = DataUtils.get_memory_usage()
        available_gb = memory_info.get('available_mb', 0) / 1024
        
        if available_gb < 8:
            logger.warning(f"Low available memory: {available_gb:.1f}GB (8GB+ recommended)")
        
        # Check disk space
        output_path = Path("data/processed/full_scale")
        if output_path.exists():
            import shutil
            free_space_gb = shutil.disk_usage(output_path).free / (1024**3)
        else:
            free_space_gb = shutil.disk_usage(Path.cwd()).free / (1024**3)
        
        if free_space_gb < 5:
            logger.error(f"Insufficient disk space: {free_space_gb:.1f}GB (5GB+ required)")
            return False
        
        # Check required packages
        required_packages = [
            'pandas', 'numpy', 'scikit-learn', 'xgboost', 'catboost',
            'requests', 'tqdm', 'pyyaml', 'psutil'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {missing_packages}")
            return False
        
        logger.info("✓ System requirements validated")
        return True
        
    except Exception as e:
        logger.error(f"System validation failed: {e}")
        return False


def setup_environment() -> None:
    """Setup environment for optimal processing."""
    import os
    
    # Set environment variables for Intel optimization
    env_vars = {
        'OMP_NUM_THREADS': '20',
        'MKL_NUM_THREADS': '20',
        'NUMEXPR_NUM_THREADS': '20',
        'OPENBLAS_NUM_THREADS': '20'
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        logger.debug(f"Set {var}={value}")
    
    # Configure pandas for large datasets
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    # Configure numpy
    np.seterr(divide='ignore', invalid='ignore')
    
    logger.info("✓ Environment configured for optimal processing")


def validate_configuration(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate and load configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Validated configuration dictionary
    """
    logger.info("Loading and validating configuration")
    
    try:
        if config_path:
            # Load custom configuration
            from src.utils.config_loader import Config_System
            config = Config_System.load_config(config_path)
        else:
            # Load project configuration
            config = load_project_config()
        
        # Validate critical settings
        assert config.get('targets', {}).get('mechanical_r2') == 0.85, "Mechanical R² target must be 0.85"
        assert config.get('targets', {}).get('ballistic_r2') == 0.80, "Ballistic R² target must be 0.80"
        
        ceramic_systems = config.get('ceramic_systems', {}).get('primary', [])
        assert len(ceramic_systems) == 5, f"Expected 5 ceramic systems, found {len(ceramic_systems)}"
        
        expected_systems = {'SiC', 'Al2O3', 'B4C', 'WC', 'TiC'}
        actual_systems = set(ceramic_systems)
        assert actual_systems == expected_systems, f"Ceramic systems mismatch: {actual_systems} vs {expected_systems}"
        
        logger.info("✓ Configuration validated")
        return config
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def execute_full_processing(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Execute the complete full-scale processing pipeline.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Processing results dictionary
    """
    logger.info("Starting full-scale dataset processing")
    start_time = time.time()
    
    try:
        # Initialize processor
        processor = FullScaleProcessor(
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            enable_parallel=True
        )
        
        # Execute processing
        results = processor.process_full_dataset(
            force_recollect=args.force_recollect,
            validate_results=True,
            generate_reports=not args.no_reports
        )
        
        # Calculate total processing time
        total_time = time.time() - start_time
        results['total_execution_time'] = total_time
        
        return results
        
    except Exception as e:
        logger.error(f"Full processing failed: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'total_execution_time': time.time() - start_time
        }


def validate_existing_data(output_dir: str) -> Dict[str, Any]:
    """
    Validate existing processed data.
    
    Args:
        output_dir: Output directory to validate
        
    Returns:
        Validation results dictionary
    """
    logger.info("Validating existing processed data")
    
    validation_results = {
        'status': 'unknown',
        'files_found': [],
        'files_missing': [],
        'data_quality': {},
        'errors': []
    }
    
    try:
        output_path = Path(output_dir)
        
        # Check for required files
        required_files = [
            'final_ceramic_materials_dataset.csv',
            'dataset_metadata.json',
            'processing_statistics.json',
            'features/comprehensive_features.csv',
            'features/feature_descriptions.json'
        ]
        
        for file_name in required_files:
            file_path = output_path / file_name
            if file_path.exists():
                validation_results['files_found'].append(str(file_path))
            else:
                validation_results['files_missing'].append(str(file_path))
        
        # Validate main dataset if it exists
        main_dataset_path = output_path / 'final_ceramic_materials_dataset.csv'
        if main_dataset_path.exists():
            try:
                df = pd.read_csv(main_dataset_path)
                
                validation_results['data_quality'] = {
                    'total_materials': len(df),
                    'total_features': df.shape[1],
                    'missing_value_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
                    'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
                    'categorical_features': len(df.select_dtypes(include=['object']).columns)
                }
                
                # Check for required derived properties
                required_properties = [
                    'specific_hardness', 'brittleness_index', 'ballistic_efficiency',
                    'thermal_shock_resistance', 'pugh_ratio', 'cauchy_pressure'
                ]
                
                missing_properties = [prop for prop in required_properties if prop not in df.columns]
                validation_results['data_quality']['missing_derived_properties'] = missing_properties
                
                # Validate targets
                if len(df) >= 5600:
                    validation_results['data_quality']['materials_target_met'] = True
                else:
                    validation_results['data_quality']['materials_target_met'] = False
                    validation_results['errors'].append(f"Materials count {len(df)} < 5600 target")
                
                if df.shape[1] >= 120:
                    validation_results['data_quality']['features_target_met'] = True
                else:
                    validation_results['data_quality']['features_target_met'] = False
                    validation_results['errors'].append(f"Feature count {df.shape[1]} < 120 target")
                
                logger.info(f"Dataset validation: {len(df):,} materials, {df.shape[1]} features")
                
            except Exception as e:
                validation_results['errors'].append(f"Error reading dataset: {e}")
        
        # Determine overall status
        if not validation_results['files_missing'] and not validation_results['errors']:
            validation_results['status'] = 'valid'
        elif validation_results['files_found']:
            validation_results['status'] = 'partial'
        else:
            validation_results['status'] = 'invalid'
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        validation_results['status'] = 'error'
        validation_results['errors'].append(str(e))
        return validation_results


def print_results_summary(results: Dict[str, Any]) -> None:
    """
    Print a comprehensive summary of processing results.
    
    Args:
        results: Processing results dictionary
    """
    print("\n" + "="*80)
    print("FULL-SCALE PROCESSING RESULTS SUMMARY")
    print("="*80)
    
    if results['status'] == 'success':
        print("✓ STATUS: SUCCESS")
        
        stats = results.get('statistics', {})
        print(f"\n📊 PROCESSING STATISTICS:")
        print(f"   • Total Materials Collected: {stats.get('total_materials_collected', 0):,}")
        print(f"   • Processing Time: {stats.get('total_processing_time_seconds', 0):.2f} seconds")
        print(f"   • Peak Memory Usage: {stats.get('memory_usage_peak_mb', 0):.1f} MB")
        
        materials_by_system = stats.get('materials_by_system', {})
        if materials_by_system:
            print(f"\n🏗️  MATERIALS BY CERAMIC SYSTEM:")
            for system, count in materials_by_system.items():
                print(f"   • {system}: {count:,} materials")
        
        quality_metrics = stats.get('data_quality_metrics', {})
        if quality_metrics:
            print(f"\n📈 DATA QUALITY METRICS:")
            print(f"   • Total Features: {quality_metrics.get('total_features', 0)}")
            print(f"   • Missing Values: {quality_metrics.get('missing_value_percentage', 0):.2f}%")
            print(f"   • Numeric Features: {quality_metrics.get('numeric_features_count', 0)}")
        
        output_dir = results.get('output_directory', 'N/A')
        print(f"\n📁 OUTPUT LOCATION:")
        print(f"   • Directory: {output_dir}")
        
        print(f"\n🎯 TARGETS ACHIEVED:")
        print(f"   • Materials Count: {'✓' if stats.get('total_materials_collected', 0) >= 5600 else '✗'} (5,600+ target)")
        print(f"   • Feature Count: {'✓' if quality_metrics.get('total_features', 0) >= 120 else '✗'} (120+ target)")
        
    else:
        print("✗ STATUS: FAILED")
        error = results.get('error', 'Unknown error')
        print(f"\n❌ ERROR: {error}")
        
        stats = results.get('statistics', {})
        if stats.get('errors_encountered'):
            print(f"\n🔍 DETAILED ERRORS:")
            for error in stats['errors_encountered']:
                print(f"   • {error}")
    
    total_time = results.get('total_execution_time', 0)
    print(f"\n⏱️  TOTAL EXECUTION TIME: {total_time:.2f} seconds")
    
    print("\n" + "="*80)


def main() -> int:
    """
    Main execution function.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        print("🚀 CERAMIC ARMOR ML - FULL-SCALE DATASET PROCESSING")
        print("="*60)
        
        # Validate system requirements
        if not validate_system_requirements():
            logger.error("System requirements not met")
            return 1
        
        # Setup environment
        setup_environment()
        
        # Validate configuration
        config = validate_configuration(args.config)
        logger.info(f"Configuration loaded: {len(config)} sections")
        
        # Execute based on mode
        if args.validate_only:
            # Validation mode
            print("\n🔍 VALIDATION MODE: Checking existing data")
            validation_results = validate_existing_data(args.output_dir)
            
            print(f"\nValidation Status: {validation_results['status'].upper()}")
            print(f"Files Found: {len(validation_results['files_found'])}")
            print(f"Files Missing: {len(validation_results['files_missing'])}")
            
            if validation_results['data_quality']:
                quality = validation_results['data_quality']
                print(f"Materials: {quality.get('total_materials', 0):,}")
                print(f"Features: {quality.get('total_features', 0)}")
                print(f"Missing Values: {quality.get('missing_value_percentage', 0):.2f}%")
            
            if validation_results['errors']:
                print("\nErrors Found:")
                for error in validation_results['errors']:
                    print(f"  • {error}")
            
            return 0 if validation_results['status'] in ['valid', 'partial'] else 1
        
        else:
            # Full processing mode
            print("\n⚙️  PROCESSING MODE: Full-scale dataset processing")
            print(f"Output Directory: {args.output_dir}")
            print(f"Max Workers: {args.max_workers}")
            print(f"Batch Size: {args.batch_size}")
            print(f"Force Recollect: {args.force_recollect}")
            print(f"Generate Reports: {not args.no_reports}")
            
            # Execute processing
            results = execute_full_processing(args)
            
            # Print results summary
            print_results_summary(results)
            
            # Return appropriate exit code
            return 0 if results['status'] == 'success' else 1
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        print("\n⚠️  Processing interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)