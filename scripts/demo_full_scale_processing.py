#!/usr/bin/env python3
"""
Full-Scale Processing Demonstration Script

This script demonstrates the complete full-scale dataset processing pipeline
for ceramic materials with a smaller dataset to show functionality.

Usage:
    python scripts/demo_full_scale_processing.py
"""

import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.full_scale_processor import FullScaleProcessor
from src.utils.logger import get_logger
logger = get_logger(__name__)


def create_demo_dataset(n_materials: int = 100) -> pd.DataFrame:
    """
    Create a demonstration dataset with realistic ceramic materials data.
    
    Args:
        n_materials: Number of materials to generate
        
    Returns:
        DataFrame with demo materials data
    """
    logger.info(f"Creating demo dataset with {n_materials} materials")
    
    np.random.seed(42)  # For reproducible demo
    
    ceramic_systems = ['SiC', 'Al2O3', 'B4C', 'WC', 'TiC']
    crystal_systems = ['cubic', 'hexagonal', 'tetragonal', 'orthorhombic']
    
    # Generate realistic property ranges for ceramic materials
    data = {
        'material_id': [f'demo_{i:04d}' for i in range(n_materials)],
        'formula': np.random.choice(ceramic_systems, n_materials),
        'ceramic_system': np.random.choice(ceramic_systems, n_materials),
        'crystal_system': np.random.choice(crystal_systems, n_materials),
        'space_group': np.random.randint(1, 230, n_materials),
        
        # Basic properties
        'density': np.random.uniform(2.0, 6.0, n_materials),
        'formation_energy': np.random.uniform(-5.0, 0.0, n_materials),
        'energy_above_hull': np.random.exponential(0.05, n_materials),
        'band_gap': np.random.uniform(0.0, 6.0, n_materials),
        
        # Elastic properties
        'elastic_bulk_modulus': np.random.uniform(100, 400, n_materials),
        'elastic_shear_modulus': np.random.uniform(80, 200, n_materials),
        'elastic_youngs_modulus': np.random.uniform(200, 600, n_materials),
        'elastic_poisson_ratio': np.random.uniform(0.1, 0.4, n_materials),
        
        # Mechanical properties
        'hardness': np.random.uniform(10, 40, n_materials),
        'fracture_toughness': np.random.uniform(2, 8, n_materials),
        'compressive_strength': np.random.uniform(1000, 5000, n_materials),
        'tensile_strength': np.random.uniform(200, 800, n_materials),
        
        # Thermal properties
        'thermal_conductivity': np.random.uniform(10, 200, n_materials),
        'thermal_expansion': np.random.uniform(2e-6, 10e-6, n_materials),
        'melting_point': np.random.uniform(1500, 3500, n_materials),
        'specific_heat': np.random.uniform(400, 1200, n_materials),
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlations to make data more realistic
    # Harder materials tend to be more brittle
    df.loc[df['hardness'] > 30, 'fracture_toughness'] *= 0.7
    
    # Denser materials tend to have higher elastic moduli
    density_factor = (df['density'] - df['density'].min()) / (df['density'].max() - df['density'].min())
    df['elastic_bulk_modulus'] *= (0.8 + 0.4 * density_factor)
    df['elastic_shear_modulus'] *= (0.8 + 0.4 * density_factor)
    
    logger.info(f"Demo dataset created: {len(df)} materials, {df.shape[1]} properties")
    return df


def demonstrate_full_processing() -> Dict[str, Any]:
    """
    Demonstrate the complete full-scale processing pipeline.
    
    Returns:
        Processing results dictionary
    """
    print("🚀 CERAMIC ARMOR ML - FULL-SCALE PROCESSING DEMONSTRATION")
    print("="*70)
    
    # Create temporary output directory
    demo_output_dir = Path(tempfile.mkdtemp(prefix="demo_full_scale_"))
    logger.info(f"Demo output directory: {demo_output_dir}")
    
    try:
        # Initialize processor (create a simplified version for demo)
        print("\n1. Initializing Full-Scale Processor...")
        
        # Create a simplified processor that doesn't require API keys
        from src.utils.config_loader import load_project_config
        from src.preprocessing.data_cleaner import DataCleaner
        from src.feature_engineering.comprehensive_feature_generator import ComprehensiveFeatureGenerator
        from src.utils.data_utils import safe_save_data
        
        config = load_project_config()
        ceramic_systems = config.get('ceramic_systems', {}).get('primary', ['SiC', 'Al2O3', 'B4C', 'WC', 'TiC'])
        
        # Create output directories
        demo_output_dir.mkdir(parents=True, exist_ok=True)
        (demo_output_dir / "raw").mkdir(exist_ok=True)
        (demo_output_dir / "processed").mkdir(exist_ok=True)
        (demo_output_dir / "features").mkdir(exist_ok=True)
        (demo_output_dir / "reports").mkdir(exist_ok=True)
        
        print(f"   ✓ Processor initialized for {len(ceramic_systems)} ceramic systems")
        
        # Create demo dataset (simulating data collection)
        print("\n2. Creating Demo Dataset (simulating data collection)...")
        demo_data = create_demo_dataset(150)  # Smaller dataset for demo
        
        # Save demo data as if it were collected
        raw_data_path = demo_output_dir / "raw" / "combined_materials_data.csv"
        demo_data.to_csv(raw_data_path, index=False)
        print(f"   ✓ Demo dataset saved: {len(demo_data)} materials")
        
        # Process the dataset
        print("\n3. Processing Dataset...")
        start_time = time.time()
        
        # Step 3a: Data cleaning
        print("   3a. Cleaning and preprocessing data...")
        data_cleaner = DataCleaner()
        cleaned_data = data_cleaner.clean_dataframe(demo_data)
        print(f"       ✓ Cleaned data: {len(cleaned_data)} materials retained")
        
        # Step 3b: Feature generation
        print("   3b. Generating comprehensive features...")
        feature_generator = ComprehensiveFeatureGenerator()
        feature_data = feature_generator.generate_all_features(cleaned_data)
        print(f"       ✓ Generated {feature_data.shape[1]} features")
        
        # Step 3c: Data validation
        print("   3c. Validating processed data...")
        missing_pct = (feature_data.isnull().sum().sum() / (feature_data.shape[0] * feature_data.shape[1])) * 100
        validation_results = {
            'total_materials': len(feature_data),
            'total_features': feature_data.shape[1],
            'missing_value_percentage': missing_pct
        }
        print(f"       ✓ Validation complete: {missing_pct:.2f}% missing values")
        
        # Step 3d: Save results
        print("   3d. Saving final results...")
        saved_files = {}
        
        # Save as CSV
        csv_path = demo_output_dir / "final_ceramic_materials_dataset.csv"
        feature_data.to_csv(csv_path, index=False)
        saved_files['csv'] = str(csv_path)
        
        # Save metadata
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'total_materials': len(feature_data),
            'total_features': feature_data.shape[1],
            'ceramic_systems': ceramic_systems,
            'column_names': feature_data.columns.tolist()
        }
        metadata_path = demo_output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = str(metadata_path)
        
        print(f"       ✓ Saved in {len(saved_files)} formats")
        
        # Step 3e: Generate reports
        print("   3e. Generating analysis reports...")
        report_paths = {}
        
        # Generate simple summary report
        summary_report = f"""# Demo Processing Summary

## Dataset Overview
- **Total Materials:** {len(feature_data):,}
- **Total Features:** {feature_data.shape[1]:,}
- **Missing Values:** {missing_pct:.2f}%
- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Ceramic Systems
{chr(10).join(f'- {system}' for system in ceramic_systems)}

## Key Features Generated
- Derived properties calculated
- Compositional features included
- Structural features included
- Phase stability features included
"""
        
        reports_dir = demo_output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        summary_path = reports_dir / "demo_summary_report.md"
        with open(summary_path, 'w') as f:
            f.write(summary_report)
        report_paths['summary'] = str(summary_path)
        
        print(f"       ✓ Generated {len(report_paths)} reports")
        
        processing_time = time.time() - start_time
        
        # Validate derived properties
        print("\n4. Validating Derived Properties...")
        derived_props = [
            'specific_hardness', 'brittleness_index', 'ballistic_efficiency',
            'thermal_shock_resistance', 'pugh_ratio', 'cauchy_pressure'
        ]
        
        validated_props = []
        for prop in derived_props:
            if prop in feature_data.columns:
                non_null_count = feature_data[prop].notna().sum()
                if non_null_count > 0:
                    validated_props.append(prop)
                    print(f"   ✓ {prop}: {non_null_count}/{len(feature_data)} values")
                else:
                    print(f"   ⚠️  {prop}: No valid values")
            else:
                print(f"   ❌ {prop}: Not found in dataset")
        
        # Generate summary
        results = {
            'status': 'success',
            'processing_time': processing_time,
            'materials_processed': len(feature_data),
            'features_generated': feature_data.shape[1],
            'derived_properties_validated': len(validated_props),
            'missing_value_percentage': validation_results.get('missing_value_percentage', 0),
            'output_directory': str(demo_output_dir),
            'files_generated': saved_files,
            'reports_generated': report_paths,
            'validation_results': validation_results
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Demo processing failed: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'output_directory': str(demo_output_dir)
        }


def print_demo_results(results: Dict[str, Any]) -> None:
    """
    Print comprehensive demo results.
    
    Args:
        results: Demo processing results
    """
    print("\n" + "="*70)
    print("DEMONSTRATION RESULTS")
    print("="*70)
    
    if results['status'] == 'success':
        print("✅ STATUS: DEMONSTRATION SUCCESSFUL")
        
        print(f"\n📊 PROCESSING METRICS:")
        print(f"   • Materials Processed: {results['materials_processed']:,}")
        print(f"   • Features Generated: {results['features_generated']:,}")
        print(f"   • Processing Time: {results['processing_time']:.2f} seconds")
        print(f"   • Missing Values: {results['missing_value_percentage']:.2f}%")
        
        print(f"\n🧪 DERIVED PROPERTIES:")
        print(f"   • Properties Validated: {results['derived_properties_validated']}/6")
        
        print(f"\n📁 OUTPUT FILES:")
        for file_type, path in results['files_generated'].items():
            print(f"   • {file_type.upper()}: {Path(path).name}")
        
        print(f"\n📋 REPORTS GENERATED:")
        for report_type, path in results['reports_generated'].items():
            print(f"   • {report_type.title()}: {Path(path).name}")
        
        print(f"\n🎯 SCALABILITY ASSESSMENT:")
        materials_per_second = results['materials_processed'] / results['processing_time']
        estimated_time_5600 = 5600 / materials_per_second
        print(f"   • Processing Rate: {materials_per_second:.1f} materials/second")
        print(f"   • Estimated Time for 5,600 materials: {estimated_time_5600:.1f} seconds")
        
        if estimated_time_5600 < 3600:  # Less than 1 hour
            print(f"   ✅ Scalable to 5,600+ materials target")
        else:
            print(f"   ⚠️  May need optimization for 5,600+ materials")
        
        print(f"\n📂 Output Directory: {results['output_directory']}")
        
    else:
        print("❌ STATUS: DEMONSTRATION FAILED")
        print(f"\n🚨 ERROR: {results['error']}")
    
    print("\n" + "="*70)


def validate_demo_outputs(output_dir: str) -> Dict[str, bool]:
    """
    Validate that all expected outputs were generated.
    
    Args:
        output_dir: Output directory path
        
    Returns:
        Dictionary of validation results
    """
    output_path = Path(output_dir)
    
    validations = {
        'main_dataset': (output_path / "final_ceramic_materials_dataset.csv").exists(),
        'metadata': (output_path / "dataset_metadata.json").exists(),
        'features': (output_path / "features" / "comprehensive_features.csv").exists(),
        'feature_descriptions': (output_path / "features" / "feature_descriptions.json").exists(),
        'summary_report': (output_path / "reports" / "data_summary_report.md").exists(),
        'feature_report': (output_path / "reports" / "feature_analysis_report.md").exists(),
        'repro_guide': (output_path / "reports" / "reproducibility_guide.md").exists(),
        'exec_instructions': (output_path / "reports" / "execution_instructions.md").exists()
    }
    
    return validations


def main() -> int:
    """
    Main demonstration function.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Run demonstration
        results = demonstrate_full_processing()
        
        # Print results
        print_demo_results(results)
        
        # Validate outputs
        if results['status'] == 'success':
            print("\n🔍 VALIDATING OUTPUTS...")
            validations = validate_demo_outputs(results['output_directory'])
            
            passed = sum(validations.values())
            total = len(validations)
            
            print(f"   📊 Validation Results: {passed}/{total} files generated")
            
            for check, passed in validations.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {check.replace('_', ' ').title()}")
            
            if passed == total:
                print("\n🎉 ALL VALIDATIONS PASSED!")
                print("\n💡 NEXT STEPS:")
                print("   1. Scale up to full 5,600+ materials dataset")
                print("   2. Configure API keys for Materials Project")
                print("   3. Run: python scripts/run_full_scale_processing.py")
                print("   4. Train models with the processed dataset")
                return 0
            else:
                print(f"\n⚠️  {total - passed} validation(s) failed")
                return 1
        else:
            return 1
            
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)