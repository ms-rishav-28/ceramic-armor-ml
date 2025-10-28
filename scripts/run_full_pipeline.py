"""
ONE-COMMAND EXECUTION SCRIPT
Runs complete pipeline from data collection to final report
"""

import sys
sys.path.append('.')

import yaml
from pathlib import Path
from loguru import logger

# Import pipeline components
from src.utils.intel_optimizer import intel_opt
from src.data_collection.materials_project_collector import MaterialsProjectCollector
from data.data_collection.aflow_collector import AFLOWCollector
from data.data_collection.jarvis_collector import JARVISCollector
from data.data_collection.comprehensive_nist_loader import ComprehensiveNISTLoader
from data.data_collection.data_integrator import DataIntegrator
from data.data_collection.literature_miner import LiteratureMiner
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.unit_standardizer import standardize
from src.preprocessing.outlier_detector import remove_iqr_outliers
from src.preprocessing.missing_value_handler import impute_knn
from src.feature_engineering.derived_properties import DerivedPropertiesCalculator
from src.feature_engineering.phase_stability import PhaseStabilityAnalyzer
from src.feature_engineering.compositional_features import CompositionalFeatureCalculator
from src.feature_engineering.microstructure_features import MicrostructureFeatureCalculator
from src.training.trainer import CeramicPropertyTrainer
from src.interpretation.shap_analyzer import SHAPAnalyzer
from src.evaluation.metrics import PerformanceChecker

def load_config(config_path='config/config.yaml'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Execute complete ML pipeline"""
    
    # ASCII Banner
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  CERAMIC ARMOR ML PIPELINE - COMPLETE EXECUTION              ║
    ║  Tree-based Models for Ballistic Property Prediction         ║
    ║  Intel i7-12700K Optimized | 20-Week Research Program        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config()
    
    # Apply Intel optimizations
    logger.info("Applying Intel CPU optimizations...")
    intel_opt.apply_optimizations()
    
    # ========================================================================
    # PHASE 1: DATA COLLECTION (Weeks 1-4)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: DATA COLLECTION")
    logger.info("="*80)
    
    # Initialize all data collectors
    api_keys_path = 'config/api_keys.yaml'
    if Path(api_keys_path).exists():
        with open(api_keys_path, 'r') as f:
            api_keys = yaml.safe_load(f)
        mp_api_key = api_keys.get('materials_project')
    else:
        logger.warning("API keys not found - using environment variables")
        import os
        mp_api_key = os.environ.get('MP_API_KEY')
    
    # Initialize collectors
    mp_collector = MaterialsProjectCollector(mp_api_key) if mp_api_key else None
    aflow_collector = AFLOWCollector()
    jarvis_collector = JARVISCollector()
    nist_loader = ComprehensiveNISTLoader()  # Use comprehensive integration
    literature_miner = LiteratureMiner()
    integrator = DataIntegrator()
    
    for system in config['ceramic_systems']['primary']:
        logger.info(f"\n--- Collecting data for {system} ---")
        
        # 1. Materials Project
        if mp_collector:
            try:
                mp_collector.collect_ceramic_data(system)
                logger.info(f"✓ Materials Project data collected for {system}")
            except Exception as e:
                logger.error(f"Materials Project collection failed for {system}: {e}")
        
        # 2. AFLOW
        try:
            aflow_collector.collect(system)
            logger.info(f"✓ AFLOW data collected for {system}")
        except Exception as e:
            logger.error(f"AFLOW collection failed for {system}: {e}")
        
        # 3. JARVIS
        try:
            jarvis_collector.collect(system)
            logger.info(f"✓ JARVIS data collected for {system}")
        except Exception as e:
            logger.error(f"JARVIS collection failed for {system}: {e}")
        
        # 4. NIST (comprehensive: manual + web scraping)
        try:
            nist_data = nist_loader.load_system(system, use_manual=True, use_scraping=True)
            if not nist_data.empty:
                # Show breakdown of data sources
                manual_count = len(nist_data[nist_data['data_source'] == 'manual']) if 'data_source' in nist_data.columns else 0
                scraped_count = len(nist_data[nist_data['data_source'] == 'scraped']) if 'data_source' in nist_data.columns else 0
                logger.info(f"✓ NIST data integrated for {system}: {len(nist_data)} total records")
                logger.info(f"  📁 Manual: {manual_count}, 🌐 Scraped: {scraped_count}")
            else:
                logger.warning(f"⚠️  No NIST data found for {system}")
        except Exception as e:
            logger.error(f"NIST integration failed for {system}: {e}")
        
        # 5. Literature Mining (optional)
        try:
            literature_miner.mine_system(system, limit=50)
            logger.info(f"✓ Literature references collected for {system}")
        except Exception as e:
            logger.warning(f"Literature mining failed for {system}: {e}")
        
        # 6. Data Integration
        try:
            source_files = {
                'materials_project': f"data/raw/materials_project/{system.lower()}_raw.csv",
                'aflow': f"data/raw/aflow/{system.lower()}_aflow_raw.csv",
                'jarvis': f"data/raw/jarvis/{system.lower()}_jarvis_raw.csv",
                'nist': f"data/raw/nist/{system.lower()}_nist_raw_merged.csv"
            }
            integrator.integrate_system(system, source_files)
            logger.info(f"✓ Data integration complete for {system}")
        except Exception as e:
            logger.error(f"Data integration failed for {system}: {e}")
    
    # ========================================================================
    # PHASE 2: PREPROCESSING (Weeks 5-6)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: DATA PREPROCESSING")
    logger.info("="*80)
    
    cleaner = DataCleaner()
    
    for system in config['ceramic_systems']['primary']:
        logger.info(f"\n--- Preprocessing {system} data ---")
        
        # Load integrated data
        integrated_file = Path(config['paths']['data']['processed']) / system.lower() / f"{system.lower()}_integrated.csv"
        if integrated_file.exists():
            import pandas as pd
            df = pd.read_csv(integrated_file)
            
            # Step 1: Unit standardization
            logger.info("Standardizing units...")
            df = standardize(df)
            
            # Step 2: Outlier removal
            logger.info("Removing outliers...")
            numeric_cols = ['density', 'youngs_modulus', 'vickers_hardness', 'fracture_toughness']
            existing_cols = [col for col in numeric_cols if col in df.columns]
            if existing_cols:
                df = remove_iqr_outliers(df, existing_cols, k=1.5)
            
            # Step 3: Missing value imputation
            logger.info("Imputing missing values...")
            df = impute_knn(df, n_neighbors=5)
            
            # Step 4: General cleaning
            logger.info("Final data cleaning...")
            df_clean = cleaner.clean_dataframe(df)
            
            # Save cleaned data
            output_dir = Path(config['paths']['data']['processed']) / system.lower()
            output_dir.mkdir(parents=True, exist_ok=True)
            df_clean.to_csv(output_dir / f"{system.lower()}_clean.csv", index=False)
            logger.info(f"✓ Preprocessing complete for {system}")
        else:
            logger.warning(f"Integrated data file not found for {system}: {integrated_file}")
    
    # ========================================================================
    # PHASE 3: FEATURE ENGINEERING (Weeks 7-8)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: FEATURE ENGINEERING")
    logger.info("="*80)
    
    # Initialize feature calculators
    derived_calc = DerivedPropertiesCalculator()
    stability_analyzer = PhaseStabilityAnalyzer()
    comp_calc = CompositionalFeatureCalculator()
    micro_calc = MicrostructureFeatureCalculator()
    
    for system in config['ceramic_systems']['primary']:
        logger.info(f"\n--- Engineering features for {system} ---")
        
        clean_file = Path(config['paths']['data']['processed']) / system.lower() / f"{system.lower()}_clean.csv"
        if clean_file.exists():
            import pandas as pd
            df = pd.read_csv(clean_file)
            
            # Step 1: Derived properties (Pugh ratio, specific properties, etc.)
            logger.info("Calculating derived properties...")
            df = derived_calc.calculate_all_derived_properties(df)
            
            # Step 2: Compositional features (atomic mass, electronegativity, etc.)
            logger.info("Adding compositional features...")
            if 'formula' in df.columns:
                df = comp_calc.augment_dataframe(df, formula_col='formula')
            
            # Step 3: Microstructure features (Hall-Petch, porosity effects)
            logger.info("Adding microstructure features...")
            df = micro_calc.add_features(df)
            
            # Step 4: Phase stability analysis
            logger.info("Analyzing phase stability...")
            df_final = stability_analyzer.analyze_dataframe(df)
            
            # Save features
            output_dir = Path(config['paths']['data']['features']) / system.lower()
            output_dir.mkdir(parents=True, exist_ok=True)
            df_final.to_csv(output_dir / f"{system.lower()}_features.csv", index=False)
            logger.info(f"✓ Feature engineering complete for {system} - {df_final.shape[1]} features")
        else:
            logger.warning(f"Clean data file not found for {system}: {clean_file}")
    
    # ========================================================================
    # PHASE 4: MODEL TRAINING (Weeks 11-14)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: MODEL TRAINING")
    logger.info("="*80)
    
    trainer = CeramicPropertyTrainer(config)
    trainer.train_all_systems()
    
    # ========================================================================
    # PHASE 5: EVALUATION (Weeks 15-17)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 5: MODEL EVALUATION")
    logger.info("="*80)
    
    checker = PerformanceChecker(config)
    results = checker.check_all_targets()
    
    # ========================================================================
    # PHASE 6: INTERPRETATION (Week 16)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 6: SHAP INTERPRETATION")
    logger.info("="*80)
    
    # Generate SHAP analysis for key properties
    # (Implementation depends on trained models - see scripts/06_interpret_results.py)
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("✓ PIPELINE EXECUTION COMPLETE")
    logger.info("="*80)
    
    logger.info("\nResults saved to:")
    logger.info(f"  Models: {config['paths']['models']}")
    logger.info(f"  Predictions: {config['paths']['predictions']}")
    logger.info(f"  Figures: {config['paths']['figures']}")
    
    logger.info("\nNext steps:")
    logger.info("  1. Review evaluation metrics in results/metrics/")
    logger.info("  2. Examine SHAP plots in results/figures/shap/")
    logger.info("  3. Run scripts/07_generate_report.py for publication figures")

if __name__ == "__main__":
    main()