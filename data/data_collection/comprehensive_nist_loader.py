# data/data_collection/comprehensive_nist_loader.py
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional

from .nist_data_integrator import NISTDataIntegrator
from .nist_web_scraper import AdvancedNISTScraper


class ComprehensiveNISTLoader:
    """
    Comprehensive NIST loader that integrates:
    1. Manual CSV files (like your TiC.csv)
    2. Automated web scraping
    3. Data standardization and quality control
    """

    def __init__(self, base_dir: str = "data/raw/nist"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.integrator = NISTDataIntegrator(base_dir)
        self.scraper = AdvancedNISTScraper(base_dir=base_dir)
        
        logger.info("✅ Comprehensive NIST loader initialized")

    def load_system(self, ceramic_system: str, 
                   use_manual: bool = True,
                   use_scraping: bool = True,
                   force_scraping: bool = False) -> pd.DataFrame:
        """
        Load NIST data for a ceramic system using all available sources.
        
        Args:
            ceramic_system: Ceramic system (SiC, Al2O3, etc.)
            use_manual: Load manual CSV files
            use_scraping: Use web scraping
            force_scraping: Force re-scraping even if cached data exists
        
        Returns:
            Integrated DataFrame with all NIST data
        """
        logger.info(f"🔍 Loading comprehensive NIST data for {ceramic_system}")
        logger.info(f"   Manual files: {'✅' if use_manual else '❌'}")
        logger.info(f"   Web scraping: {'✅' if use_scraping else '❌'}")
        
        # Check for existing integrated file
        integrated_file = self.base_dir / f"{ceramic_system.lower()}_nist_integrated.csv"
        
        if integrated_file.exists() and not force_scraping:
            logger.info(f"✅ Loading existing integrated data: {integrated_file}")
            return pd.read_csv(integrated_file)
        
        # Load manual data
        manual_data = pd.DataFrame()
        if use_manual:
            try:
                manual_data = self.integrator.load_manual_data(ceramic_system)
                if not manual_data.empty:
                    logger.info(f"✅ Manual data loaded: {len(manual_data)} records")
                else:
                    logger.info("ℹ️  No manual data found")
            except Exception as e:
                logger.error(f"❌ Manual data loading failed: {e}")
        
        # Load scraped data
        scraped_data = pd.DataFrame()
        if use_scraping:
            try:
                scraped_data = self.scraper.scrape_ceramic_system(ceramic_system)
                if not scraped_data.empty:
                    logger.info(f"✅ Scraped data loaded: {len(scraped_data)} records")
                else:
                    logger.info("ℹ️  No scraped data found")
            except Exception as e:
                logger.error(f"❌ Web scraping failed: {e}")
        
        # Integrate data
        if manual_data.empty and scraped_data.empty:
            logger.warning(f"⚠️  No NIST data found for {ceramic_system}")
            return pd.DataFrame()
        
        integrated_data = self.integrator.integrate_with_scraped_data(
            ceramic_system, manual_data, scraped_data
        )
        
        # Save integrated data
        if not integrated_data.empty:
            output_file = self.integrator.save_integrated_data(ceramic_system, integrated_data)
            logger.info(f"✅ Comprehensive NIST data ready for {ceramic_system}")
        
        return integrated_data

    def load_all_systems(self, ceramic_systems: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """Load NIST data for all ceramic systems."""
        logger.info(f"🚀 Loading NIST data for all systems: {ceramic_systems}")
        
        results = {}
        
        for system in ceramic_systems:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Processing {system}")
                logger.info(f"{'='*50}")
                
                df = self.load_system(system, **kwargs)
                results[system] = df
                
                if not df.empty:
                    logger.info(f"✅ {system}: {len(df)} total records")
                    
                    # Show data source breakdown
                    if 'data_source' in df.columns:
                        manual_count = len(df[df['data_source'] == 'manual'])
                        scraped_count = len(df[df['data_source'] == 'scraped'])
                        logger.info(f"   📁 Manual: {manual_count}, 🌐 Scraped: {scraped_count}")
                    
                    # Show available properties
                    property_cols = ['density', 'youngs_modulus', 'vickers_hardness', 
                                   'fracture_toughness', 'thermal_conductivity']
                    available_props = []
                    for prop in property_cols:
                        if prop in df.columns:
                            non_null_count = df[prop].notna().sum()
                            if non_null_count > 0:
                                available_props.append(f"{prop}({non_null_count})")
                    
                    if available_props:
                        logger.info(f"   🔬 Properties: {', '.join(available_props)}")
                else:
                    logger.warning(f"⚠️  {system}: No data found")
                
            except Exception as e:
                logger.error(f"❌ {system} failed: {e}")
                results[system] = pd.DataFrame()
        
        # Generate summary report
        self._generate_summary_report(results)
        
        return results

    def _generate_summary_report(self, results: Dict[str, pd.DataFrame]):
        """Generate a summary report of all loaded data."""
        logger.info(f"\n{'='*60}")
        logger.info("COMPREHENSIVE NIST DATA SUMMARY")
        logger.info(f"{'='*60}")
        
        total_records = 0
        successful_systems = 0
        
        for system, df in results.items():
            record_count = len(df) if not df.empty else 0
            total_records += record_count
            
            if record_count > 0:
                successful_systems += 1
                status = "✅"
                
                # Count data sources
                manual_count = 0
                scraped_count = 0
                if 'data_source' in df.columns:
                    manual_count = len(df[df['data_source'] == 'manual'])
                    scraped_count = len(df[df['data_source'] == 'scraped'])
                
                logger.info(f"{system:<8} {record_count:>4} records {status} (M:{manual_count}, S:{scraped_count})")
            else:
                logger.info(f"{system:<8} {record_count:>4} records ❌")
        
        logger.info(f"{'='*60}")
        logger.info(f"Total systems: {len(results)}")
        logger.info(f"Successful: {successful_systems}")
        logger.info(f"Total records: {total_records}")
        logger.info(f"Average per system: {total_records/len(results):.1f}")
        
        # Save summary report
        report_file = self.base_dir / "comprehensive_nist_summary.txt"
        with open(report_file, 'w') as f:
            f.write("Comprehensive NIST Data Summary\n")
            f.write("=" * 40 + "\n\n")
            
            for system, df in results.items():
                f.write(f"{system}: {len(df)} records\n")
                if not df.empty and 'data_source' in df.columns:
                    manual_count = len(df[df['data_source'] == 'manual'])
                    scraped_count = len(df[df['data_source'] == 'scraped'])
                    f.write(f"  Manual: {manual_count}, Scraped: {scraped_count}\n")
                f.write("\n")
            
            f.write(f"Total: {total_records} records across {successful_systems} systems\n")
        
        logger.info(f"📄 Summary report saved: {report_file}")

    def test_integration(self, ceramic_system: str = "TiC") -> bool:
        """Test the integration system with a specific ceramic system."""
        logger.info(f"🧪 Testing NIST integration for {ceramic_system}")
        
        try:
            # Test manual data loading
            manual_data = self.integrator.load_manual_data(ceramic_system)
            logger.info(f"Manual data test: {len(manual_data)} records")
            
            # Test web scraping (limited)
            try:
                scraped_data = self.scraper.scrape_ceramic_system(ceramic_system)
                logger.info(f"Scraping test: {len(scraped_data)} records")
            except:
                scraped_data = pd.DataFrame()
                logger.info("Scraping test: Failed (expected for testing)")
            
            # Test integration
            if not manual_data.empty or not scraped_data.empty:
                integrated_data = self.integrator.integrate_with_scraped_data(
                    ceramic_system, manual_data, scraped_data
                )
                logger.info(f"Integration test: {len(integrated_data)} final records")
                
                if not integrated_data.empty:
                    logger.info("✅ Integration test successful!")
                    
                    # Show sample data
                    logger.info("Sample integrated data:")
                    for col in ['formula', 'fracture_toughness', 'youngs_modulus', 'data_source']:
                        if col in integrated_data.columns:
                            sample_val = integrated_data[col].iloc[0] if len(integrated_data) > 0 else 'N/A'
                            logger.info(f"  {col}: {sample_val}")
                    
                    return True
                else:
                    logger.error("❌ Integration produced no data")
                    return False
            else:
                logger.warning("⚠️  No data found for integration test")
                return False
                
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            return False


# Convenience function for easy import
def load_comprehensive_nist_data(ceramic_systems: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
    """Convenience function to load comprehensive NIST data."""
    loader = ComprehensiveNISTLoader()
    return loader.load_all_systems(ceramic_systems, **kwargs)