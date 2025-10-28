"""Tests for data collection modules."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from data.data_collection.aflow_collector import AFLOWCollector
from data.data_collection.jarvis_collector import JARVISCollector
from data.data_collection.nist_downloader import NISTLoader
from data.data_collection.nist_web_scraper import AdvancedNISTScraper, EnhancedNISTLoader
from data.data_collection.data_integrator import DataIntegrator


class TestAFLOWCollector:
    """Test AFLOW data collection."""
    
    def test_init(self):
        collector = AFLOWCollector()
        assert collector.save_dir.name == "aflow"
        assert collector.timeout == 60
    
    @patch('requests.get')
    def test_query_success(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "mock_data"
        mock_get.return_value = mock_response
        
        collector = AFLOWCollector()
        result = collector.query("SiC")
        
        assert result == "mock_data"
        mock_get.assert_called_once()


class TestJARVISCollector:
    """Test JARVIS data collection."""
    
    def test_init(self):
        collector = JARVISCollector()
        assert collector.save_dir.name == "jarvis"
    
    @patch('jarvis.db.figshare.data')
    def test_load_df(self, mock_data):
        mock_data.return_value = [
            {"formula": "SiC", "formation_energy": -0.5},
            {"formula": "Al2O3", "formation_energy": -1.2}
        ]
        
        collector = JARVISCollector()
        df = collector._load_df()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "formula" in df.columns


class TestNISTLoader:
    """Test NIST data loading."""
    
    def test_init(self):
        loader = NISTLoader()
        assert loader.base_dir.name == "nist"
    
    def test_load_system_no_files(self):
        loader = NISTLoader(base_dir="tests/fixtures/empty_nist")
        result = loader.load_system("sic")
        # Should handle gracefully when no files exist


class TestAdvancedNISTScraper:
    """Test advanced NIST web scraper."""
    
    def test_init(self):
        scraper = AdvancedNISTScraper()
        assert scraper.base_dir.name == "nist"
        assert scraper.config is not None
        assert 'ceramic_search_terms' in scraper.config
    
    def test_standardize_columns(self):
        scraper = AdvancedNISTScraper()
        
        # Test column standardization
        df = pd.DataFrame({
            "material": ["SiC", "Al2O3"],
            "density_(g/cm³)": [3.2, 3.9],
            "young_modulus": [410, 370]
        })
        
        result = scraper._standardize_columns(df)
        
        # Should have standardized column names
        assert "formula" in result.columns
        assert "density" in result.columns
        assert "youngs_modulus" in result.columns
    
    def test_apply_quality_filters(self):
        scraper = AdvancedNISTScraper()
        
        # Test quality filtering
        df = pd.DataFrame({
            "density": [3.2, 100.0, 3.9],  # 100.0 should be filtered out
            "youngs_modulus": [410, 370, 2000]  # 2000 might be filtered
        })
        
        result = scraper._apply_quality_filters(df)
        
        # Should remove unrealistic values
        assert len(result) <= len(df)
        assert all(result["density"] <= 25.0)  # Max density filter


class TestEnhancedNISTLoader:
    """Test enhanced NIST loader."""
    
    def test_init(self):
        loader = EnhancedNISTLoader()
        assert loader.base_dir.name == "nist"
        assert loader.scraper is not None
    
    @patch('data.data_collection.nist_web_scraper.AdvancedNISTScraper.scrape_ceramic_system')
    def test_load_system_with_scraping(self, mock_scrape):
        # Mock scraping result
        mock_df = pd.DataFrame({
            "formula": ["SiC"],
            "density": [3.2],
            "youngs_modulus": [410]
        })
        mock_scrape.return_value = mock_df
        
        loader = EnhancedNISTLoader()
        result = loader.load_system("SiC", use_scraping=True)
        
        assert not result.empty
        assert "formula" in result.columns
        mock_scrape.assert_called_once_with("SiC")


class TestDataIntegrator:
    """Test data integration."""
    
    def test_init(self):
        integrator = DataIntegrator()
        assert integrator.output_dir.name == "processed"
    
    def test_deduplicate_by_formula(self):
        df = pd.DataFrame({
            "formula": ["SiC", "SiC", "Al2O3"],
            "density": [3.2, 3.21, 3.95],
            "source": ["MP", "AFLOW", "MP"]
        })
        
        integrator = DataIntegrator()
        result = integrator._deduplicate_by_formula(df)
        
        # Should keep only unique formulas
        assert len(result) <= len(df)
        assert len(result["formula"].unique()) == len(result)


if __name__ == "__main__":
    pytest.main([__file__])