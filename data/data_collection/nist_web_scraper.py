# data/data_collection/nist_web_scraper.py
import pandas as pd
import requests
import time
import yaml
import re
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, quote_plus
from loguru import logger
from typing import Dict, List, Optional


class AdvancedNISTScraper:
    """
    Advanced NIST web scraper with configuration-driven extraction.
    Handles multiple NIST databases and property extraction patterns.
    """

    def __init__(self, config_path: str = "config/nist_scraping_config.yaml", 
                 base_dir: str = "data/raw/nist"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup session
        self.session = requests.Session()
        settings = self.config['scraping_settings']
        self.session.headers.update({
            'User-Agent': settings['user_agent']
        })
        self.delay = settings['delay_between_requests']
        self.timeout = settings['timeout']
        self.max_retries = settings['max_retries']

    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a web page with retry logic."""
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.delay)
                logger.debug(f"Fetching: {url} (attempt {attempt + 1})")
                
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                return BeautifulSoup(response.content, 'html.parser')
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
                    return None
                time.sleep(self.delay * (attempt + 1))  # Exponential backoff
        
        return None

    def _extract_tables(self, soup: BeautifulSoup, url: str) -> List[pd.DataFrame]:
        """Extract and clean data from HTML tables."""
        tables = soup.find_all('table')
        extracted_tables = []
        
        for i, table in enumerate(tables):
            try:
                # Convert HTML table to DataFrame
                df = pd.read_html(str(table))[0]
                
                if df.empty or len(df.columns) < 2:
                    continue
                
                # Clean column names
                df.columns = [str(col).strip().lower().replace(' ', '_').replace('(', '').replace(')', '') 
                             for col in df.columns]
                
                # Check if table contains relevant ceramic data
                relevant_keywords = ['material', 'formula', 'density', 'hardness', 'modulus', 
                                   'thermal', 'ceramic', 'carbide', 'oxide', 'nitride']
                
                table_text = str(table).lower()
                if any(keyword in table_text for keyword in relevant_keywords):
                    df['source_url'] = url
                    df['table_index'] = i
                    extracted_tables.append(df)
                    logger.debug(f"Extracted table {i} with {len(df)} rows from {url}")
                
            except Exception as e:
                logger.debug(f"Could not parse table {i}: {e}")
                continue
        
        return extracted_tables

    def _extract_text_properties(self, soup: BeautifulSoup, ceramic_system: str) -> Dict:
        """Extract properties from text using regex patterns."""
        text = soup.get_text().lower()
        extracted_props = {}
        
        patterns = self.config['property_patterns']
        conversions = self.config['unit_conversions']
        
        for prop_name, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    try:
                        value, unit = matches[0]
                        value = float(value)
                        
                        # Apply unit conversions
                        if prop_name == 'density' and unit.lower() in conversions['density']:
                            value *= conversions['density'][unit.lower()]
                        elif prop_name in ['youngs_modulus', 'vickers_hardness', 'compressive_strength']:
                            if unit.lower() in conversions['pressure']:
                                value *= conversions['pressure'][unit.lower()]
                        elif prop_name == 'thermal_conductivity' and unit.lower() in conversions['thermal_conductivity']:
                            value *= conversions['thermal_conductivity'][unit.lower()]
                        
                        extracted_props[prop_name] = value
                        logger.debug(f"Extracted {prop_name}: {value} from text")
                        break  # Use first match for each property
                        
                    except (ValueError, IndexError):
                        continue
        
        if extracted_props:
            extracted_props['formula'] = ceramic_system
            extracted_props['ceramic_system'] = ceramic_system
        
        return extracted_props

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names using configuration mappings."""
        mappings = self.config['column_mappings']
        
        for standard_name, variations in mappings.items():
            for variation in variations:
                variation_clean = variation.lower().replace(' ', '_').replace('(', '').replace(')', '')
                if variation_clean in df.columns:
                    df = df.rename(columns={variation_clean: standard_name})
                    break
        
        return df

    def _apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filters to remove unrealistic values."""
        if df.empty:
            return df
        
        filters = self.config['quality_filters']
        initial_count = len(df)
        
        for prop, min_val in filters.items():
            if prop.startswith('min_'):
                prop_name = prop[4:]  # Remove 'min_' prefix
                max_prop = f'max_{prop_name}'
                
                if prop_name in df.columns and max_prop in filters:
                    max_val = filters[max_prop]
                    df = df[(df[prop_name] >= min_val) & (df[prop_name] <= max_val)]
        
        filtered_count = len(df)
        if filtered_count < initial_count:
            logger.info(f"Quality filters removed {initial_count - filtered_count} records")
        
        return df

    def search_nist_webbook(self, ceramic_system: str) -> List[pd.DataFrame]:
        """Search NIST WebBook for ceramic system data."""
        search_terms = self.config['ceramic_search_terms'].get(ceramic_system, [ceramic_system])
        base_url = self.config['search_endpoints']['webbook_search']
        
        all_data = []
        
        for term in search_terms:
            search_url = f"{base_url}?Name={quote_plus(term)}&Units=SI"
            logger.info(f"Searching NIST WebBook for: {term}")
            
            soup = self._get_page(search_url)
            if not soup:
                continue
            
            # Extract tables
            tables = self._extract_tables(soup, search_url)
            all_data.extend(tables)
            
            # Extract text properties
            text_props = self._extract_text_properties(soup, ceramic_system)
            if text_props:
                text_df = pd.DataFrame([text_props])
                text_df['source_url'] = search_url
                text_df['search_term'] = term
                all_data.append(text_df)
            
            # Follow links to detailed pages
            links = soup.find_all('a', href=True)
            for link in links[:3]:  # Limit to first 3 relevant links
                href = link['href']
                if any(keyword in href.lower() for keyword in ['thermo', 'property', 'data']):
                    detail_url = urljoin(search_url, href)
                    detail_soup = self._get_page(detail_url)
                    
                    if detail_soup:
                        detail_tables = self._extract_tables(detail_soup, detail_url)
                        all_data.extend(detail_tables)
        
        return all_data

    def scrape_specific_urls(self, ceramic_system: str) -> List[pd.DataFrame]:
        """Scrape specific NIST URLs for ceramic data."""
        specific_urls = self.config['specific_urls']
        all_data = []
        
        for category, urls in specific_urls.items():
            logger.info(f"Scraping {category} URLs for {ceramic_system}")
            
            for url in urls:
                soup = self._get_page(url)
                if not soup:
                    continue
                
                # Check if page contains relevant ceramic system data
                page_text = soup.get_text().lower()
                search_terms = self.config['ceramic_search_terms'].get(ceramic_system, [ceramic_system])
                
                if any(term.lower() in page_text for term in search_terms):
                    tables = self._extract_tables(soup, url)
                    all_data.extend(tables)
                    
                    text_props = self._extract_text_properties(soup, ceramic_system)
                    if text_props:
                        text_df = pd.DataFrame([text_props])
                        text_df['source_url'] = url
                        text_df['source_category'] = category
                        all_data.append(text_df)
        
        return all_data

    def scrape_ceramic_system(self, ceramic_system: str) -> pd.DataFrame:
        """Comprehensive scraping for a ceramic system."""
        logger.info(f"Starting comprehensive NIST scraping for {ceramic_system}")
        
        all_data = []
        
        # 1. Search NIST WebBook
        try:
            webbook_data = self.search_nist_webbook(ceramic_system)
            all_data.extend(webbook_data)
            logger.info(f"WebBook search yielded {len(webbook_data)} datasets")
        except Exception as e:
            logger.error(f"WebBook search failed: {e}")
        
        # 2. Scrape specific URLs
        try:
            specific_data = self.scrape_specific_urls(ceramic_system)
            all_data.extend(specific_data)
            logger.info(f"Specific URL scraping yielded {len(specific_data)} datasets")
        except Exception as e:
            logger.error(f"Specific URL scraping failed: {e}")
        
        if not all_data:
            logger.warning(f"No data found for {ceramic_system}")
            return pd.DataFrame()
        
        # Combine all data
        try:
            combined_df = pd.concat(all_data, ignore_index=True, sort=False)
            logger.info(f"Combined {len(all_data)} datasets into {len(combined_df)} records")
        except Exception as e:
            logger.error(f"Failed to combine data: {e}")
            return pd.DataFrame()
        
        # Clean and standardize
        combined_df = self._standardize_columns(combined_df)
        combined_df = self._apply_quality_filters(combined_df)
        
        # Add metadata
        combined_df['ceramic_system'] = ceramic_system
        combined_df['source'] = 'NIST_scraped'
        combined_df['collection_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Remove duplicates
        if 'formula' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['formula'], keep='first')
        
        # Save results
        output_file = self.base_dir / f"{ceramic_system.lower()}_nist_advanced_scraped.csv"
        combined_df.to_csv(output_file, index=False)
        logger.info(f"✓ Saved {len(combined_df)} records to {output_file}")
        
        return combined_df

    def scrape_all_systems(self, ceramic_systems: List[str]) -> Dict[str, pd.DataFrame]:
        """Scrape all ceramic systems."""
        results = {}
        
        for system in ceramic_systems:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Scraping NIST data for {system}")
                logger.info(f"{'='*50}")
                
                df = self.scrape_ceramic_system(system)
                results[system] = df
                
                if not df.empty:
                    logger.info(f"✅ Successfully scraped {len(df)} records for {system}")
                else:
                    logger.warning(f"⚠️  No data found for {system}")
                
            except Exception as e:
                logger.error(f"❌ Failed to scrape {system}: {e}")
                results[system] = pd.DataFrame()
        
        return results

    def generate_scraping_report(self, results: Dict[str, pd.DataFrame]) -> str:
        """Generate a summary report of scraping results."""
        report_lines = [
            "NIST Web Scraping Report",
            "=" * 50,
            f"Scraping completed at: {pd.Timestamp.now()}",
            ""
        ]
        
        total_records = 0
        successful_systems = 0
        
        for system, df in results.items():
            record_count = len(df) if not df.empty else 0
            total_records += record_count
            
            if record_count > 0:
                successful_systems += 1
                status = "✅ SUCCESS"
            else:
                status = "❌ NO DATA"
            
            report_lines.append(f"{system:<10} {record_count:>6} records  {status}")
        
        report_lines.extend([
            "",
            f"Summary:",
            f"  Total systems: {len(results)}",
            f"  Successful: {successful_systems}",
            f"  Total records: {total_records}",
            f"  Average per system: {total_records/len(results):.1f}"
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = self.base_dir / "nist_scraping_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Scraping report saved to {report_file}")
        return report_content


# Integration with existing NISTLoader
class EnhancedNISTLoader:
    """Enhanced NIST loader with advanced web scraping."""
    
    def __init__(self, base_dir: str = "data/raw/nist"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.scraper = AdvancedNISTScraper(base_dir=str(base_dir))
    
    def load_system(self, ceramic_system: str, use_scraping: bool = True, 
                   force_scraping: bool = False) -> pd.DataFrame:
        """Load data with enhanced scraping capabilities."""
        
        # Check for existing scraped data
        scraped_file = self.base_dir / f"{ceramic_system.lower()}_nist_advanced_scraped.csv"
        
        if scraped_file.exists() and not force_scraping:
            logger.info(f"Loading existing scraped data for {ceramic_system}")
            return pd.read_csv(scraped_file)
        
        # Check for manual CSV files
        sys_dir = self.base_dir / ceramic_system.lower()
        csv_files = list(sys_dir.glob("*.csv")) if sys_dir.exists() else []
        
        if csv_files and not use_scraping:
            logger.info(f"Loading manual CSV files for {ceramic_system}")
            dfs = []
            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    df['source_file'] = file.name
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Error reading {file}: {e}")
            
            if dfs:
                return pd.concat(dfs, ignore_index=True)
        
        # Use web scraping
        if use_scraping:
            logger.info(f"Starting web scraping for {ceramic_system}")
            return self.scraper.scrape_ceramic_system(ceramic_system)
        
        logger.warning(f"No data source available for {ceramic_system}")
        return pd.DataFrame()