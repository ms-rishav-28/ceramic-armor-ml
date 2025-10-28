# src/data_collection/nist_downloader.py
import pandas as pd
import requests
import time
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from loguru import logger

class NISTWebScraper:
    """
    Web scraper for NIST ceramic property databases.
    Extracts data from HTML tables and text content.
    """

    def __init__(self, base_dir: str = "data/raw/nist", delay: float = 1.0):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay  # Delay between requests to be respectful
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def _get_page(self, url: str) -> BeautifulSoup:
        """Fetch and parse a web page."""
        try:
            time.sleep(self.delay)  # Be respectful to the server
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def _extract_table_data(self, soup: BeautifulSoup) -> pd.DataFrame:
        """Extract data from HTML tables."""
        tables = soup.find_all('table')
        all_data = []
        
        for table in tables:
            try:
                # Try to convert table to DataFrame
                df = pd.read_html(str(table))[0]
                
                # Clean column names
                if len(df.columns) > 0:
                    df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
                    
                    # Look for relevant ceramic property columns
                    relevant_cols = []
                    for col in df.columns:
                        if any(keyword in col for keyword in [
                            'formula', 'composition', 'material',
                            'density', 'hardness', 'modulus', 'toughness',
                            'thermal', 'elastic', 'strength', 'temperature'
                        ]):
                            relevant_cols.append(col)
                    
                    if relevant_cols:
                        df_filtered = df[relevant_cols].copy()
                        all_data.append(df_filtered)
                        
            except Exception as e:
                logger.debug(f"Could not parse table: {e}")
                continue
        
        if all_data:
            return pd.concat(all_data, ignore_index=True, sort=False)
        else:
            return pd.DataFrame()

    def _extract_text_data(self, soup: BeautifulSoup) -> dict:
        """Extract property data from text content using regex patterns."""
        text = soup.get_text()
        
        # Common patterns for ceramic properties
        patterns = {
            'density': r'density[:\s]*(\d+\.?\d*)\s*(g/cm³|g/cm3|kg/m³|kg/m3)',
            'youngs_modulus': r'young[\'s\s]*modulus[:\s]*(\d+\.?\d*)\s*(gpa|mpa|pa)',
            'vickers_hardness': r'vickers[:\s]*hardness[:\s]*(\d+\.?\d*)\s*(gpa|hv|kg/mm²)',
            'fracture_toughness': r'fracture[:\s]*toughness[:\s]*(\d+\.?\d*)\s*(mpa[·√]?m|mpa\s*m\^?0\.?5)',
            'thermal_conductivity': r'thermal[:\s]*conductivity[:\s]*(\d+\.?\d*)\s*(w/m[·\s]?k|w/mk)',
            'melting_point': r'melting[:\s]*point[:\s]*(\d+\.?\d*)\s*(°c|k|celsius)',
            'compressive_strength': r'compressive[:\s]*strength[:\s]*(\d+\.?\d*)\s*(mpa|gpa)',
        }
        
        extracted_data = {}
        for prop, pattern in patterns.items():
            matches = re.findall(pattern, text.lower())
            if matches:
                # Take the first match and convert to standard units
                value, unit = matches[0]
                try:
                    value = float(value)
                    # Unit conversion to standard units
                    if prop == 'density':
                        if 'kg/m' in unit:
                            value = value / 1000  # Convert to g/cm³
                    elif prop in ['youngs_modulus', 'compressive_strength']:
                        if 'mpa' in unit:
                            value = value / 1000  # Convert to GPa
                    elif prop == 'vickers_hardness':
                        if 'kg/mm²' in unit:
                            value = value * 0.00981  # Convert to GPa
                    
                    extracted_data[prop] = value
                except ValueError:
                    continue
        
        return extracted_data

    def scrape_nist_ceramics_database(self) -> pd.DataFrame:
        """Scrape NIST ceramics database pages."""
        # NIST ceramic database URLs (these are examples - update with actual URLs)
        nist_urls = [
            'https://webbook.nist.gov/chemistry/',
            'https://www.nist.gov/mml/acmd/ceramic-properties',
            # Add more specific NIST ceramic database URLs here
        ]
        
        all_data = []
        
        for url in nist_urls:
            logger.info(f"Scraping NIST page: {url}")
            soup = self._get_page(url)
            
            if soup:
                # Extract table data
                table_data = self._extract_table_data(soup)
                if not table_data.empty:
                    table_data['source_url'] = url
                    all_data.append(table_data)
                
                # Extract text data
                text_data = self._extract_text_data(soup)
                if text_data:
                    text_df = pd.DataFrame([text_data])
                    text_df['source_url'] = url
                    all_data.append(text_df)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True, sort=False)
        else:
            return pd.DataFrame()

    def scrape_ceramic_system(self, ceramic_system: str) -> pd.DataFrame:
        """Scrape data for a specific ceramic system."""
        logger.info(f"Scraping NIST data for {ceramic_system}")
        
        # System-specific search terms
        search_terms = {
            'SiC': ['silicon carbide', 'SiC', 'carborundum'],
            'Al2O3': ['aluminum oxide', 'alumina', 'Al2O3', 'corundum'],
            'B4C': ['boron carbide', 'B4C'],
            'WC': ['tungsten carbide', 'WC'],
            'TiC': ['titanium carbide', 'TiC'],
            'Si3N4': ['silicon nitride', 'Si3N4'],
            'AlN': ['aluminum nitride', 'AlN']
        }
        
        terms = search_terms.get(ceramic_system, [ceramic_system])
        
        # NIST search URLs (update with actual search endpoints)
        search_urls = []
        for term in terms:
            # Example search URL format - update with actual NIST search API
            search_url = f"https://webbook.nist.gov/cgi/cbook.cgi?Name={term.replace(' ', '+')}&Units=SI"
            search_urls.append(search_url)
        
        all_data = []
        
        for url in search_urls:
            logger.info(f"Searching NIST for: {url}")
            soup = self._get_page(url)
            
            if soup:
                # Extract data from search results
                table_data = self._extract_table_data(soup)
                if not table_data.empty:
                    table_data['ceramic_system'] = ceramic_system
                    table_data['search_term'] = url.split('Name=')[1].split('&')[0].replace('+', ' ')
                    all_data.append(table_data)
                
                # Look for links to detailed property pages
                links = soup.find_all('a', href=True)
                for link in links[:5]:  # Limit to first 5 relevant links
                    href = link['href']
                    if any(keyword in href.lower() for keyword in ['property', 'data', 'thermo']):
                        full_url = urljoin(url, href)
                        detail_soup = self._get_page(full_url)
                        
                        if detail_soup:
                            detail_data = self._extract_table_data(detail_soup)
                            if not detail_data.empty:
                                detail_data['ceramic_system'] = ceramic_system
                                detail_data['source_url'] = full_url
                                all_data.append(detail_data)
        
        if all_data:
            result_df = pd.concat(all_data, ignore_index=True, sort=False)
            
            # Clean and standardize the data
            result_df = self._clean_scraped_data(result_df, ceramic_system)
            
            # Save to file
            output_file = self.base_dir / f"{ceramic_system.lower()}_nist_scraped.csv"
            result_df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(result_df)} records to {output_file}")
            
            return result_df
        else:
            logger.warning(f"No data found for {ceramic_system}")
            return pd.DataFrame()

    def _clean_scraped_data(self, df: pd.DataFrame, ceramic_system: str) -> pd.DataFrame:
        """Clean and standardize scraped data."""
        if df.empty:
            return df
        
        # Add ceramic system if not present
        if 'ceramic_system' not in df.columns:
            df['ceramic_system'] = ceramic_system
        
        # Add formula if not present
        if 'formula' not in df.columns:
            df['formula'] = ceramic_system
        
        # Standardize column names
        column_mapping = {
            'material': 'formula',
            'composition': 'formula',
            'compound': 'formula',
            'density_(g/cm³)': 'density',
            'density_(g/cm3)': 'density',
            'young_modulus': 'youngs_modulus',
            'elastic_modulus': 'youngs_modulus',
            'hardness': 'vickers_hardness',
            'thermal_cond': 'thermal_conductivity',
            'thermal_conductivity_(w/m·k)': 'thermal_conductivity',
        }
        
        df = df.rename(columns=column_mapping)
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Convert numeric columns
        numeric_columns = ['density', 'youngs_modulus', 'vickers_hardness', 
                          'fracture_toughness', 'thermal_conductivity', 'melting_point']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add source information
        df['source'] = 'NIST_scraped'
        df['collection_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        return df

    def scrape_all_systems(self, ceramic_systems: list) -> dict:
        """Scrape data for all ceramic systems."""
        results = {}
        
        for system in ceramic_systems:
            try:
                df = self.scrape_ceramic_system(system)
                results[system] = df
                logger.info(f"✓ Scraped {len(df)} records for {system}")
            except Exception as e:
                logger.error(f"Failed to scrape {system}: {e}")
                results[system] = pd.DataFrame()
        
        return results


class NISTLoader:
    """
    Enhanced NIST loader with web scraping capabilities.
    Loads data from local CSV files or scrapes from NIST websites.
    """

    def __init__(self, base_dir: str = "data/raw/nist"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.scraper = NISTWebScraper(base_dir)

    def load_system(self, ceramic_system: str, use_scraping: bool = True) -> pd.DataFrame:
        """Load data for a ceramic system, with optional web scraping."""
        sys_dir = self.base_dir / ceramic_system.lower()
        
        # First, try to load existing CSV files
        csv_files = list(sys_dir.glob("*.csv"))
        local_data = []
        
        if csv_files:
            logger.info(f"Found {len(csv_files)} local NIST files for {ceramic_system}")
            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    df['source_file'] = file.name
                    local_data.append(df)
                except Exception as e:
                    logger.error(f"Error reading {file}: {e}")
        
        # If no local data or scraping is enabled, try web scraping
        scraped_data = pd.DataFrame()
        if use_scraping and (not local_data or len(local_data) == 0):
            logger.info(f"No local data found for {ceramic_system}, attempting web scraping...")
            try:
                scraped_data = self.scraper.scrape_ceramic_system(ceramic_system)
            except Exception as e:
                logger.error(f"Web scraping failed for {ceramic_system}: {e}")
        
        # Combine local and scraped data
        all_data = []
        if local_data:
            all_data.extend(local_data)
        if not scraped_data.empty:
            all_data.append(scraped_data)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True, sort=False)
            
            # Remove duplicates based on formula and key properties
            if 'formula' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['formula'], keep='first')
            
            logger.info(f"✓ Loaded {len(combined_df)} NIST records for {ceramic_system}")
            return combined_df
        else:
            logger.warning(f"No NIST data found for {ceramic_system}")
            return pd.DataFrame()

    def load_all_systems(self, ceramic_systems: list, use_scraping: bool = True) -> dict:
        """Load data for all ceramic systems."""
        results = {}
        
        for system in ceramic_systems:
            results[system] = self.load_system(system, use_scraping=use_scraping)
        
        return results
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                df["__file"] = f.name
                dfs.append(df)
            except Exception as e:
                logger.error(f"NIST read error {f}: {e}")
        if not dfs:
            return pd.DataFrame()
        df = pd.concat(dfs, ignore_index=True)

        # Normalize likely column names
        rename_map = {
            "Formula": "formula",
            "Density (g/cc)": "density",
            "Vickers Hardness (GPa)": "vickers_hardness",
            "Fracture Toughness (MPa m^0.5)": "fracture_toughness_mode_i",
            "Elastic Modulus (GPa)": "youngs_modulus",
            "Compressive Strength (MPa)": "compressive_strength",
            "Thermal Conductivity (W/mK)": "thermal_conductivity",
            "CTE (1/K)": "thermal_expansion_coefficient",
            "Specific Heat (J/kgK)": "specific_heat"
        }
        for k, v in rename_map.items():
            if k in df.columns and v not in df.columns:
                df.rename(columns={k: v}, inplace=True)

        df["ceramic_system"] = ceramic_system
        df["source"] = "NIST"
        out = self.base_dir / f"{ceramic_system.lower()}_nist_raw_merged.csv"
        df.to_csv(out, index=False)
        logger.info(f"✓ NIST merged {len(df)} rows -> {out}")
        return df
