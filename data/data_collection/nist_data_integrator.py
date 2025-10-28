# data/data_collection/nist_data_integrator.py
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional, Tuple
import re


class NISTDataIntegrator:
    """
    Integrates manual NIST CSV files with automated web scraping.
    Handles data standardization, deduplication, and quality control.
    """

    def __init__(self, base_dir: str = "data/raw/nist"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard column mappings
        self.column_mappings = {
            # Formula variations
            'material': 'formula',
            'composition': 'formula', 
            'compound': 'formula',
            'chemical_formula': 'formula',
            
            # Density variations
            'density_(g/cm³)': 'density',
            'density_(g/cm3)': 'density',
            'density_g_cm3': 'density',
            'ρ': 'density',
            
            # Mechanical properties
            'young_modulus': 'youngs_modulus',
            'youngs_modulus_(gpa)': 'youngs_modulus',
            'elastic_modulus': 'youngs_modulus',
            'e_(gpa)': 'youngs_modulus',
            
            'vickers_hardness_(gpa)': 'vickers_hardness',
            'hardness': 'vickers_hardness',
            'hv': 'vickers_hardness',
            
            'fracture_toughness_(mpa√m)': 'fracture_toughness',
            'fracture_toughness_mpa_m': 'fracture_toughness',
            'k1c': 'fracture_toughness',
            'kic': 'fracture_toughness',
            
            # Thermal properties
            'thermal_conductivity_(w/m·k)': 'thermal_conductivity',
            'thermal_cond': 'thermal_conductivity',
            'λ': 'thermal_conductivity',
            
            # Other properties
            'compressive_strength_(mpa)': 'compressive_strength',
            'melting_point_(°c)': 'melting_point',
            'melting_point_c': 'melting_point',
        }

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names using mapping rules."""
        df = df.copy()
        
        # Clean column names first
        df.columns = [str(col).strip().lower().replace(' ', '_').replace('(', '').replace(')', '') 
                     for col in df.columns]
        
        # Apply mappings
        df = df.rename(columns=self.column_mappings)
        
        return df

    def _detect_file_format(self, file_path: Path) -> str:
        """Detect the format of a NIST data file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for different format indicators
            if 'Material,' in content and 'Source,' in content:
                return 'nist_detailed'  # Like your TiC.csv
            elif content.count(',') > content.count('\t'):
                return 'csv_standard'
            elif content.count('\t') > content.count(','):
                return 'tsv'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.warning(f"Could not detect format for {file_path}: {e}")
            return 'unknown'

    def _parse_nist_detailed_format(self, file_path: Path) -> pd.DataFrame:
        """Parse NIST detailed format files (like your TiC.csv)."""
        logger.info(f"Parsing NIST detailed format: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        records = []
        
        # Extract metadata
        material_name = ""
        source_info = ""
        temperature = np.nan
        
        for line in lines[:10]:  # Check first 10 lines for metadata
            if line.startswith('Material,'):
                material_name = line.split(',')[1].strip()
            elif line.startswith('Source,'):
                source_info = line.split(',')[1].strip()
            elif line.startswith('Temperature,'):
                temp_str = line.split(',')[1].strip()
                temp_match = re.search(r'(\d+)', temp_str)
                if temp_match:
                    temperature = float(temp_match.group(1))
        
        # Find data section
        data_started = False
        header_line = None
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['grain size', 'fracture toughness', 'density', 'hardness']):
                header_line = line
                data_started = True
                continue
            
            if data_started and line.strip() and not line.startswith('Material Summary'):
                parts = [p.strip() for p in line.split(',')]
                
                if len(parts) >= 3:  # Valid data row
                    record = {
                        'formula': self._extract_formula(material_name),
                        'source': 'NIST_manual',
                        'source_file': file_path.name,
                        'temperature': temperature,
                        'source_info': source_info
                    }
                    
                    # Parse data based on header
                    if header_line:
                        headers = [h.strip() for h in header_line.split(',')]
                        for j, value in enumerate(parts):
                            if j < len(headers) and value and value != '':
                                header = headers[j].lower()
                                
                                # Extract numeric value
                                numeric_value = self._extract_numeric_value(value)
                                
                                if 'grain size' in header:
                                    record['grain_size'] = numeric_value
                                elif 'porosity' in header:
                                    record['porosity'] = numeric_value
                                elif 'fracture toughness' in header:
                                    record['fracture_toughness'] = numeric_value
                                elif 'fracture energy' in header:
                                    record['fracture_energy'] = numeric_value
                                elif 'method' in header:
                                    record['measurement_method'] = value
                                elif 'environment' in header:
                                    record['measurement_environment'] = value
                                elif 'comment' in header:
                                    record['comments'] = value
                                    # Extract Young's modulus from comments
                                    if 'E =' in value:
                                        e_match = re.search(r'E = (\d+)', value)
                                        if e_match:
                                            record['youngs_modulus'] = float(e_match.group(1))
                    
                    # Only add records with actual property data
                    if any(key in record for key in ['fracture_toughness', 'youngs_modulus', 'density', 'vickers_hardness']):
                        records.append(record)
        
        if records:
            df = pd.DataFrame(records)
            # Add typical values for missing properties
            df = self._add_typical_properties(df)
            return df
        else:
            return pd.DataFrame()

    def _extract_formula(self, material_name: str) -> str:
        """Extract chemical formula from material name."""
        # Common patterns
        formula_patterns = [
            r'\b([A-Z][a-z]?\d*[A-Z][a-z]?\d*)\b',  # TiC, Al2O3, etc.
            r'\b([A-Z][a-z]?\d*)\b'  # SiC, WC, etc.
        ]
        
        for pattern in formula_patterns:
            match = re.search(pattern, material_name)
            if match:
                return match.group(1)
        
        # Fallback: extract from parentheses
        paren_match = re.search(r'\(([^)]+)\)', material_name)
        if paren_match:
            return paren_match.group(1)
        
        # Last resort: use first word
        return material_name.split()[0] if material_name else 'Unknown'

    def _extract_numeric_value(self, value_str: str) -> float:
        """Extract numeric value from string."""
        if not value_str or value_str.strip() == '':
            return np.nan
        
        # Find first number in string
        match = re.search(r'(\d+\.?\d*)', str(value_str))
        if match:
            return float(match.group(1))
        
        return np.nan

    def _add_typical_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add typical property values based on ceramic system."""
        if df.empty:
            return df
        
        # Typical values for common ceramics
        typical_values = {
            'TiC': {
                'density': 4.93,
                'melting_point': 3160,
                'vickers_hardness': 30.0,
                'thermal_conductivity': 21.0
            },
            'SiC': {
                'density': 3.21,
                'melting_point': 2730,
                'vickers_hardness': 28.0,
                'thermal_conductivity': 120.0
            },
            'Al2O3': {
                'density': 3.95,
                'melting_point': 2072,
                'vickers_hardness': 15.0,
                'thermal_conductivity': 30.0
            },
            'B4C': {
                'density': 2.52,
                'melting_point': 2763,
                'vickers_hardness': 38.0,
                'thermal_conductivity': 42.0
            },
            'WC': {
                'density': 15.6,
                'melting_point': 2870,
                'vickers_hardness': 22.0,
                'thermal_conductivity': 110.0
            }
        }
        
        for formula in df['formula'].unique():
            if formula in typical_values:
                mask = df['formula'] == formula
                for prop, value in typical_values[formula].items():
                    if prop not in df.columns or df.loc[mask, prop].isna().all():
                        df.loc[mask, prop] = value
        
        return df

    def _parse_standard_csv(self, file_path: Path) -> pd.DataFrame:
        """Parse standard CSV format files."""
        try:
            df = pd.read_csv(file_path)
            df = self._standardize_columns(df)
            df['source'] = 'NIST_manual'
            df['source_file'] = file_path.name
            return df
        except Exception as e:
            logger.error(f"Error parsing standard CSV {file_path}: {e}")
            return pd.DataFrame()

    def load_manual_data(self, ceramic_system: str) -> pd.DataFrame:
        """Load all manual NIST data for a ceramic system."""
        logger.info(f"Loading manual NIST data for {ceramic_system}")
        
        # Look for files in system-specific directory and base directory
        search_paths = [
            self.base_dir / ceramic_system.lower(),
            self.base_dir
        ]
        
        all_files = []
        for search_path in search_paths:
            if search_path.exists():
                # Prioritize converted files, then look for other matches
                converted_file = search_path / f"{ceramic_system.lower()}_converted.csv"
                if converted_file.exists():
                    all_files.append(converted_file)
                    logger.info(f"Found converted file: {converted_file}")
                else:
                    # Look for files matching the ceramic system
                    patterns = [
                        f"{ceramic_system.lower()}*.csv",
                        f"{ceramic_system.upper()}*.csv",
                        f"*{ceramic_system.lower()}*.csv"
                    ]
                    
                    for pattern in patterns:
                        found_files = list(search_path.glob(pattern))
                        # Filter out original files if converted exists
                        for f in found_files:
                            if not f.name.endswith('_converted.csv') and not f.name == f"{ceramic_system}.csv":
                                all_files.extend([f])
                            elif f.name == f"{ceramic_system}.csv" and not converted_file.exists():
                                all_files.append(f)
        
        if not all_files:
            logger.info(f"No manual NIST files found for {ceramic_system}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(all_files)} manual NIST files for {ceramic_system}")
        
        all_data = []
        
        for file_path in all_files:
            logger.info(f"Processing: {file_path}")
            
            # Detect format and parse accordingly
            file_format = self._detect_file_format(file_path)
            
            if file_format == 'nist_detailed':
                df = self._parse_nist_detailed_format(file_path)
            elif file_format in ['csv_standard', 'tsv']:
                df = self._parse_standard_csv(file_path)
            else:
                logger.warning(f"Unknown format for {file_path}, trying standard CSV")
                df = self._parse_standard_csv(file_path)
            
            if not df.empty:
                df['ceramic_system'] = ceramic_system
                all_data.append(df)
                logger.info(f"✅ Loaded {len(df)} records from {file_path.name}")
            else:
                logger.warning(f"⚠️  No data extracted from {file_path.name}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True, sort=False)
            logger.info(f"✅ Total manual NIST data for {ceramic_system}: {len(combined_df)} records")
            return combined_df
        else:
            return pd.DataFrame()

    def integrate_with_scraped_data(self, ceramic_system: str, 
                                  manual_data: pd.DataFrame, 
                                  scraped_data: pd.DataFrame) -> pd.DataFrame:
        """Integrate manual and scraped NIST data."""
        logger.info(f"Integrating manual and scraped NIST data for {ceramic_system}")
        
        all_data = []
        
        # Add manual data
        if not manual_data.empty:
            manual_data = manual_data.copy()
            manual_data['data_source'] = 'manual'
            all_data.append(manual_data)
            logger.info(f"Manual data: {len(manual_data)} records")
        
        # Add scraped data
        if not scraped_data.empty:
            scraped_data = scraped_data.copy()
            scraped_data['data_source'] = 'scraped'
            all_data.append(scraped_data)
            logger.info(f"Scraped data: {len(scraped_data)} records")
        
        if not all_data:
            logger.warning(f"No data to integrate for {ceramic_system}")
            return pd.DataFrame()
        
        # Combine data
        combined_df = pd.concat(all_data, ignore_index=True, sort=False)
        
        # Standardize columns
        combined_df = self._standardize_columns(combined_df)
        
        # Remove duplicates (prefer manual data over scraped)
        combined_df = self._deduplicate_data(combined_df)
        
        # Quality control
        combined_df = self._apply_quality_control(combined_df)
        
        # Add metadata
        combined_df['ceramic_system'] = ceramic_system
        combined_df['integration_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"✅ Integrated NIST data for {ceramic_system}: {len(combined_df)} final records")
        
        return combined_df

    def _deduplicate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records, preferring manual over scraped data."""
        if df.empty:
            return df
        
        # Sort by data source (manual first) and other criteria
        df = df.sort_values(['data_source', 'source_file'], 
                           key=lambda x: x.map({'manual': 0, 'scraped': 1}) if x.name == 'data_source' else x)
        
        # Remove duplicates based on key properties
        duplicate_cols = []
        for col in ['formula', 'density', 'youngs_modulus', 'fracture_toughness']:
            if col in df.columns:
                duplicate_cols.append(col)
        
        if duplicate_cols:
            df = df.drop_duplicates(subset=duplicate_cols, keep='first')
        
        return df

    def _apply_quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality control filters."""
        if df.empty:
            return df
        
        initial_count = len(df)
        
        # Remove rows with no property data
        property_cols = ['density', 'youngs_modulus', 'vickers_hardness', 
                        'fracture_toughness', 'thermal_conductivity']
        
        existing_prop_cols = [col for col in property_cols if col in df.columns]
        if existing_prop_cols:
            df = df.dropna(subset=existing_prop_cols, how='all')
        
        # Apply reasonable value ranges
        if 'density' in df.columns:
            df = df[(df['density'] >= 1.0) & (df['density'] <= 25.0)]
        
        if 'youngs_modulus' in df.columns:
            df = df[(df['youngs_modulus'] >= 10) & (df['youngs_modulus'] <= 1000)]
        
        if 'fracture_toughness' in df.columns:
            df = df[(df['fracture_toughness'] >= 0.5) & (df['fracture_toughness'] <= 15)]
        
        final_count = len(df)
        if final_count < initial_count:
            logger.info(f"Quality control removed {initial_count - final_count} records")
        
        return df

    def save_integrated_data(self, ceramic_system: str, df: pd.DataFrame) -> str:
        """Save integrated data to file."""
        if df.empty:
            logger.warning(f"No data to save for {ceramic_system}")
            return ""
        
        output_file = self.base_dir / f"{ceramic_system.lower()}_nist_integrated.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"✅ Saved integrated NIST data: {output_file}")
        logger.info(f"   Records: {len(df)}")
        logger.info(f"   Columns: {len(df.columns)}")
        
        # Show data summary
        manual_count = len(df[df['data_source'] == 'manual']) if 'data_source' in df.columns else 0
        scraped_count = len(df[df['data_source'] == 'scraped']) if 'data_source' in df.columns else 0
        
        logger.info(f"   Manual: {manual_count}, Scraped: {scraped_count}")
        
        return str(output_file)