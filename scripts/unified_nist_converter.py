#!/usr/bin/env python3
"""
Unified NIST Data Converter for All Ceramic Systems
Converts Al2O3, B4C, SiC, TiC, WC CSV files to pipeline-compatible format.
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import re
from pathlib import Path
from loguru import logger
from typing import Dict, List, Tuple, Optional


class UnifiedNISTConverter:
    """
    Unified converter for all NIST ceramic data formats.
    Handles multiple format types and standardizes them for the ML pipeline.
    """

    def __init__(self, base_dir: str = "data/raw/nist"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Typical property values for each ceramic system
        self.typical_properties = {
            'Al2O3': {
                'density': 3.95,
                'melting_point': 2072,
                'thermal_conductivity': 30.0,
                'formula': 'Al2O3'
            },
            'SiC': {
                'density': 3.21,
                'melting_point': 2730,
                'thermal_conductivity': 120.0,
                'formula': 'SiC'
            },
            'B4C': {
                'density': 2.52,
                'melting_point': 2763,
                'thermal_conductivity': 42.0,
                'formula': 'B4C'
            },
            'WC': {
                'density': 15.6,
                'melting_point': 2870,
                'thermal_conductivity': 110.0,
                'formula': 'WC'
            },
            'TiC': {
                'density': 4.93,
                'melting_point': 3160,
                'thermal_conductivity': 21.0,
                'formula': 'TiC'
            }
        }

    def detect_format(self, file_path: Path) -> str:
        """Detect the format of a NIST CSV file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for different format indicators
            if 'Property [unit]' in content and '20 °C' in content:
                return 'temperature_matrix'  # Al2O3, SiC format
            elif 'Material,' in content and 'Fracture Toughness' in content:
                return 'fracture_database'  # TiC, B4C, WC format
            elif content.count(',') > content.count('\t'):
                return 'standard_csv'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.warning(f"Could not detect format for {file_path}: {e}")
            return 'unknown'

    def parse_temperature_matrix_format(self, file_path: Path, ceramic_system: str) -> pd.DataFrame:
        """Parse temperature matrix format (Al2O3, SiC style)."""
        logger.info(f"Parsing temperature matrix format: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Get property names and units from first column
        properties = []
        for prop_unit in df.iloc[:, 0]:
            if '[' in str(prop_unit) and ']' in str(prop_unit):
                # Extract property name and unit
                prop_match = re.match(r'(.+?)\s*\[(.+?)\]', str(prop_unit))
                if prop_match:
                    prop_name = prop_match.group(1).strip()
                    unit = prop_match.group(2).strip()
                    properties.append((prop_name, unit))
                else:
                    properties.append((str(prop_unit), ''))
            else:
                properties.append((str(prop_unit), ''))
        
        # Get temperature columns (skip first column which is properties)
        temp_columns = df.columns[1:]
        
        records = []
        
        # Process each property row
        for i, (prop_name, unit) in enumerate(properties):
            if i >= len(df):
                continue
                
            row = df.iloc[i]
            
            # Process each temperature
            for temp_col in temp_columns:
                value_str = str(row[temp_col])
                
                if value_str and value_str != 'nan' and value_str != '':
                    # Extract numeric value (handle uncertainty in parentheses)
                    numeric_value = self._extract_numeric_value(value_str)
                    
                    if not np.isnan(numeric_value):
                        # Extract temperature
                        temp_match = re.search(r'(\d+)', temp_col)
                        temperature = float(temp_match.group(1)) if temp_match else 20.0
                        
                        # Create record
                        record = {
                            'formula': self.typical_properties[ceramic_system]['formula'],
                            'ceramic_system': ceramic_system,
                            'temperature': temperature,
                            'source': 'NIST_manual',
                            'source_file': file_path.name,
                            'measurement_unit': unit
                        }
                        
                        # Map property name to standard column
                        std_prop = self._standardize_property_name(prop_name)
                        if std_prop:
                            record[std_prop] = numeric_value
                            records.append(record)
        
        if records:
            df_result = pd.DataFrame(records)
            # Add typical properties for missing values
            df_result = self._add_typical_properties(df_result, ceramic_system)
            return df_result
        else:
            return pd.DataFrame()

    def parse_fracture_database_format(self, file_path: Path, ceramic_system: str) -> pd.DataFrame:
        """Parse fracture database format (TiC, B4C, WC style)."""
        logger.info(f"Parsing fracture database format: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        records = []
        
        # Extract metadata
        material_name = ""
        source_info = ""
        temperature = 20.0  # Default
        
        for line in lines[:10]:
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
            if 'Grain Size' in line and 'Fracture Toughness' in line:
                header_line = line
                data_started = True
                continue
            
            if data_started and line.strip() and not line.startswith('Material Summary'):
                parts = [p.strip() for p in line.split(',')]
                
                if len(parts) >= 3:
                    record = {
                        'formula': self.typical_properties[ceramic_system]['formula'],
                        'ceramic_system': ceramic_system,
                        'source': 'NIST_manual',
                        'source_file': file_path.name,
                        'temperature': temperature,
                        'source_info': source_info
                    }
                    
                    # Parse data based on header
                    if header_line:
                        headers = [h.strip() for h in header_line.split(',')]
                        for j, value in enumerate(parts):
                            if j < len(headers) and value and value != '' and value != '-':
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
                                    # Extract properties from comments
                                    self._extract_properties_from_comments(record, value)
                    
                    # Only add records with actual property data
                    if any(key in record for key in ['fracture_toughness', 'youngs_modulus', 'density', 'vickers_hardness']):
                        records.append(record)
        
        if records:
            df_result = pd.DataFrame(records)
            # Add typical properties for missing values
            df_result = self._add_typical_properties(df_result, ceramic_system)
            return df_result
        else:
            return pd.DataFrame()

    def _extract_numeric_value(self, value_str: str) -> float:
        """Extract numeric value from string, handling uncertainties and ranges."""
        if not value_str or str(value_str).strip() == '' or str(value_str) == '-':
            return np.nan
        
        value_str = str(value_str)
        
        # Handle parentheses (uncertainties)
        if '(' in value_str:
            value_str = value_str.split('(')[0]
        
        # Handle ranges (take first value)
        if '-' in value_str and not value_str.startswith('-'):
            parts = value_str.split('-')
            if len(parts) == 2:
                try:
                    return float(parts[0])
                except:
                    pass
        
        # Find first number in string
        match = re.search(r'(\d+\.?\d*)', value_str)
        if match:
            return float(match.group(1))
        
        return np.nan

    def _standardize_property_name(self, prop_name: str) -> Optional[str]:
        """Standardize property names to pipeline format."""
        prop_lower = prop_name.lower()
        
        # Property mapping
        if 'bulk modulus' in prop_lower:
            return 'bulk_modulus'
        elif 'elastic modulus' in prop_lower or 'young' in prop_lower:
            return 'youngs_modulus'
        elif 'shear modulus' in prop_lower:
            return 'shear_modulus'
        elif 'compressive strength' in prop_lower:
            return 'compressive_strength'
        elif 'flexural strength' in prop_lower:
            return 'flexural_strength'
        elif 'fracture toughness' in prop_lower:
            return 'fracture_toughness'
        elif 'hardness' in prop_lower and 'vickers' in prop_lower:
            return 'vickers_hardness'
        elif 'density' in prop_lower:
            return 'density'
        elif 'poisson' in prop_lower:
            return 'poissons_ratio'
        elif 'friction' in prop_lower:
            return 'friction_coefficient'
        elif 'creep rate' in prop_lower:
            return 'creep_rate'
        elif 'lattice parameter' in prop_lower:
            if 'a' in prop_lower:
                return 'lattice_parameter_a'
            elif 'c' in prop_lower:
                return 'lattice_parameter_c'
        elif 'sound velocity' in prop_lower:
            if 'longitudinal' in prop_lower:
                return 'sound_velocity_longitudinal'
            elif 'shear' in prop_lower:
                return 'sound_velocity_shear'
        
        return None

    def _extract_properties_from_comments(self, record: dict, comments: str):
        """Extract properties from comment strings."""
        if not comments:
            return
        
        # Extract Young's modulus
        e_match = re.search(r'E = (\d+)', comments)
        if e_match:
            record['youngs_modulus'] = float(e_match.group(1))
        
        # Extract density
        density_match = re.search(r'density = ([\d.]+)', comments)
        if density_match:
            record['density'] = float(density_match.group(1))

    def _add_typical_properties(self, df: pd.DataFrame, ceramic_system: str) -> pd.DataFrame:
        """Add typical property values for missing data."""
        if df.empty or ceramic_system not in self.typical_properties:
            return df
        
        typical = self.typical_properties[ceramic_system]
        
        for prop, value in typical.items():
            if prop not in df.columns or df[prop].isna().all():
                df[prop] = value
        
        return df

    def convert_system(self, ceramic_system: str) -> pd.DataFrame:
        """Convert NIST data for a specific ceramic system."""
        logger.info(f"🔄 Converting NIST data for {ceramic_system}")
        
        # Look for input file
        input_file = self.base_dir / f"{ceramic_system}.csv"
        
        if not input_file.exists():
            logger.warning(f"No input file found: {input_file}")
            return pd.DataFrame()
        
        # Detect format and parse
        file_format = self.detect_format(input_file)
        logger.info(f"Detected format: {file_format}")
        
        if file_format == 'temperature_matrix':
            df = self.parse_temperature_matrix_format(input_file, ceramic_system)
        elif file_format == 'fracture_database':
            df = self.parse_fracture_database_format(input_file, ceramic_system)
        else:
            logger.error(f"Unknown format for {ceramic_system}")
            return pd.DataFrame()
        
        if df.empty:
            logger.warning(f"No data extracted for {ceramic_system}")
            return df
        
        # Clean and validate data
        df = self._clean_data(df)
        
        # Save converted data
        output_file = self.base_dir / f"{ceramic_system.lower()}_converted.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"✅ Converted {ceramic_system}: {len(df)} records → {output_file}")
        
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the converted data."""
        if df.empty:
            return df
        
        # Remove rows with no property data
        property_cols = ['density', 'youngs_modulus', 'bulk_modulus', 'shear_modulus',
                        'vickers_hardness', 'fracture_toughness', 'compressive_strength']
        
        existing_prop_cols = [col for col in property_cols if col in df.columns]
        if existing_prop_cols:
            df = df.dropna(subset=existing_prop_cols, how='all')
        
        # Apply reasonable value ranges
        if 'density' in df.columns:
            df = df[(df['density'] >= 1.0) & (df['density'] <= 25.0)]
        
        if 'youngs_modulus' in df.columns:
            df = df[(df['youngs_modulus'] >= 10) & (df['youngs_modulus'] <= 1000)]
        
        if 'fracture_toughness' in df.columns:
            df = df[(df['fracture_toughness'] >= 0.5) & (df['fracture_toughness'] <= 20)]
        
        # Add metadata
        df['conversion_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return df

    def convert_all_systems(self) -> Dict[str, pd.DataFrame]:
        """Convert NIST data for all ceramic systems."""
        logger.info("🚀 Converting NIST data for all ceramic systems")
        logger.info("=" * 60)
        
        ceramic_systems = ['Al2O3', 'SiC', 'B4C', 'WC', 'TiC']
        results = {}
        
        for system in ceramic_systems:
            try:
                logger.info(f"\n{'='*20} {system} {'='*20}")
                df = self.convert_system(system)
                results[system] = df
                
                if not df.empty:
                    # Show property summary
                    property_cols = ['density', 'youngs_modulus', 'fracture_toughness', 
                                   'vickers_hardness', 'bulk_modulus']
                    available_props = []
                    
                    for prop in property_cols:
                        if prop in df.columns:
                            non_null_count = df[prop].notna().sum()
                            if non_null_count > 0:
                                values = df[prop].dropna()
                                available_props.append(f"{prop}({non_null_count}): {values.min():.1f}-{values.max():.1f}")
                    
                    if available_props:
                        logger.info(f"Properties: {', '.join(available_props)}")
                
            except Exception as e:
                logger.error(f"❌ Failed to convert {system}: {e}")
                results[system] = pd.DataFrame()
        
        # Generate summary report
        self._generate_conversion_report(results)
        
        return results

    def _generate_conversion_report(self, results: Dict[str, pd.DataFrame]):
        """Generate conversion summary report."""
        logger.info(f"\n{'='*60}")
        logger.info("UNIFIED NIST CONVERSION SUMMARY")
        logger.info(f"{'='*60}")
        
        total_records = 0
        successful_systems = 0
        
        for system, df in results.items():
            record_count = len(df) if not df.empty else 0
            total_records += record_count
            
            if record_count > 0:
                successful_systems += 1
                status = "✅"
                
                # Count unique properties
                property_cols = ['density', 'youngs_modulus', 'fracture_toughness', 
                               'vickers_hardness', 'bulk_modulus', 'shear_modulus']
                prop_count = sum(1 for col in property_cols if col in df.columns and df[col].notna().sum() > 0)
                
                logger.info(f"{system:<8} {record_count:>4} records {status} ({prop_count} properties)")
            else:
                logger.info(f"{system:<8} {record_count:>4} records ❌")
        
        logger.info(f"{'='*60}")
        logger.info(f"Total systems: {len(results)}")
        logger.info(f"Successful: {successful_systems}")
        logger.info(f"Total records: {total_records}")
        logger.info(f"Average per system: {total_records/len(results):.1f}")
        
        # Save detailed report
        report_file = self.base_dir / "unified_conversion_report.txt"
        with open(report_file, 'w') as f:
            f.write("Unified NIST Data Conversion Report\n")
            f.write("=" * 40 + "\n\n")
            
            for system, df in results.items():
                f.write(f"{system}:\n")
                f.write(f"  Records: {len(df)}\n")
                
                if not df.empty:
                    f.write(f"  Columns: {list(df.columns)}\n")
                    
                    # Property statistics
                    property_cols = ['density', 'youngs_modulus', 'fracture_toughness']
                    for prop in property_cols:
                        if prop in df.columns:
                            values = df[prop].dropna()
                            if len(values) > 0:
                                f.write(f"  {prop}: {len(values)} values, {values.min():.2f}-{values.max():.2f}\n")
                
                f.write("\n")
            
            f.write(f"Total: {total_records} records across {successful_systems} systems\n")
        
        logger.info(f"📄 Detailed report saved: {report_file}")


def main():
    """Run unified NIST data conversion."""
    logger.info("🎯 Unified NIST Data Converter")
    logger.info("Converting all ceramic system CSV files to pipeline format")
    logger.info("=" * 60)
    
    try:
        converter = UnifiedNISTConverter()
        results = converter.convert_all_systems()
        
        # Check results
        successful_count = sum(1 for df in results.values() if not df.empty)
        total_records = sum(len(df) for df in results.values())
        
        if successful_count > 0:
            logger.info(f"\n🎉 CONVERSION SUCCESSFUL!")
            logger.info(f"Converted {successful_count}/5 ceramic systems")
            logger.info(f"Total records: {total_records}")
            logger.info(f"\nConverted files:")
            
            for system, df in results.items():
                if not df.empty:
                    output_file = f"data/raw/nist/{system.lower()}_converted.csv"
                    logger.info(f"  ✅ {output_file} ({len(df)} records)")
            
            logger.info(f"\n🚀 Ready for pipeline integration!")
            logger.info(f"Run: python scripts/test_nist_integration.py")
            
            return 0
        else:
            logger.error(f"\n💥 CONVERSION FAILED!")
            logger.error(f"No data could be converted from any system")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Conversion crashed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())