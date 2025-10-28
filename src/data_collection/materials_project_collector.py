"""
Materials Project Data Collection Module

This module provides comprehensive data collection from the Materials Project API
for ceramic armor materials, including elastic tensors, thermal properties,
and mechanical data with robust error handling and retry logic.
"""

import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from tqdm import tqdm
import yaml
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..utils.logger import get_logger
from ..utils.config_loader import load_config

logger = get_logger(__name__)


@dataclass
class MaterialRecord:
    """Data structure for a single material record from Materials Project"""
    material_id: str
    formula: str
    crystal_system: str
    space_group: int
    density: float
    formation_energy: float
    energy_above_hull: float
    band_gap: float
    elastic_properties: Dict[str, float]
    thermal_properties: Dict[str, float]
    structure_data: Dict[str, Any]


class MaterialsProjectCollector:
    """
    Comprehensive Materials Project data collector for ceramic armor materials.
    
    Features:
    - Complete property coverage: elastic tensors, thermal, mechanical, electronic
    - Exponential backoff retry mechanism (2^n seconds, max 5 retries)
    - Progress bars for long queries using tqdm
    - Intermediate result saving for crash recovery
    - Comprehensive error logging with API response details
    """
    
    def __init__(self, api_key: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize Materials Project collector.
        
        Args:
            api_key: Materials Project API key. If None, loads from config
            config_path: Path to configuration file
        """
        self.api_key = api_key
        self.config = self._load_config(config_path)
        self.base_url = "https://api.materialsproject.org"
        self.session = requests.Session()
        self.session.headers.update({'X-API-KEY': self._get_api_key()})
        
        # Rate limiting and retry configuration
        self.max_retries = 5
        self.base_delay = 1  # Base delay for exponential backoff
        self.rate_limit_delay = 2  # Delay between requests to respect rate limits
        self.timeout = 30
        
        # Thread safety
        self._lock = threading.Lock()
        self._request_count = 0
        self._last_request_time = 0
        
        logger.info("MaterialsProjectCollector initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if config_path:
            return load_config(config_path)
        
        try:
            return load_config("config/config.yaml")
        except FileNotFoundError:
            logger.warning("Config file not found, using default configuration")
            return {
                'ceramic_systems': {
                    'primary': ['SiC', 'Al2O3', 'B4C', 'WC', 'TiC']
                },
                'data_sources': {
                    'materials_project': {
                        'expected_entries': 3500
                    }
                }
            }
    
    def _get_api_key(self) -> str:
        """Get API key from various sources"""
        if self.api_key:
            return self.api_key
        
        # Try to load from api_keys.yaml
        try:
            with open("config/api_keys.yaml", 'r') as f:
                keys = yaml.safe_load(f)
                if 'materials_project' in keys:
                    return keys['materials_project']
        except FileNotFoundError:
            pass
        
        # Try environment variable
        import os
        api_key = os.getenv('MATERIALS_PROJECT_API_KEY')
        if api_key:
            return api_key
        
        raise ValueError(
            "Materials Project API key not found. Please:\n"
            "1. Pass api_key parameter, or\n"
            "2. Set MATERIALS_PROJECT_API_KEY environment variable, or\n"
            "3. Create config/api_keys.yaml with materials_project key"
        )
    
    def _make_request_with_retry(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make API request with exponential backoff retry logic.
        
        Args:
            url: API endpoint URL
            params: Request parameters
            
        Returns:
            API response data or None if all retries failed
        """
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                with self._lock:
                    current_time = time.time()
                    if current_time - self._last_request_time < self.rate_limit_delay:
                        time.sleep(self.rate_limit_delay - (current_time - self._last_request_time))
                    self._last_request_time = time.time()
                    self._request_count += 1
                
                response = self.session.get(url, params=params, timeout=self.timeout)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit exceeded
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limit exceeded, waiting {retry_after} seconds")
                    time.sleep(retry_after)
                    continue
                elif response.status_code == 401:
                    logger.error("Authentication failed - check API key")
                    raise ValueError("Invalid API key")
                else:
                    logger.warning(f"Request failed with status {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error on attempt {attempt + 1}: {e}")
            
            # Exponential backoff
            if attempt < self.max_retries - 1:
                delay = self.base_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(delay)
        
        logger.error(f"All {self.max_retries} attempts failed for URL: {url}")
        return None
    
    def _query_materials_by_formula(self, formula_pattern: str) -> List[Dict[str, Any]]:
        """
        Query materials by chemical formula pattern.
        
        Args:
            formula_pattern: Chemical formula pattern (e.g., "SiC", "Al2O3")
            
        Returns:
            List of material data dictionaries
        """
        url = f"{self.base_url}/materials/summary"
        params = {
            'formula': formula_pattern,
            '_fields': 'material_id,formula_pretty,crystal_system,symmetry,density,formation_energy_per_atom,energy_above_hull,band_gap,structure'
        }
        
        logger.info(f"Querying materials with formula pattern: {formula_pattern}")
        response_data = self._make_request_with_retry(url, params)
        
        if response_data and 'data' in response_data:
            materials = response_data['data']
            logger.info(f"Found {len(materials)} materials for {formula_pattern}")
            return materials
        else:
            logger.warning(f"No materials found for {formula_pattern}")
            return []
    
    def _get_elastic_properties(self, material_id: str) -> Dict[str, float]:
        """
        Get elastic tensor properties for a material.
        
        Args:
            material_id: Materials Project ID
            
        Returns:
            Dictionary of elastic properties including C11, C12, C44 for Cauchy pressure
        """
        url = f"{self.base_url}/materials/{material_id}/elasticity"
        
        response_data = self._make_request_with_retry(url, {})
        
        if not response_data or 'data' not in response_data:
            return {}
        
        elastic_data = response_data['data']
        if not elastic_data:
            return {}
        
        # Extract elastic tensor components
        elastic_props = {}
        
        try:
            # Get the first (and usually only) elastic data entry
            elastic_entry = elastic_data[0] if isinstance(elastic_data, list) else elastic_data
            
            # Extract bulk and shear moduli
            if 'k_vrh' in elastic_entry:
                elastic_props['bulk_modulus'] = elastic_entry['k_vrh']
            if 'g_vrh' in elastic_entry:
                elastic_props['shear_modulus'] = elastic_entry['g_vrh']
            
            # Extract elastic tensor components for Cauchy pressure calculation
            if 'elastic_tensor' in elastic_entry:
                tensor = elastic_entry['elastic_tensor']
                if isinstance(tensor, list) and len(tensor) >= 6:
                    # Elastic tensor is typically 6x6 in Voigt notation
                    # C11, C12, C44 are needed for Cauchy pressure = C12 - C44
                    elastic_props['C11'] = tensor[0][0] if len(tensor[0]) > 0 else None
                    elastic_props['C12'] = tensor[0][1] if len(tensor[0]) > 1 else None
                    elastic_props['C44'] = tensor[3][3] if len(tensor) > 3 and len(tensor[3]) > 3 else None
                    
                    # Calculate Cauchy pressure if components available
                    if elastic_props.get('C12') is not None and elastic_props.get('C44') is not None:
                        elastic_props['cauchy_pressure'] = elastic_props['C12'] - elastic_props['C44']
            
            # Calculate derived properties
            if 'bulk_modulus' in elastic_props and 'shear_modulus' in elastic_props:
                B = elastic_props['bulk_modulus']
                G = elastic_props['shear_modulus']
                
                # Young's modulus: E = 9BG/(3B + G)
                elastic_props['youngs_modulus'] = (9 * B * G) / (3 * B + G)
                
                # Poisson's ratio: ν = (3B - 2G)/(6B + 2G)
                elastic_props['poisson_ratio'] = (3 * B - 2 * G) / (6 * B + 2 * G)
                
                # Pugh ratio: G/B (brittleness indicator)
                elastic_props['pugh_ratio'] = G / B
            
        except (KeyError, IndexError, TypeError, ZeroDivisionError) as e:
            logger.warning(f"Error extracting elastic properties for {material_id}: {e}")
        
        return elastic_props
    
    def _get_thermal_properties(self, material_id: str) -> Dict[str, float]:
        """
        Get thermal properties for a material using separate thermal endpoint.
        
        Args:
            material_id: Materials Project ID
            
        Returns:
            Dictionary of thermal properties
        """
        # Try multiple thermal property endpoints
        thermal_props = {}
        
        # Thermal properties from phonon calculations
        url = f"{self.base_url}/materials/{material_id}/phonon"
        response_data = self._make_request_with_retry(url, {})
        
        if response_data and 'data' in response_data:
            phonon_data = response_data['data']
            if phonon_data:
                try:
                    phonon_entry = phonon_data[0] if isinstance(phonon_data, list) else phonon_data
                    
                    # Extract thermal conductivity if available
                    if 'thermal_conductivity' in phonon_entry:
                        thermal_props['thermal_conductivity'] = phonon_entry['thermal_conductivity']
                    
                    # Extract heat capacity
                    if 'heat_capacity' in phonon_entry:
                        thermal_props['heat_capacity'] = phonon_entry['heat_capacity']
                        
                except (KeyError, IndexError, TypeError) as e:
                    logger.debug(f"Error extracting phonon data for {material_id}: {e}")
        
        # Try dielectric properties for additional thermal data
        url = f"{self.base_url}/materials/{material_id}/dielectric"
        response_data = self._make_request_with_retry(url, {})
        
        if response_data and 'data' in response_data:
            dielectric_data = response_data['data']
            if dielectric_data:
                try:
                    dielectric_entry = dielectric_data[0] if isinstance(dielectric_data, list) else dielectric_data
                    
                    # Extract refractive index (related to thermal properties)
                    if 'n' in dielectric_entry:
                        thermal_props['refractive_index'] = dielectric_entry['n']
                        
                except (KeyError, IndexError, TypeError) as e:
                    logger.debug(f"Error extracting dielectric data for {material_id}: {e}")
        
        return thermal_props
    
    def _process_material(self, material_data: Dict[str, Any]) -> Optional[MaterialRecord]:
        """
        Process a single material entry to create a complete MaterialRecord.
        
        Args:
            material_data: Raw material data from API
            
        Returns:
            MaterialRecord or None if processing failed
        """
        try:
            material_id = material_data['material_id']
            
            # Get elastic properties
            elastic_props = self._get_elastic_properties(material_id)
            
            # Get thermal properties
            thermal_props = self._get_thermal_properties(material_id)
            
            # Extract basic properties
            record = MaterialRecord(
                material_id=material_id,
                formula=material_data.get('formula_pretty', ''),
                crystal_system=material_data.get('crystal_system', ''),
                space_group=material_data.get('symmetry', {}).get('number', 0),
                density=material_data.get('density', 0.0),
                formation_energy=material_data.get('formation_energy_per_atom', 0.0),
                energy_above_hull=material_data.get('energy_above_hull', 0.0),
                band_gap=material_data.get('band_gap', 0.0),
                elastic_properties=elastic_props,
                thermal_properties=thermal_props,
                structure_data=material_data.get('structure', {})
            )
            
            return record
            
        except Exception as e:
            logger.error(f"Error processing material {material_data.get('material_id', 'unknown')}: {e}")
            return None
    
    def collect_ceramic_materials(self, 
                                ceramic_systems: Optional[List[str]] = None,
                                max_materials_per_system: Optional[int] = None,
                                save_intermediate: bool = True,
                                output_dir: str = "data/raw") -> pd.DataFrame:
        """
        Collect comprehensive ceramic materials data from Materials Project.
        
        Args:
            ceramic_systems: List of ceramic systems to collect (e.g., ['SiC', 'Al2O3'])
            max_materials_per_system: Maximum materials per system (None for all)
            save_intermediate: Whether to save intermediate results
            output_dir: Directory to save intermediate results
            
        Returns:
            DataFrame with collected materials data
        """
        if ceramic_systems is None:
            ceramic_systems = self.config.get('ceramic_systems', {}).get('primary', ['SiC', 'Al2O3', 'B4C', 'WC', 'TiC'])
        
        logger.info(f"Starting collection for ceramic systems: {ceramic_systems}")
        
        all_materials = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for system in ceramic_systems:
            logger.info(f"Collecting materials for {system}")
            
            # Query materials for this system
            materials_data = self._query_materials_by_formula(system)
            
            if max_materials_per_system:
                materials_data = materials_data[:max_materials_per_system]
            
            if not materials_data:
                logger.warning(f"No materials found for {system}")
                continue
            
            # Process materials with progress bar
            system_materials = []
            with tqdm(total=len(materials_data), desc=f"Processing {system}") as pbar:
                for material_data in materials_data:
                    record = self._process_material(material_data)
                    if record:
                        system_materials.append(record)
                    pbar.update(1)
                    
                    # Small delay to respect rate limits
                    time.sleep(0.1)
            
            logger.info(f"Successfully processed {len(system_materials)} materials for {system}")
            all_materials.extend(system_materials)
            
            # Save intermediate results
            if save_intermediate and system_materials:
                intermediate_file = output_path / f"{system}_materials.json"
                self._save_intermediate_results(system_materials, intermediate_file)
        
        # Convert to DataFrame
        df = self._materials_to_dataframe(all_materials)
        
        # Save final results
        if save_intermediate:
            final_file = output_path / "materials_project_data.csv"
            df.to_csv(final_file, index=False)
            logger.info(f"Final results saved to {final_file}")
        
        logger.info(f"Collection complete: {len(df)} total materials collected")
        return df
    
    def _save_intermediate_results(self, materials: List[MaterialRecord], filepath: Path):
        """Save intermediate results to JSON file"""
        try:
            # Convert MaterialRecord objects to dictionaries
            materials_dict = []
            for material in materials:
                material_dict = {
                    'material_id': material.material_id,
                    'formula': material.formula,
                    'crystal_system': material.crystal_system,
                    'space_group': material.space_group,
                    'density': material.density,
                    'formation_energy': material.formation_energy,
                    'energy_above_hull': material.energy_above_hull,
                    'band_gap': material.band_gap,
                    'elastic_properties': material.elastic_properties,
                    'thermal_properties': material.thermal_properties,
                    'structure_data': material.structure_data
                }
                materials_dict.append(material_dict)
            
            with open(filepath, 'w') as f:
                json.dump(materials_dict, f, indent=2, default=str)
            
            logger.info(f"Intermediate results saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving intermediate results to {filepath}: {e}")
    
    def _materials_to_dataframe(self, materials: List[MaterialRecord]) -> pd.DataFrame:
        """Convert list of MaterialRecord objects to pandas DataFrame"""
        if not materials:
            return pd.DataFrame()
        
        data = []
        for material in materials:
            row = {
                'material_id': material.material_id,
                'formula': material.formula,
                'crystal_system': material.crystal_system,
                'space_group': material.space_group,
                'density': material.density,
                'formation_energy': material.formation_energy,
                'energy_above_hull': material.energy_above_hull,
                'band_gap': material.band_gap
            }
            
            # Add elastic properties with prefix
            for key, value in material.elastic_properties.items():
                row[f'elastic_{key}'] = value
            
            # Add thermal properties with prefix
            for key, value in material.thermal_properties.items():
                row[f'thermal_{key}'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection session"""
        return {
            'total_requests': self._request_count,
            'session_duration': time.time() - self._last_request_time if self._last_request_time else 0,
            'average_request_rate': self._request_count / max(1, time.time() - self._last_request_time) if self._last_request_time else 0
        }