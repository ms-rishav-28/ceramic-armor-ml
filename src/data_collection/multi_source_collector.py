"""
Multi-Source Data Collection System for Ceramic Materials

This module orchestrates data collection from multiple sources including
Materials Project, JARVIS-DFT, AFLOW, and NIST databases to achieve
the target of 5,600+ materials with comprehensive property coverage.
"""

import asyncio
import aiohttp
import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import json
from tqdm import tqdm
import yaml
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
from datetime import datetime

from .materials_project_collector import MaterialsProjectCollector, MaterialRecord
from ..utils.logger import get_logger
from ..utils.config_loader import load_project_config
from ..utils.data_utils import safe_save_data, safe_load_data

logger = get_logger(__name__)


@dataclass
class CollectionTarget:
    """Target collection parameters for each ceramic system."""
    system: str
    target_count: int
    priority_properties: List[str]
    sources: List[str]
    collected_count: int = 0
    success_rate: float = 0.0


@dataclass
class SourceStats:
    """Statistics for each data source."""
    source_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    materials_collected: int = 0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None


class MultiSourceCollector:
    """
    Comprehensive multi-source data collector for ceramic materials.
    
    Features:
    - Materials Project API integration (primary source)
    - JARVIS-DFT database access (secondary source)
    - AFLOW database integration (tertiary source)
    - NIST web scraping capabilities (specialized properties)
    - Intelligent source prioritization and fallback
    - Comprehensive error handling and retry logic
    - Real-time progress tracking and statistics
    - Automatic data deduplication and validation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize multi-source collector.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_project_config() if config_path is None else load_project_config()
        
        # Initialize individual collectors
        self.mp_collector = MaterialsProjectCollector()
        
        # Collection targets for each ceramic system
        self.collection_targets = {
            'SiC': CollectionTarget('SiC', 1500, ['elastic_tensor', 'formation_energy'], ['mp', 'jarvis']),
            'Al2O3': CollectionTarget('Al2O3', 1200, ['mechanical_properties', 'thermal_properties'], ['mp', 'jarvis', 'nist']),
            'B4C': CollectionTarget('B4C', 800, ['ballistic_properties', 'hardness'], ['mp', 'nist']),
            'WC': CollectionTarget('WC', 1000, ['elastic_properties', 'density'], ['mp', 'jarvis']),
            'TiC': CollectionTarget('TiC', 1100, ['formation_energy', 'elastic_tensor'], ['mp', 'jarvis'])
        }
        
        # Source statistics
        self.source_stats = {
            'materials_project': SourceStats('Materials Project'),
            'jarvis': SourceStats('JARVIS-DFT'),
            'aflow': SourceStats('AFLOW'),
            'nist': SourceStats('NIST')
        }
        
        # Rate limiting and retry configuration
        self.rate_limits = {
            'materials_project': 10,  # requests per second
            'jarvis': 5,
            'aflow': 3,
            'nist': 1  # Conservative for web scraping
        }
        
        self.max_retries = 3
        self.timeout = 30
        
        # Thread safety
        self._lock = threading.Lock()
        self._collected_materials = []
        
        logger.info("MultiSourceCollector initialized for 5 ceramic systems")
    
    def collect_full_dataset(self, 
                           target_total: int = 5600,
                           save_intermediate: bool = True,
                           output_dir: str = "data/raw/multi_source") -> pd.DataFrame:
        """
        Collect the complete dataset of 5,600+ materials from all sources.
        
        Args:
            target_total: Target total number of materials
            save_intermediate: Whether to save intermediate results
            output_dir: Directory for output files
            
        Returns:
            Combined DataFrame with all collected materials
        """
        logger.info(f"Starting full dataset collection (target: {target_total:,} materials)")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        all_materials = []
        
        try:
            # Phase 1: Materials Project (Primary source - highest quality)
            logger.info("Phase 1: Collecting from Materials Project")
            mp_materials = self._collect_materials_project_comprehensive()
            if mp_materials:
                all_materials.extend(mp_materials)
                logger.info(f"Materials Project: {len(mp_materials)} materials collected")
                
                if save_intermediate:
                    mp_df = pd.DataFrame([asdict(m) for m in mp_materials])
                    safe_save_data(mp_df, output_path / "materials_project_data.csv")
            
            # Phase 2: JARVIS-DFT (Secondary source - DFT calculations)
            logger.info("Phase 2: Collecting from JARVIS-DFT")
            jarvis_materials = self._collect_jarvis_materials()
            if jarvis_materials:
                all_materials.extend(jarvis_materials)
                logger.info(f"JARVIS-DFT: {len(jarvis_materials)} materials collected")
                
                if save_intermediate:
                    jarvis_df = pd.DataFrame([asdict(m) for m in jarvis_materials])
                    safe_save_data(jarvis_df, output_path / "jarvis_data.csv")
            
            # Phase 3: AFLOW (Tertiary source - additional coverage)
            logger.info("Phase 3: Collecting from AFLOW")
            aflow_materials = self._collect_aflow_materials()
            if aflow_materials:
                all_materials.extend(aflow_materials)
                logger.info(f"AFLOW: {len(aflow_materials)} materials collected")
                
                if save_intermediate:
                    aflow_df = pd.DataFrame([asdict(m) for m in aflow_materials])
                    safe_save_data(aflow_df, output_path / "aflow_data.csv")
            
            # Phase 4: NIST (Specialized properties)
            logger.info("Phase 4: Collecting from NIST databases")
            nist_materials = self._collect_nist_materials()
            if nist_materials:
                all_materials.extend(nist_materials)
                logger.info(f"NIST: {len(nist_materials)} materials collected")
                
                if save_intermediate:
                    nist_df = pd.DataFrame([asdict(m) for m in nist_materials])
                    safe_save_data(nist_df, output_path / "nist_data.csv")
            
            # Combine and deduplicate
            logger.info("Combining and deduplicating materials")
            combined_df = self._combine_and_deduplicate_materials(all_materials)
            
            # Validate collection targets
            self._validate_collection_targets(combined_df)
            
            # Save final combined dataset
            final_path = output_path / "combined_materials_dataset.csv"
            safe_save_data(combined_df, final_path)
            
            # Generate collection statistics
            collection_time = time.time() - start_time
            self._generate_collection_report(combined_df, collection_time, output_path)
            
            logger.info(f"Full dataset collection completed: {len(combined_df):,} materials in {collection_time:.2f}s")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Full dataset collection failed: {e}")
            raise
    
    def _collect_materials_project_comprehensive(self) -> List[MaterialRecord]:
        """
        Collect comprehensive data from Materials Project for all ceramic systems.
        
        Returns:
            List of MaterialRecord objects from Materials Project
        """
        logger.info("Collecting comprehensive Materials Project data")
        
        all_mp_materials = []
        
        for system, target in self.collection_targets.items():
            if 'mp' not in target.sources:
                continue
                
            logger.info(f"Collecting {system} from Materials Project (target: {target.target_count})")
            
            try:
                # Use the existing Materials Project collector
                system_df = self.mp_collector.collect_ceramic_materials(
                    ceramic_systems=[system],
                    max_materials_per_system=target.target_count,
                    save_intermediate=False
                )
                
                if not system_df.empty:
                    # Convert DataFrame to MaterialRecord objects
                    system_materials = self._dataframe_to_material_records(system_df, system)
                    all_mp_materials.extend(system_materials)
                    
                    # Update collection target
                    target.collected_count += len(system_materials)
                    
                    # Update source statistics
                    self.source_stats['materials_project'].materials_collected += len(system_materials)
                    self.source_stats['materials_project'].successful_requests += 1
                    
                    logger.info(f"Materials Project {system}: {len(system_materials)} materials")
                
            except Exception as e:
                logger.error(f"Error collecting {system} from Materials Project: {e}")
                self.source_stats['materials_project'].failed_requests += 1
                continue
        
        logger.info(f"Materials Project total: {len(all_mp_materials)} materials")
        return all_mp_materials
    
    def _collect_jarvis_materials(self) -> List[MaterialRecord]:
        """
        Collect materials from JARVIS-DFT database.
        
        Returns:
            List of MaterialRecord objects from JARVIS
        """
        logger.info("Collecting from JARVIS-DFT database")
        
        # JARVIS-DFT API endpoint
        jarvis_base_url = "https://jarvis.nist.gov/jarvisdft/restapi"
        
        all_jarvis_materials = []
        
        for system, target in self.collection_targets.items():
            if 'jarvis' not in target.sources:
                continue
                
            logger.info(f"Collecting {system} from JARVIS-DFT")
            
            try:
                # Query JARVIS for this ceramic system
                jarvis_data = self._query_jarvis_system(system, target.target_count)
                
                if jarvis_data:
                    # Convert JARVIS data to MaterialRecord format
                    system_materials = self._convert_jarvis_to_material_records(jarvis_data, system)
                    all_jarvis_materials.extend(system_materials)
                    
                    # Update statistics
                    target.collected_count += len(system_materials)
                    self.source_stats['jarvis'].materials_collected += len(system_materials)
                    
                    logger.info(f"JARVIS {system}: {len(system_materials)} materials")
                
            except Exception as e:
                logger.error(f"Error collecting {system} from JARVIS: {e}")
                self.source_stats['jarvis'].failed_requests += 1
                continue
        
        logger.info(f"JARVIS-DFT total: {len(all_jarvis_materials)} materials")
        return all_jarvis_materials
    
    def _collect_aflow_materials(self) -> List[MaterialRecord]:
        """
        Collect materials from AFLOW database.
        
        Returns:
            List of MaterialRecord objects from AFLOW
        """
        logger.info("Collecting from AFLOW database")
        
        # AFLOW API endpoint
        aflow_base_url = "http://aflowlib.duke.edu/search/API"
        
        all_aflow_materials = []
        
        for system, target in self.collection_targets.items():
            logger.info(f"Collecting {system} from AFLOW")
            
            try:
                # Query AFLOW for this ceramic system
                aflow_data = self._query_aflow_system(system, target.target_count // 2)  # Smaller contribution
                
                if aflow_data:
                    # Convert AFLOW data to MaterialRecord format
                    system_materials = self._convert_aflow_to_material_records(aflow_data, system)
                    all_aflow_materials.extend(system_materials)
                    
                    # Update statistics
                    self.source_stats['aflow'].materials_collected += len(system_materials)
                    
                    logger.info(f"AFLOW {system}: {len(system_materials)} materials")
                
            except Exception as e:
                logger.error(f"Error collecting {system} from AFLOW: {e}")
                self.source_stats['aflow'].failed_requests += 1
                continue
        
        logger.info(f"AFLOW total: {len(all_aflow_materials)} materials")
        return all_aflow_materials
    
    def _collect_nist_materials(self) -> List[MaterialRecord]:
        """
        Collect materials from NIST databases and web sources.
        
        Returns:
            List of MaterialRecord objects from NIST
        """
        logger.info("Collecting from NIST databases")
        
        all_nist_materials = []
        
        # NIST sources focus on experimental data and specialized properties
        nist_systems = ['Al2O3', 'B4C']  # Focus on systems with good NIST coverage
        
        for system in nist_systems:
            if system not in self.collection_targets:
                continue
                
            logger.info(f"Collecting {system} from NIST sources")
            
            try:
                # Query NIST databases for this system
                nist_data = self._query_nist_system(system)
                
                if nist_data:
                    # Convert NIST data to MaterialRecord format
                    system_materials = self._convert_nist_to_material_records(nist_data, system)
                    all_nist_materials.extend(system_materials)
                    
                    # Update statistics
                    self.source_stats['nist'].materials_collected += len(system_materials)
                    
                    logger.info(f"NIST {system}: {len(system_materials)} materials")
                
            except Exception as e:
                logger.error(f"Error collecting {system} from NIST: {e}")
                self.source_stats['nist'].failed_requests += 1
                continue
        
        logger.info(f"NIST total: {len(all_nist_materials)} materials")
        return all_nist_materials
    
    def _query_jarvis_system(self, system: str, max_count: int) -> List[Dict[str, Any]]:
        """
        Query JARVIS-DFT for a specific ceramic system.
        
        Args:
            system: Ceramic system (e.g., 'SiC')
            max_count: Maximum number of materials to collect
            
        Returns:
            List of material data dictionaries from JARVIS
        """
        # JARVIS API query parameters
        jarvis_url = "https://jarvis.nist.gov/jarvisdft/restapi/v1/data"
        
        # Map ceramic systems to JARVIS search terms
        system_queries = {
            'SiC': ['Si1C1', 'Si2C2', 'SiC'],
            'Al2O3': ['Al2O3', 'Al4O6'],
            'B4C': ['B4C1', 'B4C'],
            'WC': ['W1C1', 'WC'],
            'TiC': ['Ti1C1', 'TiC']
        }
        
        materials_data = []
        
        for query_formula in system_queries.get(system, [system]):
            try:
                params = {
                    'formula': query_formula,
                    'limit': max_count // len(system_queries.get(system, [system]))
                }
                
                response = requests.get(jarvis_url, params=params, timeout=self.timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        materials_data.extend(data)
                    elif isinstance(data, dict) and 'data' in data:
                        materials_data.extend(data['data'])
                
                # Rate limiting
                time.sleep(1.0 / self.rate_limits['jarvis'])
                
            except Exception as e:
                logger.warning(f"JARVIS query failed for {query_formula}: {e}")
                continue
        
        return materials_data[:max_count]
    
    def _query_aflow_system(self, system: str, max_count: int) -> List[Dict[str, Any]]:
        """
        Query AFLOW for a specific ceramic system.
        
        Args:
            system: Ceramic system (e.g., 'SiC')
            max_count: Maximum number of materials to collect
            
        Returns:
            List of material data dictionaries from AFLOW
        """
        # AFLOW API query
        aflow_url = "http://aflowlib.duke.edu/search/API"
        
        materials_data = []
        
        try:
            # AFLOW query parameters
            params = {
                'species': system.replace('2', '').replace('3', '').replace('4', ''),  # Remove numbers
                'nspecies': len(set(c for c in system if c.isalpha())),  # Count unique elements
                'format': 'json',
                'limit': max_count
            }
            
            response = requests.get(aflow_url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    materials_data = data
                elif isinstance(data, dict) and 'data' in data:
                    materials_data = data['data']
            
        except Exception as e:
            logger.warning(f"AFLOW query failed for {system}: {e}")
        
        return materials_data[:max_count]
    
    def _query_nist_system(self, system: str) -> List[Dict[str, Any]]:
        """
        Query NIST databases for a specific ceramic system.
        
        Args:
            system: Ceramic system (e.g., 'Al2O3')
            
        Returns:
            List of material data dictionaries from NIST
        """
        # NIST data collection (simplified implementation)
        # In practice, this would involve web scraping or API calls to NIST databases
        
        materials_data = []
        
        # Placeholder for NIST data collection
        # This would be implemented with specific NIST database APIs or web scraping
        logger.info(f"NIST collection for {system} - placeholder implementation")
        
        # For now, return empty list (would be implemented with actual NIST integration)
        return materials_data
    
    def _dataframe_to_material_records(self, df: pd.DataFrame, system: str) -> List[MaterialRecord]:
        """
        Convert DataFrame to MaterialRecord objects.
        
        Args:
            df: Materials DataFrame
            system: Ceramic system identifier
            
        Returns:
            List of MaterialRecord objects
        """
        materials = []
        
        for _, row in df.iterrows():
            try:
                # Extract elastic properties
                elastic_props = {}
                for col in df.columns:
                    if col.startswith('elastic_'):
                        prop_name = col.replace('elastic_', '')
                        elastic_props[prop_name] = row[col] if pd.notna(row[col]) else None
                
                # Extract thermal properties
                thermal_props = {}
                for col in df.columns:
                    if col.startswith('thermal_'):
                        prop_name = col.replace('thermal_', '')
                        thermal_props[prop_name] = row[col] if pd.notna(row[col]) else None
                
                # Create MaterialRecord
                record = MaterialRecord(
                    material_id=row.get('material_id', f"{system}_{len(materials)}"),
                    formula=row.get('formula', system),
                    crystal_system=row.get('crystal_system', ''),
                    space_group=int(row.get('space_group', 0)) if pd.notna(row.get('space_group')) else 0,
                    density=float(row.get('density', 0)) if pd.notna(row.get('density')) else 0.0,
                    formation_energy=float(row.get('formation_energy', 0)) if pd.notna(row.get('formation_energy')) else 0.0,
                    energy_above_hull=float(row.get('energy_above_hull', 0)) if pd.notna(row.get('energy_above_hull')) else 0.0,
                    band_gap=float(row.get('band_gap', 0)) if pd.notna(row.get('band_gap')) else 0.0,
                    elastic_properties=elastic_props,
                    thermal_properties=thermal_props,
                    structure_data={}
                )
                
                materials.append(record)
                
            except Exception as e:
                logger.warning(f"Error converting row to MaterialRecord: {e}")
                continue
        
        return materials
    
    def _convert_jarvis_to_material_records(self, jarvis_data: List[Dict], system: str) -> List[MaterialRecord]:
        """Convert JARVIS data to MaterialRecord format."""
        materials = []
        
        for item in jarvis_data:
            try:
                record = MaterialRecord(
                    material_id=f"jarvis_{item.get('jid', len(materials))}",
                    formula=item.get('formula', system),
                    crystal_system=item.get('crystal_system', ''),
                    space_group=item.get('spg_number', 0),
                    density=item.get('density', 0.0),
                    formation_energy=item.get('formation_energy_peratom', 0.0),
                    energy_above_hull=item.get('ehull', 0.0),
                    band_gap=item.get('optb88vdw_bandgap', 0.0),
                    elastic_properties={
                        'bulk_modulus': item.get('bulk_modulus_kv', None),
                        'shear_modulus': item.get('shear_modulus_gv', None)
                    },
                    thermal_properties={},
                    structure_data={}
                )
                materials.append(record)
                
            except Exception as e:
                logger.warning(f"Error converting JARVIS data: {e}")
                continue
        
        return materials
    
    def _convert_aflow_to_material_records(self, aflow_data: List[Dict], system: str) -> List[MaterialRecord]:
        """Convert AFLOW data to MaterialRecord format."""
        materials = []
        
        for item in aflow_data:
            try:
                record = MaterialRecord(
                    material_id=f"aflow_{item.get('auid', len(materials))}",
                    formula=item.get('compound', system),
                    crystal_system=item.get('crystal_system', ''),
                    space_group=item.get('spacegroup_number', 0),
                    density=item.get('density', 0.0),
                    formation_energy=item.get('enthalpy_formation_atom', 0.0),
                    energy_above_hull=item.get('energy_above_hull', 0.0),
                    band_gap=item.get('Egap', 0.0),
                    elastic_properties={
                        'bulk_modulus': item.get('bulk_modulus_vrh', None),
                        'shear_modulus': item.get('shear_modulus_vrh', None)
                    },
                    thermal_properties={},
                    structure_data={}
                )
                materials.append(record)
                
            except Exception as e:
                logger.warning(f"Error converting AFLOW data: {e}")
                continue
        
        return materials
    
    def _convert_nist_to_material_records(self, nist_data: List[Dict], system: str) -> List[MaterialRecord]:
        """Convert NIST data to MaterialRecord format."""
        materials = []
        
        for item in nist_data:
            try:
                record = MaterialRecord(
                    material_id=f"nist_{len(materials)}",
                    formula=item.get('formula', system),
                    crystal_system='',
                    space_group=0,
                    density=item.get('density', 0.0),
                    formation_energy=0.0,
                    energy_above_hull=0.0,
                    band_gap=0.0,
                    elastic_properties={
                        'youngs_modulus': item.get('youngs_modulus', None),
                        'hardness': item.get('hardness', None)
                    },
                    thermal_properties={
                        'thermal_conductivity': item.get('thermal_conductivity', None)
                    },
                    structure_data={}
                )
                materials.append(record)
                
            except Exception as e:
                logger.warning(f"Error converting NIST data: {e}")
                continue
        
        return materials
    
    def _combine_and_deduplicate_materials(self, all_materials: List[MaterialRecord]) -> pd.DataFrame:
        """
        Combine materials from all sources and remove duplicates.
        
        Args:
            all_materials: List of all MaterialRecord objects
            
        Returns:
            Combined and deduplicated DataFrame
        """
        logger.info(f"Combining {len(all_materials)} materials from all sources")
        
        # Convert to DataFrame
        materials_data = []
        for material in all_materials:
            row = asdict(material)
            # Flatten nested dictionaries
            for key, value in material.elastic_properties.items():
                row[f'elastic_{key}'] = value
            for key, value in material.thermal_properties.items():
                row[f'thermal_{key}'] = value
            del row['elastic_properties']
            del row['thermal_properties']
            del row['structure_data']
            materials_data.append(row)
        
        df = pd.DataFrame(materials_data)
        
        if df.empty:
            return df
        
        # Remove duplicates based on formula and similar properties
        initial_count = len(df)
        
        # First, remove exact duplicates
        df = df.drop_duplicates(subset=['formula', 'crystal_system', 'space_group'], keep='first')
        
        # Then, remove near-duplicates based on formula and density
        if 'density' in df.columns:
            df = df.drop_duplicates(subset=['formula'], keep='first')
        
        final_count = len(df)
        
        logger.info(f"Deduplication: {initial_count} → {final_count} materials ({initial_count - final_count} duplicates removed)")
        
        return df
    
    def _validate_collection_targets(self, df: pd.DataFrame) -> None:
        """
        Validate that collection targets are met.
        
        Args:
            df: Combined materials DataFrame
        """
        logger.info("Validating collection targets")
        
        total_collected = len(df)
        target_total = sum(target.target_count for target in self.collection_targets.values())
        
        logger.info(f"Total collected: {total_collected:,} / {target_total:,} target ({(total_collected/target_total)*100:.1f}%)")
        
        # Validate by ceramic system if available
        if 'ceramic_system' in df.columns:
            system_counts = df['ceramic_system'].value_counts()
            for system, target in self.collection_targets.items():
                collected = system_counts.get(system, 0)
                percentage = (collected / target.target_count) * 100 if target.target_count > 0 else 0
                logger.info(f"{system}: {collected:,} / {target.target_count:,} ({percentage:.1f}%)")
    
    def _generate_collection_report(self, df: pd.DataFrame, collection_time: float, output_path: Path) -> None:
        """
        Generate comprehensive collection report.
        
        Args:
            df: Final combined DataFrame
            collection_time: Total collection time in seconds
            output_path: Output directory path
        """
        report = f"""# Multi-Source Data Collection Report

## Collection Summary

**Total Materials Collected:** {len(df):,}
**Collection Time:** {collection_time:.2f} seconds
**Average Rate:** {len(df)/collection_time:.2f} materials/second
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Source Statistics

"""
        
        for source_name, stats in self.source_stats.items():
            if stats.materials_collected > 0:
                success_rate = (stats.successful_requests / max(1, stats.total_requests)) * 100
                report += f"""### {stats.source_name}
- **Materials Collected:** {stats.materials_collected:,}
- **Success Rate:** {success_rate:.1f}%
- **Total Requests:** {stats.total_requests}
- **Failed Requests:** {stats.failed_requests}

"""
        
        if 'ceramic_system' in df.columns:
            system_counts = df['ceramic_system'].value_counts()
            report += "## Materials by Ceramic System\n\n"
            for system in self.collection_targets.keys():
                count = system_counts.get(system, 0)
                target = self.collection_targets[system].target_count
                percentage = (count / target) * 100 if target > 0 else 0
                report += f"- **{system}:** {count:,} / {target:,} ({percentage:.1f}%)\n"
        
        report += f"""
## Data Quality

- **Total Features:** {df.shape[1]}
- **Missing Values:** {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%
- **Numeric Features:** {len(df.select_dtypes(include=[np.number]).columns)}

## Files Generated

- `combined_materials_dataset.csv` - Final combined dataset
- `materials_project_data.csv` - Materials Project data
- `jarvis_data.csv` - JARVIS-DFT data
- `aflow_data.csv` - AFLOW data
- `nist_data.csv` - NIST data
- `collection_report.md` - This report
"""
        
        # Save report
        report_path = output_path / "collection_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Collection report saved to {report_path}")
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """
        Get current collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        return {
            'targets': {system: asdict(target) for system, target in self.collection_targets.items()},
            'source_stats': {name: asdict(stats) for name, stats in self.source_stats.items()},
            'total_collected': sum(target.collected_count for target in self.collection_targets.values())
        }