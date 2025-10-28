"""
Phase Stability Classification using DFT Hull Distance
CRITICAL: Distinguishes single-phase from multi-phase systems
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from mp_api.client import MPRester
from loguru import logger

class PhaseStabilityAnalyzer:
    """
    Analyze phase stability using DFT-calculated energy above convex hull.
    
    Classification Scheme:
    - ΔE_hull < 0.05 eV/atom → Stable single-phase
    - 0.05 ≤ ΔE_hull < 0.10 eV/atom → Metastable
    - ΔE_hull ≥ 0.10 eV/atom → Unstable (multi-phase expected)
    """
    
    def __init__(self, api_key: str = None, 
                 stable_threshold: float = 0.05,
                 metastable_threshold: float = 0.10):
        """
        Initialize phase stability analyzer
        
        Args:
            api_key: Materials Project API key
            stable_threshold: Threshold for stable classification (eV/atom)
            metastable_threshold: Threshold for metastable classification (eV/atom)
        """
        self.stable_threshold = stable_threshold
        self.metastable_threshold = metastable_threshold
        
        if api_key:
            self.mpr = MPRester(api_key)
            logger.info("✓ Materials Project API initialized for phase stability")
        else:
            self.mpr = None
            logger.warning("No API key provided - using existing hull distance values")
    
    def classify_stability(self, energy_above_hull: float) -> str:
        """
        Classify phase stability based on hull distance
        
        Args:
            energy_above_hull: Energy above hull (eV/atom)
        
        Returns:
            Classification: 'stable', 'metastable', or 'unstable'
        """
        if pd.isna(energy_above_hull):
            return 'unknown'
        
        if energy_above_hull < self.stable_threshold:
            return 'stable'
        elif energy_above_hull < self.metastable_threshold:
            return 'metastable'
        else:
            return 'unstable'    
 
    def get_hull_distance(self, material_id: str) -> float:
        """
        Query Materials Project for hull distance
        
        Args:
            material_id: Materials Project ID (e.g., 'mp-149')
        
        Returns:
            Energy above hull (eV/atom)
        """
        if not self.mpr:
            raise ValueError("Materials Project API not initialized")
        
        try:
            docs = self.mpr.materials.summary.search(
                material_ids=[material_id],
                fields=["energy_above_hull"]
            )
            if docs:
                return docs[0].energy_above_hull
            return np.nan
        except Exception as e:
            logger.error(f"Error fetching hull distance for {material_id}: {e}")
            return np.nan
    
    def analyze_dataframe(self, df: pd.DataFrame, 
                         material_id_col: str = 'material_id',
                         hull_distance_col: str = 'energy_above_hull') -> pd.DataFrame:
        """
        Analyze phase stability for entire dataset
        
        Args:
            df: DataFrame with materials data
            material_id_col: Column name for material IDs
            hull_distance_col: Column name for hull distance (if present)
        
        Returns:
            DataFrame with phase stability classifications
        """
        logger.info(f"Analyzing phase stability for {len(df)} materials...")
        
        df_stability = df.copy()
        
        # Get hull distances if not present
        if hull_distance_col not in df_stability.columns and self.mpr:
            logger.info("Fetching hull distances from Materials Project...")
            hull_distances = []
            for mat_id in df_stability[material_id_col]:
                hull_dist = self.get_hull_distance(mat_id)
                hull_distances.append(hull_dist)
            df_stability[hull_distance_col] = hull_distances
        
        # Classify stability
        df_stability['phase_stability'] = df_stability[hull_distance_col].apply(
            self.classify_stability
        )
        
        # Add binary flags for modeling
        df_stability['is_stable'] = (df_stability['phase_stability'] == 'stable').astype(int)
        df_stability['is_single_phase'] = (
            df_stability['phase_stability'].isin(['stable', 'metastable'])
        ).astype(int)
        
        # Statistics
        stability_counts = df_stability['phase_stability'].value_counts()
        logger.info("Phase stability distribution:")
        for status, count in stability_counts.items():
            pct = (count / len(df_stability)) * 100
            logger.info(f"  {status}: {count} ({pct:.1f}%)")
        
        return df_stability
    
    def get_single_phase_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract only stable/metastable (single-phase) materials
        
        Args:
            df: DataFrame with phase_stability column
        
        Returns:
            Filtered DataFrame with only single-phase materials
        """
        if 'phase_stability' not in df.columns:
            raise ValueError("DataFrame must have 'phase_stability' column")
        
        single_phase = df[df['phase_stability'].isin(['stable', 'metastable'])].copy()
        logger.info(f"Extracted {len(single_phase)} single-phase materials from {len(df)} total")
        
        return single_phase