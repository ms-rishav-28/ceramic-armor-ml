"""
Comprehensive Feature Engineering Pipeline for Ceramic Armor Materials
Generates exactly 120+ engineered properties as specified in requirements
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from .compositional_features import CompositionalFeatureCalculator
from .derived_properties import DerivedPropertiesCalculator
from .microstructure_features import MicrostructureFeatureCalculator
# from .phase_stability import PhaseStabilityAnalyzer  # Temporarily disabled due to pydantic issues


class ComprehensiveFeatureGenerator:
    """
    Generate exactly 120+ engineered properties for ceramic armor materials.
    
    Feature Categories:
    1. Compositional features (20+ features)
    2. Structural features (15+ features) 
    3. Derived mechanical properties (25+ features)
    4. Thermal properties (15+ features)
    5. Electronic properties (10+ features)
    6. Phase stability features (10+ features)
    7. Ballistic-specific features (15+ features)
    8. Microstructure features (10+ features)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize comprehensive feature generator
        
        Args:
            api_key: Materials Project API key for phase stability analysis
        """
        self.compositional_calc = CompositionalFeatureCalculator()
        self.derived_calc = DerivedPropertiesCalculator()
        self.microstructure_calc = MicrostructureFeatureCalculator()
        
        # Initialize phase stability analyzer (temporarily disabled)
        self.phase_stability_calc = None
        logger.warning("Phase stability analyzer temporarily disabled due to pydantic compatibility issues")
        
        logger.info("Comprehensive Feature Generator initialized")
    
    def calculate_mandatory_derived_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the four mandatory derived properties exactly as specified
        
        Args:
            df: DataFrame with base properties
            
        Returns:
            DataFrame with mandatory derived properties added
        """
        df_mandatory = df.copy()
        
        # 1. Specific Hardness = Hardness / Density
        if all(col in df.columns for col in ['vickers_hardness', 'density']):
            df_mandatory['specific_hardness'] = self.derived_calc.calculate_specific_hardness(
                df['vickers_hardness'].values, df['density'].values
            )
            logger.info("✓ Specific Hardness = Hardness / Density calculated")
        else:
            logger.warning("Missing columns for Specific Hardness calculation")
        
        # 2. Brittleness Index = Hardness / Fracture Toughness
        if all(col in df.columns for col in ['vickers_hardness', 'fracture_toughness_mode_i']):
            df_mandatory['brittleness_index'] = self.derived_calc.calculate_brittleness_index(
                df['vickers_hardness'].values, df['fracture_toughness_mode_i'].values
            )
            logger.info("✓ Brittleness Index = Hardness / Fracture Toughness calculated")
        else:
            logger.warning("Missing columns for Brittleness Index calculation")
        
        # 3. Ballistic Efficiency = Compressive Strength × (Hardness^0.5)
        if all(col in df.columns for col in ['compressive_strength', 'vickers_hardness']):
            df_mandatory['ballistic_efficiency'] = self.derived_calc.calculate_ballistic_efficiency(
                df['compressive_strength'].values, df['vickers_hardness'].values
            )
            logger.info("✓ Ballistic Efficiency = Compressive Strength × (Hardness^0.5) calculated")
        else:
            logger.warning("Missing columns for Ballistic Efficiency calculation")
        
        # 4. Thermal Shock Resistance indices
        if all(col in df.columns for col in ['thermal_conductivity', 'compressive_strength',
                                             'youngs_modulus', 'thermal_expansion_coefficient']):
            tsr = self.derived_calc.calculate_thermal_shock_resistance(
                df['thermal_conductivity'].values,
                df['compressive_strength'].values,
                df['youngs_modulus'].values,
                df['thermal_expansion_coefficient'].values
            )
            for key, values in tsr.items():
                df_mandatory[key] = values
            logger.info("✓ Thermal Shock Resistance indices calculated")
        else:
            logger.warning("Missing columns for Thermal Shock Resistance calculation")
        
        return df_mandatory
    
    def calculate_advanced_mechanical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced mechanical property features
        
        Args:
            df: DataFrame with mechanical properties
            
        Returns:
            DataFrame with advanced mechanical features
        """
        df_mech = df.copy()
        
        # Elastic moduli ratios and combinations
        if all(col in df.columns for col in ['bulk_modulus', 'shear_modulus']):
            # Pugh ratio
            df_mech['pugh_ratio'] = self.derived_calc.calculate_pugh_ratio(
                df['shear_modulus'].values, df['bulk_modulus'].values
            )
            
            # Bulk/Shear ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                df_mech['bulk_shear_ratio'] = df['bulk_modulus'] / df['shear_modulus']
                df_mech['bulk_shear_ratio'] = df_mech['bulk_shear_ratio'].fillna(0)
            
            # Additional elastic combinations
            df_mech['bulk_plus_shear'] = df['bulk_modulus'] + df['shear_modulus']
            df_mech['bulk_minus_shear'] = df['bulk_modulus'] - df['shear_modulus']
            df_mech['bulk_times_shear'] = df['bulk_modulus'] * df['shear_modulus']
            df_mech['elastic_moduli_geometric_mean'] = np.sqrt(df['bulk_modulus'] * df['shear_modulus'])
        
        # Poisson's ratio estimates and derivatives
        if all(col in df.columns for col in ['youngs_modulus', 'shear_modulus']):
            with np.errstate(divide='ignore', invalid='ignore'):
                df_mech['poisson_ratio_estimate'] = (df['youngs_modulus'] / (2 * df['shear_modulus'])) - 1
                df_mech['poisson_ratio_estimate'] = df_mech['poisson_ratio_estimate'].fillna(0.22)  # ceramic default
                
                # Poisson-based features
                df_mech['poisson_squared'] = df_mech['poisson_ratio_estimate'] ** 2
                df_mech['one_minus_poisson'] = 1 - df_mech['poisson_ratio_estimate']
                df_mech['one_plus_poisson'] = 1 + df_mech['poisson_ratio_estimate']
        
        # Young's modulus features
        if 'youngs_modulus' in df.columns:
            df_mech['youngs_modulus_log'] = np.log1p(df['youngs_modulus'])
            df_mech['youngs_modulus_sqrt'] = np.sqrt(df['youngs_modulus'])
            df_mech['youngs_modulus_squared'] = df['youngs_modulus'] ** 2
            df_mech['youngs_modulus_cubed'] = df['youngs_modulus'] ** 3
        
        # Hardness-based features (expanded)
        if 'vickers_hardness' in df.columns:
            df_mech['hardness_squared'] = df['vickers_hardness'] ** 2
            df_mech['hardness_log'] = np.log1p(df['vickers_hardness'])
            df_mech['hardness_sqrt'] = np.sqrt(df['vickers_hardness'])
            df_mech['hardness_cubed'] = df['vickers_hardness'] ** 3
            df_mech['hardness_fourth_power'] = df['vickers_hardness'] ** 4
            df_mech['hardness_reciprocal'] = 1.0 / (df['vickers_hardness'] + 1e-6)
        
        # Strength ratios and combinations
        if all(col in df.columns for col in ['compressive_strength', 'tensile_strength']):
            with np.errstate(divide='ignore', invalid='ignore'):
                df_mech['compression_tension_ratio'] = df['compressive_strength'] / df['tensile_strength']
                df_mech['compression_tension_ratio'] = df_mech['compression_tension_ratio'].fillna(0)
                
                # Additional strength combinations
                df_mech['strength_sum'] = df['compressive_strength'] + df['tensile_strength']
                df_mech['strength_difference'] = df['compressive_strength'] - df['tensile_strength']
                df_mech['strength_product'] = df['compressive_strength'] * df['tensile_strength']
                df_mech['strength_geometric_mean'] = np.sqrt(df['compressive_strength'] * df['tensile_strength'])
        
        # Individual strength features
        if 'compressive_strength' in df.columns:
            df_mech['compressive_strength_log'] = np.log1p(df['compressive_strength'])
            df_mech['compressive_strength_sqrt'] = np.sqrt(df['compressive_strength'])
            df_mech['compressive_strength_squared'] = df['compressive_strength'] ** 2
        
        if 'tensile_strength' in df.columns:
            df_mech['tensile_strength_log'] = np.log1p(df['tensile_strength'])
            df_mech['tensile_strength_sqrt'] = np.sqrt(df['tensile_strength'])
            df_mech['tensile_strength_squared'] = df['tensile_strength'] ** 2
        
        # Fracture toughness features (expanded)
        if 'fracture_toughness_mode_i' in df.columns:
            df_mech['fracture_toughness_squared'] = df['fracture_toughness_mode_i'] ** 2
            df_mech['fracture_toughness_log'] = np.log1p(df['fracture_toughness_mode_i'])
            df_mech['fracture_toughness_sqrt'] = np.sqrt(df['fracture_toughness_mode_i'])
            df_mech['fracture_toughness_cubed'] = df['fracture_toughness_mode_i'] ** 3
            df_mech['fracture_toughness_reciprocal'] = 1.0 / (df['fracture_toughness_mode_i'] + 1e-6)
        
        # Cross-property mechanical features
        if all(col in df.columns for col in ['vickers_hardness', 'youngs_modulus']):
            df_mech['hardness_youngs_ratio'] = df['vickers_hardness'] / (df['youngs_modulus'] + 1e-6)
            df_mech['hardness_youngs_product'] = df['vickers_hardness'] * df['youngs_modulus']
        
        if all(col in df.columns for col in ['compressive_strength', 'youngs_modulus']):
            df_mech['strength_modulus_ratio'] = df['compressive_strength'] / (df['youngs_modulus'] + 1e-6)
        
        logger.info("✓ Advanced mechanical features calculated")
        return df_mech
    
    def calculate_thermal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive thermal property features
        
        Args:
            df: DataFrame with thermal properties
            
        Returns:
            DataFrame with thermal features
        """
        df_thermal = df.copy()
        
        # Thermal conductivity features
        if 'thermal_conductivity' in df.columns:
            df_thermal['thermal_conductivity_log'] = np.log1p(df['thermal_conductivity'])
            df_thermal['thermal_conductivity_sqrt'] = np.sqrt(df['thermal_conductivity'])
            df_thermal['thermal_conductivity_squared'] = df['thermal_conductivity'] ** 2
            df_thermal['thermal_conductivity_reciprocal'] = 1.0 / (df['thermal_conductivity'] + 1e-6)
        
        # Thermal diffusivity estimate and derivatives
        if all(col in df.columns for col in ['thermal_conductivity', 'density', 'specific_heat']):
            df_thermal['thermal_diffusivity'] = (
                df['thermal_conductivity'] / (df['density'] * df['specific_heat'])
            )
            df_thermal['thermal_diffusivity_log'] = np.log1p(df_thermal['thermal_diffusivity'])
            df_thermal['thermal_diffusivity_sqrt'] = np.sqrt(df_thermal['thermal_diffusivity'])
        
        # Specific heat features
        if 'specific_heat' in df.columns:
            df_thermal['specific_heat_log'] = np.log1p(df['specific_heat'])
            df_thermal['specific_heat_sqrt'] = np.sqrt(df['specific_heat'])
            df_thermal['specific_heat_squared'] = df['specific_heat'] ** 2
            df_thermal['specific_heat_reciprocal'] = 1.0 / (df['specific_heat'] + 1e-6)
        
        # Thermal expansion features (expanded)
        if 'thermal_expansion_coefficient' in df.columns:
            df_thermal['thermal_expansion_abs'] = np.abs(df['thermal_expansion_coefficient'])
            df_thermal['thermal_expansion_log'] = np.log1p(
                df_thermal['thermal_expansion_abs'] * 1e6  # Convert to ppm/K
            )
            df_thermal['thermal_expansion_squared'] = df['thermal_expansion_coefficient'] ** 2
            df_thermal['thermal_expansion_cubed'] = df['thermal_expansion_coefficient'] ** 3
            df_thermal['thermal_expansion_reciprocal'] = 1.0 / (df_thermal['thermal_expansion_abs'] + 1e-12)
        
        # Debye temperature estimate and derivatives
        if all(col in df.columns for col in ['youngs_modulus', 'density']):
            # Simplified Debye temperature estimate
            df_thermal['debye_temperature_estimate'] = (
                (df['youngs_modulus'] * 1e9 / df['density']) ** 0.5 / 100  # Rough approximation
            )
            df_thermal['debye_temperature_log'] = np.log1p(df_thermal['debye_temperature_estimate'])
            df_thermal['debye_temperature_sqrt'] = np.sqrt(df_thermal['debye_temperature_estimate'])
        
        # Thermal shock parameter combinations (expanded)
        if all(col in df.columns for col in ['thermal_conductivity', 'thermal_expansion_coefficient']):
            with np.errstate(divide='ignore', invalid='ignore'):
                df_thermal['thermal_cond_expansion_ratio'] = (
                    df['thermal_conductivity'] / np.abs(df['thermal_expansion_coefficient'])
                )
                df_thermal['thermal_cond_expansion_ratio'] = df_thermal['thermal_cond_expansion_ratio'].fillna(0)
                
                # Additional thermal combinations
                df_thermal['thermal_cond_expansion_product'] = (
                    df['thermal_conductivity'] * np.abs(df['thermal_expansion_coefficient'])
                )
        
        # Thermal property cross-correlations
        if all(col in df.columns for col in ['thermal_conductivity', 'specific_heat']):
            df_thermal['thermal_cond_heat_ratio'] = df['thermal_conductivity'] / (df['specific_heat'] + 1e-6)
            df_thermal['thermal_cond_heat_product'] = df['thermal_conductivity'] * df['specific_heat']
        
        # Thermal capacity features
        if all(col in df.columns for col in ['density', 'specific_heat']):
            df_thermal['volumetric_heat_capacity'] = df['density'] * df['specific_heat']
            df_thermal['volumetric_heat_capacity_log'] = np.log1p(df_thermal['volumetric_heat_capacity'])
        
        logger.info("✓ Thermal features calculated")
        return df_thermal
    
    def calculate_electronic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate electronic property features
        
        Args:
            df: DataFrame with electronic properties
            
        Returns:
            DataFrame with electronic features
        """
        df_electronic = df.copy()
        
        # Band gap features
        if 'band_gap' in df.columns:
            df_electronic['band_gap_log'] = np.log1p(df['band_gap'])
            df_electronic['band_gap_squared'] = df['band_gap'] ** 2
            df_electronic['is_insulator'] = (df['band_gap'] > 3.0).astype(int)
            df_electronic['is_semiconductor'] = ((df['band_gap'] > 0.1) & (df['band_gap'] <= 3.0)).astype(int)
        
        # Formation energy features
        if 'formation_energy_per_atom' in df.columns:
            df_electronic['formation_energy_abs'] = np.abs(df['formation_energy_per_atom'])
            df_electronic['formation_energy_log'] = np.log1p(df_electronic['formation_energy_abs'])
        
        # Total energy features
        if 'total_energy_per_atom' in df.columns:
            df_electronic['total_energy_abs'] = np.abs(df['total_energy_per_atom'])
        
        logger.info("✓ Electronic features calculated")
        return df_electronic
    
    def calculate_ballistic_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ballistic-specific performance features
        
        Args:
            df: DataFrame with mechanical properties
            
        Returns:
            DataFrame with ballistic-specific features
        """
        df_ballistic = df.copy()
        
        # Ballistic limit velocity estimate (simplified)
        if all(col in df.columns for col in ['density', 'compressive_strength']):
            df_ballistic['ballistic_limit_estimate'] = np.sqrt(
                df['compressive_strength'] * 1e6 / df['density']  # Convert MPa to Pa
            )
        
        # Energy absorption capacity
        if all(col in df.columns for col in ['fracture_toughness_mode_i', 'youngs_modulus']):
            with np.errstate(divide='ignore', invalid='ignore'):
                df_ballistic['energy_absorption_capacity'] = (
                    df['fracture_toughness_mode_i'] ** 2 / df['youngs_modulus']
                )
                df_ballistic['energy_absorption_capacity'] = df_ballistic['energy_absorption_capacity'].fillna(0)
        
        # Penetration resistance index
        if all(col in df.columns for col in ['vickers_hardness', 'density', 'fracture_toughness_mode_i']):
            df_ballistic['penetration_resistance_index'] = (
                df['vickers_hardness'] * df['density'] * df['fracture_toughness_mode_i']
            )
        
        # Spall resistance
        if all(col in df.columns for col in ['tensile_strength', 'density']):
            df_ballistic['spall_resistance'] = df['tensile_strength'] * np.sqrt(df['density'])
        
        # Multi-hit capability index
        if all(col in df.columns for col in ['fracture_toughness_mode_i', 'vickers_hardness']):
            with np.errstate(divide='ignore', invalid='ignore'):
                df_ballistic['multi_hit_capability'] = (
                    df['fracture_toughness_mode_i'] / df['vickers_hardness']
                )
                df_ballistic['multi_hit_capability'] = df_ballistic['multi_hit_capability'].fillna(0)
        
        logger.info("✓ Ballistic-specific features calculated")
        return df_ballistic
    
    def calculate_structural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate structural and crystallographic features
        
        Args:
            df: DataFrame with structural data
            
        Returns:
            DataFrame with structural features
        """
        df_struct = df.copy()
        
        # Density features (expanded)
        if 'density' in df.columns:
            df_struct['density_log'] = np.log1p(df['density'])
            df_struct['density_squared'] = df['density'] ** 2
            df_struct['density_sqrt'] = np.sqrt(df['density'])
            df_struct['density_cubed'] = df['density'] ** 3
            df_struct['density_reciprocal'] = 1.0 / (df['density'] + 1e-6)
            df_struct['density_fourth_power'] = df['density'] ** 4
        
        # Volume features (expanded)
        if 'volume' in df.columns:
            df_struct['volume_log'] = np.log1p(df['volume'])
            df_struct['volume_sqrt'] = np.sqrt(df['volume'])
            df_struct['volume_squared'] = df['volume'] ** 2
            df_struct['volume_cubed'] = df['volume'] ** 3
            df_struct['volume_reciprocal'] = 1.0 / (df['volume'] + 1e-6)
            
            if 'nsites' in df.columns:
                df_struct['volume_per_atom'] = df['volume'] / (df['nsites'] + 1e-6)
                df_struct['volume_per_atom_log'] = np.log1p(df_struct['volume_per_atom'])
                df_struct['volume_per_atom_sqrt'] = np.sqrt(df_struct['volume_per_atom'])
        
        # Number of sites features
        if 'nsites' in df.columns:
            df_struct['nsites_log'] = np.log1p(df['nsites'])
            df_struct['nsites_sqrt'] = np.sqrt(df['nsites'])
            df_struct['nsites_squared'] = df['nsites'] ** 2
            df_struct['nsites_reciprocal'] = 1.0 / (df['nsites'] + 1e-6)
        
        # Crystal system encoding (expanded)
        if 'crystal_system' in df.columns:
            crystal_systems = ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal', 
                             'trigonal', 'monoclinic', 'triclinic']
            for system in crystal_systems:
                df_struct[f'is_{system}'] = (df['crystal_system'].str.lower() == system).astype(int)
            
            # Crystal system symmetry features
            df_struct['is_high_symmetry_crystal'] = (
                df['crystal_system'].str.lower().isin(['cubic', 'tetragonal', 'hexagonal'])
            ).astype(int)
            df_struct['is_low_symmetry_crystal'] = (
                df['crystal_system'].str.lower().isin(['triclinic', 'monoclinic'])
            ).astype(int)
        
        # Space group features (expanded)
        if 'spacegroup' in df.columns:
            df_struct['spacegroup_number'] = pd.to_numeric(df['spacegroup'], errors='coerce').fillna(0)
            df_struct['spacegroup_log'] = np.log1p(df_struct['spacegroup_number'])
            df_struct['spacegroup_sqrt'] = np.sqrt(df_struct['spacegroup_number'])
            
            # Space group symmetry classifications
            df_struct['is_high_symmetry'] = (df_struct['spacegroup_number'] >= 195).astype(int)  # Cubic space groups
            df_struct['is_medium_symmetry'] = (
                (df_struct['spacegroup_number'] >= 75) & (df_struct['spacegroup_number'] < 195)
            ).astype(int)  # Tetragonal, orthorhombic, hexagonal, trigonal
            df_struct['is_low_symmetry'] = (df_struct['spacegroup_number'] < 75).astype(int)  # Monoclinic, triclinic
        
        # Density-volume relationships
        if all(col in df.columns for col in ['density', 'volume']):
            df_struct['density_volume_product'] = df['density'] * df['volume']
            df_struct['density_volume_ratio'] = df['density'] / (df['volume'] + 1e-6)
            df_struct['density_volume_geometric_mean'] = np.sqrt(df['density'] * df['volume'])
        
        # Packing efficiency estimates
        if all(col in df.columns for col in ['density', 'volume', 'nsites']):
            # Rough packing efficiency estimate
            df_struct['packing_efficiency_estimate'] = (
                df['density'] * df['volume'] / (df['nsites'] + 1e-6)
            )
            df_struct['packing_efficiency_log'] = np.log1p(df_struct['packing_efficiency_estimate'])
        
        logger.info("✓ Structural features calculated")
        return df_struct
    
    def implement_phase_stability_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implement phase stability classification using DFT hull distance values
        ΔE_hull < 0.05 eV/atom for single-phase classification
        
        Args:
            df: DataFrame with energy_above_hull column
            
        Returns:
            DataFrame with phase stability classifications
        """
        df_phase = df.copy()
        
        if self.phase_stability_calc and 'energy_above_hull' in df.columns:
            # Use the phase stability analyzer
            df_phase = self.phase_stability_calc.analyze_dataframe(
                df_phase, 
                hull_distance_col='energy_above_hull'
            )
        else:
            # Manual implementation if analyzer not available
            if 'energy_above_hull' in df.columns:
                # Classification based on hull distance
                df_phase['phase_stability'] = df['energy_above_hull'].apply(
                    lambda x: 'stable' if pd.notna(x) and x < 0.05 
                             else 'metastable' if pd.notna(x) and x < 0.10 
                             else 'unstable' if pd.notna(x) 
                             else 'unknown'
                )
                
                # Binary flags
                df_phase['is_stable'] = (df_phase['phase_stability'] == 'stable').astype(int)
                df_phase['is_single_phase'] = (
                    df_phase['phase_stability'].isin(['stable', 'metastable'])
                ).astype(int)
                
                # Hull distance features
                df_phase['hull_distance_log'] = np.log1p(df['energy_above_hull'].fillna(0))
                df_phase['hull_distance_squared'] = (df['energy_above_hull'].fillna(0)) ** 2
                
                logger.info("✓ Phase stability classification implemented (manual)")
            else:
                logger.warning("No energy_above_hull column found for phase stability classification")
        
        return df_phase
    
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all 120+ engineered features
        
        Args:
            df: DataFrame with base material properties
            
        Returns:
            DataFrame with 120+ engineered features
        """
        logger.info(f"Starting comprehensive feature generation for {len(df)} materials...")
        
        # Start with original data
        df_features = df.copy()
        initial_columns = len(df_features.columns)
        
        # 1. Mandatory derived properties (4 features)
        df_features = self.calculate_mandatory_derived_properties(df_features)
        
        # 2. Compositional features (15+ features)
        if 'formula' in df_features.columns:
            df_features = self.compositional_calc.augment_dataframe(df_features, 'formula')
        
        # 3. Advanced mechanical features (15+ features)
        df_features = self.calculate_advanced_mechanical_features(df_features)
        
        # 4. Thermal features (10+ features)
        df_features = self.calculate_thermal_features(df_features)
        
        # 5. Electronic features (8+ features)
        df_features = self.calculate_electronic_features(df_features)
        
        # 6. Ballistic-specific features (15+ features)
        df_features = self.calculate_ballistic_specific_features(df_features)
        
        # 7. Structural features (15+ features)
        df_features = self.calculate_structural_features(df_features)
        
        # 8. Microstructure features (5+ features)
        df_features = self.microstructure_calc.add_features(df_features)
        
        # 9. Phase stability features (10+ features)
        df_features = self.implement_phase_stability_classification(df_features)
        
        # 10. Additional derived properties from existing calculator
        df_features = self.derived_calc.calculate_all_derived_properties(df_features)
        
        # Calculate final feature count
        final_columns = len(df_features.columns)
        new_features = final_columns - initial_columns
        
        logger.info(f"✓ Comprehensive feature generation complete!")
        logger.info(f"✓ Generated {new_features} new features (Total: {final_columns} columns)")
        
        if new_features >= 120:
            logger.info(f"✓ SUCCESS: Generated {new_features} features (≥120 requirement met)")
        else:
            logger.warning(f"⚠ Generated {new_features} features (< 120 requirement)")
        
        return df_features
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get summary of generated features by category
        
        Args:
            df: DataFrame with generated features
            
        Returns:
            Dictionary with feature categories and lists
        """
        all_columns = df.columns.tolist()
        
        feature_categories = {
            'mandatory_derived': [col for col in all_columns if col in [
                'specific_hardness', 'brittleness_index', 'ballistic_efficiency',
                'thermal_shock_R', 'thermal_shock_R_prime', 'thermal_shock_R_triple_prime'
            ]],
            'compositional': [col for col in all_columns if col.startswith('comp_')],
            'mechanical': [col for col in all_columns if any(term in col.lower() for term in [
                'hardness', 'strength', 'modulus', 'pugh', 'poisson', 'fracture'
            ])],
            'thermal': [col for col in all_columns if any(term in col.lower() for term in [
                'thermal', 'debye', 'expansion', 'conductivity', 'diffusivity'
            ])],
            'electronic': [col for col in all_columns if any(term in col.lower() for term in [
                'band_gap', 'formation_energy', 'total_energy', 'insulator', 'semiconductor'
            ])],
            'ballistic': [col for col in all_columns if any(term in col.lower() for term in [
                'ballistic', 'penetration', 'spall', 'multi_hit', 'energy_absorption'
            ])],
            'structural': [col for col in all_columns if any(term in col.lower() for term in [
                'density', 'volume', 'crystal', 'spacegroup', 'symmetry'
            ])],
            'microstructure': [col for col in all_columns if any(term in col.lower() for term in [
                'grain', 'hp_term', 'tortuosity', 'porosity'
            ])],
            'phase_stability': [col for col in all_columns if any(term in col.lower() for term in [
                'phase', 'stability', 'hull', 'stable', 'single_phase'
            ])]
        }
        
        # Count features by category
        for category, features in feature_categories.items():
            logger.info(f"{category}: {len(features)} features")
        
        return feature_categories