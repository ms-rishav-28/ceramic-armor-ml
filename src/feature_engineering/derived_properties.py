"""
Derived Property Calculations for Ceramic Armor Materials
Implements all critical feature engineering transformations
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from loguru import logger

class DerivedPropertiesCalculator:
    """
    Calculate derived properties from base measurements.
    All formulas are based on established materials science relationships.
    """
    
    def __init__(self):
        """Initialize calculator with physical constants"""
        self.constants = {
            'k_B': 1.380649e-23,  # Boltzmann constant (J/K)
            'N_A': 6.02214076e23   # Avogadro's number
        }
        logger.info("Derived Properties Calculator initialized")
    
    def calculate_specific_hardness(self, hardness: np.ndarray, density: np.ndarray) -> np.ndarray:
        """
        Specific Hardness = Hardness / Density
        Critical metric for armor applications (maximize hardness per unit mass)
        
        Args:
            hardness: Vickers hardness (GPa)
            density: Material density (g/cm³)
        
        Returns:
            Specific hardness (GPa·cm³/g)
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            specific_h = hardness / density
            specific_h = np.nan_to_num(specific_h, nan=0.0, posinf=0.0, neginf=0.0)
        return specific_h
    
    def calculate_brittleness_index(self, hardness: np.ndarray, 
                                   fracture_toughness: np.ndarray) -> np.ndarray:
        """
        Brittleness Index (BI) = Hardness / Fracture Toughness
        
        Higher BI → More brittle (ceramic behavior)
        Lower BI → More ductile (metallic behavior)
        
        Typical ranges:
        - Ceramics: BI > 4.0 μm^(-0.5)
        - Metals: BI < 1.0 μm^(-0.5)
        
        Args:
            hardness: Vickers hardness (GPa)
            fracture_toughness: Mode I fracture toughness (MPa√m)
        
        Returns:
            Brittleness index (μm^(-0.5))
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            bi = hardness / fracture_toughness
            bi = np.nan_to_num(bi, nan=0.0, posinf=0.0, neginf=0.0)
        return bi
    
    def calculate_ballistic_efficiency(self, compressive_strength: np.ndarray,
                                      hardness: np.ndarray) -> np.ndarray:
        """
        Ballistic Efficiency = Compressive Strength × (Hardness^0.5)
        
        EXACT SPECIFICATION: Compressive Strength × (Hardness^0.5)
        Critical metric for ceramic armor ballistic performance
        
        Args:
            compressive_strength: Compressive strength (MPa)
            hardness: Vickers hardness (GPa)
        
        Returns:
            Ballistic efficiency (MPa·GPa^0.5)
        """
        ballistic_eff = compressive_strength * np.power(hardness, 0.5)
        return ballistic_eff
    
    def calculate_elastic_anisotropy(self, bulk_modulus: np.ndarray,
                                    shear_modulus: np.ndarray) -> np.ndarray:
        """
        Zener Anisotropy Index
        
        For cubic crystals: A = 2C44 / (C11 - C12)
        Approximation from Voigt-Reuss-Hill averages:
        A ≈ (2 × G) / (3 × B - 2 × G)
        
        A = 1 → Isotropic
        A ≠ 1 → Anisotropic
        
        Args:
            bulk_modulus: Bulk modulus (GPa)
            shear_modulus: Shear modulus (GPa)
        
        Returns:
            Anisotropy index (dimensionless)
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            denominator = 3 * bulk_modulus - 2 * shear_modulus
            anisotropy = (2 * shear_modulus) / denominator
            anisotropy = np.nan_to_num(anisotropy, nan=1.0, posinf=1.0, neginf=1.0)
        return anisotropy
    
    def calculate_thermal_shock_resistance(self, 
                                          thermal_conductivity: np.ndarray,
                                          compressive_strength: np.ndarray,
                                          youngs_modulus: np.ndarray,
                                          thermal_expansion: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Thermal Shock Resistance Parameters
        
        R = (σ × k) / (E × α)      - First thermal shock parameter
        R' = R × (1 - ν)           - Considering Poisson effect
        R''' = R × E               - Energy-based parameter
        
        Critical for ballistic impact (generates >1000°C in microseconds)
        
        Args:
            thermal_conductivity: k (W/m·K)
            compressive_strength: σ (MPa)
            youngs_modulus: E (GPa)
            thermal_expansion: α (1/K)
        
        Returns:
            Dictionary with R, R_prime, R_triple_prime
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            # R parameter
            numerator = compressive_strength * thermal_conductivity
            denominator = youngs_modulus * 1000 * thermal_expansion  # Convert GPa to MPa
            R = numerator / denominator
            R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
            
            # R' parameter (assuming Poisson ratio ≈ 0.22 for ceramics)
            R_prime = R * (1 - 0.22)
            
            # R''' parameter
            R_triple_prime = R * youngs_modulus * 1000  # Convert back to MPa
        
        return {
            'thermal_shock_R': R,
            'thermal_shock_R_prime': R_prime,
            'thermal_shock_R_triple_prime': R_triple_prime
        }
    
    def calculate_pugh_ratio(self, shear_modulus: np.ndarray,
                            bulk_modulus: np.ndarray) -> np.ndarray:
        """
        Pugh's Modulus Ratio = G / B
        
        Empirical ductility/brittleness predictor:
        - G/B < 0.57 → Ductile behavior
        - G/B > 0.57 → Brittle behavior
        
        Args:
            shear_modulus: G (GPa)
            bulk_modulus: B (GPa)
        
        Returns:
            Pugh ratio (dimensionless)
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            pugh = shear_modulus / bulk_modulus
            pugh = np.nan_to_num(pugh, nan=0.0, posinf=0.0, neginf=0.0)
        return pugh
    
    def calculate_cauchy_pressure(self, C12: np.ndarray, C44: np.ndarray) -> np.ndarray:
        """
        Cauchy Pressure = C12 - C44
        
        Indicator of bonding character:
        - Positive → Metallic bonding
        - Negative → Covalent/ionic bonding
        
        Most ceramics have negative Cauchy pressure.
        
        Args:
            C12: Elastic constant C12 (GPa)
            C44: Elastic constant C44 (GPa)
        
        Returns:
            Cauchy pressure (GPa)
        """
        return C12 - C44
    
    def calculate_melting_temperature_estimate(self, cohesive_energy: np.ndarray) -> np.ndarray:
        """
        Lindemann Melting Temperature Estimate
        
        T_m ≈ (cohesive_energy / k_B) × constant
        
        Args:
            cohesive_energy: Cohesive energy (eV/atom)
        
        Returns:
            Estimated melting temperature (K)
        """
        # Convert eV to Joules
        cohesive_J = cohesive_energy * 1.60218e-19
        T_m = cohesive_J / (self.constants['k_B'] * 30)  # Empirical factor
        return T_m
    
    def calculate_all_derived_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all derived properties for entire dataset
        
        Args:
            df: DataFrame with base properties
        
        Returns:
            DataFrame with added derived properties
        """
        logger.info("Calculating derived properties...")
        df_derived = df.copy()
        
        # Specific Hardness
        if all(col in df.columns for col in ['vickers_hardness', 'density']):
            df_derived['specific_hardness'] = self.calculate_specific_hardness(
                df['vickers_hardness'].values, df['density'].values
            )
            logger.info("✓ Specific hardness calculated")
        
        # Brittleness Index
        if all(col in df.columns for col in ['vickers_hardness', 'fracture_toughness_mode_i']):
            df_derived['brittleness_index'] = self.calculate_brittleness_index(
                df['vickers_hardness'].values, df['fracture_toughness_mode_i'].values
            )
            logger.info("✓ Brittleness index calculated")
        
        # Ballistic Efficiency
        if all(col in df.columns for col in ['compressive_strength', 'vickers_hardness']):
            df_derived['ballistic_efficiency'] = self.calculate_ballistic_efficiency(
                df['compressive_strength'].values, df['vickers_hardness'].values
            )
            logger.info("✓ Ballistic efficiency calculated")
        
        # Elastic Anisotropy
        if all(col in df.columns for col in ['bulk_modulus', 'shear_modulus']):
            df_derived['elastic_anisotropy'] = self.calculate_elastic_anisotropy(
                df['bulk_modulus'].values, df['shear_modulus'].values
            )
            logger.info("✓ Elastic anisotropy calculated")
        
        # Thermal Shock Resistance
        if all(col in df.columns for col in ['thermal_conductivity', 'compressive_strength',
                                             'youngs_modulus', 'thermal_expansion_coefficient']):
            tsr = self.calculate_thermal_shock_resistance(
                df['thermal_conductivity'].values,
                df['compressive_strength'].values,
                df['youngs_modulus'].values,
                df['thermal_expansion_coefficient'].values
            )
            for key, values in tsr.items():
                df_derived[key] = values
            logger.info("✓ Thermal shock resistance parameters calculated")
        
        # Pugh Ratio
        if all(col in df.columns for col in ['shear_modulus', 'bulk_modulus']):
            df_derived['pugh_ratio'] = self.calculate_pugh_ratio(
                df['shear_modulus'].values, df['bulk_modulus'].values
            )
            logger.info("✓ Pugh ratio calculated")
        
        n_new_features = len(df_derived.columns) - len(df.columns)
        logger.info(f"✓ Total derived properties calculated: {n_new_features}")
        
        return df_derived