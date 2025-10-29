"""Tests for feature engineering modules."""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering.compositional_features import CompositionalFeatureCalculator
from src.feature_engineering.microstructure_features import MicrostructureFeatureCalculator
from src.feature_engineering.derived_properties import DerivedPropertiesCalculator
# from src.feature_engineering.phase_stability import PhaseStabilityAnalyzer  # Temporarily disabled
from src.feature_engineering.comprehensive_feature_generator import ComprehensiveFeatureGenerator


class TestCompositionalFeatureCalculator:
    """Test compositional feature calculation."""
    
    def test_init(self):
        calc = CompositionalFeatureCalculator()
        assert calc is not None
    
    def test_from_formula_sic(self):
        """Test compositional features for SiC."""
        calc = CompositionalFeatureCalculator()
        features = calc.from_formula("SiC")
        
        # Check expected features exist
        expected_features = [
            "comp_atomic_mass_mean",
            "comp_atomic_radius_mean", 
            "comp_en_mean",
            "comp_mixing_entropy"
        ]
        
        for feature in expected_features:
            assert feature in features
            assert not np.isnan(features[feature])
    
    def test_from_formula_al2o3(self):
        """Test compositional features for Al2O3."""
        calc = CompositionalFeatureCalculator()
        features = calc.from_formula("Al2O3")
        
        # Mixing entropy should be positive for multi-element compound
        assert features["comp_mixing_entropy"] > 0
        
        # Mean atomic mass should be reasonable
        assert 10 < features["comp_atomic_mass_mean"] < 30
    
    def test_augment_dataframe(self):
        """Test dataframe augmentation."""
        df = pd.DataFrame({
            "formula": ["SiC", "Al2O3", "B4C"],
            "density": [3.2, 3.9, 2.5]
        })
        
        calc = CompositionalFeatureCalculator()
        result = calc.augment_dataframe(df, formula_col="formula")
        
        # Should have more columns
        assert result.shape[1] > df.shape[1]
        assert result.shape[0] == df.shape[0]
        
        # Original columns should be preserved
        assert "formula" in result.columns
        assert "density" in result.columns


class TestMicrostructureFeatureCalculator:
    """Test microstructure feature calculation."""
    
    def test_init(self):
        calc = MicrostructureFeatureCalculator()
        assert calc is not None
    
    def test_safe_div(self):
        """Test safe division utility."""
        calc = MicrostructureFeatureCalculator()
        
        # Normal division
        result = calc._safe_div(10, 2)
        assert result == 5.0
        
        # Division by zero
        result = calc._safe_div(10, 0)
        assert result == 0.0
        
        # Division by very small number
        result = calc._safe_div(10, 1e-12)
        assert result == 0.0
    
    def test_add_features_with_grain_size(self):
        """Test feature addition with grain size data."""
        df = pd.DataFrame({
            "grain_size": [5.0, 10.0, 2.0],
            "density": [3.2, 3.9, 2.5],
            "compressive_strength": [2000, 1800, 2200],
            "porosity": [0.02, 0.05, 0.01]
        })
        
        calc = MicrostructureFeatureCalculator()
        result = calc.add_features(df)
        
        # Should have new features
        expected_features = ["hp_term", "tortuosity_proxy", "comp_strength_porosity_adj"]
        for feature in expected_features:
            assert feature in result.columns
        
        # Hall-Petch term should be inversely related to grain size
        assert result["hp_term"].iloc[2] > result["hp_term"].iloc[1]  # smaller grain size -> larger HP term
    
    def test_add_features_missing_data(self):
        """Test feature addition with missing microstructure data."""
        df = pd.DataFrame({
            "density": [3.2, 3.9, 2.5],
            "hardness": [25, 30, 20]
        })
        
        calc = MicrostructureFeatureCalculator()
        result = calc.add_features(df)
        
        # Should still add features (with default values)
        assert "hp_term" in result.columns
        assert "tortuosity_proxy" in result.columns
        
        # Default values should be 0
        assert all(result["hp_term"] == 0.0)
        assert all(result["tortuosity_proxy"] == 0.0)


class TestDerivedPropertiesCalculator:
    """Test derived properties calculation with mandatory specifications."""
    
    def test_init(self):
        calc = DerivedPropertiesCalculator()
        assert calc is not None
        assert 'k_B' in calc.constants
        assert 'N_A' in calc.constants
    
    def test_specific_hardness_calculation(self):
        """Test Specific Hardness = Hardness / Density calculation."""
        calc = DerivedPropertiesCalculator()
        
        hardness = np.array([25.0, 30.0, 20.0])  # GPa
        density = np.array([3.2, 4.0, 2.5])     # g/cm³
        
        result = calc.calculate_specific_hardness(hardness, density)
        expected = hardness / density
        
        np.testing.assert_array_almost_equal(result, expected)
        assert result[0] == 25.0 / 3.2  # ~7.81 GPa·cm³/g
    
    def test_brittleness_index_calculation(self):
        """Test Brittleness Index = Hardness / Fracture Toughness calculation."""
        calc = DerivedPropertiesCalculator()
        
        hardness = np.array([25.0, 30.0, 20.0])           # GPa
        fracture_toughness = np.array([4.0, 5.0, 3.5])   # MPa√m
        
        result = calc.calculate_brittleness_index(hardness, fracture_toughness)
        expected = hardness / fracture_toughness
        
        np.testing.assert_array_almost_equal(result, expected)
        assert result[0] == 25.0 / 4.0  # 6.25 μm^(-0.5)
    
    def test_ballistic_efficiency_calculation(self):
        """Test Ballistic Efficiency = Compressive Strength × (Hardness^0.5) calculation."""
        calc = DerivedPropertiesCalculator()
        
        compressive_strength = np.array([2000.0, 1800.0, 2200.0])  # MPa
        hardness = np.array([25.0, 30.0, 20.0])                    # GPa
        
        result = calc.calculate_ballistic_efficiency(compressive_strength, hardness)
        expected = compressive_strength * np.power(hardness, 0.5)
        
        np.testing.assert_array_almost_equal(result, expected)
        assert result[0] == 2000.0 * (25.0 ** 0.5)  # 2000 * 5 = 10000
    
    def test_thermal_shock_resistance_calculation(self):
        """Test Thermal Shock Resistance indices calculation."""
        calc = DerivedPropertiesCalculator()
        
        thermal_conductivity = np.array([120.0, 25.0, 80.0])      # W/m·K
        compressive_strength = np.array([2000.0, 1800.0, 2200.0]) # MPa
        youngs_modulus = np.array([400.0, 350.0, 450.0])          # GPa
        thermal_expansion = np.array([4.5e-6, 8.0e-6, 3.2e-6])   # 1/K
        
        result = calc.calculate_thermal_shock_resistance(
            thermal_conductivity, compressive_strength, youngs_modulus, thermal_expansion
        )
        
        assert isinstance(result, dict)
        assert 'thermal_shock_R' in result
        assert 'thermal_shock_R_prime' in result
        assert 'thermal_shock_R_triple_prime' in result
        
        # Check that all arrays have correct length
        for key, values in result.items():
            assert len(values) == 3


# Temporarily disabled due to pydantic compatibility issues
# class TestPhaseStabilityAnalyzer:
#     """Test phase stability classification."""
#     pass


class TestComprehensiveFeatureGenerator:
    """Test comprehensive feature generation pipeline."""
    
    def test_init(self):
        generator = ComprehensiveFeatureGenerator()
        assert generator is not None
        assert generator.compositional_calc is not None
        assert generator.derived_calc is not None
        assert generator.microstructure_calc is not None
    
    def test_mandatory_derived_properties(self):
        """Test calculation of the four mandatory derived properties."""
        df = pd.DataFrame({
            'vickers_hardness': [25.0, 30.0, 20.0],
            'density': [3.2, 4.0, 2.5],
            'fracture_toughness_mode_i': [4.0, 5.0, 3.5],
            'compressive_strength': [2000.0, 1800.0, 2200.0],
            'thermal_conductivity': [120.0, 25.0, 80.0],
            'youngs_modulus': [400.0, 350.0, 450.0],
            'thermal_expansion_coefficient': [4.5e-6, 8.0e-6, 3.2e-6]
        })
        
        generator = ComprehensiveFeatureGenerator()
        result = generator.calculate_mandatory_derived_properties(df)
        
        # Check that all mandatory properties are calculated
        mandatory_features = [
            'specific_hardness',
            'brittleness_index', 
            'ballistic_efficiency',
            'thermal_shock_R',
            'thermal_shock_R_prime',
            'thermal_shock_R_triple_prime'
        ]
        
        for feature in mandatory_features:
            assert feature in result.columns
            assert not result[feature].isna().all()
        
        # Verify specific calculations
        expected_specific_hardness = df['vickers_hardness'] / df['density']
        np.testing.assert_array_almost_equal(
            result['specific_hardness'].values, 
            expected_specific_hardness.values
        )
        
        expected_brittleness = df['vickers_hardness'] / df['fracture_toughness_mode_i']
        np.testing.assert_array_almost_equal(
            result['brittleness_index'].values,
            expected_brittleness.values
        )
        
        expected_ballistic = df['compressive_strength'] * np.power(df['vickers_hardness'], 0.5)
        np.testing.assert_array_almost_equal(
            result['ballistic_efficiency'].values,
            expected_ballistic.values
        )
    
    def test_phase_stability_classification(self):
        """Test phase stability classification implementation."""
        df = pd.DataFrame({
            'energy_above_hull': [0.02, 0.08, 0.15, np.nan],
            'formula': ['SiC', 'Al2O3', 'B4C', 'WC']
        })
        
        generator = ComprehensiveFeatureGenerator()
        result = generator.implement_phase_stability_classification(df)
        
        # Check classification columns
        assert 'phase_stability' in result.columns
        assert 'is_stable' in result.columns
        assert 'is_single_phase' in result.columns
        
        # Verify classifications based on ΔE_hull < 0.05 eV/atom for single-phase
        assert result['phase_stability'].iloc[0] == 'stable'      # 0.02 < 0.05
        assert result['phase_stability'].iloc[1] == 'metastable'  # 0.08 < 0.10
        assert result['phase_stability'].iloc[2] == 'unstable'    # 0.15 >= 0.10
        assert result['phase_stability'].iloc[3] == 'unknown'     # NaN
        
        # Verify single-phase classification
        assert result['is_single_phase'].iloc[0] == 1  # stable
        assert result['is_single_phase'].iloc[1] == 1  # metastable
        assert result['is_single_phase'].iloc[2] == 0  # unstable
    
    def test_comprehensive_feature_generation(self):
        """Test generation of 120+ features."""
        # Create comprehensive test dataset
        df = pd.DataFrame({
            'formula': ['SiC', 'Al2O3', 'B4C'],
            'vickers_hardness': [25.0, 30.0, 20.0],
            'density': [3.2, 4.0, 2.5],
            'fracture_toughness_mode_i': [4.0, 5.0, 3.5],
            'compressive_strength': [2000.0, 1800.0, 2200.0],
            'tensile_strength': [300.0, 250.0, 350.0],
            'youngs_modulus': [400.0, 350.0, 450.0],
            'bulk_modulus': [220.0, 200.0, 250.0],
            'shear_modulus': [180.0, 160.0, 200.0],
            'thermal_conductivity': [120.0, 25.0, 80.0],
            'thermal_expansion_coefficient': [4.5e-6, 8.0e-6, 3.2e-6],
            'specific_heat': [750.0, 880.0, 950.0],
            'band_gap': [2.3, 8.8, 1.5],
            'formation_energy_per_atom': [-1.2, -2.1, -0.8],
            'total_energy_per_atom': [-8.5, -12.3, -7.2],
            'energy_above_hull': [0.02, 0.08, 0.15],
            'volume': [20.1, 25.4, 18.7],
            'nsites': [2, 5, 5],
            'crystal_system': ['cubic', 'trigonal', 'trigonal'],
            'spacegroup': [216, 167, 166]
        })
        
        generator = ComprehensiveFeatureGenerator()
        result = generator.generate_all_features(df)
        
        # Check that we have significantly more features
        initial_features = len(df.columns)
        final_features = len(result.columns)
        new_features = final_features - initial_features
        
        assert new_features >= 120, f"Generated only {new_features} features, need ≥120"
        
        # Check that mandatory features are present
        mandatory_features = [
            'specific_hardness',
            'brittleness_index',
            'ballistic_efficiency'
        ]
        
        for feature in mandatory_features:
            assert feature in result.columns
        
        # Check that phase stability features are present
        phase_features = ['phase_stability', 'is_stable', 'is_single_phase']
        for feature in phase_features:
            assert feature in result.columns
    
    def test_feature_summary(self):
        """Test feature categorization summary."""
        df = pd.DataFrame({
            'formula': ['SiC'],
            'vickers_hardness': [25.0],
            'density': [3.2],
            'energy_above_hull': [0.02]
        })
        
        generator = ComprehensiveFeatureGenerator()
        result = generator.generate_all_features(df)
        summary = generator.get_feature_summary(result)
        
        # Check that summary contains expected categories
        expected_categories = [
            'mandatory_derived', 'compositional', 'mechanical', 
            'thermal', 'electronic', 'ballistic', 'structural',
            'microstructure', 'phase_stability'
        ]
        
        for category in expected_categories:
            assert category in summary
            assert isinstance(summary[category], list)


if __name__ == "__main__":
    pytest.main([__file__])