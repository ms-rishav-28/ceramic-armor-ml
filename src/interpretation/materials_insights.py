# src/interpretation/materials_insights.py
"""
Materials Science Insights for Ceramic Armor Interpretability Analysis
Provides mechanistic interpretation of feature importance with materials science rationale
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any


def interpret_feature_ranking(importance_df: pd.DataFrame, top_k: int = 10) -> str:
    """
    Convert feature importance into mechanistic interpretation text.
    Enhanced version with more comprehensive materials science insights.
    
    Args:
        importance_df: DataFrame with feature importance rankings
        top_k: Number of top features to analyze
    
    Returns:
        String with mechanistic interpretation
    """
    top = importance_df.head(top_k)["feature"].astype(str).tolist()
    statements = []
    
    # Hardness-related insights
    if any("vickers_hardness" in f or "specific_hardness" in f or "hardness" in f for f in top):
        statements.append("- **Hardness Dominance**: High hardness and specific hardness control penetration resistance through projectile blunting and dwell mechanisms, with specific hardness (hardness/density) providing optimal ballistic efficiency.")
    
    # Toughness-related insights
    if any("fracture_toughness" in f or "toughness" in f for f in top):
        statements.append("- **Fracture Toughness Control**: Fracture toughness prevents catastrophic crack propagation, enabling multi-hit survivability and controlled fragmentation under repeated impacts.")
    
    # Density and specific properties
    if any("density" in f or "specific_" in f for f in top):
        statements.append("- **Density Effects**: Density influences ballistic momentum transfer and wave impedance matching; normalized metrics (specific properties) favor high performance at lower weight.")
    
    # Thermal properties
    if any("thermal_conductivity" in f or "thermal_shock" in f or "thermal" in f for f in top):
        statements.append("- **Thermal Response**: Thermal transport and shock resistance are critical under adiabatic heating during high-velocity impact (>1000°C in microseconds), affecting local material behavior and failure modes.")
    
    # Elastic properties
    if any("pugh_ratio" in f or "elastic_anisotropy" in f or "elastic" in f or "modulus" in f for f in top):
        statements.append("- **Elastic Behavior**: Elastic moduli ratios (G/B, anisotropy) correlate with crack deflection mechanisms, spall behavior, and stress wave propagation under dynamic loading.")
    
    # Ballistic-specific properties
    if any("ballistic" in f or "v50" in f or "penetration" in f for f in top):
        statements.append("- **Direct Ballistic Metrics**: Ballistic efficiency and penetration resistance represent integrated performance measures combining hardness, toughness, and density effects.")
    
    # Compositional effects
    if any("atomic" in f or "composition" in f or "stoichiometry" in f for f in top):
        statements.append("- **Compositional Control**: Atomic composition and stoichiometry directly influence crystal structure, bonding character, and resulting mechanical properties.")
    
    # Structural properties
    if any("crystal" in f or "lattice" in f or "structure" in f for f in top):
        statements.append("- **Structural Influence**: Crystal structure and lattice parameters control fundamental bonding characteristics and mechanical response mechanisms.")
    
    # Default comprehensive statement
    if not statements:
        statements.append("- **Multi-Factor Control**: Feature importance indicates complex multi-factor control where hardness-toughness-thermal property synergy governs ceramic armor performance.")
    
    # Add overall synthesis
    statements.append("- **Synergistic Performance**: Optimal ceramic armor performance emerges from synergistic interactions between hardness (projectile defeat), toughness (damage tolerance), and thermal properties (adiabatic response), rather than individual property optimization.")
    
    return "\n".join(statements)


def generate_comprehensive_materials_insights(feature_ranking_df: pd.DataFrame, 
                                            ceramic_system: str = None,
                                            target_property: str = None) -> Dict[str, Any]:
    """
    Generate comprehensive materials science insights from feature ranking
    
    Args:
        feature_ranking_df: DataFrame with feature importance and categories
        ceramic_system: Ceramic system name (SiC, Al2O3, etc.)
        target_property: Target property being predicted
    
    Returns:
        Dictionary with comprehensive materials insights
    """
    
    insights = {
        'system_specific_insights': {},
        'property_specific_insights': {},
        'mechanistic_interpretation': {},
        'performance_controlling_factors': {},
        'materials_science_rationale': {}
    }
    
    # System-specific insights
    if ceramic_system:
        insights['system_specific_insights'] = get_system_specific_insights(ceramic_system)
    
    # Property-specific insights
    if target_property:
        insights['property_specific_insights'] = get_property_specific_insights(target_property)
    
    # Analyze feature categories
    if 'category' in feature_ranking_df.columns:
        category_analysis = analyze_feature_categories(feature_ranking_df)
        insights['mechanistic_interpretation'] = category_analysis
    
    # Identify performance controlling factors
    top_features = feature_ranking_df.head(15)
    insights['performance_controlling_factors'] = identify_controlling_factors(top_features)
    
    # Generate materials science rationale
    insights['materials_science_rationale'] = generate_materials_rationale(
        feature_ranking_df, ceramic_system, target_property
    )
    
    return insights


def get_system_specific_insights(ceramic_system: str) -> Dict[str, str]:
    """Get system-specific materials science insights"""
    
    system_insights = {
        'SiC': {
            'crystal_structure': 'Covalent bonding in SiC provides exceptional hardness (25-35 GPa) and thermal stability up to 2000°C',
            'bonding_character': 'Strong covalent Si-C bonds result in high elastic modulus (400-500 GPa) and thermal conductivity',
            'performance_drivers': 'Ultra-high hardness drives superior projectile blunting, while high thermal conductivity manages adiabatic heating',
            'limitations': 'Extreme brittleness (KIC ~3-5 MPa·m^0.5) limits multi-hit capability and damage tolerance',
            'ballistic_behavior': 'Excellent single-hit performance through dwell and erosion, but catastrophic failure under repeated impacts'
        },
        'Al2O3': {
            'crystal_structure': 'Ionic-covalent bonding in corundum structure provides balanced hardness (15-20 GPa) and toughness (3-5 MPa·m^0.5)',
            'bonding_character': 'Mixed ionic-covalent bonding enables moderate hardness with better damage tolerance than SiC',
            'performance_drivers': 'Balanced hardness-toughness combination provides good multi-hit survivability',
            'limitations': 'Lower hardness than SiC and B4C reduces single-hit ballistic performance',
            'ballistic_behavior': 'Good multi-hit performance with moderate single-hit capability, controlled fragmentation'
        },
        'B4C': {
            'crystal_structure': 'Complex icosahedral structure with covalent bonding provides ultra-high hardness (30-40 GPa)',
            'bonding_character': 'Strong covalent B-C and B-B bonds result in exceptional hardness but extreme brittleness',
            'performance_drivers': 'Highest hardness among structural ceramics drives superior ballistic performance',
            'limitations': 'Pressure-induced amorphization and extreme brittleness under dynamic loading',
            'ballistic_behavior': 'Outstanding single-hit performance but susceptible to pressure-induced phase transformation'
        },
        'WC': {
            'crystal_structure': 'Hexagonal structure with metallic W-C bonding provides high hardness (15-25 GPa) and toughness',
            'bonding_character': 'Metallic component provides better damage tolerance than purely covalent ceramics',
            'performance_drivers': 'Combination of high hardness and metallic toughening mechanisms',
            'limitations': 'Higher density reduces specific performance metrics',
            'ballistic_behavior': 'Good balance of hardness and toughness with weight penalty'
        },
        'TiC': {
            'crystal_structure': 'Face-centered cubic structure with metallic bonding character',
            'bonding_character': 'Metallic Ti-C bonding provides moderate hardness with good toughness',
            'performance_drivers': 'Balanced mechanical properties with good thermal stability',
            'limitations': 'Lower hardness compared to other carbides',
            'ballistic_behavior': 'Moderate ballistic performance with good damage tolerance'
        }
    }
    
    return system_insights.get(ceramic_system, {
        'crystal_structure': 'System-specific crystal structure influences mechanical properties',
        'bonding_character': 'Bonding character determines hardness-toughness balance',
        'performance_drivers': 'Key material factors controlling ballistic performance',
        'limitations': 'Fundamental limitations affecting armor applications',
        'ballistic_behavior': 'Expected ballistic response characteristics'
    })


def get_property_specific_insights(target_property: str) -> Dict[str, str]:
    """Get property-specific materials science insights"""
    
    property_insights = {
        'youngs_modulus': {
            'physical_meaning': 'Elastic stiffness controlling stress-strain response under loading',
            'measurement_method': 'Tensile or flexural testing with strain measurement',
            'ballistic_relevance': 'Controls stress wave propagation velocity and impedance matching',
            'controlling_factors': 'Bonding strength, crystal structure, and atomic packing density',
            'typical_values': 'SiC: 400-500 GPa, Al2O3: 350-400 GPa, B4C: 450-500 GPa'
        },
        'vickers_hardness': {
            'physical_meaning': 'Resistance to plastic deformation under indentation loading',
            'measurement_method': 'Vickers indentation with diamond pyramid indenter',
            'ballistic_relevance': 'Primary factor controlling projectile blunting and dwell time',
            'controlling_factors': 'Bonding strength, crystal structure, and defect density',
            'typical_values': 'SiC: 25-35 GPa, Al2O3: 15-20 GPa, B4C: 30-40 GPa'
        },
        'fracture_toughness_mode_i': {
            'physical_meaning': 'Critical stress intensity for crack propagation under tensile loading',
            'measurement_method': 'Single-edge notched beam or chevron notch testing',
            'ballistic_relevance': 'Controls crack propagation and multi-hit survivability',
            'controlling_factors': 'Microstructure, grain size, and toughening mechanisms',
            'typical_values': 'SiC: 3-5 MPa·m^0.5, Al2O3: 3-5 MPa·m^0.5, B4C: 2-4 MPa·m^0.5'
        },
        'v50': {
            'physical_meaning': 'Velocity at which 50% of projectiles penetrate the armor',
            'measurement_method': 'Ballistic testing with statistical analysis of penetration probability',
            'ballistic_relevance': 'Direct measure of ballistic performance and protection capability',
            'controlling_factors': 'Hardness, toughness, thickness, and projectile characteristics',
            'typical_values': 'Depends on thickness, projectile type, and ceramic system'
        },
        'ballistic_efficiency': {
            'physical_meaning': 'Composite metric combining hardness and strength for ballistic performance',
            'measurement_method': 'Calculated from mechanical properties: σc × H^0.5',
            'ballistic_relevance': 'Integrated measure of projectile defeat capability',
            'controlling_factors': 'Compressive strength and hardness combination',
            'typical_values': 'Higher values indicate better ballistic performance'
        }
    }
    
    return property_insights.get(target_property, {
        'physical_meaning': 'Material property with specific physical significance',
        'measurement_method': 'Standard testing method for property determination',
        'ballistic_relevance': 'Relevance to ceramic armor ballistic performance',
        'controlling_factors': 'Key factors controlling property magnitude',
        'typical_values': 'Representative values for ceramic materials'
    })


def analyze_feature_categories(feature_ranking_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze feature importance by materials science category"""
    
    if 'category' not in feature_ranking_df.columns:
        return {}
    
    category_analysis = feature_ranking_df.groupby('category').agg({
        'importance': ['count', 'mean', 'sum', 'std'],
        'significant': 'sum' if 'significant' in feature_ranking_df.columns else 'count'
    }).round(4)
    
    category_analysis.columns = ['feature_count', 'mean_importance', 'total_importance', 
                               'std_importance', 'significant_count']
    category_analysis = category_analysis.sort_values('total_importance', ascending=False)
    
    # Generate category-specific interpretations
    interpretations = {}
    
    for category in category_analysis.index:
        if 'Hardness' in category:
            interpretations[category] = {
                'mechanism': 'Controls projectile blunting and dwell through resistance to plastic deformation',
                'ballistic_relevance': 'Primary factor in projectile defeat and penetration resistance',
                'physical_basis': 'Bonding strength and crystal structure determine hardness magnitude'
            }
        elif 'Toughness' in category:
            interpretations[category] = {
                'mechanism': 'Prevents catastrophic crack propagation and enables damage tolerance',
                'ballistic_relevance': 'Critical for multi-hit survivability and controlled fragmentation',
                'physical_basis': 'Microstructure and grain boundary characteristics control toughness'
            }
        elif 'Thermal' in category:
            interpretations[category] = {
                'mechanism': 'Controls adiabatic heating response and thermal shock resistance',
                'ballistic_relevance': 'Manages temperature rise during high-velocity impact',
                'physical_basis': 'Phonon transport and thermal expansion characteristics'
            }
        elif 'Elastic' in category:
            interpretations[category] = {
                'mechanism': 'Controls stress wave propagation and crack deflection behavior',
                'ballistic_relevance': 'Affects spall formation and stress distribution',
                'physical_basis': 'Atomic bonding and crystal structure determine elastic constants'
            }
        elif 'Density' in category:
            interpretations[category] = {
                'mechanism': 'Influences momentum transfer and wave impedance matching',
                'ballistic_relevance': 'Weight considerations and specific performance metrics',
                'physical_basis': 'Atomic mass and crystal packing density'
            }
    
    return {
        'category_statistics': category_analysis.to_dict('index'),
        'category_interpretations': interpretations,
        'dominant_category': category_analysis.index[0] if len(category_analysis) > 0 else None
    }


def identify_controlling_factors(top_features_df: pd.DataFrame) -> Dict[str, Any]:
    """Identify primary factors controlling ballistic performance"""
    
    controlling_factors = {
        'primary_factors': [],
        'secondary_factors': [],
        'synergistic_effects': [],
        'threshold_behaviors': []
    }
    
    # Primary factors (top 5)
    for i, (_, feature) in enumerate(top_features_df.head(5).iterrows()):
        factor_info = {
            'rank': i + 1,
            'feature': feature['feature'],
            'importance': float(feature['importance']),
            'category': feature.get('category', 'Unknown'),
            'mechanism': get_feature_mechanism(feature['feature']),
            'ballistic_role': get_ballistic_role(feature['feature'])
        }
        controlling_factors['primary_factors'].append(factor_info)
    
    # Secondary factors (next 10)
    for i, (_, feature) in enumerate(top_features_df.iloc[5:15].iterrows()):
        factor_info = {
            'rank': i + 6,
            'feature': feature['feature'],
            'importance': float(feature['importance']),
            'category': feature.get('category', 'Unknown')
        }
        controlling_factors['secondary_factors'].append(factor_info)
    
    # Identify synergistic effects
    categories = top_features_df['category'].value_counts() if 'category' in top_features_df.columns else {}
    
    if 'Hardness Related' in categories and 'Toughness Related' in categories:
        controlling_factors['synergistic_effects'].append({
            'type': 'Hardness-Toughness Synergy',
            'description': 'Combined hardness and toughness features indicate synergistic control of ballistic performance',
            'mechanism': 'Hardness provides projectile defeat while toughness enables damage tolerance'
        })
    
    if 'Thermal Properties' in categories and ('Hardness Related' in categories or 'Elastic Properties' in categories):
        controlling_factors['synergistic_effects'].append({
            'type': 'Thermal-Mechanical Coupling',
            'description': 'Thermal and mechanical properties jointly control performance under dynamic loading',
            'mechanism': 'Thermal response affects local mechanical behavior during adiabatic heating'
        })
    
    return controlling_factors


def get_feature_mechanism(feature_name: str) -> str:
    """Get physical mechanism for specific feature"""
    feature_lower = feature_name.lower()
    
    if 'hardness' in feature_lower:
        return "Controls projectile blunting through resistance to plastic deformation"
    elif 'toughness' in feature_lower:
        return "Prevents catastrophic crack propagation and enables damage tolerance"
    elif 'density' in feature_lower:
        return "Influences momentum transfer, wave impedance, and specific performance"
    elif 'thermal' in feature_lower:
        return "Controls adiabatic heating response and thermal shock resistance"
    elif 'elastic' in feature_lower or 'modulus' in feature_lower:
        return "Affects stress wave propagation and crack deflection behavior"
    elif 'ballistic' in feature_lower:
        return "Direct measure of integrated ballistic performance"
    elif 'specific' in feature_lower:
        return "Normalized property providing weight-adjusted performance metric"
    else:
        return "Contributes to overall material response through specific physical mechanism"


def get_ballistic_role(feature_name: str) -> str:
    """Get ballistic role for specific feature"""
    feature_lower = feature_name.lower()
    
    if 'hardness' in feature_lower:
        return "Primary projectile defeat mechanism through blunting and dwell"
    elif 'toughness' in feature_lower:
        return "Multi-hit survivability and controlled fragmentation"
    elif 'density' in feature_lower:
        return "Weight considerations and momentum transfer effects"
    elif 'thermal' in feature_lower:
        return "Adiabatic heating management during high-velocity impact"
    elif 'elastic' in feature_lower:
        return "Stress distribution and spall formation control"
    elif 'ballistic' in feature_lower:
        return "Direct ballistic performance measurement"
    else:
        return "Supporting role in overall ballistic response"


def generate_materials_rationale(feature_ranking_df: pd.DataFrame, 
                               ceramic_system: str = None,
                               target_property: str = None) -> Dict[str, str]:
    """Generate comprehensive materials science rationale"""
    
    rationale = {
        'fundamental_principles': 'Ceramic armor performance governed by hardness-toughness-thermal property synergy',
        'microstructure_effects': 'Grain size, porosity, and phase distribution control macroscopic properties',
        'dynamic_behavior': 'High strain rate effects modify quasi-static property relationships',
        'failure_mechanisms': 'Spall, fragmentation, and through-thickness cracking determine defeat mechanisms',
        'optimization_strategy': 'Multi-objective optimization required due to property trade-offs'
    }
    
    # Add system-specific rationale
    if ceramic_system:
        if ceramic_system == 'SiC':
            rationale['system_specific'] = 'Covalent bonding provides ultra-high hardness but limits toughness'
        elif ceramic_system == 'Al2O3':
            rationale['system_specific'] = 'Ionic-covalent bonding enables balanced hardness-toughness combination'
        elif ceramic_system == 'B4C':
            rationale['system_specific'] = 'Complex structure provides highest hardness but extreme brittleness'
    
    # Add property-specific rationale
    if target_property:
        if 'ballistic' in target_property or 'v50' in target_property:
            rationale['property_specific'] = 'Ballistic performance emerges from complex interactions between multiple material properties'
        elif 'hardness' in target_property:
            rationale['property_specific'] = 'Hardness directly controls projectile blunting and dwell mechanisms'
        elif 'toughness' in target_property:
            rationale['property_specific'] = 'Toughness determines crack propagation resistance and damage tolerance'
    
    return rationale
