# src/preprocessing/unit_standardizer.py
import numpy as np
import pandas as pd
from loguru import logger

PRESSURE_TO_GPA = {
    "Pa": 1e-9, "kPa": 1e-6, "MPa": 1e-3, "GPa": 1.0, "psi": 6.89476e-6
}
DENSITY_TO_GCM3 = {"kg/m3": 1e-3, "g/cm3": 1.0}
K_TO_WMK = {"W/m·K": 1.0, "W/mK": 1.0, "W/cm·K": 100.0}
CTE_TO_1K = {"1/K": 1.0, "ppm/K": 1e-6}
TOUGHNESS_TO_MPA_SQRT_M = {"MPa√m": 1.0, "MPa·m^0.5": 1.0, "ksi√in": 1.099}

def _convert_series(x, unit_col, mapping):
    vals = x.copy()
    if unit_col not in vals:
        return vals
    units = vals[unit_col]
    data = vals.drop(columns=[unit_col])
    if isinstance(units, pd.Series):
        # Row-wise conversion
        for i, u in units.items():
            f = mapping.get(u, None)
            if f is not None and pd.notna(data.iloc[i]):
                data.iloc[i] = data.iloc[i] * f
    else:
        f = mapping.get(units, None)
        if f is not None:
            data = data * f
    return data

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize known property units to canonical SI-like units."""
    df = df.copy()
    # Pressure/moduli to GPa
    for col in ["youngs_modulus", "bulk_modulus", "shear_modulus", "vickers_hardness", "knoop_hardness"]:
        if f"{col}_unit" in df.columns and col in df.columns:
            df[col] = _convert_series(df[[col, f"{col}_unit"]], f"{col}_unit", PRESSURE_TO_GPA)[col]
            df.drop(columns=[f"{col}_unit"], inplace=True)
    # Compressive strength to MPa remains (convert to MPa)
    if "compressive_strength_unit" in df.columns and "compressive_strength" in df.columns:
        m = {"GPa": 1000.0, "MPa": 1.0, "kPa": 0.001, "psi": 0.00689476}
        df["compressive_strength"] = _convert_series(df[["compressive_strength", "compressive_strength_unit"]],
                                                     "compressive_strength_unit", m)["compressive_strength"]
        df.drop(columns=["compressive_strength_unit"], inplace=True)
    # Density to g/cm3
    if "density_unit" in df.columns and "density" in df.columns:
        df["density"] = _convert_series(df[["density", "density_unit"]], "density_unit", DENSITY_TO_GCM3)["density"]
        df.drop(columns=["density_unit"], inplace=True)
    # Thermal conductivity to W/m·K
    if "thermal_conductivity_unit" in df.columns and "thermal_conductivity" in df.columns:
        df["thermal_conductivity"] = _convert_series(df[["thermal_conductivity", "thermal_conductivity_unit"]],
                                                     "thermal_conductivity_unit", K_TO_WMK)["thermal_conductivity"]
        df.drop(columns=["thermal_conductivity_unit"], inplace=True)
    # CTE to 1/K
    if "thermal_expansion_coefficient_unit" in df.columns and "thermal_expansion_coefficient" in df.columns:
        df["thermal_expansion_coefficient"] = _convert_series(
            df[["thermal_expansion_coefficient", "thermal_expansion_coefficient_unit"]],
            "thermal_expansion_coefficient_unit", CTE_TO_1K
        )["thermal_expansion_coefficient"]
        df.drop(columns=["thermal_expansion_coefficient_unit"], inplace=True)
    # Toughness to MPa√m
    if "fracture_toughness_mode_i_unit" in df.columns and "fracture_toughness_mode_i" in df.columns:
        df["fracture_toughness_mode_i"] = _convert_series(
            df[["fracture_toughness_mode_i", "fracture_toughness_mode_i_unit"]],
            "fracture_toughness_mode_i_unit", TOUGHNESS_TO_MPA_SQRT_M
        )["fracture_toughness_mode_i"]
        df.drop(columns=["fracture_toughness_mode_i_unit"], inplace=True)

    logger.info("✓ Unit standardization complete")
    return df
