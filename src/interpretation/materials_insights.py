# src/interpretation/materials_insights.py
import pandas as pd

def interpret_feature_ranking(importance_df: pd.DataFrame, top_k: int = 10) -> str:
    """
    Convert feature importance into mechanistic interpretation text.
    """
    top = importance_df.head(top_k)["feature"].astype(str).tolist()
    statements = []
    if any("vickers_hardness" in f or "specific_hardness" in f for f in top):
        statements.append("- High hardness and specific hardness dominate penetration resistance due to increased dwell and blunting.")
    if any("fracture_toughness" in f for f in top):
        statements.append("- Fracture toughness mitigates catastrophic cracking, improving multi-hit survivability.")
    if any("density" in f for f in top):
        statements.append("- Density influences inertia and wave impedance; normalized metrics favor high hardness at lower density.")
    if any("thermal_conductivity" in f or "thermal_shock" in f for f in top):
        statements.append("- Thermal transport and shock resistance are critical under adiabatic heating during impact (>1000°C microseconds).")
    if any("pugh_ratio" in f or "elastic_anisotropy" in f for f in top):
        statements.append("- Elastic metrics (G/B, anisotropy) correlate with crack deflection and spall behavior.")
    if not statements:
        statements.append("- Feature importance indicates multi-factor control; hardness–toughness–thermal triad remains central.")
    return "\n".join(statements)
