# src/data_collection/data_integrator.py
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

CANON_COLS = [
    "material_id","formula","ceramic_system","density","band_gap",
    "youngs_modulus","bulk_modulus","shear_modulus","poisson_ratio",
    "vickers_hardness","knoop_hardness","compressive_strength",
    "fracture_toughness_mode_i","thermal_conductivity","thermal_expansion_coefficient",
    "specific_heat","energy_above_hull","formation_energy","source"
]

class DataIntegrator:
    """Merge heterogeneous sources, standardize schema, and deduplicate."""

    def __init__(self, save_dir: str = "data/processed"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _coerce_numeric(df: pd.DataFrame, cols):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def integrate_system(self, ceramic_system: str, inputs: dict):
        """
        inputs: dict with keys among {'materials_project','aflow','jarvis','nist'}
                and values as CSV file paths
        """
        dfs = []
        for name, path in inputs.items():
            try:
                df = pd.read_csv(path)
                df["source"] = df.get("source", name)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Skip {name}: {e}")
        if not dfs:
            logger.warning("No inputs provided")
            return pd.DataFrame()
        df = pd.concat(dfs, ignore_index=True)

        # Normalize column names
        norm = {c.lower(): c for c in df.columns}
        df.columns = [c.lower() for c in df.columns]
        # Create all canonical columns if missing
        for c in [c.lower() for c in CANON_COLS]:
            if c not in df.columns:
                df[c] = np.nan

        # Coerce numeric important columns
        numeric_cols = [
            "density","band_gap","youngs_modulus","bulk_modulus","shear_modulus",
            "poisson_ratio","vickers_hardness","knoop_hardness","compressive_strength",
            "fracture_toughness_mode_i","thermal_conductivity","thermal_expansion_coefficient",
            "specific_heat","energy_above_hull","formation_energy"
        ]
        df = self._coerce_numeric(df, numeric_cols)

        # Deduplication key
        df["formula_norm"] = df["formula"].astype(str).str.replace(r"\s+", "", regex=True)
        def keyrow(r):
            return (
                r["formula_norm"],
                round(r["density"], 3) if pd.notna(r["density"]) else None,
                round(r["youngs_modulus"], 2) if pd.notna(r["youngs_modulus"]) else None,
                round(r["vickers_hardness"], 2) if pd.notna(r["vickers_hardness"]) else None
            )
        df["dedup_key"] = df.apply(keyrow, axis=1)
        df.sort_values(by=["source"], inplace=True)  # prefer curated sources later if desired
        df = df.drop_duplicates(subset=["dedup_key"], keep="first").copy()

        # Finalize
        df["ceramic_system"] = ceramic_system
        out_dir = self.save_dir / ceramic_system.lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{ceramic_system.lower()}_integrated.csv"
        df.to_csv(out, index=False)
        logger.info(f"✓ Integrated {len(df)} rows -> {out}")
        return df
