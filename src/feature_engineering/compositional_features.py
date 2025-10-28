# src/feature_engineering/compositional_features.py
from typing import Dict
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from loguru import logger

class CompositionalFeatureCalculator:
    """Compute composition-based descriptors from chemical formula."""

    def __init__(self):
        pass

    @staticmethod
    def _safe(val, default=np.nan):
        try:
            return float(val) if val is not None else default
        except Exception:
            return default

    def from_formula(self, formula: str) -> Dict[str, float]:
        comp = Composition(formula)
        elements = list(comp.elements)
        fracs = [comp.get_atomic_fraction(el) for el in elements]

        # Atomic mass / radius / electronegativity
        masses, radii, en = [], [], []
        groups, periods = [], []
        for el in elements:
            masses.append(self._safe(el.atomic_mass))
            radii.append(self._safe(el.atomic_radius))
            en.append(self._safe(el.X))
            groups.append(self._safe(el.group))
            periods.append(self._safe(el.period))

        def stats(arr):
            arr = np.array([x for x in arr if not np.isnan(x)])
            if arr.size == 0:
                return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, range=np.nan)
            return dict(mean=arr.mean(), std=arr.std(), min=arr.min(), max=arr.max(), range=arr.max()-arr.min())

        m, r, e = stats(masses), stats(radii), stats(en)
        g, p = stats(groups), stats(periods)

        # Mixing entropy: -k Σ xi ln xi (k omitted: relative metric)
        mixing_entropy = -sum([x*np.log(x) for x in fracs if x > 0])

        feats = {
            "comp_atomic_mass_mean": m["mean"], "comp_atomic_mass_std": m["std"], "comp_atomic_mass_range": m["range"],
            "comp_atomic_radius_mean": r["mean"], "comp_atomic_radius_std": r["std"], "comp_atomic_radius_range": r["range"],
            "comp_en_mean": e["mean"], "comp_en_std": e["std"], "comp_en_range": e["range"],
            "comp_group_mean": g["mean"], "comp_group_std": g["std"], "comp_period_mean": p["mean"],
            "comp_mixing_entropy": mixing_entropy,
            "comp_valence_e_guess": (g["mean"] if not np.isnan(g["mean"]) else np.nan)
        }
        return feats

    def augment_dataframe(self, df: pd.DataFrame, formula_col: str = "formula") -> pd.DataFrame:
        df = df.copy()
        rows = []
        for f in df[formula_col].astype(str).tolist():
            try:
                rows.append(self.from_formula(f))
            except Exception:
                rows.append({k: np.nan for k in ["comp_atomic_mass_mean"]})  # minimal backstop
        comp_df = pd.DataFrame(rows, index=df.index)
        out = pd.concat([df, comp_df], axis=1)
        logger.info(f"✓ Added compositional descriptors: {comp_df.shape[1]} features")
        return out
