# src/feature_engineering/microstructure_features.py
import numpy as np
import pandas as pd
from loguru import logger

class MicrostructureFeatureCalculator:
    """
    Microstructure descriptors tuned for ballistic ceramics.
    - Hall–Petch-inspired terms from grain size
    - Porosity-normalized strength proxies
    - Relative density adjustments
    """

    def __init__(self):
        pass

    @staticmethod
    def _safe_div(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.true_divide(a, b)
            out[~np.isfinite(out)] = 0.0
        return out

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Grain size in micrometers is typical; handle mm or nm if indicated upstream.
        if "grain_size" in df.columns:
            gs = df["grain_size"].astype(float)
            # Hall–Petch-like term k/√d (k absorbed into scale)
            df["hp_term"] = self._safe_div(1.0, np.sqrt(np.maximum(gs, 1e-9)))
            # Crack path tortuosity proxy: 1/gs
            df["tortuosity_proxy"] = self._safe_div(1.0, np.maximum(gs, 1e-9))
        else:
            df["hp_term"] = 0.0
            df["tortuosity_proxy"] = 0.0

        # Porosity-normalized compressive strength
        if "compressive_strength" in df.columns and "porosity" in df.columns:
            por = np.clip(df["porosity"].astype(float), 0.0, 0.5)  # ceramics typically < 50%
            df["comp_strength_porosity_adj"] = df["compressive_strength"] * (1.0 - por)
        else:
            df["comp_strength_porosity_adj"] = df.get("compressive_strength", pd.Series(0.0, index=df.index))

        # Relative density scaling
        if "relative_density" in df.columns and "density" in df.columns:
            df["density_rel_scaled"] = df["density"] * df["relative_density"]
        else:
            df["density_rel_scaled"] = df.get("density", pd.Series(0.0, index=df.index))

        logger.info("✓ Microstructure features added")
        return df
