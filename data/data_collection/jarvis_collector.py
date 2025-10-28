# src/data_collection/jarvis_collector.py
import pandas as pd
from pathlib import Path
from loguru import logger
from jarvis.db.figshare import data as jdata

class JARVISCollector:
    """
    JARVIS-DFT collector using jarvis-tools Figshare datasets.
    Pulls dft_3d (and optionally dft_2d) and filters by elements.
    """

    def __init__(self, save_dir: str = "data/raw/jarvis"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _load_df(self):
        records = jdata("dft_3d")  # list of dicts
        return pd.DataFrame(records)

    def collect(self, ceramic_system: str, elements_map=None):
        if elements_map is None:
            elements_map = {
                "SiC": ["Si", "C"], "Al2O3": ["Al", "O"], "B4C": ["B", "C"],
                "WC": ["W", "C"], "TiC": ["Ti", "C"]
            }
        els = set(elements_map[ceramic_system])
        logger.info(f"JARVIS: loading dataset for {ceramic_system}")
        df = self._load_df()
        # Filter by elements present in 'elements' field
        def ok(row):
            try:
                return els.issubset(set(row["elements"]))
            except Exception:
                return False
        df = df[df.apply(ok, axis=1)].copy()
        # Normalize some columns of interest
        keep = [
            "jid", "formula", "elements", "spg_number", "spg_symbol", "formation_energy_peratom",
            "optb88vdw_bandgap", "bulk_modulus_kv", "shear_modulus_gv", "density", "magmom",
        ]
        df = df[[c for c in keep if c in df.columns]].copy()
        df.rename(columns={
            "jid": "material_id",
            "formation_energy_peratom": "formation_energy",
            "optb88vdw_bandgap": "band_gap",
            "bulk_modulus_kv": "bulk_modulus",
            "shear_modulus_gv": "shear_modulus"
        }, inplace=True)
        df["ceramic_system"] = ceramic_system
        df["source"] = "JARVIS-DFT"
        out = self.save_dir / f"{ceramic_system.lower()}_jarvis_raw.csv"
        df.to_csv(out, index=False)
        logger.info(f"✓ JARVIS saved {len(df)} rows -> {out}")
        return df
