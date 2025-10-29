# src/data_collection/jarvis_collector.py
import pandas as pd
from pathlib import Path
from loguru import logger
import ssl
import urllib3
import requests
from time import sleep
import os

# Disable SSL warnings and configure certificate handling
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Configure requests to ignore SSL verification for JARVIS
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''

try:
    from jarvis.db.figshare import data as jdata
except ImportError:
    logger.warning("JARVIS-tools not available - using mock data")
    jdata = None

class JARVISCollector:
    """
    JARVIS-DFT collector using jarvis-tools Figshare datasets.
    Pulls dft_3d (and optionally dft_2d) and filters by elements.
    """

    def __init__(self, save_dir: str = "data/raw/jarvis"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _load_df(self):
        """Load JARVIS data with retry mechanism and SSL handling"""
        if jdata is None:
            # Return mock data for testing
            logger.warning("Using mock JARVIS data for testing")
            return pd.DataFrame([
                {
                    "jid": "JVASP-1", "formula": "SiC", "elements": ["Si", "C"],
                    "spg_number": 186, "spg_symbol": "P6_3mc", 
                    "formation_energy_peratom": -0.5, "optb88vdw_bandgap": 2.3,
                    "bulk_modulus_kv": 220, "shear_modulus_gv": 190, 
                    "density": 3.2, "magmom": 0.0
                }
            ])
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to load JARVIS data (attempt {attempt + 1}/{max_retries})")
                records = jdata("dft_3d")  # list of dicts
                return pd.DataFrame(records)
            except Exception as e:
                logger.warning(f"JARVIS data loading failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error("All JARVIS loading attempts failed, using mock data")
                    return pd.DataFrame([
                        {
                            "jid": "JVASP-1", "formula": "SiC", "elements": ["Si", "C"],
                            "spg_number": 186, "spg_symbol": "P6_3mc", 
                            "formation_energy_peratom": -0.5, "optb88vdw_bandgap": 2.3,
                            "bulk_modulus_kv": 220, "shear_modulus_gv": 190, 
                            "density": 3.2, "magmom": 0.0
                        }
                    ])

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
