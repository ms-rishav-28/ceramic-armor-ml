# src/data_collection/aflow_collector.py
import time
import json
import pandas as pd
import requests
from pathlib import Path
from loguru import logger

AFLUX_URL = "https://aflowlib.duke.edu/search/API/"

class AFLOWCollector:
    """
    AFLOW/AFLUX data collector for ceramic systems.
    Uses species filters and requests commonly used properties.
    """

    def __init__(self, save_dir: str = "data/raw/aflow", timeout: int = 60):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    @staticmethod
    def _species_clause(elements):
        # AFLUX species clause expects (=El1,El2) etc.
        els = ",".join(elements)
        return f"species(={els})"

    def _query(self, elements, batch_size=1000, properties=None, retries=3, sleep=2):
        if properties is None:
            # Commonly needed properties; fields are AFLUX keywords
            properties = [
                "compound", "Egap", "enthalpy_formation_atom", "density",
                "volume_atom", "species", "spacegroup_relax", "stoichiometry",
                "aurl", "geometry", "prototype"
            ]
        props = ",".join(properties)
        clause = self._species_clause(elements)
        # AFLUX pattern: ?<properties>,paging(<skip>),<filter>
        results = []
        skip = 0
        while True:
            url = f"{AFLUX_URL}?{props},paging({skip}),{clause}"
            ok = False
            for t in range(retries):
                try:
                    r = requests.get(url, timeout=self.timeout)
                    if r.status_code == 200:
                        ok = True
                        break
                except Exception as e:
                    logger.warning(f"Aflow request error attempt {t+1}: {e}")
                time.sleep(sleep*(t+1))
            if not ok:
                break
            try:
                chunk = r.json()  # AFLUX returns JSON array of dicts
            except Exception:
                # fallback: parse text lines separated by \n
                try:
                    chunk = json.loads(r.text)
                except Exception as e:
                    logger.error(f"AFLOW parsing error: {e}")
                    break
            if not chunk:
                break
            results.extend(chunk)
            if len(chunk) < batch_size:
                break
            skip += batch_size
        return results

    def collect(self, ceramic_system: str, elements_map=None):
        """
        Collect AFLOW entries for a ceramic system.
        elements_map maps system name to element list.
        """
        if elements_map is None:
            elements_map = {
                "SiC": ["Si", "C"], "Al2O3": ["Al", "O"], "B4C": ["B", "C"],
                "WC": ["W", "C"], "TiC": ["Ti", "C"]
            }
        if ceramic_system not in elements_map:
            raise ValueError(f"Unknown system: {ceramic_system}")
        elements = elements_map[ceramic_system]
        logger.info(f"AFLOW: querying {ceramic_system} with elements {elements}")
        data = self._query(elements)
        df = pd.DataFrame(data)
        df["ceramic_system"] = ceramic_system
        df["source"] = "AFLOW"
        out = self.save_dir / f"{ceramic_system.lower()}_aflow_raw.csv"
        df.to_csv(out, index=False)
        logger.info(f"✓ AFLOW saved {len(df)} rows -> {out}")
        return df
