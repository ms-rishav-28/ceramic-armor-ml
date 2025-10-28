# src/data_collection/literature_miner.py
import os
import time
import requests
import pandas as pd
from pathlib import Path
from loguru import logger

SEM_SCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

class LiteratureMiner:
    """
    Lightweight Semantic Scholar miner (optional).
    Also supports manual CSV ingestion under data/raw/literature/<system>/*.csv
    """

    def __init__(self, api_key: str = None, base_dir: str = "data/raw/literature", timeout: int = 30):
        self.api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", None)
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def query(self, query: str, limit: int = 100):
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "title,year,authors,externalIds,url"
        }
        try:
            r = requests.get(SEM_SCH_URL, headers=headers, params=params, timeout=self.timeout)
            if r.status_code != 200:
                logger.warning(f"Semantic Scholar status={r.status_code}: {r.text[:200]}")
                return []
            data = r.json()
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Semantic Scholar error: {e}")
            return []

    def mine_system(self, ceramic_system: str, limit: int = 200) -> pd.DataFrame:
        # Example queries focused on mechanical/ballistic ceramics
        queries = [
            f"{ceramic_system} ceramic ballistic performance",
            f"{ceramic_system} fracture toughness hardness",
            f"{ceramic_system} V50 ballistic limit"
        ]
        rows = []
        for q in queries:
            results = self.query(q, limit=limit//len(queries))
            for r in results:
                rows.append({
                    "title": r.get("title"),
                    "year": r.get("year"),
                    "url": r.get("url"),
                    "doi": (r.get("externalIds") or {}).get("DOI"),
                    "ceramic_system": ceramic_system,
                    "source": "Literature"
                })
                time.sleep(0.25)
        df = pd.DataFrame(rows)
        out = self.base_dir / f"{ceramic_system.lower()}_literature_refs.csv"
        df.to_csv(out, index=False)
        logger.info(f"✓ Literature refs saved {len(df)} rows -> {out}")
        return df

    def load_curated(self, ceramic_system: str) -> pd.DataFrame:
        sys_dir = self.base_dir / ceramic_system.lower()
        files = list(sys_dir.glob("*.csv"))
        if not files:
            logger.warning(f"No curated literature CSVs found at {sys_dir}")
            return pd.DataFrame()
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                df["ceramic_system"] = ceramic_system
                df["source"] = "Literature"
                dfs.append(df)
            except Exception as e:
                logger.error(f"Curated literature load error {f}: {e}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
