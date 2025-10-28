# src/evaluation/error_analyzer.py
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

class ErrorAnalyzer:
    """Granular error analysis utilities."""

    @staticmethod
    def summarize_errors(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        err = y_pred - y_true
        df = pd.DataFrame({
            "y_true": y_true, "y_pred": y_pred,
            "error": err, "abs_error": np.abs(err)
        })
        stats = df["abs_error"].describe().to_frame(name="abs_error").T
        return df, stats

    @staticmethod
    def by_category(errors_df: pd.DataFrame, features_df: pd.DataFrame, category: str) -> pd.DataFrame:
        if category not in features_df.columns:
            return pd.DataFrame()
        joined = errors_df.join(features_df[[category]])
        grp = joined.groupby(category)["abs_error"].agg(["mean", "median", "count"]).reset_index()
        return grp

    @staticmethod
    def save_tables(df_dict: dict, out_dir: str):
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        for name, df in df_dict.items():
            df.to_csv(out / f"{name}.csv", index=False)
            logger.info(f"Saved error table: {name}")
