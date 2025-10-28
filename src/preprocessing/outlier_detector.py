# src/preprocessing/outlier_detector.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from loguru import logger

def remove_iqr_outliers(df: pd.DataFrame, columns, k: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    keep_mask = pd.Series(True, index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        keep_mask &= df[col].between(low, high) | df[col].isna()
    removed = (~keep_mask).sum()
    logger.info(f"IQR outlier removal: removed {removed}")
    return df[keep_mask].copy()

def remove_zscore_outliers(df: pd.DataFrame, columns, z: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    keep_mask = pd.Series(True, index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        mu, sigma = df[col].mean(), df[col].std()
        if sigma > 0:
            zscores = (df[col] - mu) / sigma
            keep_mask &= (zscores.abs() <= z) | df[col].isna()
    removed = (~keep_mask).sum()
    logger.info(f"Z-score outlier removal: removed {removed}")
    return df[keep_mask].copy()

def isolation_forest_filter(df: pd.DataFrame, columns, contamination: float = 0.02, random_state: int = 42):
    sub = df[columns].fillna(df[columns].median())
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    preds = iso.fit_predict(sub)
    keep = preds == 1
    removed = (~keep).sum()
    logger.info(f"IsolationForest outlier removal: removed {removed}")
    return df[keep].copy()
