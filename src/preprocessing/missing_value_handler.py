# src/preprocessing/missing_value_handler.py
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from loguru import logger

def impute_median(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].fillna(df[num].median())
    logger.info("✓ Median imputation complete")
    return df

def impute_mean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].fillna(df[num].mean())
    logger.info("✓ Mean imputation complete")
    return df

def impute_knn(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    df = df.copy()
    num = df.select_dtypes(include=[np.number]).columns
    imp = KNNImputer(n_neighbors=n_neighbors)
    df[num] = imp.fit_transform(df[num])
    logger.info("✓ KNN imputation complete")
    return df

def impute_iterative(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    df = df.copy()
    num = df.select_dtypes(include=[np.number]).columns
    imp = IterativeImputer(random_state=random_state, max_iter=20, sample_posterior=False)
    df[num] = imp.fit_transform(df[num])
    logger.info("✓ Iterative imputation complete")
    return df
