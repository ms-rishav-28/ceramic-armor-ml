# src/preprocessing/data_cleaner.py
import pandas as pd
from loguru import logger
from .unit_standardizer import standardize
from .outlier_detector import remove_iqr_outliers
from .missing_value_handler import impute_knn

class DataCleaner:
    """
    Reproducible cleaning pipeline:
    1) Standardize units
    2) Remove outliers (IQR)
    3) Impute missing values (KNN)
    """

    def __init__(self):
        pass

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = standardize(df)
        # Choose a conservative subset for outlier removal
        numeric_targets = [c for c in [
            "density","youngs_modulus","bulk_modulus","shear_modulus","vickers_hardness",
            "compressive_strength","fracture_toughness_mode_i","thermal_conductivity"
        ] if c in df.columns]
        if numeric_targets:
            df = remove_iqr_outliers(df, numeric_targets, k=1.5)
        df = impute_knn(df, n_neighbors=5)
        logger.info("✓ Data cleaning pipeline complete")
        return df
