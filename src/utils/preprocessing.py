"""
utils/preprocessing.py
======================

This module provides reusable utility functions for data cleaning,
feature engineering, and preprocessing tasks for the Spotify project.

It converts the exploratory logic from the notebook
(`notebooks/02_data_preprocessing.ipynb`)
into modular, testable, and production-ready code.

Each function performs ONE job only.
"""

import pandas as pd
from typing import List, Tuple


# -----------------------------------------------------------------------------
# 1. Handle Missing Values
# -----------------------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Steps (to be implemented later):
      • Check for missing values.
      • Drop or fill missing values depending on feature type.
      • Log any changes or warnings for large missing rates.

    Args:
        df (pd.DataFrame): Raw input dataframe.

    Returns:
        pd.DataFrame: DataFrame after handling missing values.
    """
    pass

# -----------------------------------------------------------------------------
# 2. Handle Duplicate Values
# -----------------------------------------------------------------------------
def handle_duplicate_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle duplicate values in the dataset.

    Steps (to be implemented later):
      • Check for duplicate values.
      • Drop or fill duplicate values depending on feature type.
      • Log any changes or warnings for large duplicate rates.

    Args:
        df (pd.DataFrame): Raw input dataframe.

    Returns:
        pd.DataFrame: DataFrame after handling dupliacte values.
    """
    pass


# -----------------------------------------------------------------------------
# 3. Convert Data Types
# -----------------------------------------------------------------------------
def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert feature columns to correct data types.

    Steps:
      • Convert columns like 'mode', 'key', and 'time_signature' to categorical.
      • Ensure numeric columns are float or int as needed.

    Args:
        df (pd.DataFrame): DataFrame after missing value handling.

    Returns:
        pd.DataFrame: DataFrame with correct data types.
    """
    pass


# -----------------------------------------------------------------------------
# 4. Feature Engineering
# -----------------------------------------------------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the dataset.

    Steps:
      • Apply scaling or transformation to continuous variables
        (e.g., loudness, tempo, duration_ms).
      • Encode categorical variables using one-hot encoding.
      • Add derived features if needed (e.g., song_age_scaled).

    Args:
        df (pd.DataFrame): DataFrame with correct dtypes.

    Returns:
        pd.DataFrame: DataFrame with new engineered features.
    """
    pass


# -----------------------------------------------------------------------------
# 5. Exclude Non-feature Columns
# -----------------------------------------------------------------------------
def exclude_non_feature_columns(df: pd.DataFrame, exclude_cols: List[str]) -> pd.DataFrame:
    """
    Remove columns that are not features for model training.

    Example:
        exclude_cols = ['artist_name', 'track_name', 'id']

    Args:
        df (pd.DataFrame): Cleaned DataFrame with all features.
        exclude_cols (List[str]): Columns to remove.

    Returns:
        pd.DataFrame: DataFrame containing only feature columns.
    """
    pass


# -----------------------------------------------------------------------------
# 6. Main Cleaning Pipeline
# -----------------------------------------------------------------------------
def clean_dataset_pipeline(
    df: pd.DataFrame,
    exclude_cols: List[str] = None  # ?? should we have this?
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Full cleaning pipeline combining all preprocessing steps.

    Steps:
      1. Handle missing values.
      2. Convert data types.
      3. Apply feature engineering.
      4. Exclude non-feature columns.

    Args:
        df (pd.DataFrame): Raw input dataset.
        exclude_cols (List[str], optional): Columns to exclude from features.

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            • Cleaned DataFrame ready for modeling.
            • List of final feature column names.
    """
    pass

# TODO:
# Later, src/data/preprocess_data.py will import this pipeline, call it on the raw dataset, then save the cleaned CSV (production flow).
