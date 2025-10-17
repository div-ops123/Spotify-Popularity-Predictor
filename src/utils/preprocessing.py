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
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

from src.utils.helper import get_config
from src.utils.logger import Logger

config = get_config()
logger = Logger(config.get("DEBUG_MODE")).get_logger(__name__)


# -----------------------------------------------------------------------------
# 1. Handle Missing Values
# -----------------------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        df (pd.DataFrame): Raw input dataframe.

    Returns:
        pd.DataFrame: DataFrame after handling missing values.
    """
    df = df.dropna()
    logger.info("Dropped missing values.")
    return df

# -----------------------------------------------------------------------------
# 2. Handle Duplicate Values
# -----------------------------------------------------------------------------
def handle_duplicate_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle duplicate values in the dataset.

    Args:
        df (pd.DataFrame): Raw input dataframe.

    Returns:
        pd.DataFrame: DataFrame after handling dupliacte values.
    """
    df = df.drop_duplicates()
    logger.info("Dropped duplicate values.")
    return df


# -----------------------------------------------------------------------------
# 3. Convert Data Types
# -----------------------------------------------------------------------------
def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert feature columns to correct data types.

    Args:
        df (pd.DataFrame): DataFrame after missing value handling.

    Returns:
        pd.DataFrame: DataFrame with correct data types.
    """
    
    df['track_album_release_date'] = pd.to_datetime(df['track_album_release_date'], errors='coerce')

    df['release_year'] = df['track_album_release_date'].dt.year.astype('Int64')
    df['release_month'] = df['track_album_release_date'].dt.month.astype('Int64')
    df['song_age'] = datetime.now().year - df['release_year'].astype('Int64')

    df = df.dropna(subset=['track_album_release_date', 'release_month', 'release_year', 'song_age'])
    logger.info("Converted feature columns to correct data type.")
    return df


# -----------------------------------------------------------------------------
# 4. Feature Engineering
# -----------------------------------------------------------------------------
def feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform feature engineering on the dataset.

    This function applies scaling, encoding, and transformation steps to
    numeric and categorical features. It also fits the transformers (e.g.,
    StandardScaler, MinMaxScaler, OneHotEncoder) on the data — which can later
    be saved and reused during inference.

    Args:
        df (pd.DataFrame): Input DataFrame after type conversion and cleaning.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            • The transformed DataFrame with new engineered feature columns.
            • A dictionary containing the fitted transformers:
              {
                  "scaler": StandardScaler(),
                  "minmax": MinMaxScaler(),
                  "encoder": OneHotEncoder()
              }
    """
    df['key'] = df['key'].astype('category')
    df['time_signature'] = df['time_signature'].astype('category')
    
    df['duration_ms_log'] = np.log1p(df['duration_ms'])

    scaler = StandardScaler()
    minmax = MinMaxScaler()

    df[['tempo_scaled', 'loudness_scaled', 'duration_ms_log_scaled']] = scaler.fit_transform(df[['tempo', 'loudness', 'duration_ms_log']])
    df['song_age_scaled'] = minmax.fit_transform(df[['song_age']])

    # Create and fit the encoder
    encoder = OneHotEncoder(drop='first', sparse_output=False) # drop the first column to avoid redundancy    
    
    categorical_cols = config.get("categorical_cols", [])
    for col in categorical_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found for encoding.")

    encoded = encoder.fit_transform(df[categorical_cols])

    # Convert encoded array back to DataFrame
    encoded_cols = encoder.get_feature_names_out(categorical_cols)  # Get proper column names
    df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)

    # Merge with original dataframe
    df = pd.concat([df.drop(columns=categorical_cols), df_encoded], axis=1)
    logger.info("Performed feature engineering on the dataset.")
    return df, {"scaler": scaler, "minmax": minmax, "encoder": encoder}



# -----------------------------------------------------------------------------
# 5. Exclude Non-feature Columns
# -----------------------------------------------------------------------------
def select_feature_columns(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> pd.DataFrame:
    """
    Select only feature + target columns for model training.

    Args:
        df (pd.DataFrame): Cleaned DataFrame with all features.
        feature_cols (List[str]): Feature Columns.
        target_col (str): Target column.

    Returns:
        pd.DataFrame: DataFrame containing only feature columns.
    """
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"'{col}' not found in DataFrame!")
    df_clean = df[feature_cols + [target_col]].copy()
    logger.info("Selected feature columns + target.")
    return df_clean


# -----------------------------------------------------------------------------
# 6. Main Cleaning Pipeline
# -----------------------------------------------------------------------------
def clean_dataset_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Full cleaning pipeline that combines all preprocessing steps.

    Steps performed:
      1. Handle missing values.
      2. Remove duplicate rows.
      3. Convert data types to correct formats.
      4. Apply feature engineering (scaling, encoding, etc.).
      5. Select final feature columns for modeling.

    Args:
        df (pd.DataFrame): Raw input dataset.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
            • Cleaned and fully preprocessed DataFrame ready for modeling.
            • Dictionary of fitted transformers used during feature engineering.
            • List of final feature column names (from config.yaml).
    """
    # should we use builder pattern for this?
    df = handle_missing_values(df)
    df = handle_duplicate_values(df)
    df = convert_dtypes(df)
    df, transformers = feature_engineering(df)
    df = select_feature_columns(df, config.get("features"), config.get("target"))
    return df, transformers, config.get("features")
