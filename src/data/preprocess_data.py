# This script is the entry point for running preprocessing in production or CI/CD pipelines.

"""
data/preprocess_data.py
=======================

This script orchestrates the entire data preprocessing workflow
for the Spotify project.

It:
  • Loads the raw dataset.
  • Runs all cleaning + feature engineering steps (via utils/preprocessing.py).
  • Saves the final cleaned dataset to `data/processed/spotify_clean.csv`.
  • Logs all key steps for reproducibility and debugging.

Usage (example):
    python src/data/preprocess_data.py
"""

import pandas as pd
from src.utils.logger import Logger
from src.utils.helper import load_csv, save_csv

from src.utils.preprocessing import clean_dataset_pipeline

logger = Logger.get_logger(__name__)



# -----------------------------------------------------------------------------
# 1. Load Raw Data
# -----------------------------------------------------------------------------
def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw dataset from the given file path.

    Args:
        file_path (str): Path to the raw CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return load_csv(file_path)


# -----------------------------------------------------------------------------
# 2. Save Clean Data
# -----------------------------------------------------------------------------
def save_clean_data(df: pd.DataFrame, save_path: str) -> None:
    """
    Save cleaned dataset to disk.

    Args:
        df (pd.DataFrame): Cleaned dataset.
        save_path (str): Destination file path.
    """
    save_csv(df, save_path)


# -----------------------------------------------------------------------------
# 3. Main Orchestration
# -----------------------------------------------------------------------------
def main():
    """
    Run the full preprocessing pipeline.

    Steps:
      2. Load raw dataset.
      3. Run cleaning pipeline.
      4. Save processed dataset.
    """
    raw_path = "data/raw/spotify_raw.csv"
    save_path = "data/processed/spotify_clean.csv"

    df_raw = load_raw_data(raw_path)

    exclude_columns = ['artist_name', 'track_name', 'track_id'] # ??

    df_clean, feature_columns = clean_dataset_pipeline(df_raw, exclude_columns)

    save_clean_data(df_clean, save_path)

    logger.info(f"Final dataset shape: {df_clean.shape}")
    logger.info(f"Feature columns used: {len(feature_columns)}")


# -----------------------------------------------------------------------------
# 4. Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
