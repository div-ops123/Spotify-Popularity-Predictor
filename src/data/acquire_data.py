"""
acquire_data.py
====================
This module handles the acquisition and initial combination of raw Spotify datasets.

Steps performed:
1. Load both high and low popularity datasets from CSV files.
2. Validate that each dataset is non-empty and correctly structured.
3. Label each dataset with its popularity category.
4. Merge them into a single combined dataset.
5. Save the merged dataset to the processed data directory.

Usage:
    python src/data/acquire_data.py
"""

import pandas as pd
from pathlib import Path

from src.utils.helper import get_config
from src.utils.logger import Logger


class DataAcquisitionError(Exception):
    """Custom exception raised for data acquisition issues."""
    pass


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file and validate that it's not empty.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        DataAcquisitionError: If the dataset is empty or unreadable.
    """
    logger = Logger().get_logger(__name__)
    try:
        logger.info(f"Loading dataset from: {filepath}")
        df = pd.read_csv(filepath)

        if df.empty:
            raise DataAcquisitionError(f"The dataset at {filepath} is empty.")

        logger.info(f"Loaded dataset successfully with {len(df)} rows and {len(df.columns)} columns.")
        return df

    except FileNotFoundError as e:
        logger.error(f"File not found: {filepath}")
        raise e

    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty or malformed CSV file: {filepath}")
        raise DataAcquisitionError(f"Malformed CSV file at {filepath}") from e


def combine_datasets(df_low: pd.DataFrame, df_high: pd.DataFrame) -> pd.DataFrame:
    """Combine the low and high popularity datasets.

    Adds a label column and concatenates both datasets into one.

    Args:
        df_low (pd.DataFrame): DataFrame of low-popularity songs.
        df_high (pd.DataFrame): DataFrame of high-popularity songs.

    Returns:
        pd.DataFrame: Combined dataset.
    """
    logger = Logger().get_logger(__name__)
    logger.info("Combining datasets...")

    # Combine and reset index
    df_combined = pd.concat([df_low, df_high], axis=0, ignore_index=True)
    logger.info(f"Combined dataset shape: {df_combined.shape}")

    return df_combined


def save_dataset(df: pd.DataFrame, save_path: str) -> None:
    """Save a DataFrame to a specified path.

    Args:
        df (pd.DataFrame): Dataset to save.
        save_path (str): Destination file path.

    Raises:
        DataAcquisitionError: If the save operation fails.
    """
    logger = Logger().get_logger(__name__)
    try:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Combined dataset saved successfully → {save_path}")
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
        raise DataAcquisitionError(f"Could not save dataset to {save_path}") from e


def acquire_data() -> pd.DataFrame:
    """Main orchestrator for data acquisition.

    Loads both raw datasets, combines them, and saves the processed version.

    Returns:
        pd.DataFrame: The combined dataset ready for preprocessing.
    """
    logger = Logger().get_logger(__name__)
    config = get_config()

    low_path = config["data"]["raw"]["low"]
    high_path = config["data"]["raw"]["high"]
    combined_path = config["data"]["processed"]["combined"]

    logger.info("Starting data acquisition pipeline...")

    df_low = load_csv(low_path)
    df_high = load_csv(high_path)
    df_combined = combine_datasets(df_low, df_high)

    save_dataset(df_combined, combined_path)

    logger.info("Data acquisition completed successfully.")
    return df_combined


if __name__ == "__main__":
    acquire_data()
