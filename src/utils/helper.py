"""
helper.py
====================
Utility functions for common tasks across the project.
Currently includes configuration loading utilities.
"""

import os
import yaml
import pickle
import pandas as pd
from pathlib import Path

from src.utils.logger import Logger


def get_config() -> dict:
    """Load configuration from the `configs/config.yaml` file.

    This function dynamically locates the project root (two levels up from
    this file), reads the YAML configuration, and returns it as a dictionary.

    Returns:
        dict: Dictionary containing all configuration values.

    Raises:
        FileNotFoundError: If `config.yaml` does not exist.
        yaml.YAMLError: If the YAML file is invalid or cannot be parsed.
    """
    # Step 1: Resolve project root (2 levels up from this file)
    project_root = Path(__file__).resolve().parents[2]

    # Step 2: Point to the config file
    config_path = project_root / "configs" / "config.yaml"

    # Step 3: Read and parse YAML
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing config file: {e}")

    return config


config = get_config()
DEBUG_MODE = config.get("DEBUG_MODE", False)
logger = Logger(debug=DEBUG_MODE).get_logger(__name__)

# -------------------------------------------------------------------
# 1. Load CSV
# -------------------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame with validation and logging.

    Args:
        path (str): The file path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
    """
    if not os.path.exists(path):
        logger.error(f"❌ File not found: {path}")
        raise FileNotFoundError(f"CSV file not found: {path}")

    try:
        df = pd.read_csv(path)
        logger.info(f"✅ Loaded CSV: {path} — {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    except pd.errors.EmptyDataError:
        logger.error(f"❌ CSV file is empty: {path}")
        raise

    except Exception as e:
        logger.exception(f"❌ Error loading CSV: {path} — {str(e)}")
        raise


# -------------------------------------------------------------------
# 2. Save CSV
# -------------------------------------------------------------------
def save_csv(df: pd.DataFrame, path: str) -> None:
    """
    Save a pandas DataFrame to a CSV file with validation and logging.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): The destination file path.

    Raises:
        ValueError: If df is not a valid pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("❌ save_csv() expected a pandas DataFrame")
        raise ValueError("save_csv() expected a pandas DataFrame")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        df.to_csv(path, index=False)
        logger.info(f"✅ Saved CSV: {path} ({df.shape[0]} rows, {df.shape[1]} columns)")
    except Exception as e:
        logger.exception(f"❌ Error saving CSV to {path}: {str(e)}")
        raise



def save_pickle(obj: object, filepath: str) -> None:
    """
    Save a Python object to disk using pickle.

    Args:
        obj (object): The Python object to serialize (e.g., scalers, encoders, models).
        filepath (str): Path where the pickle file will be saved.

    Raises:
        OSError: If the file cannot be written due to permission or path errors.
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved successfully at: {path}")
    except Exception as e:
        logger.error(f"Failed to save pickle file at {filepath}: {e}")
        raise


def load_pickle(filepath: str) -> object:
    """
    Load a pickled Python object from disk.

    Args:
        filepath (str): Path to the pickle file.

    Returns:
        object: The loaded Python object.

    Raises:
        FileNotFoundError: If the pickle file does not exist.
        OSError: If reading or unpickling fails.
    """
    try:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Pickle file not found at {filepath}")

        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Object loaded successfully from: {path}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load pickle file at {filepath}: {e}")
        raise