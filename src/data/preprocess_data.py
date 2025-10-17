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

from src.utils.logger import Logger
from src.utils.helper import load_csv, save_csv, get_config, save_pickle

from src.utils.preprocessing import clean_dataset_pipeline

config = get_config()
logger = Logger(debug=config.get("DEBUG_MODE")).get_logger(__name__)


# -----------------------------------------------------------------------------
# 1. Main Orchestration
# -----------------------------------------------------------------------------
def main():
    """
    Run the full preprocessing pipeline.

    Steps:
      1. Load raw dataset.
      2. Run cleaning pipeline.
      3. Save processed dataset and preprocessing artifacts.
    """  
    raw_path = config["data"]["processed"]["combined"]
    save_path = config['data']['processed']['clean']

    df_raw = load_csv(raw_path)
    logger.info("Calling `clean_dataset_pipeline`...")
    df_clean, transformers, feature_columns = clean_dataset_pipeline(df_raw)

    # Save cleaned dataser
    save_csv(df_clean, save_path)

    # Save scalers (StandardScaler + MinMaxScaler)
    scaler_bundle = {
        "scaler": transformers["scaler"],
        "minmax": transformers["minmax"]
    }
    save_pickle(scaler_bundle, config["artifacts"]["scalers"])

    # Save encoder
    save_pickle(transformers["encoder"], config["artifacts"]["encoders"])


    logger.info(f"Final dataset shape: {df_clean.shape}")
    logger.info(f"Feature columns used: {len(feature_columns)}")
    logger.info("Preprocessing completed and artifacts saved successfully.")



# -----------------------------------------------------------------------------
# 2. Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
