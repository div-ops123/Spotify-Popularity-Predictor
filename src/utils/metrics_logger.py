import json
import os
from src.utils.logger import Logger

logger = Logger().get_logger(__name__)

def save_metrics(model_name: str, metrics: dict, filepath: str) -> None:
    """
    Save model evaluation metrics into a single JSON file.
    Each model's results are stored separately to avoid overwriting.
    
    Args:
        model_name (str): Name of the model (e.g. "LinearRegression", "Lasso")
        metrics (dict): Dictionary of metrics (r2, mae, rmse, etc.)
        filepath (str): Path to metrics.json file
    """
    # ✅ Ensure the directory exists
    folder = os.path.dirname(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)

    # ✅ Load existing data if file exists
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            # file exists but is empty or corrupted
            data = {}
            logger.warning("⚠️ metrics.json was empty or corrupted. Recreating file.")
    else:
        data = {}

    # ✅ Add or update this model's metrics
    data[model_name] = metrics

    # ✅ Save back to file (formatted)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"✅ Metrics for {model_name} saved successfully to {os.path.abspath(filepath)}")


# TODO:
# read_display_metric() that reads from the metrics.json and shows a table comparing or visually compare.