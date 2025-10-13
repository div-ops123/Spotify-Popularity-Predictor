"""
compute_vif.py
====================
Utilities to compute Variance Inflation Factor (VIF) for numeric features,
and to optionally perform iterative feature removal based on a VIF threshold.

Why VIF?
- VIF quantifies how much a feature is explained by the other features.
- High VIF indicates multicollinearity which can destabilize linear models.

Usage:
    from src.data.compute_vif import calculate_vif, iterative_vif_reduction

    df = pd.read_csv("data/processed/spotify_combined.csv")
    numeric = [col for col in df.select_dtypes(include="number").columns if col != "track_popularity"]
    vif_df = calculate_vif(df, numeric)
"""

from typing import List, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

# statsmodels provides the VIF utility
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
except ImportError as e:
    raise ImportError(
        "statsmodels is required to compute VIF. Install with `pip install statsmodels`."
    ) from e

from src.utils.helper import get_config
from src.utils.logger import Logger


logger = Logger().get_logger(__name__)


def _validate_inputs(df: pd.DataFrame, features: List[str]) -> None:
    """Simple validation for inputs."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not features:
        raise ValueError("features list is empty.")
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"The following features are not in the dataframe: {missing}")


def calculate_vif(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Calculate VIF for the provided features.

    Args:
        df: DataFrame containing the data.
        features: List of numeric feature names to compute VIF for.

    Returns:
        DataFrame with columns ['feature', 'vif'] sorted by descending VIF.
    """
    _validate_inputs(df, features)
    logger.info("Calculating VIF for features: %s", features)

    # Prepare matrix: drop rows with NaNs in the selected features
    X = df[features].dropna()
    if X.shape[0] == 0:
        raise ValueError("No rows remaining after dropping NaNs for the selected features.")

    # Add intercept (constant) for VIF calculation
    X_const = add_constant(X, has_constant='add')

    vif_values = []
    # compute VIF for each feature (skip the intercept column)
    for i in range(1, X_const.shape[1]):  # index 0 is 'const'
        feature_name = X_const.columns[i]
        try:
            vif = float(variance_inflation_factor(X_const.values, i))
        except np.linalg.LinAlgError as e:
            logger.error("Linear algebra error while computing VIF for %s: %s", feature_name, e)
            vif = np.inf
        vif_values.append({"feature": feature_name, "vif": vif})

    vif_df = pd.DataFrame(vif_values).sort_values("vif", ascending=False).reset_index(drop=True)
    logger.info("VIF calculation complete.")
    return vif_df


def iterative_vif_reduction(
    df: pd.DataFrame,
    features: List[str],
    threshold: float = 5.0,
    drop_highest: bool = True
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """Iteratively remove features with VIF >= threshold.

    Args:
        df: DataFrame with the data.
        features: Initial list of numeric features to test.
        threshold: VIF threshold above which features are removed (default 5.0).
        drop_highest: If True remove only the single highest VIF each iteration.
                      If False, remove all features above threshold in one step.

    Returns:
        kept_features: list of features left after reduction
        dropped_features: list of features removed
        final_vif_df: DataFrame of VIFs for kept features
    """
    _validate_inputs(df, features)
    features = features.copy()
    dropped = []

    while True:
        vif_df = calculate_vif(df, features)
        max_vif = vif_df["vif"].max()
        if np.isinf(max_vif):
            logger.warning("Infinite VIF encountered; stopping iterative removal.")
            break

        if max_vif < threshold:
            logger.info("All features have VIF < %s. Stopping.", threshold)
            break

        # identify features above threshold
        high_vif = vif_df[vif_df["vif"] >= threshold]

        if drop_highest:
            to_drop = [high_vif.iloc[0]["feature"]]
        else:
            to_drop = list(high_vif["feature"])

        logger.info("Dropping features due to high VIF: %s", to_drop)
        for f in to_drop:
            features.remove(f)
            dropped.append(f)

        # stop if we are down to 1 feature (can't compute VIF meaningfully)
        if len(features) <= 1:
            logger.info("One or fewer features remaining; stopping iterative removal.")
            break

    final_vif_df = calculate_vif(df, features) if len(features) > 0 else pd.DataFrame(columns=["feature", "vif"])
    return features, dropped, final_vif_df


# small helper for notebook usage
def compute_and_print_vif_from_config(exclude: List[str] = None) -> pd.DataFrame:
    """Load processed data path from config, compute VIF for numeric features and print result.

    Args:
        exclude: list of column names to exclude from VIF (like the target column).
    """
    exclude = exclude or []
    cfg = get_config()
    data_path = cfg["data"].get("processed", {}).get("combined")
    if data_path is None:
        raise KeyError("processed_path not found in config. Please set data.processed_path or nested key.")
    df = pd.read_csv(data_path)
    numeric = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    vif_df = calculate_vif(df, numeric)
    print(vif_df.to_string(index=False))
    return vif_df


# TODO:
# 1. Tests & edge cases (quick checklist)

# Write unit tests (pytest) for these cases:

# Empty DataFrame → expect ValueError.

# Non-existent features → expect ValueError.

# Single numeric feature → VIF cannot be computed meaningfully (function should either return empty DataFrame or handle gracefully).

# Perfect multicollinearity (X2 = 2*X1) → VIF becomes large or infinite; ensure code handles np.inf.