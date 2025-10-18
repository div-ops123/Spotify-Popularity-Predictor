import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Adds project root to sys.path

"""
Unit tests for src/utils/preprocessing.py

This module tests all key preprocessing functions:
- handle_missing_values
- handle_duplicate_values
- convert_dtypes
- feature_engineering
- select_feature_columns
- clean_dataset_pipeline

Edge cases are also covered, e.g., missing columns, empty DataFrames, invalid dates.

Run tests using:
    pytest -v
"""

import pytest
import pandas as pd
import numpy as np
from src.utils.preprocessing import (
    handle_missing_values,
    handle_duplicate_values,
    convert_dtypes,
    feature_engineering,
    select_feature_columns,
    clean_dataset_pipeline
)

# --------------------------------------------------------------------------
# 1️⃣ Test handle_missing_values
# --------------------------------------------------------------------------
def test_handle_missing_values_removes_na():
    """Rows with any NaN should be dropped"""
    df = pd.DataFrame({
        "col1": [1, 2, None],
        "col2": ["a", None, "c"]
    })
    df_clean = handle_missing_values(df)
    
    # Only first row should remain
    assert df_clean.shape[0] == 1
    assert df_clean.isna().sum().sum() == 0

# Edge case: empty DataFrame should return empty DataFrame
def test_handle_missing_values_empty_df():
    df = pd.DataFrame(columns=["a", "b"])
    df_clean = handle_missing_values(df)
    assert df_clean.empty

# --------------------------------------------------------------------------
# 2️⃣ Test handle_duplicate_values
# --------------------------------------------------------------------------
def test_handle_duplicate_values_removes_duplicates():
    """Duplicate rows should be removed"""
    df = pd.DataFrame({
        "col1": [1, 1, 2],
        "col2": ["a", "a", "b"]
    })
    df_clean = handle_duplicate_values(df)
    
    # One duplicate removed
    assert df_clean.shape[0] == 2

def test_handle_duplicate_values_no_duplicates():
    """DataFrame with no duplicates should remain unchanged"""
    df = pd.DataFrame({"col1": [1,2], "col2": ["a","b"]})
    df_clean = handle_duplicate_values(df)
    assert df_clean.shape[0] == 2

# --------------------------------------------------------------------------
# 3️⃣ Test convert_dtypes
# --------------------------------------------------------------------------
def test_convert_dtypes_creates_new_columns():
    """Should create release_year, release_month, and song_age columns"""
    df = pd.DataFrame({
        "track_album_release_date": ["2020-01-01", "2019-06-15"],
        "other_col": [1, 2]
    })
    df_clean = convert_dtypes(df)
    
    # Columns created
    for col in ["release_year", "release_month", "song_age"]:
        assert col in df_clean.columns

# Edge case: invalid dates should be dropped
def test_convert_dtypes_invalid_dates():
    df = pd.DataFrame({
        "track_album_release_date": ["invalid", "2020-01-01"],
        "other_col": [1,2]
    })
    df_clean = convert_dtypes(df)
    # Only valid date row should remain
    assert df_clean.shape[0] == 1

# --------------------------------------------------------------------------
# 4️⃣ Test feature_engineering
# --------------------------------------------------------------------------
def test_feature_engineering_creates_scaled_and_encoded(monkeypatch):
    """Checks that scaling and encoding are applied"""
    # Monkeypatch config to include categorical columns
    monkeypatch.setattr("src.utils.preprocessing.config", {"categorical_cols": ["key"]})

    df = pd.DataFrame({
        "tempo": [100, 120],
        "loudness": [-5, -6],
        "duration_ms": [200000, 250000],
        "song_age": [2, 3],
        "key": [0, 1],
        "time_signature": [4, 4]
    })

    df_out, transformers = feature_engineering(df)
    
    # Scaled columns should exist
    scaled_cols = ["tempo_scaled", "loudness_scaled", "duration_ms_log_scaled", "song_age_scaled"]
    for col in scaled_cols:
        assert col in df_out.columns

    # Encoder should be returned
    assert "encoder" in transformers

# Edge case: missing categorical column should raise ValueError
def test_feature_engineering_missing_categorical(monkeypatch):
    monkeypatch.setattr("src.utils.preprocessing.config", {"categorical_cols": ["nonexistent"]})
    df = pd.DataFrame({
        "tempo": [100, 120],
        "loudness": [-5, -6],
        "duration_ms": [200000, 250000],
        "song_age": [2, 3]
    })
    with pytest.raises(ValueError, match="Column nonexistent not found"):
        feature_engineering(df)

# --------------------------------------------------------------------------
# 5️⃣ Test select_feature_columns
# --------------------------------------------------------------------------
def test_select_feature_columns_raises_error_for_missing_column():
    """If a feature column is missing, raise ValueError"""
    df = pd.DataFrame({
        "a": [1,2],
        "b": [3,4]
    })
    with pytest.raises(ValueError, match="'c' not found in DataFrame!"):
        select_feature_columns(df, feature_cols=["a","c"], target_col="b")

# --------------------------------------------------------------------------
# 6️⃣ Test clean_dataset_pipeline (full pipeline)
# --------------------------------------------------------------------------
def test_clean_dataset_pipeline_runs_without_error(monkeypatch):
    """Full pipeline should run without crashing on minimal valid input"""
    df = pd.DataFrame({
        "tempo": [100, 120],
        "loudness": [-5, -6],
        "duration_ms": [200000, 250000],
        "song_age": [2, 3],
        "track_album_release_date": ["2020-01-01", "2019-06-15"],
        "key": [0, 1],
        "time_signature": [4, 4],
        "target": [0,1]
    })

    # Monkeypatch config for features and categorical columns
    monkeypatch.setattr("src.utils.preprocessing.config", {
        "features": ["tempo_scaled", "loudness_scaled", "duration_ms_log_scaled", "song_age_scaled"],
        "target": "target",
        "categorical_cols": ["key"]
    })

    df_clean, transformers, feature_cols = clean_dataset_pipeline(df)

    # Check shape and presence of transformer keys
    assert df_clean.shape[0] == df.shape[0]
    assert "encoder" in transformers
