"""
Unit tests for src.utils.helper.py

Covers:
- get_config()
- load_csv()
- save_csv()

These tests use pytest fixtures like `tmp_path` to create temporary
files and directories safely during tests (no real file I/O).
"""

import os
import yaml
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from src.utils.helper import get_config, load_csv, save_csv


# -----------------------------------------------------------------------------
# 🧪 TEST: get_config()
# -----------------------------------------------------------------------------
def test_get_config_success(tmp_path, monkeypatch):
    """Should load a valid YAML config file successfully."""
    # Arrange
    config_content = {"data": {"path": "data/raw"}}
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    config_file = configs_dir / "config.yaml"

    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    # Mock project root to tmp_path (simulate structure)
    fake_helper_path = tmp_path / "src" / "utils" / "helper.py"
    fake_helper_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("src.utils.helper.Path.resolve", lambda x: fake_helper_path)

    # Act
    result = get_config()

    # Assert
    assert result == config_content


def test_get_config_missing_file(monkeypatch):
    """Should raise FileNotFoundError if config.yaml is missing."""
    fake_helper_path = Path("/fake/path/src/utils/helper.py")
    monkeypatch.setattr("src.utils.helper.Path.resolve", lambda x: fake_helper_path)

    with pytest.raises(FileNotFoundError):
        get_config()


def test_get_config_invalid_yaml(tmp_path, monkeypatch):
    """Should raise yaml.YAMLError for invalid YAML syntax."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    config_file = configs_dir / "config.yaml"
    config_file.write_text("invalid_yaml: [this is broken", encoding="utf-8")

    fake_helper_path = tmp_path / "src" / "utils" / "helper.py"
    fake_helper_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("src.utils.helper.Path.resolve", lambda x: fake_helper_path)

    with pytest.raises(yaml.YAMLError):
        get_config()


# -----------------------------------------------------------------------------
# 🧪 TEST: load_csv()
# -----------------------------------------------------------------------------
def test_load_csv_success(tmp_path):
    """Should successfully load a valid CSV file."""
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    file_path = tmp_path / "test.csv"
    data.to_csv(file_path, index=False)

    result = load_csv(str(file_path))
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)


def test_load_csv_missing_file():
    """Should raise FileNotFoundError when file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_csv("non_existent.csv")


def test_load_csv_empty_file(tmp_path):
    """Should raise pd.errors.EmptyDataError when CSV is empty."""
    file_path = tmp_path / "empty.csv"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(pd.errors.EmptyDataError):
        load_csv(str(file_path))


# -----------------------------------------------------------------------------
# 🧪 TEST: save_csv()
# -----------------------------------------------------------------------------
def test_save_csv_success(tmp_path):
    """Should successfully save a DataFrame to CSV."""
    df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
    save_path = tmp_path / "output.csv"

    save_csv(df, str(save_path))
    assert save_path.exists()

    loaded = pd.read_csv(save_path)
    assert loaded.equals(df)


def test_save_csv_invalid_input(tmp_path):
    """Should raise ValueError if input is not a DataFrame."""
    invalid_data = {"x": [1, 2, 3]}
    save_path = tmp_path / "invalid.csv"

    with pytest.raises(ValueError):
        save_csv(invalid_data, str(save_path))
