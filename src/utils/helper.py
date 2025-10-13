"""
helper.py
====================
Utility functions for common tasks across the project.
Currently includes configuration loading utilities.
"""

from pathlib import Path
import yaml


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
