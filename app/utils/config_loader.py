from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return its parsed content."""
    file_path = Path(config_path)
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path

    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_system_config() -> dict[str, Any]:
    """Load system-level configuration."""
    return load_yaml_config("config/system.yaml")


def load_classes_config() -> dict[str, Any]:
    """Load food class configuration."""
    return load_yaml_config("config/classes.yaml")


def load_fridge_regions_config() -> dict[str, Any]:
    """Load fridge region configuration."""
    return load_yaml_config("config/fridge_regions.yaml")
