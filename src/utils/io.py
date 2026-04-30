"""I/O helpers: YAML config loading, JSON result saving/loading.

All paths are expected to be absolute or relative to the repository root.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML as a Python dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save *data* as a pretty-printed JSON file.

    Args:
        data: JSON-serialisable object.
        path: Output file path (parent directories must exist).
        indent: JSON indentation level.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """Load a JSON file and return the parsed object.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON object.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create *path* as a directory if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        ``Path`` object for the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
