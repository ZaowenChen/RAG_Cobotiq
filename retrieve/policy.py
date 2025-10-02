"""Retrieval policy loader and helpers."""

from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_POLICY_PATH = Path("configs/policy.yaml")


def load_policy(path: Path | None = None) -> Dict[str, Any]:
    """Load the retrieval policy YAML as a dictionary."""
    target = path or DEFAULT_POLICY_PATH
    with target.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
