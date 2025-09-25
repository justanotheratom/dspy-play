"""Experiment configuration schema for DSPyLab."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json


EXPERIMENT_SCHEMA_PATH: Path = Path(__file__).with_name("schema.json")


def _load_schema() -> Dict[str, Any]:
    with EXPERIMENT_SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


EXPERIMENT_SCHEMA: Dict[str, Any] = _load_schema()


__all__ = ["EXPERIMENT_SCHEMA", "EXPERIMENT_SCHEMA_PATH"]

