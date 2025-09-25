"""Utilities for loading and validating experiment configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import json

import yaml
from jsonschema import Draft7Validator, ValidationError

from .schema import EXPERIMENT_SCHEMA


@dataclass
class ExperimentDocument:
    """Raw experiment config data and metadata."""

    path: Path
    data: Dict[str, Any]


class ExperimentConfigError(Exception):
    """Raised when the experiment config fails validation."""


def load_experiment_config(path: Path) -> ExperimentDocument:
    """Load and validate an experiment YAML file.

    Args:
        path: Path to YAML file.

    Returns:
        ExperimentDocument containing parsed data.

    Raises:
        ExperimentConfigError: If file cannot be read or validation fails.
    """

    if not path.exists():
        raise ExperimentConfigError(f"Config file not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw_text) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ExperimentConfigError(f"Failed to load config {path}: {exc}") from exc

    validator = Draft7Validator(EXPERIMENT_SCHEMA)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if errors:
        message_lines = ["Experiment config validation failed:"]
        for err in errors:
            location = ".".join(str(p) for p in err.path) or "<root>"
            message_lines.append(f" - {location}: {err.message}")
        raise ExperimentConfigError("\n".join(message_lines))

    return ExperimentDocument(path=path, data=data)


def dump_example_config(path: Path) -> None:
    """Write an example config to path for documentation/testing."""

    example = {
        "experiment_name": "preference_validator_baseline",
        "program": {
            "module": "programs.preferencevalidator.01.program",
            "factory": "build_program",
            "dataset_loader": "load_dataset",
            "metrics": {
                "accuracy": "compute_accuracy"
            }
        },
        "models": [
            {
                "id": "gpt4o",
                "provider": "openai",
                "model": "gpt-4o",
                "api_key_env": "OPENAI_API_KEY",
                "temperature": 0.3,
                "max_output_tokens": 512,
                "pricing": {
                    "input_per_1k": 0.01,
                    "output_per_1k": 0.03
                }
            }
        ],
        "matrix": [
            {
                "name": "baseline",
                "model": "gpt4o",
                "optimizer": "bootstrap",
                "strategy": "cot"
            }
        ],
        "dataset": {
            "source": "local",
            "path": "data/dev_test.jsonl",
            "split": {"dev_ratio": 0.2, "seed": 42}
        },
        "outputs": {
            "root_dir": "programs/preferencevalidator/01/results"
        }
    }

    path.write_text(yaml.safe_dump(example, sort_keys=False), encoding="utf-8")

