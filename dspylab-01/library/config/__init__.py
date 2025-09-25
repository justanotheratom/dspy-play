"""Configuration handling for DSPyLab."""

from .schema import EXPERIMENT_SCHEMA, EXPERIMENT_SCHEMA_PATH
from .loader import load_experiment_config, ExperimentDocument, ExperimentConfigError

__all__ = [
    "EXPERIMENT_SCHEMA",
    "EXPERIMENT_SCHEMA_PATH",
    "load_experiment_config",
    "ExperimentConfigError",
    "ExperimentDocument",
]

