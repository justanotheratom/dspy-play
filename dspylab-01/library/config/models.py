"""Pydantic models representing experiment configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class Pricing(BaseModel):
    input_per_1k: Optional[float] = Field(default=None, ge=0)
    output_per_1k: Optional[float] = Field(default=None, ge=0)


class ModelConfig(BaseModel):
    id: str
    provider: str
    model: str
    api_key_env: str
    base_url: Optional[str] = None
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_output_tokens: Optional[int] = Field(default=None, ge=1)
    pricing: Optional[Pricing] = None
    metadata: Dict[str, str] | None = None
    cache: bool = True


class OptimizerConfig(BaseModel):
    id: str
    type: str
    params: Dict[str, object] = Field(default_factory=dict)
    metrics: List[str] | None = None


class MatrixEntry(BaseModel):
    name: Optional[str] = None
    model: str
    optimizer: Optional[str] = None
    metrics: List[str] | None = None
    overrides: Dict[str, object] = Field(default_factory=dict)
    tags: List[str] | None = None


class DatasetConfig(BaseModel):
    source: str
    path: str
    split: Dict[str, object] = Field(default_factory=dict)


class ProgramConfig(BaseModel):
    module: str
    factory: str
    dataset_loader: Optional[str] = None
    metrics: Dict[str, str] | None = None
    extras: Dict[str, object] = Field(default_factory=dict)


class OutputConfig(BaseModel):
    root_dir: str
    raw_filename: str = "raw.jsonl"
    summary_filename: str = "summary.json"


class LoggingConfig(BaseModel):
    level: str = "info"
    verbose: bool = False


class ExperimentConfig(BaseModel):
    experiment_name: str
    description: Optional[str] = None
    tags: List[str] | None = None
    program: ProgramConfig
    models: List[ModelConfig]
    optimizers: List[OptimizerConfig] = Field(default_factory=list)
    matrix: List[MatrixEntry]
    dataset: DatasetConfig
    outputs: OutputConfig
    logging: LoggingConfig = LoggingConfig()
    environment: Dict[str, str] | None = None


@dataclass
class RunSpec:
    name: str
    model: ModelConfig
    optimizer: OptimizerConfig | None
    metrics: List[str]
    overrides: Dict[str, object]


def expand_matrix(config: ExperimentConfig) -> List[RunSpec]:
    model_map = {m.id: m for m in config.models}
    optimizer_map = {o.id: o for o in config.optimizers}

    runs: List[RunSpec] = []
    for idx, entry in enumerate(config.matrix):
        if entry.model not in model_map:
            raise ValueError(f"Matrix entry references unknown model '{entry.model}'")
        optimizer = None
        if entry.optimizer:
            if entry.optimizer not in optimizer_map:
                raise ValueError(f"Matrix entry references unknown optimizer '{entry.optimizer}'")
            optimizer = optimizer_map[entry.optimizer]

        name = entry.name or f"run_{idx+1}"

        metrics = entry.metrics or []
        if optimizer and optimizer.metrics:
            metrics = metrics or optimizer.metrics

        runs.append(
            RunSpec(
                name=name,
                model=model_map[entry.model],
                optimizer=optimizer,
                metrics=metrics,
                overrides=entry.overrides or {},
            )
        )

    return runs

