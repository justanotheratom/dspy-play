"""Dynamic import helpers for DSPy programs."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Dict, Optional


class ProgramLoaderError(Exception):
    """Raised when program module loading fails."""


@dataclass
class ProgramBundle:
    module: ModuleType
    factory: Callable[..., Any]
    dataset_loader: Optional[Callable[..., Any]]
    metrics: Dict[str, Callable[..., Any]]


def _safe_getattr(module: ModuleType, name: str) -> Any:
    if not hasattr(module, name):
        raise ProgramLoaderError(f"Module '{module.__name__}' missing attribute '{name}'")
    return getattr(module, name)


def load_program_components(module_path: str, factory_name: str, dataset_loader_name: Optional[str], metrics: Dict[str, str] | None) -> ProgramBundle:
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ProgramLoaderError(f"Failed to import module '{module_path}': {exc}") from exc

    factory = _safe_getattr(module, factory_name)
    dataset_loader = _safe_getattr(module, dataset_loader_name) if dataset_loader_name else None

    resolved_metrics: Dict[str, Callable[..., Any]] = {}
    if metrics:
        for metric_name, attr_name in metrics.items():
            resolved_metrics[metric_name] = _safe_getattr(module, attr_name)

    return ProgramBundle(
        module=module,
        factory=factory,
        dataset_loader=dataset_loader,
        metrics=resolved_metrics,
    )

