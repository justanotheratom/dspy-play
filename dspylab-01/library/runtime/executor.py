"""Experiment execution engine for DSPyLab."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from library.config.models import ExperimentConfig, RunSpec
from library.program.loader import ProgramBundle

from .dspy_runtime import DSpyUnavailableError, configure_dspy_runtime


LOGGER = logging.getLogger(__name__)


@dataclass
class RunOutcome:
    """Result of a single experiment run."""

    name: str
    metrics: Dict[str, Any]
    records: List[Dict[str, Any]]
    optimizer_id: Optional[str] = None
    strategy_id: Optional[str] = None
    provider_settings: Dict[str, Any] = field(default_factory=dict)


class ExperimentExecutor:
    """Coordinates execution of DSPy experiments across run matrix entries."""

    def __init__(
        self,
        config: ExperimentConfig,
        bundle: ProgramBundle,
        runs: List[RunSpec],
    ) -> None:
        self._config = config
        self._bundle = bundle
        self._runs = runs
        self._dataset_cache: Optional[Dict[str, Iterable[Any]]] = None

    def execute(self) -> List[RunOutcome]:
        LOGGER.info("Starting experiment '%s' with %d run(s)", self._config.experiment_name, len(self._runs))
        dataset = self._get_dataset_splits()
        train_split = list(dataset.get("train") or [])
        eval_split = dataset.get("dev") or dataset.get("test") or []
        eval_examples = list(eval_split)

        if not eval_examples:
            raise RuntimeError("Dataset loader must provide a 'dev' or 'test' split for evaluation")

        outcomes: List[RunOutcome] = []
        for run in self._runs:
            LOGGER.info("→ Run %s using model '%s'", run.name, run.model.id)
            provider_settings = configure_dspy_runtime(run.model)
            program_instance = self._build_program_instance(run)
            compiled_program = self._maybe_optimize(program_instance, run, train_split)
            records = self._evaluate_program(compiled_program, eval_examples)
            metrics = self._compute_metrics(run, records, eval_examples)

            outcomes.append(
                RunOutcome(
                    name=run.name,
                    metrics=metrics,
                    records=records,
                    optimizer_id=run.optimizer.id if run.optimizer else None,
                    strategy_id=run.strategy.id if run.strategy else None,
                    provider_settings=provider_settings,
                )
            )

        return outcomes

    def _get_dataset_splits(self) -> Dict[str, Iterable[Any]]:
        if self._dataset_cache is not None:
            return self._dataset_cache

        if self._bundle.dataset_loader is None:
            raise RuntimeError("Program must define a dataset_loader to run experiments")

        loader = self._bundle.dataset_loader
        LOGGER.info("Loading dataset using %s", loader.__name__)

        dataset_config = self._config.dataset

        try:
            splits = loader(dataset_config)
        except TypeError:
            # Retry with a plain dict if signature mismatch occurs.
            try:
                splits = loader(dataset_config.model_dump())  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover - escalated below
                raise RuntimeError(f"Dataset loader '{loader.__name__}' failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - escalated below
            raise RuntimeError(f"Dataset loader '{loader.__name__}' failed: {exc}") from exc

        if not isinstance(splits, dict):
            raise RuntimeError("Dataset loader must return a mapping of split names to iterables")

        self._dataset_cache = {str(k): v for k, v in splits.items()}
        return self._dataset_cache

    def _build_program_instance(self, run: RunSpec):
        factory_kwargs: Dict[str, Any] = dict(self._config.program.extras or {})
        factory_kwargs.update(run.overrides or {})

        if run.strategy and "strategy_config" not in factory_kwargs:
            factory_kwargs["strategy_config"] = run.strategy

        LOGGER.debug("Instantiating program with kwargs: %s", factory_kwargs)
        return self._bundle.factory(**factory_kwargs)

    def _maybe_optimize(self, program_instance, run: RunSpec, train_split: Iterable[Any]):
        if not run.optimizer:
            LOGGER.info("Run %s has no optimizer; using raw program", run.name)
            return program_instance

        optimizer_config = run.optimizer
        metrics_fns = [self._bundle.metrics[name] for name in self._resolve_metric_names(run) if name in self._bundle.metrics]

        optimizer = self._instantiate_optimizer(optimizer_config.type, optimizer_config.params, metrics_fns)

        if hasattr(optimizer, "compile"):
            LOGGER.info("Compiling program using optimizer '%s'", optimizer_config.id)
            return optimizer.compile(train_split, program_instance)

        raise RuntimeError(f"Optimizer '{optimizer_config.type}' does not support compile()")

    def _instantiate_optimizer(self, optimizer_type: str, params: Dict[str, Any], metrics_fns: List[Any]):
        try:
            import dspy  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised in runtime tests
            raise DSpyUnavailableError("DSPy library is required for optimizer execution") from exc

        optimize_module = getattr(dspy, "optimize", None)
        if optimize_module is None:
            raise RuntimeError("dspy.optimize module is unavailable")

        optimizer_cls = getattr(optimize_module, optimizer_type, None)
        if optimizer_cls is None:
            raise RuntimeError(f"Optimizer type '{optimizer_type}' not found in dspy.optimize")

        init_kwargs = dict(params or {})
        if metrics_fns and "metric" not in init_kwargs and "metrics" not in init_kwargs:
            init_kwargs["metrics" if len(metrics_fns) > 1 else "metric"] = metrics_fns if len(metrics_fns) > 1 else metrics_fns[0]

        LOGGER.debug("Instantiating optimizer '%s' with kwargs: %s", optimizer_type, init_kwargs)
        return optimizer_cls(**init_kwargs)

    def _evaluate_program(self, program, eval_examples: Sequence[Any]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for example in eval_examples:
            output = program(example)
            records.append({"example": example, "output": output})
        return records

    def _compute_metrics(self, run: RunSpec, records: List[Dict[str, Any]], eval_examples: Sequence[Any]) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        metric_names = self._resolve_metric_names(run)

        for metric_name in metric_names:
            metric_fn = self._bundle.metrics.get(metric_name)
            if metric_fn is None:
                raise RuntimeError(f"Metric '{metric_name}' not provided by program module")

            LOGGER.debug("Calculating metric '%s'", metric_name)
            metrics[metric_name] = metric_fn(records=records, examples=list(eval_examples))

        return metrics

    def _resolve_metric_names(self, run: RunSpec) -> List[str]:
        if run.metrics:
            return run.metrics
        if self._config.program.metrics:
            return list(self._config.program.metrics.keys())
        return []


__all__ = ["ExperimentExecutor", "RunOutcome"]


