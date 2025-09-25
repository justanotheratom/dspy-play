"""Experiment execution engine for DSPyLab."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence
import json
import inspect
from pathlib import Path

from library.config.models import ExperimentConfig, RunSpec
from library.program.loader import ProgramBundle

from library.metrics import LatencyTracker, TimedCompletion, compute_latency_metrics

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
        config_path: Optional[Path] = None,
    ) -> None:
        self._config = config
        self._bundle = bundle
        self._runs = runs
        self._dataset_cache: Optional[Dict[str, Iterable[Any]]] = None
        self._latency_tracker = LatencyTracker()
        self._config_path = config_path or Path.cwd()

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

        dataset_config = self._config.dataset

        if self._bundle.dataset_loader is not None:
            loader = self._bundle.dataset_loader
            LOGGER.info("Loading dataset using %s", loader.__name__)

            try:
                splits = loader(dataset_config)
            except TypeError:
                try:
                    splits = loader(dataset_config.model_dump())  # type: ignore[arg-type]
                except Exception as exc:  # pragma: no cover - escalated below
                    raise RuntimeError(f"Dataset loader '{loader.__name__}' failed: {exc}") from exc
            except Exception as exc:  # pragma: no cover - escalated below
                raise RuntimeError(f"Dataset loader '{loader.__name__}' failed: {exc}") from exc
        else:
            splits = self._load_dataset_from_path(dataset_config)

        if not isinstance(splits, dict):
            raise RuntimeError("Dataset loader must return a mapping of split names to iterables")

        self._dataset_cache = {str(k): v for k, v in splits.items()}
        return self._dataset_cache

    def _load_dataset_from_path(self, dataset_config) -> Dict[str, List[Dict[str, Any]]]:
        path_value = getattr(dataset_config, "path", None)
        if isinstance(dataset_config, dict):
            path_value = dataset_config.get("path", path_value)

        if not path_value:
            raise RuntimeError("Dataset configuration must specify a 'path' when no loader is provided")

        dataset_path = Path(path_value)
        if not dataset_path.is_absolute():
            candidates = [self._config_path.parent / dataset_path, Path.cwd() / dataset_path]
            for candidate in candidates:
                if candidate.exists():
                    dataset_path = candidate.resolve()
                    break
            else:
                dataset_path = candidates[0].resolve()

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        examples: List[Dict[str, Any]] = []
        with dataset_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                examples.append({"input": record["input"], "output": record["output"]})

        if not examples:
            raise RuntimeError("Dataset file is empty; cannot run experiment")

        split_config = getattr(dataset_config, "split", None)
        if isinstance(dataset_config, dict):
            split_config = dataset_config.get("split", split_config)

        dev_ratio = 0.2
        if isinstance(split_config, dict):
            dev_ratio = split_config.get("dev_ratio", dev_ratio)

        dev_count = max(1, int(len(examples) * dev_ratio))
        dev_split = examples[:dev_count]
        train_split = examples[dev_count:] or dev_split

        return {"train": train_split, "dev": dev_split}

    def _build_program_instance(self, run: RunSpec):
        factory_kwargs: Dict[str, Any] = dict(self._config.program.extras or {})
        factory_kwargs.update(run.overrides or {})

        strategy_module = None
        if run.strategy:
            strategy_module = self._instantiate_strategy(run.strategy)
            factory_kwargs["chain_module"] = strategy_module

        LOGGER.debug("Instantiating program with kwargs: %s", factory_kwargs)
        return self._bundle.factory(**factory_kwargs)

    def _maybe_optimize(self, program_instance, run: RunSpec, train_split: Iterable[Any]):
        if not run.optimizer:
            LOGGER.info("Run %s has no optimizer; using raw program", run.name)
            return program_instance

        optimizer_config = run.optimizer
        metrics_fns = [self._bundle.metrics[name] for name in self._resolve_metric_names(run) if name in self._bundle.metrics]

        optimizer = self._instantiate_optimizer(optimizer_config.type, optimizer_config.params)

        prepared_trainset = []
        for example in train_split:
            if isinstance(example, dict) and "input" in example and "output" in example:
                prepared_trainset.append(example)
            else:
                raise RuntimeError("Train split examples must be dicts with 'input' and 'output'")

        if hasattr(optimizer, "compile"):
            LOGGER.info("Compiling program using optimizer '%s'", optimizer_config.id)
            compile_sig = inspect.signature(optimizer.compile)
            kwargs: Dict[str, Any] = {}
            if "trainset" in compile_sig.parameters:
                kwargs["trainset"] = prepared_trainset
            if "student" in compile_sig.parameters:
                kwargs["student"] = program_instance
            if metrics_fns:
                if "metric" in compile_sig.parameters and len(metrics_fns) == 1:
                    kwargs["metric"] = metrics_fns[0]
                elif "metrics" in compile_sig.parameters:
                    kwargs["metrics"] = metrics_fns
            return optimizer.compile(**kwargs)

        raise RuntimeError(f"Optimizer '{optimizer_config.type}' does not support compile()")

    def _instantiate_optimizer(self, optimizer_type: str, params: Dict[str, Any]):
        try:
            import dspy  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised in runtime tests
            raise DSpyUnavailableError("DSPy library is required for optimizer execution") from exc

        init_kwargs = dict(params or {})

        teleprompt_module = getattr(dspy, "teleprompt", None)
        if teleprompt_module:
            camel_name = "".join(part.capitalize() for part in optimizer_type.split("_"))
            if hasattr(teleprompt_module, camel_name):
                optimizer_cls = getattr(teleprompt_module, camel_name)
                return optimizer_cls(**init_kwargs)

        optimize_module = getattr(dspy, "optimize", None)
        if optimize_module and hasattr(optimize_module, optimizer_type):
            optimizer_cls = getattr(optimize_module, optimizer_type)
            return optimizer_cls(**init_kwargs)

        if optimize_module:
            camel_name = "".join(part.capitalize() for part in optimizer_type.split("_"))
            if hasattr(optimize_module, camel_name):
                optimizer_cls = getattr(optimize_module, camel_name)
                return optimizer_cls(**init_kwargs)

        raise RuntimeError(f"Optimizer type '{optimizer_type}' not found in dspy")

    def _evaluate_program(self, program, eval_examples: Sequence[Any]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for example in eval_examples:
            input_value = example["input"] if isinstance(example, dict) and "input" in example else example
            result = program(input_value)

            timing = result.get("timing") if isinstance(result, dict) else None
            if timing:
                completion = TimedCompletion(
                    started_at=timing.get("started_at", 0.0),
                    first_token_at=timing.get("first_token_at", timing.get("started_at", 0.0)),
                    finished_at=timing.get("finished_at", timing.get("started_at", 0.0)),
                    prompt_tokens=timing.get("prompt_tokens", 0),
                    completion_tokens=timing.get("completion_tokens", 0),
                )
                self._latency_tracker.extend([completion])

            records.append({"example": example, "output": result})
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

        metrics.update(compute_latency_metrics(self._latency_tracker.captures))

        return metrics

    def _resolve_metric_names(self, run: RunSpec) -> List[str]:
        if run.metrics:
            return run.metrics
        if self._config.program.metrics:
            return list(self._config.program.metrics.keys())
        return []

    def _instantiate_strategy(self, strategy_config) -> Any:
        try:
            import dspy  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised in runtime tests
            raise DSpyUnavailableError("DSPy library is required for strategies") from exc

        strategies_module = getattr(dspy, "strategies", None)
        params = strategy_config.params or {}

        if strategies_module and hasattr(strategies_module, strategy_config.type):
            strategy_cls = getattr(strategies_module, strategy_config.type)
            try:
                return strategy_cls(**params)
            except TypeError:
                return strategy_cls

        strategy_cls = getattr(dspy, strategy_config.type, None)
        if strategy_cls:
            try:
                return strategy_cls(**params)
            except TypeError:
                return strategy_cls

        def passthrough(module):  # type: ignore
            return module

        return passthrough


__all__ = ["ExperimentExecutor", "RunOutcome"]


