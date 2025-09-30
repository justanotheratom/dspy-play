"""Experiment execution engine for DSPyLab."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence
import json
import inspect
from pathlib import Path

from library.config.models import ExperimentConfig, RunSpec
from library.program.loader import ProgramBundle

from library.metrics import LatencyTracker, compute_latency_metrics
from library.costs import UsageTracker, BudgetManager, BudgetExceededError, RunPhase
from library.pricing.manager import ensure_pricing_for_models

from .dspy_runtime import DSpyUnavailableError, configure_dspy_runtime, build_lm_instance


LOGGER = logging.getLogger(__name__)


@dataclass
class RunOutcome:
    """Result of a single experiment run."""

    name: str
    metrics: Dict[str, Any]
    records: List[Dict[str, Any]]
    optimizer_id: Optional[str] = None
    provider_settings: Dict[str, Any] = field(default_factory=dict)
    cost_summary: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    cost_events: List[Dict[str, Any]] = field(default_factory=list)


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
            latency_tracker = LatencyTracker()
            pricing_entries = ensure_pricing_for_models(models=[run.model], non_interactive=True)
            pricing_entry = pricing_entries[0]
            usage_tracker = UsageTracker(pricing_entry)
            budget_manager = BudgetManager(self._config.budget)

            provider_settings = configure_dspy_runtime(
                run.model,
                latency_tracker=latency_tracker,
                usage_tracker=usage_tracker,
                budget_manager=budget_manager,
            )
            program_instance = self._build_program_instance(run)

            if run.optimizer:
                usage_tracker.set_phase(RunPhase.TRAIN)
            try:
                compiled_program = self._maybe_optimize(program_instance, run, train_split)
            except BudgetExceededError as exc:
                outcomes.append(
                    self._build_budget_exceeded_outcome(
                        run,
                        records=[],
                        latency_tracker=latency_tracker,
                        usage_tracker=usage_tracker,
                        budget_manager=budget_manager,
                        provider_settings=provider_settings,
                        error=exc,
                    )
                )
                continue
            finally:
                usage_tracker.set_phase(RunPhase.INFER)

            instrumented_program = self._wrap_program_with_latency(compiled_program, latency_tracker)
            try:
                records = self._evaluate_program(instrumented_program, eval_examples)
            except BudgetExceededError as exc:
                outcomes.append(
                    self._build_budget_exceeded_outcome(
                        run,
                        records=[],
                        latency_tracker=latency_tracker,
                        usage_tracker=usage_tracker,
                        budget_manager=budget_manager,
                        provider_settings=provider_settings,
                        error=exc,
                    )
                )
                continue

            metrics = self._compute_metrics(run, records, eval_examples, latency_tracker)

            program_calls = len(records)
            cost_summary = usage_tracker.summary(program_calls=program_calls if program_calls else None)
            cost_summary["program_call_count"] = program_calls
            warnings = usage_tracker.warning_messages() + budget_manager.snapshot.warnings

            outcomes.append(
                RunOutcome(
                    name=run.name,
                    metrics=metrics,
                    records=records,
                    optimizer_id=run.optimizer.id if run.optimizer else None,
                    provider_settings=provider_settings,
                    cost_summary=cost_summary,
                    warnings=warnings,
                    cost_events=usage_tracker.event_records(),
                )
            )

        return outcomes

    def summarize_dataset(self) -> Dict[str, Any]:
        dataset = self._get_dataset_splits()
        train_split = list(dataset.get("train") or [])
        eval_split = dataset.get("dev") or dataset.get("test") or []
        eval_examples = list(eval_split)

        train_prompt_tokens_total = sum(_approx_prompt_tokens(example) for example in train_split)
        eval_prompt_tokens_total = sum(_approx_prompt_tokens(example) for example in eval_examples)

        return {
            "train_count": len(train_split),
            "eval_count": len(eval_examples),
            "train_prompt_tokens_total": train_prompt_tokens_total,
            "eval_prompt_tokens_total": eval_prompt_tokens_total,
        }

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

        LOGGER.debug("Instantiating program with kwargs: %s", factory_kwargs)
        return self._bundle.factory(**factory_kwargs)

    def _maybe_optimize(self, program_instance, run: RunSpec, train_split: Iterable[Any]):
        if not run.optimizer:
            LOGGER.info("Run %s has no optimizer; using raw program", run.name)
            return program_instance

        optimizer_config = run.optimizer
        optimizer_metric_names: Iterable[str] = getattr(optimizer_config, "metrics", []) or []
        optimizer_params = dict(optimizer_config.params or {})
        teacher_model_id = optimizer_params.pop("teacher_model", None)
        metrics_fns = []
        for metric_name in optimizer_metric_names:
            metric_fn = self._bundle.metrics.get(metric_name)
            if metric_fn is None:
                raise RuntimeError(
                    f"Optimizer '{optimizer_config.id}' requested unknown metric '{metric_name}'",
                )
            metrics_fns.append(metric_fn)

        optimizer = self._instantiate_optimizer(optimizer_config.type, optimizer_params, metrics_fns)

        teacher_lm = None
        if teacher_model_id:
            try:
                model_lookup = {model.id: model for model in self._config.models}
                if teacher_model_id not in model_lookup:
                    raise RuntimeError(
                        f"Optimizer '{optimizer_config.id}' requested unknown teacher model '{teacher_model_id}'",
                    )

                teacher_config = model_lookup[teacher_model_id]
                lm_instance, _ = build_lm_instance(teacher_config)
                teacher_lm = lm_instance
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize teacher model '{teacher_model_id}': {exc}") from exc

        prepared_trainset = []
        for example in train_split:
            if isinstance(example, dict) and "input" in example and "output" in example:
                prepared_trainset.append(example)
            else:
                raise RuntimeError("Train split examples must be dicts with 'input' and 'output'")

        if optimizer_config.type in {"bootstrap_few_shot", "bootstrap_few_shot_with_random_search"}:
            try:
                import dspy  # type: ignore

                extras = getattr(program_instance, "_dspylab_io", None)
                if not extras:
                    extras = self._config.program.extras or {}

                input_field = extras.get("input_field")
                output_field = extras.get("output_field")
                if not input_field or not output_field:
                    raise RuntimeError(
                        "BootstrapFewShot requires program extras to define 'input_field' and 'output_field'"
                    )

                prepared_trainset = [
                    dspy.Example(**{input_field: example["input"], output_field: example["output"]})
                    .with_inputs(input_field)
                    for example in prepared_trainset
                ]
            except Exception as exc:
                raise RuntimeError(f"Failed to prepare training data for BootstrapFewShot: {exc}") from exc

        if hasattr(optimizer, "compile"):
            LOGGER.info("Compiling program using optimizer '%s'", optimizer_config.id)
            compile_sig = inspect.signature(optimizer.compile)
            kwargs: Dict[str, Any] = {}
            accepts_metric_param = "metric" in compile_sig.parameters
            accepts_metrics_param = "metrics" in compile_sig.parameters
            if teacher_lm is not None:
                teacher_settings = getattr(optimizer, "teacher_settings", None)
                if teacher_settings is None:
                    teacher_settings = {}
                else:
                    teacher_settings = dict(teacher_settings)
                teacher_settings["lm"] = teacher_lm
                optimizer.teacher_settings = teacher_settings
            if "trainset" in compile_sig.parameters:
                kwargs["trainset"] = prepared_trainset
            if "student" in compile_sig.parameters:
                kwargs["student"] = program_instance
            if metrics_fns:
                if accepts_metric_param and accepts_metrics_param:
                    if len(metrics_fns) == 1:
                        kwargs["metric"] = metrics_fns[0]
                    else:
                        kwargs["metrics"] = metrics_fns
                elif accepts_metric_param:
                    if len(metrics_fns) != 1:
                        raise RuntimeError(
                            f"Optimizer '{optimizer_config.id}' accepts a single metric but {len(metrics_fns)} were provided",
                        )
                    kwargs["metric"] = metrics_fns[0]
                elif accepts_metrics_param:
                    kwargs["metrics"] = metrics_fns
                elif hasattr(optimizer, "metric") and len(metrics_fns) == 1:
                    optimizer.metric = metrics_fns[0]
                elif hasattr(optimizer, "metrics"):
                    optimizer.metrics = metrics_fns
                else:
                    raise RuntimeError(
                        f"Optimizer '{optimizer_config.id}' does not accept metrics but metrics were provided",
                    )
            return optimizer.compile(**kwargs)

        raise RuntimeError(f"Optimizer '{optimizer_config.type}' does not support compile()")

    def _instantiate_optimizer(
        self,
        optimizer_type: str,
        params: Dict[str, Any],
        metrics_fns: Iterable[Any],
    ):
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
                return self._instantiate_with_metrics(optimizer_cls, init_kwargs, metrics_fns)

        optimize_module = getattr(dspy, "optimize", None)
        if optimize_module and hasattr(optimize_module, optimizer_type):
            optimizer_cls = getattr(optimize_module, optimizer_type)
            return self._instantiate_with_metrics(optimizer_cls, init_kwargs, metrics_fns)

        if optimize_module:
            camel_name = "".join(part.capitalize() for part in optimizer_type.split("_"))
            if hasattr(optimize_module, camel_name):
                optimizer_cls = getattr(optimize_module, camel_name)
                return self._instantiate_with_metrics(optimizer_cls, init_kwargs, metrics_fns)

        raise RuntimeError(f"Optimizer type '{optimizer_type}' not found in dspy")

    def _instantiate_with_metrics(self, optimizer_cls, init_kwargs: Dict[str, Any], metrics_fns: Iterable[Any]):
        metrics_list = list(metrics_fns or [])

        init_sig = inspect.signature(optimizer_cls.__init__)
        accepts_metric = "metric" in init_sig.parameters
        accepts_metrics = "metrics" in init_sig.parameters

        if accepts_metric and accepts_metrics:
            if metrics_list:
                if len(metrics_list) == 1:
                    init_kwargs.setdefault("metric", metrics_list[0])
                else:
                    init_kwargs.setdefault("metrics", metrics_list)
        elif accepts_metric and metrics_list:
            if len(metrics_list) != 1:
                raise RuntimeError(
                    f"Optimizer '{optimizer_cls.__name__}' requires a single metric callable but {len(metrics_list)} were provided",
                )
            init_kwargs.setdefault("metric", metrics_list[0])
        elif accepts_metrics and metrics_list:
            init_kwargs.setdefault("metrics", metrics_list)

        return optimizer_cls(**init_kwargs)

    def _evaluate_program(
        self,
        program,
        eval_examples: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for example in eval_examples:
            input_value = example["input"] if isinstance(example, dict) and "input" in example else example
            result = program(input_value)
            records.append({"example": example, "output": result})
        return records

    def _compute_metrics(
        self,
        run: RunSpec,
        records: List[Dict[str, Any]],
        eval_examples: Sequence[Any],
        latency_tracker: LatencyTracker,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        metric_names = self._resolve_metric_names(run)

        for metric_name in metric_names:
            metric_fn = self._bundle.metrics.get(metric_name)
            if metric_fn is None:
                raise RuntimeError(f"Metric '{metric_name}' not provided by program module")

            LOGGER.debug("Calculating metric '%s'", metric_name)
            metrics[metric_name] = metric_fn(records=records, examples=list(eval_examples))

        metrics.update(compute_latency_metrics(latency_tracker.captures))

        return metrics

    def _resolve_metric_names(self, run: RunSpec) -> List[str]:
        if run.metrics:
            return run.metrics
        if self._config.program.metrics:
            return list(self._config.program.metrics.keys())
        return []

    def _wrap_program_with_latency(self, program, tracker: LatencyTracker):
        if program is None or not callable(program):
            return program

        class LatencyLoggingProgram:
            def __init__(self, inner):
                self._inner = inner

            def __call__(self, *args, **kwargs):
                start = time.perf_counter()
                try:
                    return self._inner(*args, **kwargs)
                finally:
                    elapsed = time.perf_counter() - start
                    tracker.capture(latency=elapsed)
                    LOGGER.debug("Program call latency: %.6f seconds", elapsed)

            def __getattr__(self, item):
                return getattr(self._inner, item)

        return LatencyLoggingProgram(program)

    def _build_budget_exceeded_outcome(
        self,
        run: RunSpec,
        *,
        records: List[Dict[str, Any]],
        latency_tracker: LatencyTracker,
        usage_tracker: UsageTracker,
        budget_manager: BudgetManager,
        provider_settings: Dict[str, Any],
        error: BudgetExceededError,
    ) -> RunOutcome:
        warnings = usage_tracker.warning_messages() + budget_manager.snapshot.warnings
        warnings.append(str(error))

        metrics = {
            "budget_exceeded": True,
            "budget_phase": error.phase.value,
            "budget_limit_usd": error.budget_limit,
            "budget_total_spend_usd": error.total_spend,
        }
        metrics.update(compute_latency_metrics(latency_tracker.captures))

        program_calls = len(records)
        cost_summary = usage_tracker.summary(program_calls=program_calls if program_calls else None)

        return RunOutcome(
            name=run.name,
            metrics=metrics,
            records=records,
            optimizer_id=run.optimizer.id if run.optimizer else None,
            provider_settings=provider_settings,
            cost_summary=cost_summary,
            warnings=warnings,
            cost_events=usage_tracker.event_records(),
        )


__all__ = ["ExperimentExecutor", "RunOutcome"]


