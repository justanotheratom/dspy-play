import sys
import types
from typing import Any, Dict, Iterable, List

import pytest

from library.config.models import (
    DatasetConfig,
    ExperimentConfig,
    MatrixEntry,
    ModelConfig,
    OptimizerConfig,
    OutputConfig,
    ProgramConfig,
    RunSpec,
    StrategyConfig,
)
from library.program.loader import ProgramBundle
from library.runtime.executor import ExperimentExecutor


def _build_fake_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="demo",
        program=ProgramConfig(
            module="programs.preferencevalidator.01.program",
            factory="build_program",
            metrics={"accuracy": "compute_accuracy"},
        ),
        models=[
            ModelConfig(
                id="model-a",
                provider="openai",
                model="gpt-4o",
                api_key_env="OPENAI_API_KEY",
            )
        ],
        strategies=[
            StrategyConfig(id="cot", type="chain_of_thought", params={"enabled": True})
        ],
        optimizers=[
            OptimizerConfig(
                id="bootstrap",
                type="bootstrap_few_shot",
                params={"max_bootstraps": 1},
                metrics=["accuracy"],
            )
        ],
        matrix=[
            MatrixEntry(name="baseline", model="model-a", optimizer="bootstrap", strategy="cot")
        ],
        dataset=DatasetConfig(source="local", path="data/dev.jsonl", split={"dev_ratio": 0.2}),
        outputs=OutputConfig(root_dir="results/demo"),
    )


def _fake_dataset_loader(dataset_config: DatasetConfig) -> Dict[str, Iterable[Any]]:
    return {
        "train": [{"text": "train example"}],
        "dev": [
            {"text": "example1", "label": 1},
            {"text": "example2", "label": 0},
        ],
    }


class DummyProgram:
    def __init__(self):
        self.call_inputs: List[Dict[str, Any]] = []

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        self.call_inputs.append(example)
        return {"prediction": example.get("label", 0)}


def _fake_program_factory(**kwargs):
    return DummyProgram()


def _accuracy_metric(records: List[Dict[str, Any]], examples: Iterable[Dict[str, Any]]) -> float:
    example_list = list(examples)
    correct = 0
    for record, example in zip(records, example_list):
        if record["output"]["prediction"] == example.get("label"):
            correct += 1
    return correct / len(example_list)


class DummyOptimizer:
    def __init__(self, metric=None, metrics=None, **kwargs):
        self.metric = metric or metrics
        self.kwargs = kwargs

    def compile(self, train_split, program_instance):
        # pretend to adjust the program based on train data
        return program_instance


def _install_fake_dspy(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    fake = types.SimpleNamespace()

    def configure(**kwargs):
        fake.configure_kwargs = kwargs

    fake.configure = configure
    fake.optimize = types.SimpleNamespace(bootstrap_few_shot=DummyOptimizer)

    monkeypatch.setitem(sys.modules, "dspy", fake)
    return fake


def test_executor_runs_matrix_with_optimizer(monkeypatch):
    config = _build_fake_config()
    run_specs = [
        RunSpec(
            name="baseline",
            model=config.models[0],
            optimizer=config.optimizers[0],
            strategy=config.strategies[0],
            metrics=["accuracy"],
            overrides={},
        )
    ]

    bundle = ProgramBundle(
        module=types.SimpleNamespace(),
        factory=_fake_program_factory,
        dataset_loader=_fake_dataset_loader,
        metrics={"accuracy": _accuracy_metric},
    )

    fake_dspy = _install_fake_dspy(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    executor = ExperimentExecutor(config, bundle, run_specs)
    outcomes = executor.execute()

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.name == "baseline"
    assert pytest.approx(outcome.metrics["accuracy"], 0.01) == 1.0
    assert fake_dspy.configure_kwargs["model_config"]["provider"] == "openai"


def test_executor_requires_dataset_split(monkeypatch):
    config = _build_fake_config()
    run_specs = [
        RunSpec(
            name="baseline",
            model=config.models[0],
            optimizer=None,
            strategy=None,
            metrics=[],
            overrides={},
        )
    ]

    def empty_loader(dataset_config: DatasetConfig):
        return {"train": []}

    bundle = ProgramBundle(
        module=types.SimpleNamespace(),
        factory=_fake_program_factory,
        dataset_loader=empty_loader,
        metrics={},
    )

    _install_fake_dspy(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    executor = ExperimentExecutor(config, bundle, run_specs)

    with pytest.raises(RuntimeError, match="dev|test"):
        executor.execute()


