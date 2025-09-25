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
)
from library.metrics import TimedCompletion
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
        optimizers=[
            OptimizerConfig(
                id="bootstrap",
                type="bootstrap_few_shot",
                params={"max_bootstraps": 1},
                metrics=["accuracy"],
            )
        ],
        matrix=[
            MatrixEntry(name="baseline", model="model-a", optimizer="bootstrap")
        ],
        dataset=DatasetConfig(source="local", path="data/dev.jsonl", split={"dev_ratio": 0.2}),
        outputs=OutputConfig(root_dir="results/demo"),
    )


def _fake_dataset_loader(_config):
    return {
        "train": [{"input": "A", "output": "report_success(\"demo\")"}],
        "dev": [{"input": "B", "output": "report_success(\"demo\")"}],
    }


class DummyProgram:
    def __init__(self):
        self.call_inputs: List[Dict[str, Any]] = []

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(example, dict):
            input_value = example.get("input")
        else:
            input_value = example
        completion = TimedCompletion(
            started_at=0.0,
            first_token_at=0.05,
            finished_at=0.3,
            prompt_tokens=10,
            completion_tokens=5,
        )
        return {
            "output": {
                "prediction": input_value
            },
            "timing": {
                "started_at": completion.started_at,
                "first_token_at": completion.first_token_at,
                "finished_at": completion.finished_at,
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
            },
        }


def _fake_program_factory(**kwargs):
    return DummyProgram()


def _accuracy_metric(records: List[Dict[str, Any]], examples: Iterable[Dict[str, Any]]) -> float:
    return 1.0


class DummyOptimizer:
    def __init__(self, metric=None, metrics=None, **kwargs):
        self.metric = metric or metrics
        self.kwargs = kwargs

    def compile(self, train_split, program_instance):
        # pretend to adjust the program based on train data
        return program_instance


def _install_fake_dspy(monkeypatch):
    fake_dspy = types.SimpleNamespace()

    def configure(**kwargs):
        fake_dspy.configured = kwargs

    class DummyLM:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

    class DummyChainOfThought:
        def __init__(self, signature):
            self.signature = signature

        def __call__(self, preference_request):
            return types.SimpleNamespace(normalized_action="report_success(\"demo\")")

    fake_dspy.configure = configure
    fake_dspy.LM = DummyLM
    fake_dspy.ChainOfThought = DummyChainOfThought
    fake_dspy.teleprompt = types.SimpleNamespace(LabeledFewShot=lambda k=None: types.SimpleNamespace(compile=lambda student, trainset: student))
    fake_dspy.optimize = types.SimpleNamespace(bootstrap_few_shot=lambda **kwargs: types.SimpleNamespace(compile=lambda student, trainset: student))

    monkeypatch.setitem(sys.modules, "dspy", fake_dspy)
    return fake_dspy


def test_executor_runs_matrix_with_optimizer(monkeypatch):
    config = _build_fake_config()
    run_specs = [
        RunSpec(
            name="baseline",
            model=config.models[0],
            optimizer=config.optimizers[0],
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
    assert fake_dspy.configured["lm"].model == "gpt-4o"


def test_executor_requires_dataset_split(monkeypatch):
    config = _build_fake_config()
    run_specs = [
        RunSpec(
            name="baseline",
            model=config.models[0],
            optimizer=None,
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

    fake_dspy = _install_fake_dspy(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    executor = ExperimentExecutor(config, bundle, run_specs)

    with pytest.raises(RuntimeError, match="dev|test"):
        executor.execute()


