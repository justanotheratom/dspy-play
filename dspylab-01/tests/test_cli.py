import json
from pathlib import Path

import sys
import types

import pytest
import yaml

from library.cli import run_cli
from library.config.loader import dump_example_config
from unittest import mock


def _install_fake_dspy(monkeypatch: pytest.MonkeyPatch) -> None:
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

        def __call__(self, preference_request: str):
            return types.SimpleNamespace(normalized_action="report_success(\"demo\")")

    class DummyExample:
        def __init__(self, **kwargs):
            self.data = kwargs

        def with_inputs(self, *args, **kwargs):
            return self

    fake_dspy.Example = DummyExample
    fake_dspy.configure = configure
    fake_dspy.LM = DummyLM
    fake_dspy.ChainOfThought = DummyChainOfThought
    fake_dspy.LabeledFewShot = lambda k=None: types.SimpleNamespace(compile=lambda student, trainset: student)
    class DummyOptimizer:
        def __init__(self, **kwargs):
            self.metric = kwargs.get("metric")
            self.metrics = kwargs.get("metrics")

        def compile(self, student=None, trainset=None):
            return student

    fake_dspy.teleprompt = types.SimpleNamespace(
        LabeledFewShot=lambda k=None: DummyOptimizer(metric=None)
    )
    fake_dspy.optimize = types.SimpleNamespace(
        bootstrap_few_shot=lambda **kwargs: DummyOptimizer(**kwargs),
        gepa=lambda **kwargs: DummyOptimizer(**kwargs),
    )

    monkeypatch.setitem(sys.modules, "dspy", fake_dspy)


def _prepare_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "experiment.yaml"
    dump_example_config(config_path)

    data = yaml.safe_load(config_path.read_text())
    data["program"]["module"] = "tests.fake_program"
    data["program"]["dataset_loader"] = "load_dataset"
    data["program"].setdefault("extras", {"input_field": "input", "output_field": "output"})
    data["outputs"]["root_dir"] = str(tmp_path / "results")
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return config_path


def _install_fake_program(monkeypatch: pytest.MonkeyPatch) -> str:
    module_name = "tests.fake_program"
    module = types.ModuleType(module_name)

    def build_program(chain_module=None, **extras):
        def program(preference_request: str):
            return {"output": "report_success(\"demo\")"}

        return program

    def load_dataset(_config):
        return {
            "train": [{"input": "A", "output": "report_success(\"demo\")"}],
            "dev": [{"input": "B", "output": "report_success(\"demo\")"}],
        }

    def compute_accuracy(records, examples):
        return 1.0

    module.build_program = build_program
    module.load_dataset = load_dataset
    module.compute_accuracy = compute_accuracy

    monkeypatch.setitem(sys.modules, module_name, module)
    return module_name


def test_cli_validates_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _prepare_config(tmp_path)
    _install_fake_program(monkeypatch)

    _install_fake_dspy(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    exit_code = run_cli(["run", str(config_path), "--hypothesis", "baseline check"])
    assert exit_code == 0

    data = yaml.safe_load(config_path.read_text())
    results_root = Path(data["outputs"]["root_dir"])
    ledger_path = results_root / "ledger.jsonl"
    assert ledger_path.exists()

    lines = [line for line in ledger_path.read_text().splitlines() if line.strip()]
    assert lines, "ledger should contain at least one entry"

    last_entry = json.loads(lines[-1])
    run_id = last_entry["run_id"]
    run_dir = results_root / run_id
    assert run_dir.exists()

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["hypothesis"] == "baseline check"

    aggregated_raw = run_dir / "raw.jsonl"
    aggregated_summary = run_dir / "summary.json"

    assert aggregated_raw.exists()
    assert aggregated_summary.exists()

    subruns = manifest["sub_runs"]
    assert len(subruns) == 1
    subrun = subruns[0]
    subrun_raw = run_dir / subrun["paths"]["raw"]
    subrun_metrics = run_dir / subrun["paths"]["metrics"]
    assert subrun_raw.exists()
    assert subrun_metrics.exists()


@mock.patch("builtins.input", return_value="hypo")
def test_cli_handles_validation_errors(mock_input: mock.Mock, tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("experiment_name: test\n", encoding="utf-8")

    exit_code = run_cli(["run", str(config_path)])
    assert exit_code == 2

