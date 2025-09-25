import sys
import types

import pytest

from library.program.loader import ProgramLoaderError, load_program_components


def test_loads_program_components(monkeypatch):
    module_name = "tests.fake_program"
    module = types.ModuleType(module_name)

    def build_program():
        return "program"

    def load_dataset():
        return "dataset"

    def metric_fn():
        return 1

    module.build_program = build_program
    module.load_dataset = load_dataset
    module.metric_fn = metric_fn

    monkeypatch.setitem(sys.modules, module_name, module)

    bundle = load_program_components(
        module_path=module_name,
        factory_name="build_program",
        dataset_loader_name="load_dataset",
        metrics={"accuracy": "metric_fn"},
    )

    assert bundle.factory() == "program"
    assert bundle.dataset_loader() == "dataset"
    assert bundle.metrics["accuracy"]() == 1


def test_missing_attribute_raises(monkeypatch):
    module_name = "tests.missing_program"
    module = types.ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, module)

    with pytest.raises(ProgramLoaderError):
        load_program_components(module_name, "build_program", None, None)

