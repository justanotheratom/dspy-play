from pathlib import Path

import pytest

from library.config.loader import (
    ExperimentConfigError,
    ExperimentDocument,
    dump_example_config,
    load_experiment_config,
)


def test_dump_and_load_example_config(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    dump_example_config(config_path)

    doc = load_experiment_config(config_path)

    assert isinstance(doc, ExperimentDocument)
    assert doc.path == config_path
    assert doc.data["experiment_name"] == "preference_validator_baseline"


def test_missing_config_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"
    with pytest.raises(ExperimentConfigError):
        load_experiment_config(missing_path)


def test_invalid_config_missing_required(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment_name: test\n", encoding="utf-8")

    with pytest.raises(ExperimentConfigError) as excinfo:
        load_experiment_config(config_path)

    assert "program" in str(excinfo.value)

