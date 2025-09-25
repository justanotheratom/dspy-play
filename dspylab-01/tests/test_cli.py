from pathlib import Path

import pytest

from library.cli import run_cli
from library.config.loader import dump_example_config


def test_cli_requires_run_command(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    dump_example_config(config_path)

    exit_code = run_cli([str(config_path)])
    assert exit_code == 1


def test_cli_validates_config(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    dump_example_config(config_path)

    exit_code = run_cli(["run", str(config_path)])
    assert exit_code == 0


def test_cli_handles_validation_errors(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("experiment_name: test\n", encoding="utf-8")

    exit_code = run_cli(["run", str(config_path)])
    assert exit_code == 2

