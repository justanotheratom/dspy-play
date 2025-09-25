import pytest

from library.config.loader import dump_example_config, load_experiment_config
from library.config.models import ExperimentConfig, expand_matrix


def test_expand_matrix_generates_runs(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    dump_example_config(config_path)
    doc = load_experiment_config(config_path)

    # add optimizer for testing matrix expansion
    doc.data.setdefault("optimizers", []).append({
        "id": "bootstrap",
        "type": "bootstrap_few_shot"
    })

    experiment = ExperimentConfig(**doc.data)
    runs = expand_matrix(experiment)

    assert len(runs) == 1
    run = runs[0]
    assert run.name == "baseline"
    assert run.model.id == "gpt4o"
    assert run.optimizer is not None


def test_expand_matrix_validates_unknown_references(tmp_path):
    doc_data = {
        "experiment_name": "invalid",
        "program": {"module": "x", "factory": "y"},
        "models": [{"id": "model1", "provider": "p", "model": "m", "api_key_env": "KEY"}],
        "matrix": [{"model": "missing"}],
        "dataset": {"source": "local", "path": "data.json"},
        "outputs": {"root_dir": "out"}
    }

    experiment = ExperimentConfig(**doc_data)
    with pytest.raises(ValueError):
        expand_matrix(experiment)

