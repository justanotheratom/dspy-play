import sys
import types

import pytest

from library.config.models import ModelConfig
from library.runtime.dspy_runtime import DSpyUnavailableError, configure_dspy_runtime


class DummyDSPy:
    def __init__(self):
        self.configured = None

    def configure(self, **kwargs):
        self.configured = kwargs


def test_configure_dspy_runtime(monkeypatch):
    dummy = DummyDSPy()
    dspy_module = types.SimpleNamespace(configure=dummy.configure)
    monkeypatch.setitem(sys.modules, "dspy", dspy_module)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    model = ModelConfig(id="m1", provider="openai", model="gpt-4o", api_key_env="OPENAI_API_KEY")

    settings = configure_dspy_runtime(model)
    assert settings["api_key"] == "sk-test"
    assert dummy.configured is not None


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "dspy", types.SimpleNamespace(configure=lambda **_: None))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    model = ModelConfig(id="m1", provider="openai", model="gpt-4o", api_key_env="OPENAI_API_KEY")

    with pytest.raises(DSpyUnavailableError):
        configure_dspy_runtime(model)


def test_missing_dspy_module(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

