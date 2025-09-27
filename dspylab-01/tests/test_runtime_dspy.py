import sys
import types

import pytest

from library.config.models import ModelConfig
from library.metrics import LatencyTracker
from library.runtime.dspy_runtime import DSpyUnavailableError, configure_dspy_runtime


class DummyDSPy:
    def __init__(self):
        self.configured = None

    def configure(self, **kwargs):
        self.configured = kwargs


def test_configure_dspy_runtime(monkeypatch):
    dummy = DummyDSPy()
    class DummyLM:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

        def forward(self, *args, **kwargs):  # pragma: no cover - simple stub
            return types.SimpleNamespace(
                usage={
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_time": 0.05,
                },
                response_metadata={"time_to_first_token": 0.01},
            )

    dspy_module = types.SimpleNamespace(configure=dummy.configure, LM=DummyLM)
    monkeypatch.setitem(sys.modules, "dspy", dspy_module)
    monkeypatch.setattr("library.runtime.dspy_runtime.dspy", dspy_module)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    model = ModelConfig(id="m1", provider="openai", model="gpt-4o", api_key_env="OPENAI_API_KEY")

    tracker = LatencyTracker()
    settings = configure_dspy_runtime(model, latency_tracker=tracker)
    assert settings["api_key"] == "sk-test"
    assert getattr(configure_dspy_runtime, "_last_configured", None) is not None
    lm_instance = configure_dspy_runtime._last_configured
    assert isinstance(lm_instance, DummyLM)

    lm_instance.forward(prompt="demo")
    captures = tracker.captures
    assert len(captures) == 1
    captured = captures[0]
    assert captured.latency == pytest.approx(0.05, rel=1e-3)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "dspy", types.SimpleNamespace(configure=lambda **_: None, LM=object()))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    model = ModelConfig(id="m1", provider="openai", model="gpt-4o", api_key_env="OPENAI_API_KEY")

    with pytest.raises(DSpyUnavailableError):
        configure_dspy_runtime(model, latency_tracker=LatencyTracker())


def test_missing_dspy_module(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

