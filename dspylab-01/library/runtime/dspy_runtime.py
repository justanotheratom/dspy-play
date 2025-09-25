"""Integration helpers for DSPy + LiteLLM configuration."""

from __future__ import annotations

import os
from typing import Dict, Optional

from library.config.models import ModelConfig


class DSpyUnavailableError(RuntimeError):
    """Raised when DSPy or LiteLLM is unavailable."""


def configure_dspy_runtime(model_config: ModelConfig) -> Dict[str, object]:
    try:
        import dspy  # type: ignore
    except ImportError as exc:
        raise DSpyUnavailableError("DSPy library is required but not installed") from exc

    provider_settings = _build_provider_settings(model_config)

    lm_class = getattr(dspy, "LM", None)
    if lm_class:
        lm_instance = lm_class(model_config.model, **provider_settings)
        dspy.configure(lm=lm_instance)
    else:
        dspy.configure(model=model_config.model, model_config=provider_settings)

    return provider_settings


def _build_provider_settings(model_config: ModelConfig) -> Dict[str, object]:
    api_key = os.environ.get(model_config.api_key_env)
    if not api_key:
        raise DSpyUnavailableError(
            f"Environment variable '{model_config.api_key_env}' not set for model '{model_config.id}'"
        )

    settings: Dict[str, object] = {
        "provider": model_config.provider,
        "api_key": api_key,
        "temperature": model_config.temperature,
    }

    if model_config.base_url:
        settings["base_url"] = model_config.base_url

    if model_config.max_output_tokens:
        settings["max_output_tokens"] = model_config.max_output_tokens

    if model_config.metadata:
        settings.update(model_config.metadata)

    return settings

