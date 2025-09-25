"""Integration helpers for DSPy + LiteLLM configuration."""

from __future__ import annotations

import logging
import os
import time
import types
from typing import Dict, Optional

from library.config.models import ModelConfig
from library.metrics import LatencyTracker


LOGGER = logging.getLogger(__name__)


class DSpyUnavailableError(RuntimeError):
    """Raised when DSPy or LiteLLM is unavailable."""


def configure_dspy_runtime(
    model_config: ModelConfig,
    *,
    latency_tracker: Optional[LatencyTracker] = None,
) -> Dict[str, object]:
    try:
        import dspy  # type: ignore
    except ImportError as exc:
        raise DSpyUnavailableError("DSPy library is required but not installed") from exc

    provider_settings = _build_provider_settings(model_config)

    model_name = model_config.model
    if model_config.provider.lower() == "groq":
        provider_settings.setdefault("base_url", "https://api.groq.com/openai/v1")
        if not model_name.startswith("groq/"):
            model_name = f"groq/{model_name}"

    lm_class = getattr(dspy, "LM", None)
    if lm_class is None:
        raise DSpyUnavailableError("DSPy >=3.0.3 with dspy.LM is required")

    lm_kwargs = dict(provider_settings)
    lm_kwargs["cache"] = model_config.cache

    lm_instance = lm_class(model_name, **lm_kwargs)
    if latency_tracker is not None:
        _install_latency_hooks(lm_instance, latency_tracker)

    dspy.configure(lm=lm_instance)

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


def _extract_usage_tokens(usage: object, key: str) -> int:
    if usage is None:
        return 0
    if key in ("prompt_tokens", "completion_tokens") and isinstance(usage, dict):
        value = usage.get(key)
        if value is None:
            # LiteLLM Usage objects may store fields as methods; try calling if callable
            maybe_callable = usage.get(key)
            if callable(maybe_callable):  # pragma: no cover - defensive
                try:
                    value = maybe_callable()
                except Exception:  # pragma: no cover - best effort
                    value = None
        if value is not None:
            return int(value)
    if isinstance(usage, dict):
        return int(usage.get(key, 0) or 0)
    attr = getattr(usage, key, None)
    if callable(attr):  # pragma: no cover - defensive
        try:
            attr = attr()
        except Exception:
            attr = None
    return int(getattr(usage, key, 0) or 0)


def _extract_first_token_delay(result: object, fallback: float) -> float:
    meta = getattr(result, "response_metadata", None)
    if isinstance(meta, dict):
        value = meta.get("time_to_first_token") or meta.get("first_token_latency")
        if isinstance(value, (int, float)):
            return float(value)
    if isinstance(meta, list):
        for entry in meta:
            if isinstance(entry, dict):
                value = entry.get("time_to_first_token") or entry.get("first_token_latency")
                if isinstance(value, (int, float)):
                    return float(value)
    usage = getattr(result, "usage", None)
    if isinstance(usage, dict):
        value = usage.get("time_to_first_token") or usage.get("first_token_latency")
        if isinstance(value, (int, float)):
            return float(value)
    return fallback


def _extract_latency_from_result(result: object) -> float | None:
    latency_ms = getattr(result, "latency_ms", None)
    if isinstance(latency_ms, (int, float)):
        return float(latency_ms) / 1000.0

    response_ms = getattr(result, "response_ms", None)
    if isinstance(response_ms, (int, float)):
        return float(response_ms) / 1000.0

    usage = getattr(result, "usage", None)
    if usage is not None:
        for attr in ("total_time", "completion_time", "response_time"):
            if isinstance(usage, dict):
                value = usage.get(attr)
            else:
                value = getattr(usage, attr, None)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _install_latency_hooks(lm_instance, tracker: LatencyTracker) -> None:
    original_forward = lm_instance.forward

    def instrumented_forward(self, *args, **kwargs):  # pragma: no cover - integration path
        start = time.perf_counter()
        result = original_forward(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _record_latency_metrics(tracker, result, elapsed)
        return result

    lm_instance.forward = types.MethodType(instrumented_forward, lm_instance)  # type: ignore[assignment]

    if hasattr(lm_instance, "__call__"):
        original_call = lm_instance.__call__

        def instrumented_call(self, *args, **kwargs):  # pragma: no cover - integration path
            start = time.perf_counter()
            output = original_call(*args, **kwargs)
            elapsed = time.perf_counter() - start
            completion = getattr(self, "last_completion", None)
            response = completion or getattr(self, "last_response", None) or output
            _record_latency_metrics(tracker, response, elapsed)
            return output

        lm_instance.__call__ = types.MethodType(instrumented_call, lm_instance)  # type: ignore[assignment]


def _record_latency_metrics(tracker: LatencyTracker, response: object, elapsed: float) -> None:
    if response is None:
        return

    litellm_latency = _extract_latency_from_result(response)
    latency = litellm_latency if litellm_latency is not None else max(elapsed, 0.0)

    usage = getattr(response, "usage", None)
    prompt_tokens = _extract_usage_tokens(usage, "prompt_tokens")
    completion_tokens = _extract_usage_tokens(usage, "completion_tokens")
    first_token_delay = _extract_first_token_delay(response, latency)

    if litellm_latency is not None:
        first_token_delay = min(first_token_delay, latency)

    LOGGER.debug(
        "Latency capture: latency=%s (raw=%s elapsed=%s) prompt_tokens=%s completion_tokens=%s usage=%s",
        latency,
        litellm_latency,
        elapsed,
        prompt_tokens,
        completion_tokens,
        usage,
    )

    tracker.capture(latency=latency)


