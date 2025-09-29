"""Integration helpers for DSPy + LiteLLM configuration."""

from __future__ import annotations

import logging
import os
import time
import types
from typing import Dict, Optional

import dspy  # type: ignore

from library.config.models import ModelConfig
from library.metrics import LatencyTracker
from library.costs import UsageTracker, BudgetManager, BudgetExceededError, RunPhase

LOGGER = logging.getLogger(__name__)


class DSpyUnavailableError(RuntimeError):
    """Raised when DSPy or LiteLLM is unavailable."""


def build_lm_instance(model_config: ModelConfig):
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

    return lm_instance, provider_settings


def configure_dspy_runtime(
    model_config: ModelConfig,
    *,
    latency_tracker: Optional[LatencyTracker] = None,
    usage_tracker: Optional[UsageTracker] = None,
    budget_manager: Optional[BudgetManager] = None,
) -> Dict[str, object]:
    lm_instance, provider_settings = build_lm_instance(model_config)

    if usage_tracker is not None:
        _install_cost_hooks(lm_instance, usage_tracker, budget_manager)

    if latency_tracker is not None:
        _install_latency_hooks(lm_instance, latency_tracker)

    dspy.configure(lm=lm_instance)
    configure_dspy_runtime._last_configured = lm_instance  # type: ignore[attr-defined]

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


def _install_cost_hooks(
    lm_instance,
    usage_tracker: UsageTracker,
    budget_manager: Optional[BudgetManager],
) -> None:
    original_forward = lm_instance.forward

    def instrumented_forward(self, *args, **kwargs):  # pragma: no cover - integration path
        if budget_manager is not None:
            _preflight_budget_check(self, usage_tracker, budget_manager, args, kwargs)
        result = original_forward(*args, **kwargs)
        _record_cost_metrics(usage_tracker, budget_manager, result)
        return result

    lm_instance.forward = types.MethodType(instrumented_forward, lm_instance)  # type: ignore[assignment]

    if hasattr(lm_instance, "__call__"):
        original_call = lm_instance.__call__

        def instrumented_call(self, *args, **kwargs):  # pragma: no cover - integration path
            if budget_manager is not None:
                _preflight_budget_check(self, usage_tracker, budget_manager, args, kwargs)
            output = original_call(*args, **kwargs)
            completion = getattr(self, "last_completion", None)
            response = completion or getattr(self, "last_response", None) or output
            _record_cost_metrics(usage_tracker, budget_manager, response)
            return output

        lm_instance.__call__ = types.MethodType(instrumented_call, lm_instance)  # type: ignore[assignment]


def _preflight_budget_check(lm_instance, usage_tracker: UsageTracker, budget_manager: BudgetManager, args, kwargs) -> None:
    prompt_estimate = _estimate_prompt_tokens(args, kwargs)
    completion_estimate = _estimate_completion_tokens(lm_instance, kwargs)

    if prompt_estimate == 0 and completion_estimate == 0:
        return

    cost_estimate = usage_tracker.estimate_cost(
        prompt_tokens=prompt_estimate,
        cached_prompt_tokens=0,
        completion_tokens=completion_estimate,
        cache_hit=False,
        phase=usage_tracker.phase,
    )
    budget_manager.register_expected_call(usage_tracker.phase, cost_estimate)


def _estimate_prompt_tokens(args, kwargs) -> int:
    messages = kwargs.get("messages")
    if isinstance(messages, list):
        total = 0
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    total += _approx_token_count(content)
        if total > 0:
            return total

    prompt = kwargs.get("prompt")
    if isinstance(prompt, str):
        return _approx_token_count(prompt)

    if args:
        first = args[0]
        if isinstance(first, str):
            return _approx_token_count(first)

    return 0


def _estimate_completion_tokens(lm_instance, kwargs) -> int:
    for key in ("max_output_tokens", "max_tokens"):
        value = kwargs.get(key)
        if isinstance(value, int) and value > 0:
            return value
    lm_value = getattr(lm_instance, "max_output_tokens", None)
    if isinstance(lm_value, int) and lm_value > 0:
        return lm_value
    return 0


def _approx_token_count(text: str) -> int:
    if not text:
        return 0
    return max(len(text) // 4, 1)


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


def _record_cost_metrics(
    usage_tracker: UsageTracker,
    budget_manager: Optional[BudgetManager],
    response: object,
) -> None:
    if response is None:
        return

    usage = getattr(response, "usage", None)
    provider_usage = usage
    prompt_tokens = _extract_usage_tokens(usage, "prompt_tokens")
    completion_tokens = _extract_usage_tokens(usage, "completion_tokens")
    cached_prompt_tokens = _extract_usage_tokens(usage, "cached_prompt_tokens")
    usage_missing = usage is None or (prompt_tokens == 0 and completion_tokens == 0)
    cache_hit = bool(getattr(response, "cache_hit", False))

    event = usage_tracker.record_tokens(
        prompt_tokens=prompt_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
        completion_tokens=completion_tokens,
        cache_hit=cache_hit,
        provider_usage=provider_usage,
        usage_missing=usage_missing,
    )

    if budget_manager is not None:
        budget_manager.record_event(event)


def _extract_usage_tokens(usage: object, key: str) -> int:
    if usage is None:
        return 0
    if key in ("prompt_tokens", "completion_tokens", "cached_prompt_tokens") and isinstance(usage, dict):
        value = usage.get(key)
        if value is None:
            maybe_callable = usage.get(key)
            if callable(maybe_callable):  # pragma: no cover - defensive
                try:
                    value = maybe_callable()
                except Exception:
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


__all__ = [
    "configure_dspy_runtime",
    "build_lm_instance",
    "DSpyUnavailableError",
]


