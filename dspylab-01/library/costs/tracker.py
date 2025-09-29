"""Token usage and cost tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from library.pricing.catalog import PriceRate, PricingEntry

TOKENS_PER_MILLION = 1_000_000


class RunPhase(str, Enum):
    """Enumerates experiment phases for cost accounting."""

    TRAIN = "train"
    INFER = "infer"


@dataclass
class CallCost:
    """Represents the USD cost of a single call."""

    actual_usd: float
    economic_usd: float


@dataclass
class UsageWarning:
    """Structured warning emitted during usage tracking."""

    message: str
    code: str
    phase: RunPhase


@dataclass
class UsageEvent:
    """Single language model invocation cost details."""

    phase: RunPhase
    prompt_tokens: int
    cached_prompt_tokens: int
    completion_tokens: int
    cost: CallCost
    cache_hit: bool
    provider_usage: Any
    usage_missing: bool = False
    warnings: List[UsageWarning] = field(default_factory=list)

    @classmethod
    def from_tokens(
        cls,
        *,
        phase: RunPhase,
        prompt_tokens: int,
        cached_prompt_tokens: int,
        completion_tokens: int,
        pricing: PricingEntry,
        cache_hit: bool,
        provider_usage: Any,
        usage_missing: bool = False,
    ) -> "UsageEvent":
        prompt = max(int(prompt_tokens), 0)
        cached = min(max(int(cached_prompt_tokens), 0), prompt)
        completion = max(int(completion_tokens), 0)

        cost, warnings = _calculate_costs(
            phase=phase,
            prompt_tokens=prompt,
            cached_prompt_tokens=cached,
            completion_tokens=completion,
            pricing=pricing,
            cache_hit=cache_hit,
            usage_missing=usage_missing,
        )

        return cls(
            phase=phase,
            prompt_tokens=prompt,
            cached_prompt_tokens=cached,
            completion_tokens=completion,
            cost=cost,
            cache_hit=cache_hit,
            provider_usage=provider_usage,
            usage_missing=usage_missing,
            warnings=warnings,
        )


@dataclass
class _PhaseTotals:
    actual_usd: float = 0.0
    economic_usd: float = 0.0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cache_hits: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "actual_usd": self.actual_usd,
            "economic_usd": self.economic_usd,
            "prompt_tokens": self.prompt_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
        }


class UsageTracker:
    """Aggregates usage events into per-phase cost summaries."""

    def __init__(self, pricing: PricingEntry) -> None:
        self._pricing = pricing
        self._phase: RunPhase = RunPhase.INFER
        self._events: List[UsageEvent] = []
        self._totals: Dict[RunPhase, _PhaseTotals] = {
            RunPhase.TRAIN: _PhaseTotals(),
            RunPhase.INFER: _PhaseTotals(),
        }
        self._warnings: List[UsageWarning] = []

    @property
    def phase(self) -> RunPhase:
        return self._phase

    def set_phase(self, phase: RunPhase) -> None:
        self._phase = phase

    @property
    def pricing(self) -> PricingEntry:
        return self._pricing

    def record_event(self, event: UsageEvent) -> None:
        self._events.append(event)
        totals = self._totals[event.phase]
        totals.actual_usd += event.cost.actual_usd
        totals.economic_usd += event.cost.economic_usd
        totals.prompt_tokens += event.prompt_tokens
        totals.cached_prompt_tokens += event.cached_prompt_tokens
        totals.completion_tokens += event.completion_tokens
        totals.calls += 1
        if event.cache_hit:
            totals.cache_hits += 1
        self._warnings.extend(event.warnings)

    def record_tokens(
        self,
        *,
        prompt_tokens: int,
        cached_prompt_tokens: int,
        completion_tokens: int,
        cache_hit: bool,
        provider_usage: Any,
        usage_missing: bool = False,
    ) -> UsageEvent:
        event = UsageEvent.from_tokens(
            phase=self._phase,
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            completion_tokens=completion_tokens,
            pricing=self._pricing,
            cache_hit=cache_hit,
            provider_usage=provider_usage,
            usage_missing=usage_missing,
        )
        self.record_event(event)
        return event

    def estimate_cost(
        self,
        *,
        prompt_tokens: int,
        cached_prompt_tokens: int,
        completion_tokens: int,
        cache_hit: bool,
        phase: Optional[RunPhase] = None,
    ) -> CallCost:
        deltas, _warnings = _calculate_costs(
            phase=phase or self._phase,
            prompt_tokens=max(prompt_tokens, 0),
            cached_prompt_tokens=max(min(cached_prompt_tokens, prompt_tokens), 0),
            completion_tokens=max(completion_tokens, 0),
            pricing=self._pricing,
            cache_hit=cache_hit,
            usage_missing=False,
        )
        return deltas

    def totals(self, phase: RunPhase) -> Dict[str, object]:
        return dict(self._totals[phase].to_dict())

    def warnings(self) -> List[UsageWarning]:
        return list(self._warnings)

    def warning_messages(self) -> List[str]:
        return [warning.message for warning in self._warnings]

    def events(self) -> List[UsageEvent]:
        return list(self._events)

    def summary(self) -> Dict[str, object]:
        train_totals = self._totals[RunPhase.TRAIN].to_dict()
        infer_totals = self._totals[RunPhase.INFER].to_dict()
        return {
            "phases": {
                RunPhase.TRAIN.value: train_totals,
                RunPhase.INFER.value: infer_totals,
            },
            "actual_usd": train_totals["actual_usd"] + infer_totals["actual_usd"],
            "economic_usd": train_totals["economic_usd"] + infer_totals["economic_usd"],
            "events": len(self._events),
        }


def _calculate_costs(
    *,
    phase: RunPhase,
    prompt_tokens: int,
    cached_prompt_tokens: int,
    completion_tokens: int,
    pricing: PricingEntry,
    cache_hit: bool,
    usage_missing: bool,
) -> tuple[CallCost, List[UsageWarning]]:
    warnings: List[UsageWarning] = []
    input_rate = pricing.input
    cached_rate = pricing.cached_input or pricing.input
    output_rate = pricing.output

    if usage_missing:
        warnings.append(
            UsageWarning(
                message="Provider did not supply usage details; recorded tokens may be incomplete.",
                code="usage-missing",
                phase=phase,
            )
        )

    prompt_uncached = max(prompt_tokens - cached_prompt_tokens, 0)

    prompt_actual = _usd(prompt_uncached, input_rate)
    cached_actual = 0.0
    if cached_prompt_tokens:
        if phase == RunPhase.TRAIN:
            cached_actual = 0.0
        else:
            if pricing.cached_input is None:
                warnings.append(
                    UsageWarning(
                        message="Cached prompt tokens observed but no cached_input rate configured; using input rate.",
                        code="cached-rate-missing",
                        phase=phase,
                    )
                )
            cached_actual = _usd(cached_prompt_tokens, cached_rate)
    completion_actual = 0.0 if cache_hit else _usd(completion_tokens, output_rate)

    if cache_hit:
        prompt_actual = 0.0
        cached_actual = 0.0

    actual_usd = prompt_actual + cached_actual + completion_actual

    if cache_hit and phase == RunPhase.INFER:
        economic_prompt = _usd(prompt_tokens, input_rate)
        economic_completion = _usd(completion_tokens, output_rate)
    elif phase == RunPhase.INFER:
        economic_prompt = _usd(prompt_tokens, input_rate)
        economic_completion = _usd(completion_tokens, output_rate)
    else:
        economic_prompt = prompt_actual + cached_actual
        economic_completion = completion_actual

    economic_usd = economic_prompt + economic_completion

    return CallCost(actual_usd=actual_usd, economic_usd=economic_usd), warnings


def _usd(tokens: int, rate: PriceRate) -> float:
    return (tokens / TOKENS_PER_MILLION) * float(rate.usd_per_1m)
