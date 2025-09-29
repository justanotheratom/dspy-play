import pytest

from datetime import datetime, timezone

from library.costs.tracker import UsageTracker, RunPhase
from library.pricing.catalog import PricingEntry, PriceRate


@pytest.fixture
def pricing_entry():
    return PricingEntry(
        provider="openai",
        model="gpt-4o",
        tier=None,
        fine_tuned=False,
        price_version=1,
        effective_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        expires_at=None,
        input=PriceRate(usd_per_1m=30.0),
        cached_input=PriceRate(usd_per_1m=3.0),
        output=PriceRate(usd_per_1m=60.0),
        training=None,
        notes=None,
        pricing_id="openai/gpt-4o",
        was_estimated=False,
    )


def test_usage_tracker_records_actual_and_economic_costs(pricing_entry):
    tracker = UsageTracker(pricing_entry)

    event = tracker.record_tokens(
        prompt_tokens=2000,
        cached_prompt_tokens=500,
        completion_tokens=1000,
        cache_hit=False,
        provider_usage={"prompt_tokens": 2000, "cached_prompt_tokens": 500, "completion_tokens": 1000},
    )

    assert pytest.approx(event.cost.actual_usd, rel=1e-6) == ((1500 / 1_000_000) * 30.0) + ((500 / 1_000_000) * 3.0) + ((1000 / 1_000_000) * 60.0)
    assert pytest.approx(event.cost.economic_usd, rel=1e-6) == ((2000 / 1_000_000) * 30.0) + ((1000 / 1_000_000) * 60.0)

    summary = tracker.summary()
    assert summary["phases"][RunPhase.INFER.value]["calls"] == 1
    assert summary["phases"][RunPhase.INFER.value]["prompt_tokens"] == 2000
    assert summary["actual_usd"] == pytest.approx(event.cost.actual_usd, rel=1e-6)


def test_usage_tracker_cache_hit_economic_cost(pricing_entry):
    tracker = UsageTracker(pricing_entry)

    event = tracker.record_tokens(
        prompt_tokens=4000,
        cached_prompt_tokens=0,
        completion_tokens=1000,
        cache_hit=True,
        provider_usage={"prompt_tokens": 4000, "completion_tokens": 1000},
    )

    assert event.cost.actual_usd == pytest.approx(0.0)
    assert event.cost.economic_usd == pytest.approx(((4000 / 1_000_000) * 30.0) + ((1000 / 1_000_000) * 60.0))


def test_usage_tracker_training_phase_ignores_cached_cost(pricing_entry):
    tracker = UsageTracker(pricing_entry)
    tracker.set_phase(RunPhase.TRAIN)

    event = tracker.record_tokens(
        prompt_tokens=3000,
        cached_prompt_tokens=2000,
        completion_tokens=1000,
        cache_hit=False,
        provider_usage={"prompt_tokens": 3000, "cached_prompt_tokens": 2000, "completion_tokens": 1000},
    )

    assert event.cost.actual_usd == pytest.approx(((1000 / 1_000_000) * 30.0) + ((1000 / 1_000_000) * 60.0))
    assert event.cost.economic_usd == pytest.approx(event.cost.actual_usd)


def test_usage_tracker_usage_missing_warning(pricing_entry):
    tracker = UsageTracker(pricing_entry)

    event = tracker.record_tokens(
        prompt_tokens=0,
        cached_prompt_tokens=0,
        completion_tokens=0,
        cache_hit=False,
        provider_usage=None,
        usage_missing=True,
    )

    assert len(event.warnings) == 1
    assert event.warnings[0].code == "usage-missing"
    assert "Provider did not supply usage details" in event.warnings[0].message


def test_usage_tracker_estimate_cost(pricing_entry):
    tracker = UsageTracker(pricing_entry)

    estimate = tracker.estimate_cost(
        prompt_tokens=5000,
        cached_prompt_tokens=1000,
        completion_tokens=2000,
        cache_hit=False,
        phase=RunPhase.INFER,
    )

    assert estimate.actual_usd == pytest.approx(((4000 / 1_000_000) * 30.0) + ((1000 / 1_000_000) * 3.0) + ((2000 / 1_000_000) * 60.0))
    assert estimate.economic_usd == pytest.approx(((5000 / 1_000_000) * 30.0) + ((2000 / 1_000_000) * 60.0))
