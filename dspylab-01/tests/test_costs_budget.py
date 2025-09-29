import pytest

from library.costs.budget import BudgetManager, BudgetExceededError
from library.costs.tracker import CallCost, RunPhase, UsageEvent, UsageWarning
from library.config.models import BudgetConfig


def _event(phase: RunPhase, actual: float) -> UsageEvent:
    return UsageEvent(
        phase=phase,
        prompt_tokens=0,
        cached_prompt_tokens=0,
        completion_tokens=0,
        cost=CallCost(actual_usd=actual, economic_usd=actual),
        cache_hit=False,
        provider_usage={},
    )


def test_budget_manager_records_and_warns():
    manager = BudgetManager(
        BudgetConfig(max_usd=10.0, warn_usd=6.0, max_train_usd=8.0, max_infer_usd=9.0)
    )

    manager.record_event(_event(RunPhase.TRAIN, 3.0))
    manager.record_event(_event(RunPhase.INFER, 2.0))

    assert manager.snapshot.actual_spend[RunPhase.TRAIN] == pytest.approx(3.0)
    assert manager.snapshot.actual_spend[RunPhase.INFER] == pytest.approx(2.0)
    assert manager.snapshot.warnings == []

    manager.record_event(_event(RunPhase.INFER, 4.0))
    assert any("warning threshold" in message for message in manager.snapshot.warnings)


def test_budget_manager_enforces_phase_limits():
    manager = BudgetManager(BudgetConfig(max_usd=20.0, max_train_usd=5.0))

    manager.record_event(_event(RunPhase.TRAIN, 4.0))
    with pytest.raises(BudgetExceededError):
        manager.record_event(_event(RunPhase.TRAIN, 2.0))


def test_budget_manager_enforces_total_limit():
    manager = BudgetManager(BudgetConfig(max_usd=5.0))

    manager.record_event(_event(RunPhase.TRAIN, 3.0))
    with pytest.raises(BudgetExceededError):
        manager.record_event(_event(RunPhase.INFER, 3.0))
