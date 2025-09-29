"""Budget enforcement for DSPyLab cost tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from library.config.models import BudgetConfig

from .tracker import CallCost, RunPhase, UsageEvent


class BudgetExceededError(RuntimeError):
    """Raised when an experiment budget is exceeded."""

    def __init__(self, message: str, *, phase: RunPhase, total_spend: float, budget_limit: float | None) -> None:
        super().__init__(message)
        self.phase = phase
        self.total_spend = total_spend
        self.budget_limit = budget_limit


@dataclass
class BudgetSnapshot:
    actual_spend: Dict[RunPhase, float] = field(default_factory=lambda: {RunPhase.TRAIN: 0.0, RunPhase.INFER: 0.0})
    warnings: List[str] = field(default_factory=list)


class BudgetManager:
    """Tracks experiment spend against configured budgets."""

    def __init__(self, budget: Optional[BudgetConfig]) -> None:
        self._budget = budget
        self._snapshot = BudgetSnapshot()
        self._warned_overall = False

    @property
    def snapshot(self) -> BudgetSnapshot:
        return self._snapshot

    def register_expected_call(self, phase: RunPhase, cost: CallCost) -> None:
        if self._budget is None:
            return
        projected_total = self._snapshot.actual_spend[phase] + cost.actual_usd
        self._check_limits(phase, projected_total, preflight=True)

    def record_event(self, event: UsageEvent) -> None:
        if self._budget is None:
            return
        phase = event.phase
        self._snapshot.actual_spend[phase] += event.cost.actual_usd
        self._check_limits(phase, self._snapshot.actual_spend[phase], preflight=False)

    def _check_limits(self, phase: RunPhase, actual_total: float, *, preflight: bool) -> None:
        if self._budget is None:
            return

        max_map = {
            RunPhase.TRAIN: self._budget.max_train_usd,
            RunPhase.INFER: self._budget.max_infer_usd,
        }
        limit = max_map.get(phase)
        if limit is None:
            limit = self._budget.max_usd

        if limit is not None and actual_total > limit + 1e-12:
            raise BudgetExceededError(
                f"{phase.value} budget of ${limit:0.2f} exceeded (spend ${actual_total:0.2f})",
                phase=phase,
                total_spend=actual_total,
                budget_limit=limit,
            )

        if not preflight:
            overall = sum(self._snapshot.actual_spend.values())
            warn_limit = self._budget.warn_usd
            if warn_limit is not None and overall >= warn_limit and not self._warned_overall:
                message = f"Overall spend ${overall:0.2f} crossed warning threshold ${warn_limit:0.2f}"
                self._snapshot.warnings.append(message)
                self._warned_overall = True

        if not preflight and self._budget.max_usd is not None:
            overall_total = sum(self._snapshot.actual_spend.values())
            if overall_total > self._budget.max_usd + 1e-12:
                raise BudgetExceededError(
                    f"Total budget of ${self._budget.max_usd:0.2f} exceeded (spend ${overall_total:0.2f})",
                    phase=phase,
                    total_spend=overall_total,
                    budget_limit=self._budget.max_usd,
                )
