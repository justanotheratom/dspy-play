"""Timing-based metrics utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class TimedCompletion:
    """Represents a single model completion with timing data."""

    started_at: float
    finished_at: float

    @property
    def latency(self) -> float:
        return self.finished_at - self.started_at


class LatencyTracker:
    """Collects timing information for completions."""

    def __init__(self) -> None:
        self._captures: List[TimedCompletion] = []

    def capture(self, *, latency: float) -> None:
        started_at = time.perf_counter() - latency
        finished_at = started_at + latency
        self._captures.append(
            TimedCompletion(
                started_at=started_at,
                finished_at=finished_at,
            )
        )

    def extend(self, completions: Iterable[TimedCompletion]) -> None:
        self._captures.extend(completions)

    @property
    def captures(self) -> List[TimedCompletion]:
        return list(self._captures)


def compute_latency_metrics(completions: Iterable[TimedCompletion]) -> Dict[str, float]:
    completions_list = list(completions)
    if not completions_list:
        return {"latency_avg": 0.0}

    latency_avg = sum(c.latency for c in completions_list) / len(completions_list)

    return {
        "latency_avg": latency_avg,
    }


