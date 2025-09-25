"""Timing-based metrics utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class TimedCompletion:
    """Represents a single model completion with timing data."""

    started_at: float
    first_token_at: float
    finished_at: float
    prompt_tokens: int
    completion_tokens: int

    @property
    def latency(self) -> float:
        return self.finished_at - self.started_at

    @property
    def time_to_first_token(self) -> float:
        return self.first_token_at - self.started_at

    @property
    def tokens_per_second(self) -> float:
        elapsed = self.finished_at - self.started_at
        total_tokens = self.prompt_tokens + self.completion_tokens
        return total_tokens / elapsed if elapsed > 0 else 0.0


class LatencyTracker:
    """Collects timing information for completions."""

    def __init__(self) -> None:
        self._captures: List[TimedCompletion] = []

    def capture(self, *, prompt_tokens: int, completion_tokens: int, first_token_delay: float, latency: float) -> None:
        started_at = time.perf_counter() - latency
        first_token_at = started_at + first_token_delay
        finished_at = started_at + latency
        self._captures.append(
            TimedCompletion(
                started_at=started_at,
                first_token_at=first_token_at,
                finished_at=finished_at,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
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
        return {"latency_avg": 0.0, "ttft_avg": 0.0, "tokens_per_second_avg": 0.0}

    latency_avg = sum(c.latency for c in completions_list) / len(completions_list)
    ttft_avg = sum(c.time_to_first_token for c in completions_list) / len(completions_list)
    tokens_per_second_avg = sum(c.tokens_per_second for c in completions_list) / len(completions_list)

    return {
        "latency_avg": latency_avg,
        "ttft_avg": ttft_avg,
        "tokens_per_second_avg": tokens_per_second_avg,
    }


