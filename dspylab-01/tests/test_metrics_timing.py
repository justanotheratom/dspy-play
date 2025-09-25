from library.metrics import LatencyTracker, TimedCompletion, compute_latency_metrics


def test_latency_tracker_capture_and_metrics():
    tracker = LatencyTracker()
    tracker.capture(prompt_tokens=10, completion_tokens=5, first_token_delay=0.1, latency=0.5)

    captures = tracker.captures
    assert len(captures) == 1
    completion = captures[0]
    assert completion.prompt_tokens == 10
    assert completion.completion_tokens == 5

    metrics = compute_latency_metrics(captures)
    assert metrics["ttft_avg"] > 0
    assert metrics["latency_avg"] > 0
    assert metrics["tokens_per_second_avg"] > 0


def test_compute_latency_metrics_empty():
    metrics = compute_latency_metrics([])
    assert metrics == {
        "latency_avg": 0.0,
        "ttft_avg": 0.0,
        "tokens_per_second_avg": 0.0,
    }


