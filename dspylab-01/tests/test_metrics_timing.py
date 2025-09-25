from library.metrics import LatencyTracker, TimedCompletion, compute_latency_metrics


def test_latency_tracker_capture_and_metrics():
    tracker = LatencyTracker()
    tracker.capture(latency=0.5)

    captures = tracker.captures
    assert len(captures) == 1

    metrics = compute_latency_metrics(captures)
    assert metrics["latency_avg"] > 0


def test_compute_latency_metrics_empty():
    metrics = compute_latency_metrics([])
    assert metrics == {"latency_avg": 0.0}


