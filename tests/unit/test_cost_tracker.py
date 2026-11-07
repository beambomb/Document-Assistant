"""Unit tests for CostAndLatencyTracker."""

from src.evaluation.cost_tracker import CostAndLatencyTracker


def test_estimate_tokens():
    text = "Halo dunia, ini adalah pengujian token estimator."
    tokens = CostAndLatencyTracker.estimate_tokens(text)
    assert tokens > 5


def test_latency_percentiles():
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    percentiles = CostAndLatencyTracker.compute_latency_percentiles(latencies)

    assert percentiles["p50_ms"] == 55.0
    assert percentiles["p95_ms"] > 90.0
    assert percentiles["mean_ms"] == 55.0


def test_estimate_cost():
    # 10,000 prompt tokens and 2,000 completion tokens on gemini-1.5-flash
    cost_info = CostAndLatencyTracker.estimate_cost(
        provider="gemini-1.5-flash",
        total_prompt_tokens=100_000,
        total_completion_tokens=20_000,
        num_queries=100,
    )

    assert cost_info["total_cost_usd"] > 0.0
    assert cost_info["cost_per_1k_queries_usd"] > 0.0
    assert cost_info["cost_per_1k_queries_idr"] > 0.0
