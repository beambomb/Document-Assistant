"""Operational Latency & Token Cost Estimator for Enterprise RAG queries."""

import numpy as np
from typing import List, Dict, Any


class CostAndLatencyTracker:
    """Tracks latency percentiles and estimates token costs across LLM providers."""

    PRICING_PER_1M_TOKENS = {
        "openrouter_free": {"prompt": 0.00, "completion": 0.00},
        "gemini-1.5-flash": {"prompt": 0.075, "completion": 0.30},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
        "mock": {"prompt": 0.00, "completion": 0.00},
    }

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Heuristic token count estimation (~4 characters per token)."""
        return max(1, len(text) // 4)

    @staticmethod
    def compute_latency_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
        """Calculates p50 (median) and p95 latency."""
        if not latencies_ms:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
        arr = np.array(latencies_ms)
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "mean_ms": float(np.mean(arr)),
        }

    @classmethod
    def estimate_cost(
        cls,
        provider: str,
        total_prompt_tokens: int,
        total_completion_tokens: int,
        num_queries: int = 1,
    ) -> Dict[str, Any]:
        """Estimates USD cost per query and per 1,000 queries."""
        pricing = cls.PRICING_PER_1M_TOKENS.get(
            provider.lower(), cls.PRICING_PER_1M_TOKENS["openrouter_free"]
        )

        prompt_cost = (total_prompt_tokens / 1_000_000) * pricing["prompt"]
        comp_cost = (total_completion_tokens / 1_000_000) * pricing["completion"]
        total_cost = prompt_cost + comp_cost

        cost_per_query = total_cost / max(1, num_queries)
        cost_per_1k = cost_per_query * 1000

        return {
            "provider": provider,
            "total_cost_usd": round(total_cost, 6),
            "cost_per_query_usd": round(cost_per_query, 6),
            "cost_per_1k_queries_usd": round(cost_per_1k, 4),
            "cost_per_1k_queries_idr": round(cost_per_1k * 16200, 2),  # USD to IDR ~16,200
        }
