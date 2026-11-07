"""Integration test for EvaluationHarness execution."""

from src.evaluation.harness import EvaluationHarness


def test_evaluation_harness_runs_ablation():
    harness = EvaluationHarness(dataset_name="id_enterprise", use_mock_embeddings=True)
    results = harness.run_ablation()

    assert len(results) == 4
    modes = [r["mode"] for r in results]
    assert "BM25 Lexical Only" in modes
    assert "Dense Vector Only" in modes
    assert "Hybrid (BM25 + Dense RRF)" in modes
    assert "Hybrid + Cross-Reranker" in modes

    # Rigorous ACL guarantee: every mode must have 0.00% leakage
    for r in results:
        assert r["leakage_rate_pct"] == 0.00

    # Cross-reranker mode must achieve high trap rejection
    rerank_mode = next(r for r in results if "Reranker" in r["mode"])
    assert rerank_mode["trap_rejection_pct"] >= 80.0
