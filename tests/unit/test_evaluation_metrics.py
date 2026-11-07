"""Unit tests for IR and Security evaluation metrics."""

from src.evaluation.metrics import (
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    check_acl_leakage,
)


def test_recall_at_k_calculation():
    retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
    relevant = ["doc_2", "doc_4"]

    assert recall_at_k(retrieved, relevant, k=1) == 0.0
    assert recall_at_k(retrieved, relevant, k=2) == 0.5  # doc_2 in top 2
    assert recall_at_k(retrieved, relevant, k=5) == 1.0  # both in top 5


def test_mrr_calculation():
    # First relevant at rank 2 -> RR = 0.5
    assert reciprocal_rank(["doc_x", "doc_rel", "doc_y"], ["doc_rel"]) == 0.5
    # First relevant at rank 1 -> RR = 1.0
    assert reciprocal_rank(["doc_rel", "doc_x"], ["doc_rel"]) == 1.0
    # None found -> RR = 0.0
    assert reciprocal_rank(["doc_x", "doc_y"], ["doc_rel"]) == 0.0


def test_ndcg_calculation():
    retrieved = ["doc_rel", "doc_unrelated"]
    relevant = ["doc_rel"]
    score = ndcg_at_k(retrieved, relevant, k=2)
    # Perfect ranking of 1 relevant doc at rank 1 has nDCG = 1.0
    assert round(score, 2) == 1.00


def test_ndcg_duplicate_chunks_bounded_between_0_and_1():
    # If multiple chunks of the same relevant document are retrieved, nDCG must never exceed 1.0
    retrieved = ["doc_rel", "doc_rel", "doc_rel", "doc_rel", "doc_rel"]
    relevant = ["doc_rel"]
    score = ndcg_at_k(retrieved, relevant, k=5)
    assert 0.0 <= score <= 1.0
    assert round(score, 4) == 1.0000


def test_acl_leakage_detection():
    retrieved = ["PUB-001", "SOP-IT-001", "KEB-SEC-005"]
    prohibited = ["KEB-SEC-005"]
    # Should detect leakage
    assert check_acl_leakage(retrieved, prohibited) is True

    clean_retrieved = ["PUB-001", "SOP-IT-001"]
    # Should have zero leakage
    assert check_acl_leakage(clean_retrieved, prohibited) is False
