"""Evaluation metrics for Information Retrieval (IR), Access Control Security, and Abstention."""

import math
from typing import List, Set


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """Computes Recall@k: fraction of ground truth relevant items retrieved in top-k."""
    if not relevant_ids:
        return 1.0  # Vacuously true if nothing was relevant (e.g. traps)

    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(top_k_retrieved.intersection(relevant_set))
    return hits / len(relevant_set)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Computes Reciprocal Rank (RR): 1 / rank of first relevant item found."""
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank k (nDCG@k)."""
    if not relevant_ids:
        return 1.0

    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    # Calculate Discounted Cumulative Gain (DCG)
    dcg = 0.0
    for i, item_id in enumerate(top_k, start=1):
        rel = 1.0 if item_id in relevant_set else 0.0
        dcg += rel / math.log2(i + 1)

    # Calculate Ideal Discounted Cumulative Gain (IDCG)
    idcg = 0.0
    num_ideal = min(k, len(relevant_set))
    for i in range(1, num_ideal + 1):
        idcg += 1.0 / math.log2(i + 1)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def check_acl_leakage(retrieved_ids: List[str], prohibited_ids: List[str]) -> bool:
    """Returns True if any prohibited document ID appears in retrieved list."""
    if not prohibited_ids:
        return False
    retrieved_set = set(retrieved_ids)
    prohibited_set = set(prohibited_ids)
    return bool(retrieved_set.intersection(prohibited_set))
