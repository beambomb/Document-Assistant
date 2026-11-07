"""Unit tests for Reciprocal Rank Fusion (RRF)."""

from src.retrieval.hybrid import HybridRetriever
from src.core.models import Chunk, RetrievalCandidate


def make_cand(chunk_id: str, title: str, rank: int, score: float = 1.0) -> RetrievalCandidate:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("_")[0],
        title=title,
        text=f"Content for {chunk_id}",
        access_roles=["employee"],
    )
    return RetrievalCandidate(chunk=chunk, lexical_score=score, dense_score=score, rank=rank)


def test_rrf_fusion_combines_ranks():
    # retriever with dummy indexes for unit test of mathematical fusion
    retriever = HybridRetriever(bm25_index=None, vector_index=None, rrf_k=60)

    # BM25 found: A (rank 1), B (rank 2)
    bm25_list = [make_cand("A_1", "Doc A", 1, 5.0), make_cand("B_1", "Doc B", 2, 3.0)]
    # Dense found: B (rank 1), C (rank 2)
    dense_list = [make_cand("B_1", "Doc B", 1, 0.9), make_cand("C_1", "Doc C", 2, 0.8)]

    fused = retriever.reciprocal_rank_fusion(bm25_list, dense_list, top_k=3)

    assert len(fused) == 3
    # B was ranked high by both engines (rank 2 in BM25, rank 1 in Dense)
    # RRF score for B: 1/(60+2) + 1/(60+1) = 0.016129 + 0.016393 = 0.03252
    # RRF score for A: 1/(60+1) = 0.016393
    # Therefore, B MUST be #1 overall
    assert fused[0].chunk.chunk_id == "B_1"
    assert fused[0].rank == 1
    assert fused[1].chunk.chunk_id == "A_1"
    assert fused[2].chunk.chunk_id == "C_1"
