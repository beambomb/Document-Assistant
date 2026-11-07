"""Hybrid Retrieval combining BM25 lexical and Dense vector search with Reciprocal Rank Fusion (RRF)."""

from typing import List, Dict
from src.core.models import RetrievalCandidate, Chunk, UserContext
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_index import DenseVectorIndex
from config.settings import settings


class HybridRetriever:
    """Orchestrates BM25 and Dense search, combining results via RRF or weighted fusion."""

    def __init__(
        self,
        bm25_index: BM25Index,
        vector_index: DenseVectorIndex,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ):
        self.bm25_index = bm25_index
        self.vector_index = vector_index
        self.rrf_k = rrf_k or settings.rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def reciprocal_rank_fusion(
        self,
        bm25_candidates: List[RetrievalCandidate],
        dense_candidates: List[RetrievalCandidate],
        top_k: int = 10,
    ) -> List[RetrievalCandidate]:
        """Merges two ranked lists using standard Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[str, float] = {}
        candidate_map: Dict[str, RetrievalCandidate] = {}

        # Add scores from BM25 ranks
        for rank, cand in enumerate(bm25_candidates, start=1):
            cid = cand.chunk.chunk_id
            candidate_map[cid] = cand
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # Add scores from Dense ranks
        for rank, cand in enumerate(dense_candidates, start=1):
            cid = cand.chunk.chunk_id
            if cid not in candidate_map:
                candidate_map[cid] = cand
            else:
                # Merge dense score into existing candidate
                candidate_map[cid].dense_score = cand.dense_score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort descending by RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_cids = sorted_cids[:top_k]

        fused_results: List[RetrievalCandidate] = []
        for rank, cid in enumerate(top_cids, start=1):
            cand = candidate_map[cid]
            cand.hybrid_score = rrf_scores[cid]
            cand.rank = rank
            fused_results.append(cand)

        return fused_results

    def search(
        self, query: str, user: UserContext, top_k: int = 10
    ) -> List[RetrievalCandidate]:
        """Runs dual search and fuses candidate chunks."""
        bm25_results = self.bm25_index.search(
            query=query, user=user, top_k=settings.top_k_bm25
        )
        dense_results = self.vector_index.search(
            query=query, user=user, top_k=settings.top_k_dense
        )

        return self.reciprocal_rank_fusion(
            bm25_candidates=bm25_results,
            dense_candidates=dense_results,
            top_k=top_k,
        )
