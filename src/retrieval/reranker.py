"""Cross-Encoder Reranker for fine-grained semantic relevance scoring."""

import math
from typing import List, Optional
from src.core.models import RetrievalCandidate
from config.settings import settings


def sigmoid(x: float) -> float:
    """Standard sigmoid function to bound logits between 0.0 and 1.0."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class CrossEncoderReranker:
    """Reranks candidate chunks using a deep cross-encoder model."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_mock: bool = False,
    ):
        self.model_name = model_name or settings.reranker_model_name
        self.use_mock = use_mock
        self._model = None

    def _get_model(self):
        """Lazy loads sentence-transformers CrossEncoder."""
        if self._model is None and not self.use_mock:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception:
                self.use_mock = True
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_k: int = 3,
    ) -> List[RetrievalCandidate]:
        """Scores (query, chunk_text) pairs and reranks candidates."""
        if not candidates:
            return []

        pairs = [
            (query, f"{c.chunk.title} {c.chunk.section}: {c.chunk.text}")
            for c in candidates
        ]

        if self.use_mock:
            # Deterministic mock scoring based on word overlap for fast tests
            query_words = set(query.lower().split())
            scores = []
            for _, doc_text in pairs:
                doc_words = set(doc_text.lower().split())
                overlap = len(query_words.intersection(doc_words))
                # Normalize between 0.1 and 0.95
                score = min(0.95, 0.1 + (overlap / (len(query_words) + 1e-5)) * 0.8)
                scores.append(score)
        else:
            model = self._get_model()
            if model is not None:
                raw_scores = model.predict(pairs)
                # Map logits to 0-1 probability via sigmoid
                scores = [sigmoid(float(s)) for s in raw_scores]
            else:
                scores = [c.hybrid_score for c in candidates]

        # Attach rerank_score to each candidate
        for cand, score in zip(candidates, scores):
            cand.rerank_score = score

        # Sort descending by rerank score
        reranked = sorted(candidates, key=lambda x: x.rerank_score or 0.0, reverse=True)

        # Update ranks and return top_k
        final_results: List[RetrievalCandidate] = []
        for rank, cand in enumerate(reranked[:top_k], start=1):
            cand.rank = rank
            final_results.append(cand)

        return final_results
