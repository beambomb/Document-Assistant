"""Confidence Gate and Abstention Decision Engine to prevent hallucinations and out-of-scope replies."""

from typing import List, Tuple, Optional
from src.core.models import RetrievalCandidate
from config.settings import settings


class ConfidenceGate:
    """Evaluates candidate relevance scores against an empirical threshold to trigger abstention."""

    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold if threshold is not None else settings.confidence_threshold

    def evaluate(
        self, candidates: List[RetrievalCandidate]
    ) -> Tuple[bool, Optional[str], float]:
        """
        Determines whether to answer or abstain.
        Returns:
            should_abstain (bool)
            refusal_reason (Optional[str])
            confidence_score (float)
        """
        # Uniform refusal message to prevent oracle/side-channel leakage
        UNIFORM_REFUSAL = "Maaf, informasi tidak ditemukan dalam dokumen resmi yang tersedia untuk akun Anda."

        # Case 1: Zero candidates passed ACL or index search
        if not candidates:
            return (True, UNIFORM_REFUSAL, 0.0)

        # Case 2: Examine top candidate score
        top_cand = candidates[0]
        # Prefer rerank_score if available, otherwise check hybrid_score
        score = top_cand.rerank_score if top_cand.rerank_score is not None else top_cand.hybrid_score

        # Case 3: Score is below confidence threshold
        if score < self.threshold:
            return (True, UNIFORM_REFUSAL, score)

        # Case 4: Evidence is grounded and passes gate
        return (False, None, score)
