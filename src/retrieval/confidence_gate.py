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
        # Case 1: Zero candidates passed ACL or index search
        if not candidates:
            return (
                True,
                "Informasi tidak ditemukan dalam dokumen resmi yang dapat diakses oleh akun Anda.",
                0.0,
            )

        # Case 2: Examine top candidate score
        top_cand = candidates[0]
        # Prefer rerank_score if available, otherwise check hybrid_score
        score = top_cand.rerank_score if top_cand.rerank_score is not None else top_cand.hybrid_score

        # Case 3: Score is below empirical confidence threshold
        if score < self.threshold:
            return (
                True,
                f"Tingkat keyakinan bukti dokumen rendah ({score:.2f} < {self.threshold:.2f}). "
                "Pertanyaan berada di luar cakupan dokumen yang tersedia.",
                score,
            )

        # Case 4: Evidence is grounded and passes gate
        return (False, None, score)
