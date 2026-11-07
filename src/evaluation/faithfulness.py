"""Faithfulness & Grounding evaluation module based on RAGAS / DeepEval principles."""

import re
from typing import List, Tuple
from src.core.models import RetrievalCandidate


class FaithfulnessEvaluator:
    """Evaluates factual faithfulness: measures if statements in the answer are supported by retrieved context."""

    @staticmethod
    def extract_claims(answer: str) -> List[str]:
        """Splits answer into discrete factual claims / sentences."""
        clean_text = re.sub(r"\[Doc:[^\]]+\]", "", answer).strip()
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", clean_text) if len(s.strip()) > 8]
        return sentences

    @staticmethod
    def evaluate_faithfulness(
        answer: str,
        retrieved_chunks: List[RetrievalCandidate],
    ) -> Tuple[float, List[str], List[str]]:
        """
        Calculates faithfulness score between 0.0 and 1.0.
        Returns:
            faithfulness_score (float)
            supported_claims (List[str])
            unsupported_claims (List[str])
        """
        claims = FaithfulnessEvaluator.extract_claims(answer)
        if not claims:
            return 1.0, [], []

        context_corpus = " ".join([c.chunk.text.lower() for c in retrieved_chunks])

        supported: List[str] = []
        unsupported: List[str] = []

        for claim in claims:
            words = [w for w in re.findall(r"\w+", claim.lower()) if len(w) >= 3]
            if not words:
                continue

            # Match either exact word or root stem (first 4 chars)
            matches = sum(1 for w in words if w in context_corpus or (len(w) >= 4 and w[:4] in context_corpus))
            overlap_ratio = matches / len(words)

            if overlap_ratio >= 0.35:
                supported.append(claim)
            else:
                unsupported.append(claim)

        total_evaluated = len(supported) + len(unsupported)
        score = len(supported) / total_evaluated if total_evaluated > 0 else 1.0
        return float(score), supported, unsupported
