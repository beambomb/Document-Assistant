"""Document Summarization Evaluator measuring ROUGE and Factual Faithfulness."""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from rouge_score import rouge_scorer
from src.generation.citation_verifier import CitationVerifier


@dataclass
class SummaryEvaluationResult:
    """Holds quantitative evaluation results for a generated summary."""
    rouge1_f1: float
    rouge2_f1: float
    rougeL_f1: float
    faithfulness_score: float
    supported_claims: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    citation_precision: float = 1.0
    citations_found: List[str] = field(default_factory=list)
    is_faithful: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rouge1_f1": round(self.rouge1_f1, 4),
            "rouge2_f1": round(self.rouge2_f1, 4),
            "rougeL_f1": round(self.rougeL_f1, 4),
            "faithfulness_score": round(self.faithfulness_score, 4),
            "is_faithful": self.is_faithful,
            "supported_claims_count": len(self.supported_claims),
            "unsupported_claims_count": len(self.unsupported_claims),
            "citation_precision": round(self.citation_precision, 4),
            "citations_found": self.citations_found,
        }


class SummarizationEvaluator:
    """Evaluates generated document summaries using ROUGE (salience) and Faithfulness (grounding)."""

    def __init__(self, rouge_types: Optional[List[str]] = None):
        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]
        self.scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)
        self.citation_verifier = CitationVerifier()

    @staticmethod
    def extract_claims(text: str) -> List[str]:
        """Extracts discrete sentences/claims from summary, stripping markdown bullets and citation tags."""
        clean = re.sub(r"\[Doc:[^\]]+\]", "", text)
        clean = re.sub(r"^[*-]\s+", "", clean, flags=re.MULTILINE)
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", clean) if len(s.strip()) > 10]
        return sentences

    def evaluate_faithfulness(
        self,
        summary: str,
        source_text: str,
    ) -> tuple[float, List[str], List[str]]:
        """
        Evaluates whether every claim in the summary is factually grounded in the source document.
        Returns:
            score (0.0 to 1.0)
            supported_claims (List[str])
            unsupported_claims (List[str])
        """
        claims = self.extract_claims(summary)
        if not claims:
            return 1.0, [], []

        source_lower = source_text.lower()
        supported = []
        unsupported = []

        for claim in claims:
            words = [w for w in re.findall(r"\w+", claim.lower()) if len(w) >= 3]
            if not words:
                continue

            # Check overlap against source document text
            matches = sum(
                1 for w in words
                if w in source_lower or (len(w) >= 4 and w[:4] in source_lower)
            )
            ratio = matches / len(words)

            # A factual claim must have strong overlap with source context (>= 40%)
            if ratio >= 0.40:
                supported.append(claim)
            else:
                unsupported.append(claim)

        total = len(supported) + len(unsupported)
        score = len(supported) / total if total > 0 else 1.0
        return float(score), supported, unsupported

    def evaluate(
        self,
        generated_summary: str,
        reference_summary: str,
        source_document_text: str,
        expected_doc_id: Optional[str] = None,
    ) -> SummaryEvaluationResult:
        """
        Runs comprehensive evaluation on a generated summary.

        Args:
            generated_summary: The summary output produced by the LLM / generator.
            reference_summary: The golden human/reference summary for ROUGE comparison.
            source_document_text: The full source document text for faithfulness grounding.
            expected_doc_id: Optional expected document ID for citation verification.
        """
        # 1. Compute ROUGE Scores against reference
        rouge_scores = self.scorer.score(reference_summary, generated_summary)
        r1 = rouge_scores["rouge1"].fmeasure
        r2 = rouge_scores["rouge2"].fmeasure
        rL = rouge_scores["rougeL"].fmeasure

        # 2. Compute Factual Faithfulness against source document
        faith_score, supported, unsupported = self.evaluate_faithfulness(
            generated_summary, source_document_text
        )

        # 3. Verify Citations
        citation_matches = CitationVerifier.CITATION_PATTERN.findall(generated_summary)
        citations_found = [m[0].strip() for m in citation_matches]

        if not citation_matches:
            cit_precision = 0.0
        elif expected_doc_id:
            valid_cits = sum(1 for c in citations_found if c.lower() == expected_doc_id.lower())
            cit_precision = valid_cits / len(citations_found)
        else:
            cit_precision = 1.0

        is_faithful = faith_score >= 0.90 and len(unsupported) == 0

        return SummaryEvaluationResult(
            rouge1_f1=r1,
            rouge2_f1=r2,
            rougeL_f1=rL,
            faithfulness_score=faith_score,
            supported_claims=supported,
            unsupported_claims=unsupported,
            citation_precision=cit_precision,
            citations_found=citations_found,
            is_faithful=is_faithful,
        )
