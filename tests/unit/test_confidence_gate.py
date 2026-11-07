"""Unit tests for ConfidenceGate abstention decisions."""

from src.retrieval.confidence_gate import ConfidenceGate
from src.core.models import Chunk, RetrievalCandidate


def make_candidate(score: float) -> RetrievalCandidate:
    chunk = Chunk(
        chunk_id="chk_1",
        doc_id="DOC_1",
        title="Sample",
        text="Sample text",
        access_roles=["employee"],
    )
    return RetrievalCandidate(chunk=chunk, rerank_score=score, rank=1)


def test_confidence_gate_abstains_on_empty_candidates():
    gate = ConfidenceGate(threshold=0.35)
    should_abstain, reason, score = gate.evaluate([])

    assert should_abstain is True
    assert "tidak ditemukan" in reason
    assert score == 0.0


def test_confidence_gate_abstains_on_low_score():
    gate = ConfidenceGate(threshold=0.35)
    # Score 0.20 is below threshold 0.35 (e.g. out of scope query)
    candidate = make_candidate(score=0.20)
    should_abstain, reason, score = gate.evaluate([candidate])

    assert should_abstain is True
    assert "rendah" in reason
    assert score == 0.20


def test_confidence_gate_allows_high_score():
    gate = ConfidenceGate(threshold=0.35)
    # Score 0.85 is confident
    candidate = make_candidate(score=0.85)
    should_abstain, reason, score = gate.evaluate([candidate])

    assert should_abstain is False
    assert reason is None
    assert score == 0.85
