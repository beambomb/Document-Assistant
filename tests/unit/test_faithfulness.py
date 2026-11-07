"""Unit tests for FaithfulnessEvaluator."""

from src.evaluation.faithfulness import FaithfulnessEvaluator
from src.core.models import Chunk, RetrievalCandidate


def make_candidate(text: str) -> RetrievalCandidate:
    chunk = Chunk(
        chunk_id="chk_1",
        doc_id="SOP-01",
        title="SOP Kata Sandi",
        text=text,
        access_roles=["employee"],
    )
    return RetrievalCandidate(chunk=chunk, rank=1)


def test_faithfulness_grounded_answer():
    context_text = "Kata sandi laptop karyawan minimal 12 karakter dan wajib diperbarui setiap 90 hari kalender."
    cand = make_candidate(context_text)

    grounded_answer = "Kata sandi laptop minimal 12 karakter. Karyawan wajib memperbarui kata sandi setiap 90 hari kalender."

    score, supported, unsupported = FaithfulnessEvaluator.evaluate_faithfulness(
        answer=grounded_answer,
        retrieved_chunks=[cand],
    )

    assert score >= 0.80
    assert len(supported) > 0


def test_faithfulness_detects_unsupported_hallucinations():
    context_text = "Kata sandi laptop minimal 12 karakter."
    cand = make_candidate(context_text)

    hallucinated_answer = (
        "Kata sandi laptop minimal 12 karakter. "
        "Perusahaan memberikan tiket liburan gratis ke Tokyo Jepang setiap akhir tahun bagi karyawan berprestasi."
    )

    score, supported, unsupported = FaithfulnessEvaluator.evaluate_faithfulness(
        answer=hallucinated_answer,
        retrieved_chunks=[cand],
    )

    # Tokyo trip is hallucinated, so score must drop
    assert score < 1.0
    assert len(unsupported) > 0
    assert any("tokyo" in u.lower() or "jepang" in u.lower() for u in unsupported)
