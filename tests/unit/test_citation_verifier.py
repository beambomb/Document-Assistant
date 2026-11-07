"""Unit tests for CitationVerifier."""

from src.generation.citation_verifier import CitationVerifier
from src.core.models import Chunk, RetrievalCandidate


def make_candidate(doc_id: str, title: str) -> RetrievalCandidate:
    chunk = Chunk(
        chunk_id=f"{doc_id}_chk_1",
        doc_id=doc_id,
        title=title,
        text="Teks resmi kebijakan internal.",
        section="Pasal 1",
        page=2,
        access_roles=["employee"],
    )
    return RetrievalCandidate(chunk=chunk, rank=1)


def test_extract_valid_citations():
    verifier = CitationVerifier()
    retrieved = [
        make_candidate("SOP-IT-001", "SOP IT"),
        make_candidate("KEB-HR-002", "Kebijakan Cuti"),
    ]

    answer = (
        "Kata sandi harus diganti tiap 90 hari [Doc: SOP-IT-001, Hal: 2]. "
        "Cuti tahunan berjumlah 12 hari [Doc: KEB-HR-002]."
    )

    citations, precision, hallucination_count = verifier.extract_and_verify(answer, retrieved)

    assert len(citations) == 2
    assert precision == 1.0
    assert hallucination_count == 0
    assert citations[0].doc_id == "SOP-IT-001"
    assert citations[0].page == 2
    assert citations[1].doc_id == "KEB-HR-002"


def test_detect_hallucinated_citations():
    verifier = CitationVerifier()
    # Context ONLY has SOP-IT-001
    retrieved = [make_candidate("SOP-IT-001", "SOP IT")]

    # LLM hallucinates a citation to an unretrieved document FAKE-DOC-999
    answer = "Aturan lembur dibayar 2x lipat [Doc: FAKE-DOC-999, Hal: 10]."

    citations, precision, hallucination_count = verifier.extract_and_verify(answer, retrieved)

    assert len(citations) == 1
    assert precision == 0.0
    assert hallucination_count == 1
    assert citations[0].chunk_id == "HALLUCINATED"
