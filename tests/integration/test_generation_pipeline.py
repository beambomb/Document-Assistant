"""Integration test for end-to-end generation and citation verification."""

from src.generation.generator import MockGenerator
from src.generation.citation_verifier import CitationVerifier
from src.core.models import Chunk, RetrievalCandidate


def test_generation_and_citation_verification_flow():
    chunk = Chunk(
        chunk_id="SOP-FIN-003_chk_1",
        doc_id="SOP-FIN-003",
        title="SOP Reimbursement",
        text="Klaim harus diajukan maksimal 14 hari kalender setelah transaksi.",
        section="Ketentuan Pengajuan Klaim",
        page=1,
        access_roles=["employee", "finance"],
    )
    cand = RetrievalCandidate(chunk=chunk, rank=1)

    generator = MockGenerator()
    answer = generator.generate(
        query="Berapa hari batas klaim reimbursement?",
        retrieved_chunks=[cand],
        user_roles=["employee"],
    )

    verifier = CitationVerifier()
    citations, precision, hallucination_count = verifier.extract_and_verify(
        answer=answer,
        retrieved_chunks=[cand],
    )

    assert len(citations) == 1
    assert citations[0].doc_id == "SOP-FIN-003"
    assert precision == 1.0
    assert hallucination_count == 0
