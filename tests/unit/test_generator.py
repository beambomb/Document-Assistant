"""Unit tests for BaseGenerator implementations."""

from src.generation.generator import MockGenerator, OpenRouterGenerator
from src.core.models import Chunk, RetrievalCandidate


def make_candidate(doc_id: str, title: str, text: str) -> RetrievalCandidate:
    chunk = Chunk(
        chunk_id=f"{doc_id}_chk_1",
        doc_id=doc_id,
        title=title,
        text=text,
        section="Pasal 1",
        page=1,
        access_roles=["employee"],
    )
    return RetrievalCandidate(chunk=chunk, rank=1)


def test_mock_generator_returns_grounded_citation():
    generator = MockGenerator()
    cand = make_candidate("SOP-IT-001", "SOP Kata Sandi", "Password minimal 12 karakter.")

    answer = generator.generate("Berapa panjang password?", [cand], ["employee"])

    assert "SOP Kata Sandi" in answer
    assert "[Doc: SOP-IT-001, Hal: 1]" in answer


def test_mock_generator_abstains_on_empty_context():
    generator = MockGenerator()
    answer = generator.generate("Pertanyaan tanpa dokumen", [], ["employee"])

    assert "tidak ditemukan" in answer


def test_openrouter_generator_fallback_without_key():
    # When api_key is empty, it safely delegates to mock generator
    gen = OpenRouterGenerator(api_key="")
    cand = make_candidate("PUB-GEN-001", "Profil Perusahaan", "PT Dua Empat Tujuh Nusantara.")
    answer = gen.generate("Apa nama perusahaan?", [cand], ["guest"])

    assert "Profil Perusahaan" in answer
