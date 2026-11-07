"""Unit tests for Document Summarization Evaluator (ROUGE, Faithfulness, Citation)."""

import pytest
from src.evaluation.summarization import SummarizationEvaluator


@pytest.fixture
def evaluator():
    return SummarizationEvaluator()


@pytest.fixture
def sample_source_text():
    return (
        "Setiap karyawan wajib menggunakan kata sandi dengan panjang minimal 12 karakter "
        "yang mengandung kombinasi huruf besar, huruf kecil, angka, dan simbol khusus. "
        "Kata sandi wajib diperbarui setiap 90 hari kalender. Dilarang menggunakan kata sandi yang sama. "
        "Laptop inventaris perusahaan hanya boleh digunakan untuk kepentingan kedinasan."
    )


@pytest.fixture
def sample_reference_summary():
    return (
        "Kata sandi laptop wajib minimal 12 karakter dengan kombinasi huruf, angka, dan simbol [Doc: SOP-IT-001]. "
        "Pembaruan sandi wajib setiap 90 hari [Doc: SOP-IT-001]. "
        "Laptop dinas hanya untuk keperluan kedinasan [Doc: SOP-IT-001]."
    )


def test_rouge_identical_text_scores_perfect_1(evaluator, sample_reference_summary, sample_source_text):
    result = evaluator.evaluate(
        generated_summary=sample_reference_summary,
        reference_summary=sample_reference_summary,
        source_document_text=sample_source_text,
        expected_doc_id="SOP-IT-001",
    )

    assert result.rouge1_f1 == 1.0
    assert result.rouge2_f1 == 1.0
    assert result.rougeL_f1 == 1.0
    assert result.citation_precision == 1.0
    assert result.is_faithful is True


def test_rouge_partial_overlap(evaluator, sample_reference_summary, sample_source_text):
    candidate = "Kata sandi minimal 12 karakter diperbarui setiap 90 hari [Doc: SOP-IT-001]."
    result = evaluator.evaluate(
        generated_summary=candidate,
        reference_summary=sample_reference_summary,
        source_document_text=sample_source_text,
        expected_doc_id="SOP-IT-001",
    )

    assert 0.0 < result.rouge1_f1 < 1.0
    assert 0.0 < result.rouge2_f1 < 1.0
    assert 0.0 < result.rougeL_f1 < 1.0
    assert result.citation_precision == 1.0
    assert result.faithfulness_score == 1.0


def test_faithfulness_catches_hallucinated_facts(evaluator, sample_reference_summary, sample_source_text):
    # Candidate injects a fabricated policy claim not present in source text
    hallucinated_summary = (
        "Kata sandi laptop wajib minimal 12 karakter [Doc: SOP-IT-001]. "
        "Karyawan berhak membawa pulang laptop dan menjualnya ke pasar loak secara bebas [Doc: SOP-IT-001]. "
        "Perusahaan memberikan bonus tunjangan liburan ke Paris setiap akhir bulan [Doc: SOP-IT-001]."
    )

    result = evaluator.evaluate(
        generated_summary=hallucinated_summary,
        reference_summary=sample_reference_summary,
        source_document_text=sample_source_text,
        expected_doc_id="SOP-IT-001",
    )

    assert result.faithfulness_score < 1.0
    assert result.is_faithful is False
    assert len(result.unsupported_claims) >= 1
    # Check that the hallucination was caught
    assert any("paris" in c.lower() or "pasar loak" in c.lower() for c in result.unsupported_claims)


def test_citation_precision_detects_wrong_doc_id(evaluator, sample_reference_summary, sample_source_text):
    # Cited wrong document ID (KEB-SEC-005 instead of SOP-IT-001)
    wrong_citation_summary = (
        "Kata sandi laptop wajib minimal 12 karakter [Doc: KEB-SEC-005]."
    )

    result = evaluator.evaluate(
        generated_summary=wrong_citation_summary,
        reference_summary=sample_reference_summary,
        source_document_text=sample_source_text,
        expected_doc_id="SOP-IT-001",
    )

    assert result.citation_precision == 0.0
