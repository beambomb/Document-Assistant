"""Unit tests for DocumentChunker."""

import pytest
from src.ingestion.chunker import DocumentChunker
from src.core.models import Document


@pytest.fixture
def chunker():
    return DocumentChunker(target_chunk_size=300)


def test_markdown_metadata_parsing(chunker):
    sample_md = """# SOP KEAMANAN
**Nomor Dokumen**: SOP-IT-999
**Versi**: 4.0
**Tanggal Berlaku**: 10 Maret 2026
**Klasifikasi Akses**: internal, employee
**Divisi**: Cyber Security

## 1. Pendahuluan
Teks pendahuluan di sini.
"""
    meta, _ = chunker.parse_markdown_metadata(sample_md)

    assert meta["doc_id"] == "SOP-IT-999"
    assert meta["version"] == "4.0"
    assert meta["effective_date"] == "10 Maret 2026"
    assert meta["access_roles"] == ["internal", "employee"]
    assert meta["department"] == "Cyber Security"
    assert meta["title"] == "SOP KEAMANAN"


def test_chunking_preserves_section_and_roles(chunker):
    doc = Document(
        doc_id="SOP-FIN-003",
        title="SOP Reimbursement",
        content="""## 1. Ketentuan Klaim
Klaim harus diajukan maksimal 14 hari kalender dengan bukti sah.

## 2. Plafon Perjalanan Dinas
Level staf per diem 350 ribu per hari.
""",
        access_roles=["finance", "employee"],
        department="Finance",
    )

    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "SOP-FIN-003_chk_1"
    assert chunks[0].section == "1. Ketentuan Klaim"
    assert chunks[0].access_roles == ["finance", "employee"]
    assert "14 hari kalender" in chunks[0].text

    assert chunks[1].chunk_id == "SOP-FIN-003_chk_2"
    assert chunks[1].section == "2. Plafon Perjalanan Dinas"
    assert chunks[1].access_roles == ["finance", "employee"]
    assert "350 ribu" in chunks[1].text
