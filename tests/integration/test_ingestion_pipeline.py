"""Integration test for DocumentLoader with actual raw documents."""

from pathlib import Path
from src.ingestion.loader import DocumentLoader
from config.settings import settings


def test_load_all_enterprise_documents():
    loader = DocumentLoader()
    enterprise_dir = settings.raw_data_dir / "id_enterprise"
    assert enterprise_dir.exists(), f"Directory not found: {enterprise_dir}"

    docs, chunks = loader.load_directory_markdown(enterprise_dir)

    # Ensure all foundational enterprise SOP documents are present
    assert len(docs) >= 5
    doc_ids = {d.doc_id for d in docs}
    expected_ids = {"PUB-GEN-001", "SOP-IT-001", "KEB-HR-002", "SOP-FIN-003", "KEB-SEC-005"}
    assert expected_ids.issubset(doc_ids)

    # Chunks must have been created
    assert len(chunks) >= 10
    # Confidential chunk check
    sec_chunks = [c for c in chunks if c.doc_id == "KEB-SEC-005"]
    assert len(sec_chunks) >= 1
    for sc in sec_chunks:
        assert "confidential" in sc.access_roles or "management" in sc.access_roles
