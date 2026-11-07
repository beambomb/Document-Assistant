"""Unit tests for BM25Index and its ACL pre-filtering mechanism."""

import pytest
from src.indexing.bm25_index import BM25Index
from src.core.models import Chunk, UserContext


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id="chk_pub",
            doc_id="PUB-001",
            title="Profil PT Majelis Debu Bintang",
            text="Perusahaan penyedia teknologi analitik big data di Jakarta.",
            access_roles=["public"],
        ),
        Chunk(
            chunk_id="chk_it",
            doc_id="SOP-IT-001",
            title="SOP Password Laptop",
            text="Panjang minimal kata sandi adalah 12 karakter wajib diperbarui 90 hari.",
            access_roles=["employee"],
        ),
        Chunk(
            chunk_id="chk_sec",
            doc_id="KEB-SEC-005",
            title="Kompensasi Gaji Direksi",
            text="Gaji pokok CEO adalah 120 juta rupiah per bulan dengan tantiem laba.",
            access_roles=["management", "confidential"],
        ),
    ]


def test_bm25_exact_keyword_retrieval(sample_chunks):
    index = BM25Index()
    index.build(sample_chunks)

    emp_user = UserContext(user_id="emp_1", roles=["employee"])
    results = index.search(query="kata sandi 12 karakter 90 hari", user=emp_user, top_k=5)

    assert len(results) > 0
    assert results[0].chunk.chunk_id == "chk_it"
    assert results[0].lexical_score > 0.0


def test_bm25_acl_prefilter_blocks_confidential_docs(sample_chunks):
    index = BM25Index()
    index.build(sample_chunks)

    # Employee tries to search for confidential director salary keywords
    emp_user = UserContext(user_id="emp_attacker", roles=["employee"])
    results = index.search(query="Gaji pokok CEO 120 juta tantiem", user=emp_user, top_k=5)

    # Must NOT return chk_sec under any circumstance
    retrieved_chunk_ids = {r.chunk.chunk_id for r in results}
    assert "chk_sec" not in retrieved_chunk_ids


def test_bm25_guest_isolation(sample_chunks):
    index = BM25Index()
    index.build(sample_chunks)

    guest = UserContext(user_id="guest_1", roles=["guest"])
    results = index.search(query="SOP Password Laptop", user=guest, top_k=5)

    # Guest cannot see IT SOP
    assert len(results) == 0
