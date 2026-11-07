"""Unit tests for DenseVectorIndex and its ACL pre-filtering mechanism."""

import pytest
from src.indexing.vector_index import DenseVectorIndex
from src.core.models import Chunk, UserContext


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id="chk_pub",
            doc_id="PUB-001",
            title="Profil Perusahaan",
            text="Solusi analitik big data dan kecerdasan buatan.",
            access_roles=["public"],
        ),
        Chunk(
            chunk_id="chk_emp",
            doc_id="SOP-IT-001",
            title="Keamanan Komputer",
            text="Kebijakan keamanan perangkat kerja dan koneksi VPN.",
            access_roles=["employee"],
        ),
        Chunk(
            chunk_id="chk_sec",
            doc_id="KEB-SEC-005",
            title="Gaji Direksi",
            text="Remunerasi rahasia pejabat tinggi dan komisaris.",
            access_roles=["management", "confidential"],
        ),
    ]


def test_vector_index_search_and_acl_filter(sample_chunks):
    # Use mock embeddings for sub-second deterministic test
    index = DenseVectorIndex(use_mock=True)
    index.build(sample_chunks)

    emp_user = UserContext(user_id="emp_1", roles=["employee"])
    results = index.search(query="kebijakan vpn kantor", user=emp_user, top_k=5)

    assert len(results) > 0
    retrieved_ids = {r.chunk.chunk_id for r in results}
    # Security test: confidential chunk must NEVER be in retrieved results
    assert "chk_sec" not in retrieved_ids


def test_vector_management_access(sample_chunks):
    index = DenseVectorIndex(use_mock=True)
    index.build(sample_chunks)

    mgmt_user = UserContext(user_id="boss_1", roles=["management"])
    results = index.search(query="remunerasi rahasia", user=mgmt_user, top_k=5)

    assert len(results) > 0
    assert any(r.chunk.chunk_id == "chk_sec" for r in results)
