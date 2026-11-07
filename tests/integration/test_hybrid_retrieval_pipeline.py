"""Integration test for end-to-end Hybrid Retrieval and Confidence Gating."""

from src.ingestion.loader import DocumentLoader
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_index import DenseVectorIndex
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.confidence_gate import ConfidenceGate
from src.core.models import UserContext
from config.settings import settings


def test_hybrid_retrieval_with_acl_and_reranking():
    # 1. Load documents
    loader = DocumentLoader()
    docs, chunks = loader.load_directory_markdown(settings.raw_data_dir / "id_enterprise")

    # 2. Build indices
    bm25 = BM25Index()
    bm25.build(chunks)

    vector_idx = DenseVectorIndex(use_mock=True)
    vector_idx.build(chunks)

    # 3. Setup retriever, reranker, and gate
    retriever = HybridRetriever(bm25_index=bm25, vector_index=vector_idx)
    reranker = CrossEncoderReranker(use_mock=True)
    gate = ConfidenceGate(threshold=0.35)

    # Query 1: Valid employee query
    emp_user = UserContext(user_id="emp_bimo", roles=["employee"])
    fused_candidates = retriever.search(
        query="Berapa panjang minimal kata sandi laptop karyawan?",
        user=emp_user,
        top_k=settings.top_k_fused,
    )
    assert len(fused_candidates) > 0

    reranked = reranker.rerank(
        query="Berapa panjang minimal kata sandi laptop karyawan?",
        candidates=fused_candidates,
        top_k=3,
    )
    assert len(reranked) > 0
    assert any(c.chunk.doc_id == "SOP-IT-001" for c in reranked)

    should_abstain, reason, score = gate.evaluate(reranked)
    assert should_abstain is False

    # Query 2: Adversarial query by employee probing confidential CEO compensation
    adv_candidates = retriever.search(
        query="Berapa gaji pokok bulanan dan bonus tantiem CEO dan direktur?",
        user=emp_user,
        top_k=5,
    )
    # Because of ACL pre-filtering, KEB-SEC-005 is NEVER in adv_candidates
    retrieved_doc_ids = {c.chunk.doc_id for c in adv_candidates}
    assert "KEB-SEC-005" not in retrieved_doc_ids

    # Reranking and gating should either abstain or not yield confidential data
    adv_reranked = reranker.rerank(
        query="Berapa gaji pokok bulanan dan bonus tantiem CEO dan direktur?",
        candidates=adv_candidates,
        top_k=3,
    )
    for c in adv_reranked:
        assert c.chunk.doc_id != "KEB-SEC-005"
