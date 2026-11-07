"""FastAPI backend service for Enterprise Document Assistant."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.core.models import UserContext, QueryResult, Citation, RetrievalCandidate
from src.ingestion.loader import DocumentLoader
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_index import DenseVectorIndex
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.confidence_gate import ConfidenceGate
from src.generation.generator import get_generator
from src.generation.citation_verifier import CitationVerifier
from src.evaluation.harness import EvaluationHarness
from config.settings import settings

# Pipeline instances (singleton in memory)
loader = DocumentLoader()
bm25_index = BM25Index()
vector_index = DenseVectorIndex(use_mock=True)
reranker = CrossEncoderReranker(use_mock=True)
confidence_gate = ConfidenceGate(threshold=settings.confidence_threshold)
generator = get_generator()
citation_verifier = CitationVerifier()
is_initialized = False


def init_pipeline():
    """Initializes document corpus and indices."""
    global is_initialized
    if not is_initialized:
        corpus_dir = settings.raw_data_dir / "id_enterprise"
        _, chunks = loader.load_directory_markdown(corpus_dir)
        bm25_index.build(chunks)
        vector_index.build(chunks)
        is_initialized = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI modern lifespan handler."""
    init_pipeline()
    yield


app = FastAPI(
    title="Enterprise Document Assistant API",
    description="Production-grade RAG with Role-Based Access Control, Hybrid Search, and Abstention Gate",
    version=settings.version,
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str
    user_id: str = "user_001"
    roles: List[str] = Field(default_factory=lambda: ["employee"])
    top_k: int = 3


class HealthResponse(BaseModel):
    status: str
    version: str
    total_indexed_chunks: int


@app.get("/health", response_model=HealthResponse)
def health_check():
    init_pipeline()
    return HealthResponse(
        status="healthy",
        version=settings.version,
        total_indexed_chunks=len(bm25_index.chunks),
    )


@app.post("/v1/query", response_model=QueryResult)
def process_query(req: QueryRequest):
    init_pipeline()
    t0 = time.perf_counter()

    user = UserContext(user_id=req.user_id, roles=req.roles)
    retriever = HybridRetriever(bm25_index=bm25_index, vector_index=vector_index)

    # 1. Hybrid Search with strict ACL pre-filtering
    fused_candidates = retriever.search(
        query=req.query,
        user=user,
        top_k=settings.top_k_fused,
    )

    # 2. Cross-Encoder Reranking
    reranked_candidates = reranker.rerank(
        query=req.query,
        candidates=fused_candidates,
        top_k=req.top_k,
    )

    # 3. Confidence Gating & Abstention Decision
    should_abstain, refusal_reason, score = confidence_gate.evaluate(reranked_candidates)

    if should_abstain:
        latency = (time.perf_counter() - t0) * 1000
        return QueryResult(
            query=req.query,
            user_id=req.user_id,
            user_roles=req.roles,
            answer=refusal_reason or "Informasi tidak ditemukan dalam dokumen resmi yang tersedia.",
            citations=[],
            retrieved_chunks=reranked_candidates,
            abstained=True,
            refusal_reason=refusal_reason,
            confidence_score=score,
            latency_ms=latency,
            model_name=settings.llm_provider,
        )

    # 4. LLM Generation
    raw_answer = generator.generate(
        query=req.query,
        retrieved_chunks=reranked_candidates,
        user_roles=req.roles,
    )

    # 5. Citation Verification
    citations, precision, _ = citation_verifier.extract_and_verify(
        answer=raw_answer,
        retrieved_chunks=reranked_candidates,
    )

    latency = (time.perf_counter() - t0) * 1000

    return QueryResult(
        query=req.query,
        user_id=req.user_id,
        user_roles=req.roles,
        answer=raw_answer,
        citations=citations,
        retrieved_chunks=reranked_candidates,
        abstained=False,
        confidence_score=score,
        latency_ms=latency,
        model_name=settings.llm_provider,
    )


@app.post("/v1/evaluate")
def run_evaluation(dataset: str = "id_enterprise"):
    harness = EvaluationHarness(dataset_name=dataset, use_mock_embeddings=True)
    results = harness.run_ablation()
    return {"dataset": dataset, "ablation_results": results}
