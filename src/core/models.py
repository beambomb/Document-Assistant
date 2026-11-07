"""Core data models and schemas for Enterprise Document Assistant."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class UserContext(BaseModel):
    """Represents the authenticated user and their access rights."""
    user_id: str
    username: str = "Anonymous"
    roles: List[str] = Field(default_factory=lambda: ["guest"])
    department: Optional[str] = None

    def get_effective_roles(self, role_hierarchy: Dict[str, List[str]] | None = None) -> List[str]:
        """Resolves user roles to all accessible role tags based on hierarchy."""
        effective = set()
        for role in self.roles:
            effective.add(role.lower())
            if role_hierarchy and role.lower() in role_hierarchy:
                effective.update([r.lower() for r in role_hierarchy[role.lower()]])
        return sorted(list(effective))


class Document(BaseModel):
    """Raw document before chunking."""
    doc_id: str
    title: str
    content: str
    access_roles: List[str] = Field(default_factory=lambda: ["internal"])
    version: str = "1.0"
    effective_date: str = "2026-01-01"
    department: str = "General"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A granular chunk of a document with search metadata and access tags."""
    chunk_id: str
    doc_id: str
    title: str
    text: str
    section: str = "General"
    page: int = 1
    access_roles: List[str] = Field(default_factory=lambda: ["internal"])
    version: str = "1.0"
    effective_date: str = "2026-01-01"
    department: str = "General"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalCandidate(BaseModel):
    """A chunk retrieved from lexical or dense index with matching score."""
    chunk: Chunk
    lexical_score: float = 0.0
    dense_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: Optional[float] = None
    rank: int = 0


class Citation(BaseModel):
    """Source reference supporting generated text."""
    doc_id: str
    chunk_id: str
    title: str
    section: str
    page: int
    quote: str = ""


class QueryResult(BaseModel):
    """End-to-end response returned to user or API client."""
    query: str
    user_id: str
    user_roles: List[str]
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    retrieved_chunks: List[RetrievalCandidate] = Field(default_factory=list)
    abstained: bool = False
    refusal_reason: Optional[str] = None
    confidence_score: float = 0.0
    latency_ms: float = 0.0
    model_name: str = "mock"


class EvaluationSample(BaseModel):
    """A golden test query for IR, ACL, or Abstention testing."""
    sample_id: str
    query: str
    user_roles: List[str] = Field(default_factory=lambda: ["employee"])
    expected_doc_ids: List[str] = Field(default_factory=list)
    expected_chunk_ids: List[str] = Field(default_factory=list)
    is_trap: bool = False  # If True, system should abstain
    is_adversarial_acl: bool = False  # Query probes forbidden docs
    prohibited_doc_ids: List[str] = Field(default_factory=list)
    ground_truth_answer: Optional[str] = None
    category: str = "general"


class RetrievalMetrics(BaseModel):
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0


class SecurityMetrics(BaseModel):
    total_adversarial_queries: int = 0
    leaked_queries_count: int = 0
    leakage_rate_pct: float = 0.0  # Must be 0.0%


class AbstentionMetrics(BaseModel):
    total_traps: int = 0
    correctly_abstained_traps: int = 0
    trap_rejection_rate_pct: float = 0.0
    total_valid_queries: int = 0
    false_refusals_count: int = 0
    false_refusal_rate_pct: float = 0.0

