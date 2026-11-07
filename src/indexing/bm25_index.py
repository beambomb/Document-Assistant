"""BM25 Lexical Index with Role-Based Access Control (RBAC) pre-filtering."""

import re
from typing import List, Optional
from rank_bm25 import BM25Okapi
from src.core.models import Chunk, RetrievalCandidate, UserContext
from src.core.security import access_manager


def default_tokenizer(text: str) -> List[str]:
    """Tokenizes text preserving alphanumeric codes, document IDs, and words."""
    # Keeps words, numbers, and hyphenated terms like 'SOP-IT-001'
    tokens = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", text.lower())
    return tokens


class BM25Index:
    """In-memory BM25 index supporting strict ACL pre-filtering."""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or default_tokenizer
        self.chunks: List[Chunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def build(self, chunks: List[Chunk]) -> None:
        """Builds the BM25 index from a list of chunks."""
        self.chunks = chunks
        self.tokenized_corpus = [
            self.tokenizer(f"{chunk.title} {chunk.section} {chunk.text}")
            for chunk in chunks
        ]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def search(
        self, query: str, user: UserContext, top_k: int = 10
    ) -> List[RetrievalCandidate]:
        """Searches BM25 index enforcing strict ACL pre-filtering."""
        if not self.bm25 or not self.chunks:
            return []

        # Step 1: Pre-filter candidate indices by user role permissions
        allowed_indices: List[int] = []
        for idx, chunk in enumerate(self.chunks):
            if access_manager.is_chunk_accessible(chunk, user):
                allowed_indices.append(idx)

        # If user has no access to any documents, immediately return empty
        if not allowed_indices:
            return []

        # Step 2: Score query against tokenized corpus
        query_tokens = self.tokenizer(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Step 3: Collect scores strictly for authorized indices
        scored_candidates: List[tuple[int, float]] = []
        for idx in allowed_indices:
            score = float(scores[idx])
            if score > 0.0:  # Only consider chunks with lexical overlap
                scored_candidates.append((idx, score))

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored_candidates[:top_k]

        # Step 4: Construct RetrievalCandidate objects
        results: List[RetrievalCandidate] = []
        for rank, (idx, score) in enumerate(top_candidates, start=1):
            candidate = RetrievalCandidate(
                chunk=self.chunks[idx],
                lexical_score=score,
                rank=rank,
            )
            results.append(candidate)

        return results
