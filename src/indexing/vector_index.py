"""Dense Vector Index with cosine similarity and Role-Based Access Control (RBAC) pre-filtering."""

from typing import List, Optional
import numpy as np
from src.core.models import Chunk, RetrievalCandidate, UserContext
from src.core.security import access_manager
from config.settings import settings


class DenseVectorIndex:
    """In-memory dense vector index with strict ACL pre-filtering."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_mock: bool = False,
    ):
        self.model_name = model_name or settings.embedding_model_name
        self.use_mock = use_mock or settings.use_mock_embeddings_for_fast_test
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._model = None

    def _get_model(self):
        """Lazy loads sentence-transformers model."""
        if self._model is None and not self.use_mock:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                # Fallback to mock if model download fails or offline
                self.use_mock = True
        return self._model

    def _embed_text(self, texts: List[str]) -> np.ndarray:
        """Generates normalized dense embeddings."""
        if self.use_mock:
            # Deterministic pseudo-embedding for fast offline tests
            dim = settings.embedding_dim
            vecs = []
            for t in texts:
                np.random.seed(abs(hash(t)) % (2**32))
                v = np.random.randn(dim).astype(np.float32)
                norm = np.linalg.norm(v)
                vecs.append(v / (norm + 1e-9))
            return np.array(vecs, dtype=np.float32)

        model = self._get_model()
        if model is not None:
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.array(embeddings, dtype=np.float32)
        else:
            return np.zeros((len(texts), settings.embedding_dim), dtype=np.float32)

    def build(self, chunks: List[Chunk]) -> None:
        """Builds index by encoding chunk texts and storing normalized vectors."""
        self.chunks = chunks
        if not chunks:
            self.embeddings = None
            return

        texts = [f"{c.title} {c.section} {c.text}" for c in chunks]
        self.embeddings = self._embed_text(texts)

    def search(
        self, query: str, user: UserContext, top_k: int = 10
    ) -> List[RetrievalCandidate]:
        """Searches dense index enforcing strict ACL pre-filtering."""
        if self.embeddings is None or not self.chunks:
            return []

        # Step 1: Identify indices permitted for the user (ACL Pre-filter)
        allowed_indices = [
            idx for idx, chunk in enumerate(self.chunks)
            if access_manager.is_chunk_accessible(chunk, user)
        ]

        if not allowed_indices:
            return []

        # Step 2: Embed query
        query_vec = self._embed_text([query])[0]

        # Step 3: Compute dot product (cosine similarity since vectors are normalized)
        # Sub-select only allowed embeddings matrix
        allowed_embeddings = self.embeddings[allowed_indices]
        similarities = np.dot(allowed_embeddings, query_vec)

        # Step 4: Sort top-k
        top_sub_indices = np.argsort(similarities)[::-1][:top_k]

        results: List[RetrievalCandidate] = []
        for rank, sub_idx in enumerate(top_sub_indices, start=1):
            real_chunk_idx = allowed_indices[sub_idx]
            sim_score = float(similarities[sub_idx])
            candidate = RetrievalCandidate(
                chunk=self.chunks[real_chunk_idx],
                dense_score=sim_score,
                rank=rank,
            )
            results.append(candidate)

        return results
