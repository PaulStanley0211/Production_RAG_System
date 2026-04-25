"""Embedder — generates dense + sparse vectors for chunks.

Dense vectors (384-dim, BAAI/bge-small-en-v1.5) capture semantic similarity.
Sparse vectors (BM25 via Qdrant/bm25) capture lexical similarity.
Both run locally on CPU via fastembed — no API costs for indexing.

Phase 3 will fuse the two via Reciprocal Rank Fusion (RRF) for hybrid retrieval.
"""

import logging
from collections.abc import Iterable

import numpy as np
from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding

log = logging.getLogger(__name__)

DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"
DEFAULT_BATCH_SIZE = 64


class Embedder:
    """Local dense + sparse embedder via fastembed."""

    def __init__(
        self,
        dense_model: str = DEFAULT_DENSE_MODEL,
        sparse_model: str = DEFAULT_SPARSE_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        log.info("Loading dense model: %s", dense_model)
        self._dense = TextEmbedding(model_name=dense_model)
        log.info("Loading sparse model: %s", sparse_model)
        self._sparse = SparseTextEmbedding(model_name=sparse_model)
        self.batch_size = batch_size

    def embed_dense(self, texts: list[str]) -> list[np.ndarray]:
        """Generate dense vectors for a list of texts."""
        if not texts:
            return []
        return list(self._dense.embed(texts, batch_size=self.batch_size))

    def embed_sparse(self, texts: list[str]) -> list[SparseEmbedding]:
        """Generate sparse vectors for a list of texts."""
        if not texts:
            return []
        return list(self._sparse.embed(texts, batch_size=self.batch_size))

    def embed_query_dense(self, query: str) -> np.ndarray:
        """Single-query dense embedding (Phase 3 retrieval)."""
        return list(self._dense.query_embed([query]))[0]

    def embed_query_sparse(self, query: str) -> SparseEmbedding:
        """Single-query sparse embedding (Phase 3 retrieval)."""
        return list(self._sparse.query_embed([query]))[0]