"""Hybrid retriever — dense + sparse search with RRF fusion.

For each query:
1. Embed the query as both a dense vector and a sparse vector
2. Search Qdrant twice (named vector "dense", named vector "sparse")
3. Fuse the two ranked lists with Reciprocal Rank Fusion (RRF)
4. Return the merged ranking

The output of this stage feeds into the reranker, which produces the
final top-N. RRF is unaffected by score-scale differences between
dense (cosine, 0-1) and sparse (BM25, unbounded) — it only uses ranks.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Filter,
    NamedSparseVector,
    NamedVector,
    ScoredPoint,
    SparseVector,
)

from app.config import settings
from pipeline.embedder import Embedder

log = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """One result from hybrid retrieval — a Qdrant point + its fused score."""

    point: ScoredPoint
    rrf_score: float
    dense_rank: int | None
    sparse_rank: int | None


class HybridRetriever:
    """Dense + sparse retrieval with Reciprocal Rank Fusion."""

    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        embedder: Embedder,
        collection: str | None = None,
    ):
        self.qdrant = qdrant
        self.embedder = embedder
        self.collection = collection or settings.qdrant_collection
        self.rrf_k = settings.rrf_k  # default 60

    async def retrieve(
        self,
        query: str,
        filters: Filter | None = None,
        top_k_dense: int | None = None,
        top_k_sparse: int | None = None,
        top_k_fused: int | None = None,
    ) -> list[RetrievalResult]:
        """Run both searches, fuse, return top_k_fused results."""
        top_k_dense = top_k_dense or settings.retrieval_top_k_dense
        top_k_sparse = top_k_sparse or settings.retrieval_top_k_sparse
        top_k_fused = top_k_fused or settings.retrieval_top_k_dense

        # Embed the query both ways
        dense_vec = self.embedder.embed_query_dense(query)
        sparse_vec = self.embedder.embed_query_sparse(query)

        # Run both searches
        dense_results = await self._search_dense(
            dense_vec.tolist(), top_k_dense, filters
        )
        sparse_results = await self._search_sparse(
            sparse_vec, top_k_sparse, filters
        )

        log.debug(
            "Hybrid: dense=%d sparse=%d", len(dense_results), len(sparse_results)
        )

        # Fuse and return top_k
        fused = self._rrf_fuse(dense_results, sparse_results)
        return fused[:top_k_fused]

    async def _search_dense(
        self,
        vector: list[float],
        limit: int,
        filters: Filter | None,
    ) -> list[ScoredPoint]:
        return await self.qdrant.search(
            collection_name=self.collection,
            query_vector=NamedVector(name="dense", vector=vector),
            limit=limit,
            query_filter=filters,
            with_payload=True,
        )

    async def _search_sparse(
        self,
        sparse_embedding,
        limit: int,
        filters: Filter | None,
    ) -> list[ScoredPoint]:
        return await self.qdrant.search(
            collection_name=self.collection,
            query_vector=NamedSparseVector(
                name="sparse",
                vector=SparseVector(
                    indices=sparse_embedding.indices.tolist(),
                    values=sparse_embedding.values.tolist(),
                ),
            ),
            limit=limit,
            query_filter=filters,
            with_payload=True,
        )

    def _rrf_fuse(
        self,
        dense: list[ScoredPoint],
        sparse: list[ScoredPoint],
    ) -> list[RetrievalResult]:
        """Reciprocal Rank Fusion over two ranked lists.

        score(doc) = 1/(k + rank_dense) + 1/(k + rank_sparse)
        Documents in both lists get the sum; documents in one get partial credit.
        """
        scores: dict = defaultdict(lambda: {"score": 0.0, "dense_rank": None, "sparse_rank": None})
        points: dict = {}

        for rank, hit in enumerate(dense):
            scores[hit.id]["score"] += 1 / (self.rrf_k + rank)
            scores[hit.id]["dense_rank"] = rank
            points[hit.id] = hit

        for rank, hit in enumerate(sparse):
            scores[hit.id]["score"] += 1 / (self.rrf_k + rank)
            scores[hit.id]["sparse_rank"] = rank
            points[hit.id] = hit

        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda kv: kv[1]["score"], reverse=True)

        return [
            RetrievalResult(
                point=points[pid],
                rrf_score=info["score"],
                dense_rank=info["dense_rank"],
                sparse_rank=info["sparse_rank"],
            )
            for pid, info in ranked
        ]