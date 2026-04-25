"""Reranker — cross-encoder rescoring of retrieval candidates.

Takes the top-N from hybrid retrieval, evaluates each (query, document) pair
directly with a cross-encoder, and returns the best top-K.

Cross-encoders are slower than embedding-based scoring but more accurate
because they see query and document together — they can model term
interactions that bi-encoder embeddings cannot.

Standard pattern: retrieve top-20 cheaply, rerank to top-5 accurately.
"""

import logging
from dataclasses import dataclass

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.config import settings
from app.retrieval.hybrid_retrieval import RetrievalResult

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@dataclass
class RerankedResult:
    """Result after reranking — adds rerank_score to RetrievalResult."""

    point: object  # ScoredPoint from Qdrant
    rrf_score: float
    rerank_score: float
    dense_rank: int | None
    sparse_rank: int | None


class Reranker:
    """Cross-encoder rescoring of retrieval candidates."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        log.info("Loading cross-encoder: %s", model_name)
        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        """Rescore candidates against query, return top_k by rerank_score."""
        top_k = top_k or settings.retrieval_top_k_reranked

        if not candidates:
            return []

        # Extract content from each candidate's payload
        texts = [c.point.payload.get("content", "") for c in candidates]

        # Cross-encoder produces a relevance score per (query, doc) pair
        scores = list(self._model.rerank(query, texts))

        # Pair each candidate with its new score
        reranked = [
            RerankedResult(
                point=c.point,
                rrf_score=c.rrf_score,
                rerank_score=float(score),
                dense_rank=c.dense_rank,
                sparse_rank=c.sparse_rank,
            )
            for c, score in zip(candidates, scores)
        ]

        # Sort by rerank score, descending
        reranked.sort(key=lambda r: r.rerank_score, reverse=True)
        return reranked[:top_k]