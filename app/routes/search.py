"""Retrieval debugger endpoint.

GET /api/search?q=...
  → Bypasses the LLM entirely. Returns raw retrieval + reranking results.
  Use it to tune retrieval and to debug bad answers in production.

Shows: RRF score, rerank score, per-list ranks, source, snippet.
"""

from fastapi import APIRouter, Query, Request

router = APIRouter()


@router.get("/api/search")
async def search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=2000, description="Search query"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to return"),
    snippet_chars: int = Query(300, ge=50, le=2000, description="Snippet length"),
):
    """Run hybrid retrieval + rerank, return diagnostics for each result."""
    retriever = request.app.state.retriever
    reranker = request.app.state.reranker

    # Stage 1: hybrid retrieval (dense + sparse + RRF)
    candidates = await retriever.retrieve(q)

    # Stage 2: cross-encoder rerank
    reranked = reranker.rerank(q, candidates, top_k=top_k)

    # Format response — diagnostics + content
    return {
        "query": q,
        "result_count": len(reranked),
        "results": [
            {
                "rank": i + 1,
                "chunk_id": str(r.point.id),
                "rerank_score": round(r.rerank_score, 4),
                "rrf_score": round(r.rrf_score, 4),
                "dense_rank": r.dense_rank,
                "sparse_rank": r.sparse_rank,
                "source": r.point.payload.get("source"),
                "doc_id": r.point.payload.get("doc_id"),
                "snippet": (r.point.payload.get("content") or "")[:snippet_chars],
                "metadata": {
                    k: v for k, v in r.point.payload.items()
                    if k not in {"content", "source", "doc_id"}
                },
            }
            for i, r in enumerate(reranked)
        ],
    }