"""Main query endpoint — runs the full RAG pipeline.

POST /api/query
    body: { "query": "...", "conversation_id": "..." (optional), "stream": false }
    returns: { answer, citations, conversation_id, cache_hit }

Streaming (stream=true) will be added in Phase 7. Phase 4 returns the full
response as JSON once generation completes.
"""

import logging

from fastapi import APIRouter, HTTPException, Request, status

from app.models import QueryRequest, QueryResponse

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request) -> QueryResponse:
    """Run a query through the full RAG pipeline."""
    pipeline = request.app.state.rag_pipeline

    try:
        result = await pipeline.run(
            query=req.query,
            conversation_id=req.conversation_id,
        )
    except Exception as e:
        log.exception("Pipeline failed for query: %r", req.query[:200])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {type(e).__name__}",
        ) from e

    return result