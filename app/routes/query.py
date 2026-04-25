"""Main query endpoint with SSE streaming.

Phase 7 will implement:
- POST /api/query   — main RAG endpoint, streams tokens + citations + trace
- Request validation via QueryRequest schema
- Pipes through app.state.rag_pipeline.stream(...)
- Returns EventSourceResponse with events: token, citation, trace, done

For now: a placeholder that returns 501 Not Implemented so the route
exists and main.py can import it cleanly.
"""

from fastapi import APIRouter, HTTPException, status

from app.models import QueryRequest

router = APIRouter()


@router.post("/api/query", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def query(req: QueryRequest):
    """Phase 7 — to be implemented."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Query endpoint will be implemented in Phase 7",
    )