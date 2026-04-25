"""Retrieval debugger endpoint.

Phase 7 will implement:
- GET /api/search  — bypasses LLM, returns raw retrieval + reranking results
- Invaluable for tuning retrieval without LLM cost
- Shows: dense scores, sparse scores, RRF-fused order, rerank scores

For now: a placeholder that returns 501 Not Implemented.
"""

from fastapi import APIRouter, HTTPException, status

router = APIRouter()


@router.get("/api/search", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def search(q: str):
    """Phase 7 — to be implemented."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Search endpoint will be implemented in Phase 7",
    )