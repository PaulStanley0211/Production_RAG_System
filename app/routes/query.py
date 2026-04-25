"""Main query endpoint.

POST /api/query
    body: {
        "query": "...",
        "conversation_id": "..." (optional),
        "stream": false  (default false; true returns SSE)
    }

When stream=false: returns QueryResponse JSON.
When stream=true:  returns Server-Sent Events with these event types:
    - status     : pipeline progress markers
    - citation   : one source chunk reference
    - token      : a token of the generated answer
    - redacted   : sent if output guard caught PII (after streaming)
    - done       : final summary
    - error      : something went wrong
"""

import json
import logging

from fastapi import APIRouter, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from app.models import QueryRequest, QueryResponse

log = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/api/query",
    response_model=QueryResponse,
    responses={
        200: {
            "description": "Either a JSON QueryResponse or an SSE stream "
                           "(depending on the stream flag)."
        }
    },
)
async def query(req: QueryRequest, request: Request):
    """Run a query through the full RAG pipeline."""
    pipeline = request.app.state.rag_pipeline

    if req.stream:
        return EventSourceResponse(_stream_events(pipeline, req))

    # Non-streaming path
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


async def _stream_events(pipeline, req: QueryRequest):
    """Adapter: pipeline.stream() yields dicts; SSE wants {event, data} strings."""
    try:
        async for event in pipeline.stream(
            query=req.query,
            conversation_id=req.conversation_id,
        ):
            yield {
                "event": event["event"],
                "data": json.dumps(event["data"]),
            }
    except Exception as e:
        log.exception("Streaming pipeline failed for query: %r", req.query[:200])
        # Emit an error event before closing — frontend can show a clean message
        yield {
            "event": "error",
            "data": json.dumps({
                "message": "Internal pipeline error",
                "type": type(e).__name__,
            }),
        }