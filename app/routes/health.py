"""Health and readiness endpoints.

`/health`  — liveness: process is up. Returns 200 always if reachable.
`/ready`   — readiness: actually pings Qdrant + Redis. 503 if any down.

The split between liveness and readiness is a production pattern:
- Liveness failure → restart the container
- Readiness failure → don't route traffic, but don't restart
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Request, Response, status

from app.models import ComponentHealth, HealthResponse

router = APIRouter()


async def _check_qdrant(request: Request) -> ComponentHealth:
    """Ping Qdrant by listing collections. Cheap and reliable."""
    try:
        await request.app.state.qdrant.get_collections()
        return ComponentHealth(name="qdrant", status="ok")
    except Exception as e:
        return ComponentHealth(name="qdrant", status="down", detail=str(e))


async def _check_redis(request: Request) -> ComponentHealth:
    """Ping Redis. PONG response confirms connection."""
    try:
        pong = await request.app.state.redis.ping()
        return ComponentHealth(
            name="redis",
            status="ok" if pong else "degraded",
        )
    except Exception as e:
        return ComponentHealth(name="redis", status="down", detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness — is the process running?

    Always returns 200 if the app responds. Used by orchestrators
    to decide whether to restart the container.
    """
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc),
        components=[],
    )


@router.get("/ready", response_model=HealthResponse)
async def ready(request: Request, response: Response) -> HealthResponse:
    """Readiness — are dependencies reachable?

    Returns 503 if any dependency is down. Used by load balancers
    to decide whether to route traffic to this instance.
    """
    components = [
        await _check_qdrant(request),
        await _check_redis(request),
    ]

    if any(c.status == "down" for c in components):
        overall = "down"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif any(c.status == "degraded" for c in components):
        overall = "degraded"
    else:
        overall = "ok"

    return HealthResponse(
        status=overall,
        timestamp=datetime.now(timezone.utc),
        components=components,
    )