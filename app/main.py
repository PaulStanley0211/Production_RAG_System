"""FastAPI application entrypoint.

Expensive clients (Qdrant, Redis, Anthropic) are created once during the
`lifespan` context and attached to `app.state`. Routes receive them via
`request.app.state.<client>` — no re-instantiation per request.

This is THE wiring file for the entire app. Every other file ultimately
flows through here.
"""

import logging
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from anthropic import AsyncAnthropic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient

from app.config import settings
from app.routes import health, query, search

# Configure logging once, globally
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Instantiate external clients at startup; clean up at shutdown.

    Everything before `yield` runs at startup.
    Everything after `yield` runs at shutdown.
    """
    log.info("Starting up. env=%s", settings.app_env)

    # External clients — created once, reused for all requests
    app.state.qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    app.state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    app.state.anthropic = AsyncAnthropic(api_key=settings.anthropic_api_key)

    log.info("Clients initialized (Qdrant, Redis, Anthropic)")

    # Phase 2+ services will be initialized here:
    #   app.state.embedder = Embedder()
    #   app.state.retriever = HybridRetriever(...)
    #   app.state.reranker = Reranker()
    #   app.state.rag_pipeline = RAGPipeline(...)

    yield

    # Teardown
    log.info("Shutting down")
    await app.state.qdrant.close()
    await app.state.redis.aclose()
    # AsyncAnthropic doesn't require explicit cleanup


app = FastAPI(
    title="Production RAG System",
    description="CRAG-based RAG with hybrid retrieval and 3-layer security",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allows browser-based frontends from configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers
app.include_router(health.router, tags=["health"])
app.include_router(query.router, tags=["query"])
app.include_router(search.router, tags=["search"])


@app.get("/")
async def root():
    """Friendly landing page with links to common endpoints."""
    return {
        "name": "production-rag-system",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
    }