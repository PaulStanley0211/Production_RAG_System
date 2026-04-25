"""FastAPI application entrypoint.

Lifespan constructs all clients, retrieval components, and Phase 4 services,
wires them into the RAG pipeline, and attaches everything to app.state.
Routes receive what they need via request.app.state.<name>.
"""

import logging
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from anthropic import AsyncAnthropic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient

from app.config import settings
from app.retrieval.hybrid_retrieval import HybridRetriever
from app.retrieval.reranker import Reranker
from app.routes import health, query, search
from app.services.conversation import ConversationMemory
from app.services.document_grader import DocumentGrader
from app.services.query_decomposer import QueryDecomposer
from app.services.query_router import QueryRouter
from app.services.rag_pipeline import RAGPipeline
from app.services.semantic_cache import SemanticCache
from pipeline.embedder import Embedder

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Construct all infrastructure at startup; clean up at shutdown."""
    log.info("Starting up. env=%s", settings.app_env)

    # ---- External clients ----
    app.state.qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    app.state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    app.state.anthropic = AsyncAnthropic(api_key=settings.anthropic_api_key)
    log.info("Clients initialized (Qdrant, Redis, Anthropic)")

    # ---- Retrieval components (load embedder + reranker models) ----
    log.info("Loading retrieval models...")
    app.state.embedder = Embedder()
    app.state.retriever = HybridRetriever(
        qdrant=app.state.qdrant,
        embedder=app.state.embedder,
    )
    app.state.reranker = Reranker()
    log.info("Retrieval components ready")

    # ---- Phase 4 services ----
    app.state.memory = ConversationMemory(
        redis_client=app.state.redis,
        anthropic_client=app.state.anthropic,
    )
    app.state.cache = SemanticCache(
        redis_client=app.state.redis,
        embedder=app.state.embedder,
    )
    app.state.router = QueryRouter(anthropic_client=app.state.anthropic)
    app.state.decomposer = QueryDecomposer(anthropic_client=app.state.anthropic)
    app.state.grader = DocumentGrader(anthropic_client=app.state.anthropic)

    # ---- The RAG pipeline orchestrator ----
    app.state.rag_pipeline = RAGPipeline(
        anthropic_client=app.state.anthropic,
        retriever=app.state.retriever,
        reranker=app.state.reranker,
        memory=app.state.memory,
        cache=app.state.cache,
        router=app.state.router,
        decomposer=app.state.decomposer,
        grader=app.state.grader,
    )
    log.info("RAG pipeline ready")

    yield

    # ---- Teardown ----
    log.info("Shutting down")
    await app.state.qdrant.close()
    await app.state.redis.aclose()


app = FastAPI(
    title="Production RAG System",
    description="CRAG-based RAG with hybrid retrieval and 3-layer security",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        "search": "/api/search?q=...",
        "query": "POST /api/query",
    }