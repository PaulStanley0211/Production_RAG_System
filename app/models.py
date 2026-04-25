"""Pydantic request/response schemas.

Shared contract between API, services, and frontend. Defining these
once means everything agrees on the shape of data.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# Query API
# ============================================================

class QueryRequest(BaseModel):
    """User query coming in from the frontend or API consumer."""

    query: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = None
    stream: bool = True
    filters: dict | None = None  # metadata filters (Phase 3)


class Citation(BaseModel):
    """One source chunk referenced in the answer."""

    source: str
    chunk_id: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    """Non-streaming query response (streaming uses SSE events)."""

    answer: str
    citations: list[Citation] = []
    conversation_id: str
    cache_hit: bool = False
    trace_id: str | None = None


# ============================================================
# Health / readiness
# ============================================================

class ComponentHealth(BaseModel):
    name: str
    status: Literal["ok", "degraded", "down"]
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    timestamp: datetime
    components: list[ComponentHealth]


# ============================================================
# Pipeline (Phase 2 — defined early for shared types)
# ============================================================

class Document(BaseModel):
    """A raw extracted document before chunking."""

    id: str
    source: str
    content: str
    metadata: dict = {}


class Chunk(BaseModel):
    """A chunk ready for embedding and indexing."""

    id: str
    doc_id: str
    source: str
    content: str
    metadata: dict = {}


# ============================================================
# CRAG / agentic (Phase 5)
# ============================================================

class GradeResult(BaseModel):
    """Output of the document grader."""

    chunk_id: str
    grade: Literal["relevant", "partially_relevant", "irrelevant"]
    reason: str | None = None


class CRAGDecision(BaseModel):
    """The CRAG self-correction loop's decision."""

    verdict: Literal["correct", "ambiguous", "incorrect"]
    relevant_chunks: list[Chunk] = []
    needs_decomposition: bool = False
    needs_refusal: bool = False


# ============================================================
# Security (Phase 6)
# ============================================================

class GuardResult(BaseModel):
    """Output of input/content/output guards."""

    passed: bool
    sanitized_text: str
    reason: str | None = None
    flags: list[str] = []