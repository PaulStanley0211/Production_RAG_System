"""Centralized application settings.

Single source of truth for all configuration. Every module reads
`from app.config import settings`. Pydantic validates at startup —
misconfiguration fails loudly rather than silently.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------
    # Application
    # ------------------------------------------------------------
    app_env: Literal["development", "production"] = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    # ------------------------------------------------------------
    # Anthropic — tiered model selection
    # ------------------------------------------------------------
    anthropic_api_key: str = Field(..., description="Claude API key")
    model_generation: str = "claude-sonnet-4-6"          # Workhorse — answers
    model_routing: str = "claude-haiku-4-5-20251001"     # Cheap+fast — routing/grading
    model_complex: str = "claude-opus-4-7"               # Reserved — hardest queries

    # ------------------------------------------------------------
    # Qdrant (vector DB)
    # ------------------------------------------------------------
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "docs"
    qdrant_vector_size: int = 384

    # ------------------------------------------------------------
    # Redis (semantic cache + conversation memory)
    # ------------------------------------------------------------
    redis_url: str = "redis://redis:6379/0"
    semantic_cache_ttl_seconds: int = 3600
    semantic_cache_threshold: float = 0.92

    # ------------------------------------------------------------
    # Retrieval (Phase 3)
    # ------------------------------------------------------------
    retrieval_top_k_dense: int = 20
    retrieval_top_k_sparse: int = 20
    retrieval_top_k_reranked: int = 5
    rrf_k: int = 60

    # ------------------------------------------------------------
    # Security (Phase 6)
    # ------------------------------------------------------------
    guard_deny_on_injection: bool = True
    guard_pii_redaction: bool = True

    # ------------------------------------------------------------
    # Observability (Phase 8)
    # ------------------------------------------------------------
    opik_api_key: str | None = None
    opik_project: str = "production-rag"

    @property
    def cors_origins_list(self) -> list[str]:
        """Convert comma-separated string into a list for FastAPI CORS."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — call this everywhere you need config."""
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
