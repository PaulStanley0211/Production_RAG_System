"""Application configuration via pydantic-settings.

All settings load from environment variables (with .env file support).
Defaults match production behavior — ablation flags default to off.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All app config in one place. Settings are loaded once at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------
    # App
    # ------------------------------------------------------------
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # ------------------------------------------------------------
    # External services
    # ------------------------------------------------------------
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "docs"
    redis_url: str = "redis://redis:6379/0"
    anthropic_api_key: str = ""

    # ------------------------------------------------------------
    # LLM models — tiered for cost/quality
    # ------------------------------------------------------------
    model_generation: str = "claude-sonnet-4-6"
    model_routing: str = "claude-haiku-4-5-20251001"
    model_complex: str = "claude-opus-4-7"

    # ------------------------------------------------------------
    # Retrieval (Phase 3)
    # ------------------------------------------------------------
    retrieval_top_k_dense: int = 20
    retrieval_top_k_sparse: int = 20
    retrieval_top_k_reranked: int = 5
    rrf_k: int = 60
    embedding_model_dense: str = "BAAI/bge-small-en-v1.5"
    embedding_model_sparse: str = "Qdrant/bm25"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # ------------------------------------------------------------
    # Caching (Phase 4)
    # ------------------------------------------------------------
    semantic_cache_threshold: float = 0.92
    semantic_cache_ttl_seconds: int = 3600

    # ------------------------------------------------------------
    # Security guards (Phase 6)
    # ------------------------------------------------------------
    guard_deny_on_injection: bool = True
    guard_pii_redaction: bool = True

    # ------------------------------------------------------------
    # CRAG / Agentic (Phase 5)
    # ------------------------------------------------------------
    enable_web_fallback: bool = False
    crag_max_iterations: int = 3

    # ------------------------------------------------------------
    # Ablation flags (Phase 8 — eval only)
    # ------------------------------------------------------------
    # All default False / "hybrid" so production behavior is unchanged.
    # The ablation runner sets these via env vars when comparing variants.

    # Skip cross-encoder reranking; use raw RRF order from hybrid retrieval
    disable_reranker: bool = False

    # Skip the CRAG self-correction loop. The pipeline does one retrieval
    # pass + grading + generation, with no decompose retry or fallback.
    disable_crag: bool = False

    # Skip the content filter on retrieved chunks (Phase 6 security guard).
    disable_content_filter: bool = False

    # Retrieval mode: which signals to use.
    #   "hybrid"      → dense + sparse + RRF (default)
    #   "dense_only"  → just dense embeddings
    #   "sparse_only" → just BM25
    retrieval_mode: str = "hybrid"

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    @property
    def cors_origins_list(self) -> list[str]:
        """CORS origins as a list — env var stores them comma-separated."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


# Module-level singleton — all services import this
settings = Settings()