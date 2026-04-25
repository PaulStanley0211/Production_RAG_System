"""Semantic cache — Redis-backed cache keyed by query meaning.

Embeds incoming queries and finds cached answers from semantically similar
past queries. Cache hit threshold (cosine similarity) is configurable;
default 0.92 — high enough to avoid wrong answers, low enough to catch
common rephrasings.

Why not exact match? Users phrase the same question many ways:
  "what is X?" / "tell me about X" / "X overview please"
Exact-match would cache all three; semantic caches them as one.

Implementation note: this version uses a linear scan over recent cache
entries. For Phase 4's scale (tens of cached queries per minute) this is
plenty fast. For thousands per second, swap to Redis Stack + RediSearch
HNSW index — the cache interface stays the same.
"""

import json
import logging
import time
from dataclasses import dataclass

import numpy as np
import redis.asyncio as aioredis

from app.config import settings
from pipeline.embedder import Embedder

log = logging.getLogger(__name__)

# All cache entries live under this key prefix in Redis.
CACHE_KEY_PREFIX = "rag:cache:"

# Maximum number of recent entries to scan per lookup.
MAX_SCAN_ENTRIES = 100


@dataclass
class CacheHit:
    """Result of a cache lookup that found a match."""

    query: str
    answer: str
    citations: list[dict]
    similarity: float
    cached_at: float


class SemanticCache:
    """Cache answers by query semantic similarity."""

    def __init__(
        self,
        redis_client: aioredis.Redis,
        embedder: Embedder,
        threshold: float | None = None,
        ttl_seconds: int | None = None,
    ):
        self.redis = redis_client
        self.embedder = embedder
        self.threshold = threshold or settings.semantic_cache_threshold
        self.ttl = ttl_seconds or settings.semantic_cache_ttl_seconds

    async def lookup(self, query: str) -> CacheHit | None:
        """Return a cache hit if a similar query is cached, else None."""
        query_vec = self.embedder.embed_query_dense(query)
        query_vec = np.asarray(query_vec, dtype=np.float32)

        keys = await self._recent_keys()
        if not keys:
            return None

        raw_entries = await self.redis.mget(keys)

        best: CacheHit | None = None
        best_sim = -1.0

        for raw in raw_entries:
            if raw is None:
                continue
            entry = json.loads(raw)
            cached_vec = np.asarray(entry["embedding"], dtype=np.float32)
            sim = self._cosine(query_vec, cached_vec)

            if sim >= self.threshold and sim > best_sim:
                best_sim = sim
                best = CacheHit(
                    query=entry["query"],
                    answer=entry["answer"],
                    citations=entry.get("citations", []),
                    similarity=sim,
                    cached_at=entry["cached_at"],
                )

        if best:
            log.info(
                "Cache HIT (similarity=%.3f) for query: %r",
                best.similarity,
                query[:80],
            )
        return best

    async def store(
        self,
        query: str,
        answer: str,
        citations: list[dict] | None = None,
    ) -> None:
        """Cache the answer for this query. Embeds query for future lookups."""
        query_vec = self.embedder.embed_query_dense(query)

        entry = {
            "query": query,
            "answer": answer,
            "citations": citations or [],
            "embedding": query_vec.tolist(),
            "cached_at": time.time(),
        }

        key = f"{CACHE_KEY_PREFIX}{time.time_ns()}"
        await self.redis.set(key, json.dumps(entry), ex=self.ttl)

        log.info("Cache STORE for query: %r", query[:80])

    async def _recent_keys(self) -> list[str]:
        """Get up to MAX_SCAN_ENTRIES most recent cache keys."""
        keys: list[str] = []
        async for key in self.redis.scan_iter(match=f"{CACHE_KEY_PREFIX}*", count=200):
            keys.append(key)
            if len(keys) >= MAX_SCAN_ENTRIES * 2:
                break
        keys.sort(reverse=True)
        return keys[:MAX_SCAN_ENTRIES]

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors. Returns value in [-1, 1]."""
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-10
        return float(np.dot(a, b) / denom)
