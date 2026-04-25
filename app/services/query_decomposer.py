"""Query decomposer — break complex queries into focused sub-queries.

Used for comparative or multi-part queries (router classifies as 'comparative').
Each sub-query is retrieved independently; the pipeline merges results.

For simple queries, decomposition returns a single-item list — the prompt
explicitly avoids over-decomposition.
"""

import json
import logging
import re

from anthropic import AsyncAnthropic

from app.config import settings
from app.prompts import PromptName, registry

log = logging.getLogger(__name__)

# Regex for parsing structured <output> response
RE_OUTPUT_TAG = re.compile(r"<output>(.*?)</output>", re.DOTALL)

# Sanity bounds — over-decomposition is wasteful, under-decomposition defeats the point
MIN_SUBQUERIES = 1
MAX_SUBQUERIES = 5


class QueryDecomposer:
    """Decompose complex queries into focused sub-queries."""

    def __init__(
        self,
        anthropic_client: AsyncAnthropic,
        model: str | None = None,
    ):
        self.anthropic = anthropic_client
        self.model = model or settings.model_routing  # Haiku — cheap structured task

    async def decompose(self, query: str) -> list[str]:
        """Return 1-N sub-queries. Falls back to [query] on any parse issue."""
        prompt = registry.get(PromptName.QUERY_DECOMPOSER).format(query=query)

        response = await self.anthropic.messages.create(
            model=self.model,
            max_tokens=400,  # room for ~4 sub-queries
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text

        sub_queries = self._parse(raw, fallback=[query])

        log.info(
            "Decomposed: %r → %d sub-queries",
            query[:80],
            len(sub_queries),
        )
        return sub_queries

    def _parse(self, raw: str, fallback: list[str]) -> list[str]:
        """Extract a JSON list from <output> tags. Robust to common LLM mistakes."""
        match = RE_OUTPUT_TAG.search(raw)
        if not match:
            log.warning("Decomposer output had no <output> tags: %r", raw[:200])
            return fallback

        json_str = match.group(1).strip()
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            log.warning("Decomposer JSON parse failed: %s | raw=%r", e, json_str[:200])
            return fallback

        # Validate shape: list of non-empty strings
        if not isinstance(parsed, list) or not parsed:
            log.warning("Decomposer didn't return a non-empty list: %r", parsed)
            return fallback

        sub_queries = [s.strip() for s in parsed if isinstance(s, str) and s.strip()]
        if not sub_queries:
            return fallback

        # Clamp to sane bounds
        if len(sub_queries) > MAX_SUBQUERIES:
            log.info("Truncating %d sub-queries to %d", len(sub_queries), MAX_SUBQUERIES)
            sub_queries = sub_queries[:MAX_SUBQUERIES]

        return sub_queries