"""Web search tool — Anthropic's built-in web_search as a CRAG fallback.

When indexed retrieval fails (CRAG verdict: INCORRECT), the agent can
optionally fall back to web search. Returns results in the same ToolResult
shape as VectorSearchTool so the agent doesn't care which tool ran.

Web search costs more than vector retrieval, so this tool is opt-in via
config. See app/config.py for the enable flag (added in Phase 5).
"""

import logging
from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic

from app.agents.tools.vector_search import ToolResult
from app.config import settings
from app.retrieval.reranker import RerankedResult

log = logging.getLogger(__name__)

# Max number of web results to return back to the pipeline as "chunks"
DEFAULT_TOP_K = 5

# Anthropic web search tool spec — see docs.claude.com for current version
WEB_SEARCH_TOOL_SPEC = {
    "type": "web_search_20250305",
    "name": "web_search",
}


@dataclass
class _SyntheticPoint:
    """Mimics a Qdrant ScoredPoint enough that downstream code works.

    Web results don't come from Qdrant, but the rest of the pipeline
    expects RerankedResult objects with .point.payload. We synthesize
    a payload-shaped dict so the citation extractor and grader work
    without special-casing web results.
    """

    id: str
    payload: dict[str, Any]


class WebSearchTool:
    """Web search via Anthropic's built-in web_search tool."""

    name = "web_search"
    description = "Search the public web when the indexed corpus has no relevant info."

    def __init__(
        self,
        anthropic_client: AsyncAnthropic,
        model: str | None = None,
    ):
        self.anthropic = anthropic_client
        # Use the cheap routing model for the search dispatch — accuracy
        # comes from search quality, not model size
        self.model = model or settings.model_routing

    async def call(self, query: str, top_k: int = DEFAULT_TOP_K) -> ToolResult:
        """Run a web search; return up to top_k results as RerankedResult."""
        log.info("WebSearchTool: searching for %r", query[:80])

        try:
            response = await self.anthropic.messages.create(
                model=self.model,
                max_tokens=2048,
                tools=[WEB_SEARCH_TOOL_SPEC],
                messages=[{"role": "user", "content": query}],
            )
        except Exception as e:
            log.warning("Web search failed: %s — returning empty result", e)
            return ToolResult(tool_name=self.name, chunks=[])

        chunks = self._extract_results(response, top_k)
        log.info("WebSearchTool: returned %d results", len(chunks))
        return ToolResult(tool_name=self.name, chunks=chunks)

    def _extract_results(self, response, top_k: int) -> list[RerankedResult]:
        """Parse Anthropic's web search response into RerankedResult chunks."""
        results: list[RerankedResult] = []

        # The response contains content blocks. Web search results show up
        # as "web_search_tool_result" blocks (or as embedded citations in text).
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type != "web_search_tool_result":
                continue

            search_results = getattr(block, "content", []) or []
            for idx, sr in enumerate(search_results):
                if len(results) >= top_k:
                    break
                title = getattr(sr, "title", "") or ""
                url = getattr(sr, "url", "") or ""
                snippet = getattr(sr, "encrypted_content", "") or ""

                # Truncate very long snippets — they're already retrieved+ranked
                content = self._format_result(title, url, snippet)

                point = _SyntheticPoint(
                    id=f"web::{url or idx}",
                    payload={
                        "content": content,
                        "source": url or "web",
                        "doc_id": f"web::{url or idx}",
                        "title": title,
                        "from_tool": self.name,
                    },
                )
                results.append(
                    RerankedResult(
                        point=point,
                        rrf_score=0.0,
                        rerank_score=0.0,  # web results don't have rerank scores
                        dense_rank=None,
                        sparse_rank=None,
                    )
                )

        return results

    @staticmethod
    def _format_result(title: str, url: str, snippet: str) -> str:
        """Format a web result as a chunk-shaped text block."""
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if url:
            parts.append(f"URL: {url}")
        if snippet:
            parts.append(snippet[:1500])
        return "\n".join(parts)