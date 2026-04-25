"""Vector search tool — wraps hybrid retrieval + reranking behind a tool interface.

The CRAG agent treats every retrieval source (vector store, web search) as
a "tool" with the same input/output shape:
    input:  query string
    output: list of RerankedResult

This uniform shape lets the adaptive router pick a tool by name without
caring which underlying machinery it dispatches to.
"""

import logging
from dataclasses import dataclass

from app.retrieval.hybrid_retrieval import HybridRetriever
from app.retrieval.reranker import RerankedResult, Reranker

log = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Standardized output from any retrieval tool."""

    tool_name: str
    chunks: list[RerankedResult]


class VectorSearchTool:
    """Wraps HybridRetriever + Reranker as a single 'tool' the agent can call."""

    name = "vector_search"
    description = "Retrieve relevant chunks from the indexed document corpus."

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker,
    ):
        self.retriever = retriever
        self.reranker = reranker

    async def call(self, query: str, top_k: int = 5) -> ToolResult:
        """Retrieve + rerank for the given query."""
        log.info("VectorSearchTool: retrieving for %r", query[:80])

        candidates = await self.retriever.retrieve(query)
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)

        log.info("VectorSearchTool: returned %d chunks", len(reranked))
        return ToolResult(tool_name=self.name, chunks=reranked)