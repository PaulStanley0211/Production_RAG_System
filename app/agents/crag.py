"""CRAG agent — self-correcting retrieval loop.

Orchestrates retrieve → grade → decide → maybe-retry. Each iteration may
add new chunks (from decomposition or web fallback) and re-grade.

The loop terminates when:
- The router says GENERATE or REFUSE (we have an answer or admit defeat)
- We hit settings.crag_max_iterations
"""

import logging
from dataclasses import dataclass

from app.agents.adaptive_router import Action, AdaptiveRouter, Decision
from app.agents.tools.vector_search import VectorSearchTool
from app.agents.tools.web_search import WebSearchTool
from app.services.document_grader import DocumentGrader, GradedChunk
from app.services.query_decomposer import QueryDecomposer

log = logging.getLogger(__name__)


@dataclass
class CRAGResult:
    """Final output of one CRAG run."""

    graded_chunks: list[GradedChunk]
    final_decision: Decision
    iterations: int
    sub_queries_used: list[str]
    tools_used: list[str]


class CRAGAgent:
    """Self-correcting retrieval orchestrator."""

    def __init__(
        self,
        vector_tool: VectorSearchTool,
        web_tool: WebSearchTool,
        decomposer: QueryDecomposer,
        grader: DocumentGrader,
        adaptive_router: AdaptiveRouter,
    ):
        self.vector_tool = vector_tool
        self.web_tool = web_tool
        self.decomposer = decomposer
        self.grader = grader
        self.adaptive_router = adaptive_router

    async def run(self, query: str) -> CRAGResult:
        """Run the CRAG loop until we GENERATE, REFUSE, or hit the iteration cap."""
        log.info("CRAG: starting for query %r", query[:80])

        # --- Iteration 0: initial vector retrieval + grade ---
        initial = await self.vector_tool.call(query)
        graded = await self.grader.grade(query, initial.chunks)

        sub_queries_used: list[str] = []
        tools_used: list[str] = [self.vector_tool.name]
        iteration = 0

        # --- Loop ---
        while True:
            decision = self.adaptive_router.decide(graded, iteration=iteration)

            if decision.action in (Action.GENERATE, Action.REFUSE):
                # Terminal — return whatever we have
                break

            if decision.action == Action.DECOMPOSE:
                # Break query into focused sub-queries; retrieve each
                sub_queries = await self.decomposer.decompose(query)
                sub_queries_used.extend(sub_queries)

                new_chunks = []
                for sq in sub_queries:
                    sub_result = await self.vector_tool.call(sq)
                    new_chunks.extend(sub_result.chunks)

                # Merge with existing chunks, dedup by point id
                all_chunks = self._merge_unique(
                    [g for g in graded] , new_chunks,
                )
                # Re-grade the combined set against the original query
                graded = await self.grader.grade(query, all_chunks)
                iteration += 1
                continue

            if decision.action == Action.FALLBACK:
                # External source (web search)
                tools_used.append(self.web_tool.name)
                web_result = await self.web_tool.call(query)

                all_chunks = self._merge_unique(
                    [g for g in graded], web_result.chunks,
                )
                graded = await self.grader.grade(query, all_chunks)
                iteration += 1
                continue

            # Defensive: unknown action shouldn't happen
            log.warning("CRAG: unexpected action %r, breaking", decision.action)
            break

        log.info(
            "CRAG: done in %d iteration(s), final action=%s",
            iteration, decision.action.value,
        )
        return CRAGResult(
            graded_chunks=graded,
            final_decision=decision,
            iterations=iteration,
            sub_queries_used=sub_queries_used,
            tools_used=tools_used,
        )

    @staticmethod
    def _merge_unique(existing_graded, new_chunks):
        """Combine chunks, deduplicating by point id.

        `existing_graded` is list[GradedChunk]; we use their .point.
        `new_chunks` is list[RerankedResult] from a tool call.
        Returns a flat list[RerankedResult] — re-graded by caller.
        """
        seen_ids: set[str] = set()
        merged = []

        # Existing chunks come from a previous grade — wrap their .point + scores
        # back into RerankedResult shape for the grader's input
        from app.retrieval.reranker import RerankedResult

        for g in existing_graded:
            pid = str(g.point.id)
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            merged.append(
                RerankedResult(
                    point=g.point,
                    rrf_score=g.rrf_score,
                    rerank_score=g.rerank_score,
                    dense_rank=None,
                    sparse_rank=None,
                )
            )

        for c in new_chunks:
            pid = str(c.point.id)
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            merged.append(c)

        return merged