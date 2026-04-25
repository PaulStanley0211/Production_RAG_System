"""Document grader — per-chunk relevance scoring before generation.

Each retrieved chunk gets graded by Haiku as relevant / partially_relevant /
irrelevant. Filters out chunks that don't actually address the query.

Why this exists despite the reranker (Phase 3):
- Reranker = statistical similarity score, doesn't *understand* the query.
- Grader = LLM reads chunk + query together, makes a categorical judgment.
- Catches keyword-matched-but-off-topic chunks the reranker rewards.

The three-way categorization (not just relevant/irrelevant) drives the
CRAG self-correction loop in Phase 5. Phase 4 just filters by it.

Grading runs concurrently across chunks via asyncio.gather — single LLM
call per chunk, but all in parallel.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from enum import Enum

from anthropic import AsyncAnthropic

from app.config import settings
from app.prompts import PromptName, registry
from app.retrieval.reranker import RerankedResult

log = logging.getLogger(__name__)

# Regex for parsing structured <output> response
RE_OUTPUT_TAG = re.compile(r"<output>(.*?)</output>", re.DOTALL | re.IGNORECASE)

# Cap chunk text sent to grader — long chunks waste tokens on classification
MAX_CHUNK_CHARS = 1500


class Grade(str, Enum):
    """Grader verdicts."""

    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    IRRELEVANT = "irrelevant"


VALID_GRADES = {g.value for g in Grade}


@dataclass
class GradedChunk:
    """Reranked chunk + its relevance grade."""

    point: object  # ScoredPoint from Qdrant
    rrf_score: float
    rerank_score: float
    grade: Grade


class DocumentGrader:
    """Score retrieved chunks against the user's query."""

    def __init__(
        self,
        anthropic_client: AsyncAnthropic,
        model: str | None = None,
    ):
        self.anthropic = anthropic_client
        self.model = model or settings.model_routing

    async def grade(
        self,
        query: str,
        chunks: list[RerankedResult],
    ) -> list[GradedChunk]:
        """Grade each chunk concurrently. Returns same order as input."""
        if not chunks:
            return []

        # Run all grading calls in parallel
        grades = await asyncio.gather(
            *(self._grade_one(query, c) for c in chunks),
            return_exceptions=False,
        )

        graded = [
            GradedChunk(
                point=c.point,
                rrf_score=c.rrf_score,
                rerank_score=c.rerank_score,
                grade=g,
            )
            for c, g in zip(chunks, grades)
        ]

        relevant_count = sum(1 for g in graded if g.grade == Grade.RELEVANT)
        log.info(
            "Graded %d chunks: %d relevant, %d partial, %d irrelevant",
            len(graded),
            relevant_count,
            sum(1 for g in graded if g.grade == Grade.PARTIALLY_RELEVANT),
            sum(1 for g in graded if g.grade == Grade.IRRELEVANT),
        )
        return graded

    @staticmethod
    def filter_relevant(graded: list[GradedChunk]) -> list[GradedChunk]:
        """Keep only relevant + partially_relevant chunks for generation."""
        return [g for g in graded if g.grade != Grade.IRRELEVANT]

    # -------- Internal helpers --------

    async def _grade_one(self, query: str, chunk: RerankedResult) -> Grade:
        """Single chunk grading. Defaults to RELEVANT on parse failure."""
        chunk_text = (chunk.point.payload.get("content") or "")[:MAX_CHUNK_CHARS]
        prompt = registry.get(PromptName.DOCUMENT_GRADER).format(
            query=query,
            chunk=chunk_text,
        )

        try:
            response = await self.anthropic.messages.create(
                model=self.model,
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            log.warning("Grader API call failed: %s — defaulting to RELEVANT", e)
            return Grade.RELEVANT

        raw = response.content[0].text
        return self._parse(raw)

    def _parse(self, raw: str) -> Grade:
        match = RE_OUTPUT_TAG.search(raw)
        candidate = (match.group(1) if match else raw).strip().lower()

        if candidate in VALID_GRADES:
            return Grade(candidate)
        for valid in VALID_GRADES:
            if valid in candidate:
                return Grade(valid)

        log.warning("Grader output unrecognized, defaulting to RELEVANT: %r", raw[:200])
        return Grade.RELEVANT