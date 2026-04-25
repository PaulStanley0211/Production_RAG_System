"""Adaptive router — decides next action based on grading verdict.

The CRAG loop calls this after grading. The router looks at the distribution
of relevant / partially_relevant / irrelevant grades and picks one of:

    GENERATE   — we have at least one relevant chunk; proceed to generation
    DECOMPOSE  — only partial matches; break the query into sub-queries
    FALLBACK   — nothing usable; try web search if enabled
    REFUSE     — nothing usable and no fallback; return graceful refusal

This file is small by design. Keep the decision logic isolated so it's
trivial to test and tune without touching the agent loop.
"""

import logging
from dataclasses import dataclass
from enum import Enum

from app.config import settings
from app.services.document_grader import Grade, GradedChunk

log = logging.getLogger(__name__)


class Action(str, Enum):
    """Next step the CRAG loop should take."""

    GENERATE = "generate"
    DECOMPOSE = "decompose"
    FALLBACK = "fallback"
    REFUSE = "refuse"


@dataclass
class Decision:
    """The router's choice plus the reason for traceability."""

    action: Action
    reason: str
    relevant_count: int
    partial_count: int
    irrelevant_count: int


class AdaptiveRouter:
    """Maps grading outcomes to next-action decisions."""

    def __init__(
        self,
        web_fallback_enabled: bool | None = None,
    ):
        self.web_fallback_enabled = (
            web_fallback_enabled
            if web_fallback_enabled is not None
            else settings.enable_web_fallback
        )

    def decide(
        self,
        graded_chunks: list[GradedChunk],
        iteration: int = 0,
    ) -> Decision:
        """Return the next action based on the grade distribution.

        `iteration` is the current CRAG loop iteration. After max iterations,
        DECOMPOSE is no longer offered — we fall back or refuse instead.
        """
        relevant = sum(1 for c in graded_chunks if c.grade == Grade.RELEVANT)
        partial = sum(1 for c in graded_chunks if c.grade == Grade.PARTIALLY_RELEVANT)
        irrelevant = sum(1 for c in graded_chunks if c.grade == Grade.IRRELEVANT)

        # Have at least one fully relevant chunk → generate
        if relevant >= 1:
            return self._build(
                Action.GENERATE,
                f"{relevant} relevant chunk(s) found",
                relevant, partial, irrelevant,
            )

        # Have only partials → try decomposition (unless we already retried)
        if partial >= 1:
            if iteration < settings.crag_max_iterations:
                return self._build(
                    Action.DECOMPOSE,
                    f"{partial} partial match(es), retrying with decomposition",
                    relevant, partial, irrelevant,
                )
            # Exhausted retries with only partials — accept what we have
            return self._build(
                Action.GENERATE,
                f"max iterations reached, generating from {partial} partial(s)",
                relevant, partial, irrelevant,
            )

        # Nothing useful → fall back to web if enabled, else refuse
        if self.web_fallback_enabled and iteration < settings.crag_max_iterations:
            return self._build(
                Action.FALLBACK,
                "no relevant chunks; trying web search",
                relevant, partial, irrelevant,
            )
        return self._build(
            Action.REFUSE,
            "no relevant chunks; no fallback available",
            relevant, partial, irrelevant,
        )

    @staticmethod
    def _build(
        action: Action,
        reason: str,
        relevant: int,
        partial: int,
        irrelevant: int,
    ) -> Decision:
        decision = Decision(
            action=action,
            reason=reason,
            relevant_count=relevant,
            partial_count=partial,
            irrelevant_count=irrelevant,
        )
        log.info(
            "Adaptive router: action=%s reason=%r (R=%d, P=%d, I=%d)",
            action.value, reason, relevant, partial, irrelevant,
        )
        return decision