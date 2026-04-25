"""Prompt registry — single import location for all prompts.

Services import from `app.prompts` rather than from the underlying files.
This decouples callers from how prompts are organized internally.

Usage:
    from app.prompts import registry, PromptName

    template = registry.get(PromptName.FACTUAL_QA)
    filled = template.format(query=q, context=c)
"""

from enum import Enum

from prompts import grading, templates


class PromptName(str, Enum):
    """All available prompts. Add new ones here when registering."""

    # Generation
    FACTUAL_QA = "factual_qa"
    REFUSAL = "refusal"

    # Grading / decisions
    QUERY_ROUTER = "query_router"
    QUERY_DECOMPOSER = "query_decomposer"
    DOCUMENT_GRADER = "document_grader"
    MEMORY_REWRITER = "memory_rewriter"


# Mapping from name → current default template string
_REGISTRY: dict[PromptName, str] = {
    PromptName.FACTUAL_QA: templates.FACTUAL_QA,
    PromptName.REFUSAL: templates.REFUSAL,
    PromptName.QUERY_ROUTER: grading.QUERY_ROUTER,
    PromptName.QUERY_DECOMPOSER: grading.QUERY_DECOMPOSER,
    PromptName.DOCUMENT_GRADER: grading.DOCUMENT_GRADER,
    PromptName.MEMORY_REWRITER: grading.MEMORY_REWRITER,
}


class PromptRegistry:
    """Lookup-by-name access to all prompts."""

    def get(self, name: PromptName) -> str:
        """Return the current default template for the given name."""
        if name not in _REGISTRY:
            raise KeyError(f"No prompt registered for {name!r}")
        return _REGISTRY[name]

    def list_all(self) -> list[PromptName]:
        """Useful for diagnostic endpoints."""
        return list(_REGISTRY.keys())


# Module-level singleton — import this everywhere
registry = PromptRegistry()