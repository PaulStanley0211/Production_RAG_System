"""Query router — classifies user intent into a category.

Routes determine which pipeline path to take:
    factual      → standard RAG (retrieve + ground + answer)
    comparative  → decompose into sub-queries, retrieve each
    chitchat     → skip retrieval, brief conversational reply
    no_retrieval → skip retrieval, generic LLM answer

Classification runs on every query, so it must be cheap. We use Haiku.
On parse failure, fall back to 'factual' — the safe default.
"""

import logging
import re
from enum import Enum

from anthropic import AsyncAnthropic

from app.config import settings
from prompts import PromptName, registry

log = logging.getLogger(__name__)

# Regex for parsing structured <output> response
RE_OUTPUT_TAG = re.compile(r"<output>(.*?)</output>", re.DOTALL | re.IGNORECASE)


class Intent(str, Enum):
    """All valid query intents. Pipeline behavior is keyed off these."""

    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    CHITCHAT = "chitchat"
    NO_RETRIEVAL = "no_retrieval"


# Set of valid intent strings — used to validate parsed output
VALID_INTENTS = {i.value for i in Intent}


class QueryRouter:
    """Classifies a user query into an Intent."""

    def __init__(
        self,
        anthropic_client: AsyncAnthropic,
        model: str | None = None,
    ):
        self.anthropic = anthropic_client
        self.model = model or settings.model_routing

    async def classify(self, query: str) -> Intent:
        """Return the best Intent for the query. Defaults to FACTUAL on failure."""
        prompt = registry.get(PromptName.QUERY_ROUTER).format(query=query)

        response = await self.anthropic.messages.create(
            model=self.model,
            max_tokens=20,  # one word is plenty
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text

        intent = self._parse(raw)
        log.info("Routed: %r → %s", query[:80], intent.value)
        return intent

    def _parse(self, raw: str) -> Intent:
        """Extract intent from <output> tags; fall back to FACTUAL on failure."""
        match = RE_OUTPUT_TAG.search(raw)
        candidate = (match.group(1) if match else raw).strip().lower()

        # Direct match against valid intents
        if candidate in VALID_INTENTS:
            return Intent(candidate)

        # Tolerate small variants ("factual question", "FACTUAL", etc.)
        for valid in VALID_INTENTS:
            if valid in candidate:
                return Intent(valid)

        log.warning("Router output unrecognized, defaulting to FACTUAL: %r", raw[:200])
        return Intent.FACTUAL