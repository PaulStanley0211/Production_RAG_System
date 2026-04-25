"""Conversation memory + query rewriting.

Two responsibilities:
1. Store recent conversation turns per session in Redis.
2. Rewrite follow-up queries to be self-contained before retrieval.

The rewriting step is critical for multi-turn quality. Without it,
follow-ups like "what about its features?" retrieve garbage because
the retriever has no idea what "it" refers to.
"""

import json
import logging
import re
from dataclasses import dataclass

import redis.asyncio as aioredis
from anthropic import AsyncAnthropic

from app.config import settings
from app.prompts import PromptName, registry

log = logging.getLogger(__name__)

# Redis key prefix for conversation lists
CONV_KEY_PREFIX = "rag:conv:"

# Sliding window — keep at most this many recent turns
MAX_HISTORY_TURNS = 10

# Conversation TTL — expire after 24 hours of inactivity
CONV_TTL_SECONDS = 86400

# Regex for parsing the rewriter's structured output
RE_OUTPUT_TAG = re.compile(r"<output>(.*?)</output>", re.DOTALL)


@dataclass
class Turn:
    """One exchange in a conversation."""

    user_query: str
    assistant_answer: str


class ConversationMemory:
    """Maintains conversation state and rewrites follow-up queries."""

    def __init__(
        self,
        redis_client: aioredis.Redis,
        anthropic_client: AsyncAnthropic,
        model: str | None = None,
    ):
        self.redis = redis_client
        self.anthropic = anthropic_client
        self.model = model or settings.model_routing  # Haiku — cheap+fast

    async def get_history(self, conversation_id: str) -> list[Turn]:
        """Return recent turns for a conversation, oldest first."""
        key = self._key(conversation_id)
        # LRANGE 0 -1 returns all elements; we trim to MAX_HISTORY_TURNS on write
        raw = await self.redis.lrange(key, 0, MAX_HISTORY_TURNS - 1)
        # Stored newest-first; reverse so oldest comes first
        turns = [self._deserialize(r) for r in reversed(raw)]
        return turns

    async def append(
        self,
        conversation_id: str,
        user_query: str,
        assistant_answer: str,
    ) -> None:
        """Add a turn to the conversation. Trims to MAX_HISTORY_TURNS."""
        key = self._key(conversation_id)
        turn_json = json.dumps({
            "user_query": user_query,
            "assistant_answer": assistant_answer,
        })

        # LPUSH adds to head; LTRIM keeps only the last N entries
        async with self.redis.pipeline(transaction=True) as pipe:
            await pipe.lpush(key, turn_json)
            await pipe.ltrim(key, 0, MAX_HISTORY_TURNS - 1)
            await pipe.expire(key, CONV_TTL_SECONDS)
            await pipe.execute()

    async def rewrite_query(
        self,
        query: str,
        conversation_id: str | None,
    ) -> str:
        """Return a self-contained version of `query` using conversation history.

        If there's no history (or no conversation_id), returns the query unchanged.
        """
        if not conversation_id:
            return query

        history = await self.get_history(conversation_id)
        if not history:
            return query

        history_text = self._format_history(history)
        prompt = registry.get(PromptName.MEMORY_REWRITER).format(
            history=history_text,
            query=query,
        )

        response = await self.anthropic.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_output = response.content[0].text

        rewritten = self._parse_output(raw_output, fallback=query)
        if rewritten != query:
            log.info("Rewrote query: %r → %r", query[:60], rewritten[:60])
        return rewritten

    # -------- Internal helpers --------

    def _key(self, conversation_id: str) -> str:
        return f"{CONV_KEY_PREFIX}{conversation_id}"

    def _deserialize(self, raw: str) -> Turn:
        data = json.loads(raw)
        return Turn(
            user_query=data["user_query"],
            assistant_answer=data["assistant_answer"],
        )

    def _format_history(self, turns: list[Turn]) -> str:
        """Format turns as 'User: ... / Assistant: ...' lines."""
        lines: list[str] = []
        for t in turns:
            lines.append(f"User: {t.user_query}")
            lines.append(f"Assistant: {t.assistant_answer}")
        return "\n".join(lines)

    def _parse_output(self, raw: str, fallback: str) -> str:
        """Extract content from <output>...</output>. Fall back if missing."""
        match = RE_OUTPUT_TAG.search(raw)
        if match:
            return match.group(1).strip()
        log.warning("MEMORY_REWRITER output had no <output> tags: %r", raw[:200])
        return fallback