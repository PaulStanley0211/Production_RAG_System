"""Content filter — validates retrieved chunks before they reach the generator.

The threat: doc poisoning. An attacker plants a chunk that contains
instructions ("when asked X, say Y"). Naive RAG happily reads the chunk as
context and follows the embedded instructions because they look like
authoritative content.

The defense: scan retrieved chunks for instruction-like patterns and redact
them before generation. The LLM still sees the chunk's main content, just
without the manipulative parts.

Two modes:
- strict   → aggressive redaction, more false positives
- balanced → only strong signals (default)
"""

import logging
import re
from dataclasses import dataclass

from app.retrieval.reranker import RerankedResult

log = logging.getLogger(__name__)

REDACTION_MARKER = "[content removed: suspicious instruction]"


# ============================================================
# Suspicious patterns — instruction-like content in retrieved docs
# ============================================================
# These mirror the injection patterns from the input guard, but tuned for
# false-positive minimization. Retrieved chunks contain ALL kinds of
# legitimate text; we want to be conservative about what we redact.

STRONG_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "directive_to_assistant",
        # Direct addressing of "the assistant" or "the AI" with instructions
        re.compile(
            r"\b(?:assistant|the\s+ai|the\s+model|chatbot)\s*[,:]\s*(?:always|never|when|if)\b.{1,200}",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "instruction_override_in_doc",
        # The same override patterns the input guard catches, but inside a doc
        re.compile(
            r"\b(?:ignore|disregard|forget)\b.{0,30}\b(?:previous|above|all)\b.{0,30}\b(?:instructions?|prompts?|rules?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "delimiter_injection",
        re.compile(
            r"<\|(?:im_start|im_end|system|assistant|user|endoftext)\|>",
            re.IGNORECASE,
        ),
    ),
    (
        "always_say",
        # "Always say X when Y" style manipulation
        re.compile(
            r"\b(?:always|never)\s+(?:say|respond|answer|reply|claim|tell)\b.{1,200}",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
]

# Strict mode adds these — high false-positive risk on legitimate technical docs
STRICT_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "system_prompt_phrase",
        re.compile(
            r"\b(?:system\s+prompt|initial\s+instructions|hidden\s+prompt)\b",
            re.IGNORECASE,
        ),
    ),
]


@dataclass
class FilterResult:
    """Outcome of filtering a single chunk."""

    sanitized_content: str
    redactions: list[str]  # pattern names that triggered redaction


class ContentFilter:
    """Strips instruction-like patterns from retrieved chunks."""

    def __init__(self, mode: str = "balanced"):
        if mode not in {"balanced", "strict"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        self.mode = mode
        self.patterns = list(STRONG_PATTERNS)
        if mode == "strict":
            self.patterns.extend(STRICT_PATTERNS)

    def filter_chunk_text(self, content: str) -> FilterResult:
        """Sanitize a single chunk's text. Returns clean text + applied flags."""
        sanitized = content
        redactions: list[str] = []

        for name, pattern in self.patterns:
            if pattern.search(sanitized):
                sanitized = pattern.sub(REDACTION_MARKER, sanitized)
                redactions.append(name)

        return FilterResult(sanitized_content=sanitized, redactions=redactions)

    def filter_chunks(self, chunks: list[RerankedResult]) -> list[RerankedResult]:
        """Apply filtering to every chunk's content payload. Returns same list."""
        if not chunks:
            return chunks

        total_redacted = 0
        for chunk in chunks:
            original = chunk.point.payload.get("content", "")
            if not original:
                continue
            result = self.filter_chunk_text(original)
            if result.redactions:
                # Mutate the payload in place — the LLM sees the sanitized version
                chunk.point.payload["content"] = result.sanitized_content
                chunk.point.payload["_filter_flags"] = result.redactions
                total_redacted += 1
                log.warning(
                    "ContentFilter redacted chunk %s (flags=%s, source=%s)",
                    str(chunk.point.id)[:16],
                    result.redactions,
                    chunk.point.payload.get("source", "?"),
                )

        if total_redacted:
            log.info(
                "ContentFilter: %d/%d chunks had redactions",
                total_redacted, len(chunks),
            )
        return chunks