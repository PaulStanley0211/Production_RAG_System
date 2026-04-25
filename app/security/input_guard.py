"""Input guard — detect prompt injection attempts before they reach the pipeline.

Two-tier model:
- Tier 1: regex patterns for known injection signatures (implemented here)
- Tier 2: LLM-based classification for borderline cases (future enhancement)

Tier 1 catches 90%+ of real-world injection attempts at zero cost. Tier 2
adds defense against novel/obfuscated attacks at the cost of a Haiku call.

Configurable behavior:
- `guard_deny_on_injection=True`  → reject the query, return refusal
- `guard_deny_on_injection=False` → log + flag, but let it through
"""

import logging
import re
from dataclasses import dataclass

from app.config import settings

log = logging.getLogger(__name__)


# ============================================================
# Injection patterns
# ============================================================
# Each pattern represents a known injection technique. Keep the regexes
# tight — false positives on legitimate queries are very annoying.

INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|above|prior|earlier|all)\b.{0,30}\b(instructions?|prompts?|rules?|directions?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "role_jailbreak",
        re.compile(
            r"\b(you are|act as|pretend to be|roleplay as|simulate)\b.{0,80}\b(dan|developer mode|jailbroken|unrestricted|no rules|no restrictions|evil)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "system_prompt_extraction",
        re.compile(
            r"\b(reveal|show|print|output|repeat|tell me|what (is|are|were))\b.{0,30}\b(your|the)\b.{0,30}\b(system prompt|initial instructions|original instructions|hidden prompt|starter prompt)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "delimiter_injection",
        re.compile(
            r"<\|(im_start|im_end|system|assistant|user|endoftext)\|>",
            re.IGNORECASE,
        ),
    ),
    (
        "fake_tool_response",
        re.compile(
            r"\b(tool[_\s]result|function[_\s]response|assistant[_\s]message)\s*[:=]",
            re.IGNORECASE,
        ),
    ),
    (
        "exfiltration_request",
        re.compile(
            r"\b(send|post|exfiltrate|leak|transmit)\b.{0,50}\b(api[_\s]?key|secret|token|password|credentials)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
]

# Heuristic flags — suspicious but not necessarily malicious
SUSPICIOUS_HEURISTICS: list[tuple[str, re.Pattern]] = [
    (
        "excessive_length",
        # Very long queries are sometimes used to push real instructions out
        # of the model's attention. We flag at 3000 chars, separately from blocking.
        re.compile(r".{3000,}", re.DOTALL),
    ),
    (
        "encoded_payload",
        # Base64-looking payloads (40+ chars of base64 alphabet)
        re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),
    ),
]


@dataclass
class GuardCheckResult:
    """Outcome of an input guard check."""

    passed: bool
    sanitized_text: str
    reason: str | None = None
    flags: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.flags is None:
            self.flags = []


class InputGuard:
    """Detects prompt injection in user queries."""

    def __init__(self, deny_on_match: bool | None = None):
        self.deny_on_match = (
            deny_on_match
            if deny_on_match is not None
            else settings.guard_deny_on_injection
        )

    async def check(self, query: str) -> GuardCheckResult:
        """Scan a query. Block if injection patterns match (when deny enabled)."""
        flags: list[str] = []
        injection_match: str | None = None

        # Tier 1: hard injection patterns
        for name, pattern in INJECTION_PATTERNS:
            if pattern.search(query):
                flags.append(name)
                if injection_match is None:
                    injection_match = name

        # Heuristics — flag but don't block on these alone
        for name, pattern in SUSPICIOUS_HEURISTICS:
            if pattern.search(query):
                flags.append(name)

        # Decide
        if injection_match and self.deny_on_match:
            log.warning(
                "InputGuard BLOCKED query (matched=%s, all_flags=%s): %r",
                injection_match,
                flags,
                query[:200],
            )
            return GuardCheckResult(
                passed=False,
                sanitized_text="",
                reason=f"Query blocked by input guard ({injection_match})",
                flags=flags,
            )

        if flags:
            log.info("InputGuard flags (passed): %s for query %r", flags, query[:80])

        return GuardCheckResult(
            passed=True,
            sanitized_text=query,
            flags=flags,
        )