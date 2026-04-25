"""Output guard — redacts PII from generated answers before they leave the system.

Final defense layer. Runs on every generated answer (not on cached answers,
which were already guarded when first generated).

Detects and redacts:
- Email addresses
- Phone numbers (US + international common formats)
- Credit card numbers (Luhn-validated to suppress false positives)
- SSN-like patterns
- IP addresses

When `settings.guard_pii_redaction` is False, the guard scans + flags
without modifying the answer — useful for testing what would have been
redacted before flipping the switch.
"""

import logging
import re
from dataclasses import dataclass, field

from app.config import settings

log = logging.getLogger(__name__)


# ============================================================
# PII patterns
# ============================================================
# Regex tuned for high precision — false positives degrade UX more than
# false negatives degrade safety (assuming this is layered with other controls).

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# US-style phone numbers (optionally with country code) and common international
PHONE_PATTERN = re.compile(
    r"""
    (?:
        \+?\d{1,3}[\s.-]?  # optional country code
    )?
    \(?\d{3}\)?[\s.-]?     # area code
    \d{3}[\s.-]?           # prefix
    \d{4}                  # line
    \b
    """,
    re.VERBOSE,
)

# 13-19 digit sequences with optional separators — Luhn-checked below
CC_CANDIDATE_PATTERN = re.compile(
    r"\b(?:\d[ -]?){12,18}\d\b"
)

# US SSN — three digits, two digits, four digits
SSN_PATTERN = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)

# Naive IPv4
IPV4_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)


# ============================================================
# Result type
# ============================================================

@dataclass
class OutputGuardResult:
    """Outcome of running the output guard on an answer."""

    sanitized_text: str
    redactions: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


# ============================================================
# The guard
# ============================================================

class OutputGuard:
    """Redacts PII from generated answers."""

    def __init__(self, redact: bool | None = None):
        self.redact = (
            redact if redact is not None else settings.guard_pii_redaction
        )

    async def check(self, text: str) -> OutputGuardResult:
        """Scan + redact (if enabled). Returns the (possibly modified) text."""
        if not text:
            return OutputGuardResult(sanitized_text=text)

        result_text = text
        counts: dict[str, int] = {}

        # Order matters — check most-specific patterns first to avoid
        # email-with-numbers being misread as a phone number, etc.
        result_text, counts["email"] = self._scan_replace(
            result_text, EMAIL_PATTERN, "[REDACTED:email]"
        )
        result_text, counts["ssn"] = self._scan_replace(
            result_text, SSN_PATTERN, "[REDACTED:ssn]"
        )
        result_text, counts["credit_card"] = self._scan_replace_validated(
            result_text, CC_CANDIDATE_PATTERN, "[REDACTED:credit_card]",
            validator=self._is_valid_credit_card,
        )
        result_text, counts["phone"] = self._scan_replace(
            result_text, PHONE_PATTERN, "[REDACTED:phone]"
        )
        result_text, counts["ipv4"] = self._scan_replace(
            result_text, IPV4_PATTERN, "[REDACTED:ip]"
        )

        # Filter to only types that actually fired
        applied = [name for name, n in counts.items() if n > 0]
        applied_counts = {k: v for k, v in counts.items() if v > 0}

        if applied:
            log.warning(
                "OutputGuard %s PII: %s",
                "REDACTED" if self.redact else "DETECTED",
                applied_counts,
            )

        return OutputGuardResult(
            sanitized_text=result_text if self.redact else text,
            redactions=applied,
            counts=applied_counts,
        )

    # -------- Internal helpers --------

    def _scan_replace(
        self,
        text: str,
        pattern: re.Pattern,
        replacement: str,
    ) -> tuple[str, int]:
        """Replace all matches; return (new_text, count)."""
        matches = pattern.findall(text)
        if not matches:
            return text, 0
        return pattern.sub(replacement, text), len(matches)

    def _scan_replace_validated(
        self,
        text: str,
        pattern: re.Pattern,
        replacement: str,
        validator,
    ) -> tuple[str, int]:
        """Like _scan_replace but only replaces matches that pass validator()."""
        count = 0

        def maybe_replace(match: re.Match) -> str:
            nonlocal count
            raw = match.group(0)
            if validator(raw):
                count += 1
                return replacement
            return raw

        new_text = pattern.sub(maybe_replace, text)
        return new_text, count

    @staticmethod
    def _is_valid_credit_card(candidate: str) -> bool:
        """Luhn algorithm — checks if a digit sequence is a valid card number."""
        digits = [int(c) for c in candidate if c.isdigit()]
        if len(digits) < 13 or len(digits) > 19:
            return False
        # Luhn: double every second digit from the right, sum digits of result,
        # add to sum of remaining digits, check if total mod 10 == 0
        total = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
        return total % 10 == 0