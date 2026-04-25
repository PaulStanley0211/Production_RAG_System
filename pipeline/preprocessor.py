"""Preprocessor — clean and normalize extracted text before chunking.

Runs four passes:
1. Unicode normalization (NFKC) — canonicalize visually-similar characters
2. Boilerplate stripping       — page numbers, copyright lines, etc.
3. Whitespace normalization    — collapse runs, strip per-line trailing space
4. Empty paragraph collapse    — multiple blank lines → single blank line

Order matters: normalize Unicode first so regex patterns match consistently.
"""

import logging
import re
import unicodedata

log = logging.getLogger(__name__)

# Boilerplate patterns. Conservative — better to leave a little cruft than
# accidentally strip real content. Add patterns here as you encounter them.
BOILERPLATE_PATTERNS = [
    # "Page 5 of 42" or "Page 5"
    re.compile(r"^\s*Page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE | re.MULTILINE),
    # Bare page number on its own line
    re.compile(r"^\s*-?\s*\d+\s*-?\s*$", re.MULTILINE),
    # Copyright lines
    re.compile(r"^\s*©\s*\d{4}.*$", re.MULTILINE),
    re.compile(r"^\s*Copyright\s+(©|\(c\))?\s*\d{4}.*$", re.IGNORECASE | re.MULTILINE),
    # "Confidential — do not distribute" boilerplate
    re.compile(r"^\s*confidential.*$", re.IGNORECASE | re.MULTILINE),
]

# Whitespace cleanup
RE_MULTIPLE_SPACES = re.compile(r"[ \t]+")
RE_TRAILING_WHITESPACE = re.compile(r"[ \t]+$", re.MULTILINE)
RE_MULTIPLE_BLANK_LINES = re.compile(r"\n\s*\n\s*\n+")


class Preprocessor:
    """Cleans and normalizes raw extracted text."""

    def clean(self, text: str) -> str:
        """Run all cleanup passes in order. Idempotent."""
        text = self._normalize_unicode(text)
        text = self._strip_boilerplate(text)
        text = self._normalize_whitespace(text)
        return text.strip()

    def _normalize_unicode(self, text: str) -> str:
        """NFKC: canonicalize ligatures, full-width chars, curly quotes."""
        return unicodedata.normalize("NFKC", text)

    def _strip_boilerplate(self, text: str) -> str:
        for pattern in BOILERPLATE_PATTERNS:
            text = pattern.sub("", text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse runs of spaces/tabs (but preserve newlines)
        text = RE_MULTIPLE_SPACES.sub(" ", text)
        # Strip trailing whitespace from each line
        text = RE_TRAILING_WHITESPACE.sub("", text)
        # Collapse 3+ blank lines into 2 (one blank line between paragraphs)
        text = RE_MULTIPLE_BLANK_LINES.sub("\n\n", text)
        return text