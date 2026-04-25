"""Deduplicator — skip documents we've already seen this run.

SHA256 hash of normalized content. Two files with identical content but
different filenames get flagged as duplicates. Two files with cosmetic
differences (whitespace, case) but same words also flagged.

For near-duplicates (95% identical), Phase 3 will add embedding-similarity
checks. Hash-based dedup is fast and catches the common cases.
"""

import hashlib
import logging
import re

from app.models import Document

log = logging.getLogger(__name__)

# Aggressive normalization for hashing — collapse all whitespace, lowercase
RE_ALL_WHITESPACE = re.compile(r"\s+")


class Deduplicator:
    """Tracks document content hashes; flags duplicates."""

    def __init__(self):
        self._seen_hashes: set[str] = set()
        self._duplicate_count = 0

    def is_duplicate(self, doc: Document) -> bool:
        """Return True if we've seen this content before in this run."""
        content_hash = self._hash_content(doc.content)
        if content_hash in self._seen_hashes:
            self._duplicate_count += 1
            log.info("Duplicate detected: %s", doc.source)
            return True
        self._seen_hashes.add(content_hash)
        return False

    def _hash_content(self, content: str) -> str:
        """SHA256 of aggressively-normalized content."""
        normalized = RE_ALL_WHITESPACE.sub(" ", content).strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @property
    def stats(self) -> dict:
        """For end-of-run reporting."""
        return {
            "unique_documents": len(self._seen_hashes),
            "duplicates_skipped": self._duplicate_count,
        }