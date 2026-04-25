"""Text extractor — handles .txt, .md, and any unmatched text-like file.

Acts as the fallback in the format detector chain. Robust to encoding issues
that are common on Windows-saved files.
"""

from pathlib import Path

import chardet

from app.models import Document
from pipeline.extractors.base import Extractor

# Files this extractor handles by extension. Anything unmatched also falls
# through to here as the last-resort extractor in the format detector chain.
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv"}


class TextExtractor(Extractor):
    """Reads plain text files with robust encoding detection."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in TEXT_EXTENSIONS

    def extract(self, path: Path) -> Document:
        content = self._read_with_encoding_fallback(path)

        return Document(
            id=str(path.resolve()),
            source=path.name,
            content=content,
            metadata={
                "format": path.suffix.lower().lstrip("."),
                "size_bytes": path.stat().st_size,
            },
        )

    def _read_with_encoding_fallback(self, path: Path) -> str:
        """Detect encoding from a sample, fall back through common encodings."""
        # Read a sample for chardet to sniff
        with open(path, "rb") as f:
            sample = f.read(10_000)

        detected = chardet.detect(sample)
        encodings_to_try: list[str] = []

        # Use detected encoding first if confidence is reasonable
        if detected["encoding"] and detected["confidence"] > 0.7:
            encodings_to_try.append(detected["encoding"])

        # Fallback chain — latin-1 always succeeds (accepts any byte)
        encodings_to_try.extend(["utf-8", "utf-8-sig", "cp1252", "latin-1"])

        # De-duplicate while preserving order
        seen = set()
        ordered = [e for e in encodings_to_try if not (e in seen or seen.add(e))]

        for enc in ordered:
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, LookupError):
                continue

        # Should never reach here because latin-1 accepts anything
        raise RuntimeError(f"Could not decode {path} with any encoding")