"""Format detector — routes files to the right extractor.

Maintains an ordered list of extractors. For each file, asks each extractor
'can you handle this?' until one says yes. Returns that extractor.

Order matters: register specific extractors first, fallbacks last.
"""

from pathlib import Path

from pipeline.extractors.base import Extractor
from pipeline.extractors.docx_extractor import DOCXExtractor
from pipeline.extractors.html_extractor import HTMLExtractor
from pipeline.extractors.image_extractor import ImageExtractor
from pipeline.extractors.pdf_extractor import PDFExtractor
from pipeline.extractors.text_extractor import TextExtractor


class FormatDetector:
    """Picks the right extractor for any given file path."""

    def __init__(self):
        # Order matters: more specific formats before catch-all text
        self.extractors: list[Extractor] = [
            PDFExtractor(),
            DOCXExtractor(),
            HTMLExtractor(),
            ImageExtractor(),
            TextExtractor(),  # fallback for .txt, .md, anything unmatched
        ]

    def detect(self, path: Path) -> Extractor | None:
        """Return the first extractor that can handle the file, or None."""
        for ex in self.extractors:
            if ex.can_handle(path):
                return ex
        return None