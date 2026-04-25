"""Base class for all extractors.

Every concrete extractor (PDF, HTML, DOCX, image, text) inherits from
Extractor and implements two methods:
- can_handle(path): return True if this extractor knows the format
- extract(path):    do the actual extraction, return a Document

The format detector uses can_handle() to pick the right extractor at runtime.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from app.models import Document


class Extractor(ABC):
    """Abstract base for all file-format extractors."""

    @abstractmethod
    def can_handle(self, path: Path) -> bool:
        """Return True if this extractor can process the given file."""

    @abstractmethod
    def extract(self, path: Path) -> Document:
        """Extract content from the file. Raise on failure.

        Returns a Document with content + metadata (source, format, etc.).
        """