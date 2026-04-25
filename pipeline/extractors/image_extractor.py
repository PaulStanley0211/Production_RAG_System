"""Image extractor — OCR via Tesseract.

Handles common image formats. Requires the Tesseract binary installed and
on PATH. Falls back with a clear error if missing — no cryptic stack traces.
"""

import logging
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.models import Document
from pipeline.extractors.base import Extractor

log = logging.getLogger(__name__)

# Pytesseract import — wraps the Tesseract binary in shell calls
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}


class ImageExtractor(Extractor):
    """OCR text out of image files."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in IMAGE_EXTENSIONS

    def extract(self, path: Path) -> Document:
        if not OCR_AVAILABLE:
            raise RuntimeError(
                "pytesseract not installed. Run: uv pip install pytesseract Pillow"
            )

        try:
            image = Image.open(path)
        except UnidentifiedImageError as e:
            raise RuntimeError(f"Cannot open {path.name}: not a valid image") from e

        try:
            text = pytesseract.image_to_string(image)
        except pytesseract.TesseractNotFoundError as e:
            raise RuntimeError(
                "Tesseract binary not found. Install from "
                "https://github.com/UB-Mannheim/tesseract/wiki and add to PATH."
            ) from e

        return Document(
            id=str(path.resolve()),
            source=path.name,
            content=text,
            metadata={
                "format": path.suffix.lower().lstrip("."),
                "image_size": f"{image.width}x{image.height}",
                "size_bytes": path.stat().st_size,
            },
        )