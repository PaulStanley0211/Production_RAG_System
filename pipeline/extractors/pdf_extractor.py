"""PDF extractor — text-layer extraction with OCR fallback.

Strategy:
1. Try pypdf first — fast, works for most PDFs (Word/LaTeX exports, browser saves)
2. If extracted text is suspiciously short, assume scanned PDF and fall back to OCR
3. OCR uses pdf2image + pytesseract; gracefully degrades if poppler isn't installed

A scanned PDF detection threshold of 100 chars across all pages reliably distinguishes
text PDFs (thousands of chars) from scanned ones (~0 chars from pypdf).
"""

import logging
from pathlib import Path

from pypdf import PdfReader

from app.models import Document
from pipeline.extractors.base import Extractor

log = logging.getLogger(__name__)

# OCR is optional — degrades gracefully if poppler isn't installed
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# If pypdf returns less than this many chars total, assume scanned PDF
SCANNED_PDF_THRESHOLD = 100


class PDFExtractor(Extractor):
    """Extracts text from PDFs. Falls back to OCR for scanned documents."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def extract(self, path: Path) -> Document:
        # Try fast path first
        text, page_count = self._extract_text_layer(path)

        # If text layer is empty/tiny, fall back to OCR
        used_ocr = False
        if len(text.strip()) < SCANNED_PDF_THRESHOLD:
            if OCR_AVAILABLE:
                log.info("Text layer empty for %s, falling back to OCR", path.name)
                text = self._extract_via_ocr(path)
                used_ocr = True
            else:
                log.warning(
                    "PDF %s appears to be scanned but OCR unavailable "
                    "(install pdf2image + poppler for OCR support)",
                    path.name,
                )

        return Document(
            id=str(path.resolve()),
            source=path.name,
            content=text,
            metadata={
                "format": "pdf",
                "page_count": page_count,
                "used_ocr": used_ocr,
                "size_bytes": path.stat().st_size,
            },
        )

    def _extract_text_layer(self, path: Path) -> tuple[str, int]:
        """Use pypdf to extract embedded text. Returns (text, page_count)."""
        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception as e:
                log.warning("Failed to extract page from %s: %s", path.name, e)
                pages.append("")
        return "\n\n".join(pages), len(reader.pages)

    def _extract_via_ocr(self, path: Path) -> str:
        """Rasterize each page, run Tesseract OCR. Slow but reliable."""
        images = convert_from_path(str(path))
        pages: list[str] = []
        for i, img in enumerate(images, start=1):
            try:
                pages.append(pytesseract.image_to_string(img))
            except Exception as e:
                log.warning("OCR failed on page %d of %s: %s", i, path.name, e)
                pages.append("")
        return "\n\n".join(pages)