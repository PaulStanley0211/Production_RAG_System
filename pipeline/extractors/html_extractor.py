"""HTML extractor — strips boilerplate, preserves table structure.

Removes navigation, scripts, styles, footers, and other non-content elements
before extracting text. Without this, every chunk is polluted with menu items
and cookie banners that destroy retrieval quality.

Tables are flattened to tab-separated rows so structure isn't lost.
"""

import logging
from pathlib import Path

from bs4 import BeautifulSoup

from app.models import Document
from pipeline.extractors.base import Extractor

log = logging.getLogger(__name__)

# Tags whose content is almost never useful for retrieval
BOILERPLATE_TAGS = {
    "script", "style", "nav", "header", "footer", "aside",
    "form", "noscript", "iframe", "svg", "button",
}


class HTMLExtractor(Extractor):
    """Extracts main content from HTML, dropping navigation and scripts."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in {".html", ".htm", ".xhtml"}

    def extract(self, path: Path) -> Document:
        with open(path, "rb") as f:
            raw = f.read()

        # BeautifulSoup auto-detects encoding from <meta charset> or BOM
        soup = BeautifulSoup(raw, "html.parser")

        # Pull the title before we strip anything
        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        # Strip boilerplate elements completely
        for tag in soup.find_all(BOILERPLATE_TAGS):
            tag.decompose()

        # Extract tables as tab-separated text before they get flattened
        tables_text = self._extract_tables(soup)

        # Remove tables from the soup so we don't double-count them
        for table in soup.find_all("table"):
            table.decompose()

        # Get remaining text with newlines between blocks
        body_text = soup.get_text(separator="\n", strip=True)

        # Recombine — body text first, then tables
        parts = [p for p in [body_text, tables_text] if p]
        content = "\n\n".join(parts)

        return Document(
            id=str(path.resolve()),
            source=path.name,
            content=content,
            metadata={
                "format": "html",
                "title": title,
                "size_bytes": path.stat().st_size,
            },
        )

    def _extract_tables(self, soup: BeautifulSoup) -> str:
        """Flatten each <table> into tab-separated rows."""
        tables: list[str] = []
        for table in soup.find_all("table"):
            rows: list[str] = []
            for tr in table.find_all("tr"):
                cells = [
                    cell.get_text(strip=True)
                    for cell in tr.find_all(["td", "th"])
                ]
                if cells:
                    rows.append("\t".join(cells))
            if rows:
                tables.append("\n".join(rows))
        return "\n\n".join(tables)