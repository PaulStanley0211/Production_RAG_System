"""Chunker — splits documents into overlapping chunks for embedding.

Recursive split strategy: paragraphs → sentences → hard cut.
Overlap preserves context across boundaries.
Token counting via tiktoken (cl100k_base) — matches most embedding models.

Chunk size and overlap are configurable. Defaults are 512 tokens with
50-token overlap, a tested middle-ground for most embedding models.
"""

import hashlib
import logging
import re

import tiktoken

from app.models import Chunk, Document

log = logging.getLogger(__name__)

# Sentence boundary detector — handles ., !, ?, with whitespace after
RE_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Paragraph boundary — one or more blank lines
RE_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


class Chunker:
    """Splits documents into overlapping chunks of approximately N tokens."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split a document into chunks. Returns ordered list of Chunks."""
        # Split into paragraphs first
        paragraphs = [
            p.strip() for p in RE_PARAGRAPH_BREAK.split(doc.content) if p.strip()
        ]

        # Pack paragraphs into chunks, splitting any oversized paragraphs
        text_chunks = self._pack_paragraphs(paragraphs)

        # Apply overlap by carrying tokens forward
        text_chunks = self._apply_overlap(text_chunks)

        # Wrap as Chunk objects with deterministic IDs
        return [
            self._make_chunk(doc, idx, text)
            for idx, text in enumerate(text_chunks)
        ]

    def _pack_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """Pack paragraphs into chunks ≤ chunk_size tokens."""
        chunks: list[str] = []
        buffer: list[str] = []
        buffer_tokens = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            # Oversized single paragraph — split by sentence
            if para_tokens > self.chunk_size:
                # Flush whatever we have buffered first
                if buffer:
                    chunks.append("\n\n".join(buffer))
                    buffer, buffer_tokens = [], 0
                chunks.extend(self._split_long_text(para))
                continue

            # Adding this paragraph would overflow → flush buffer first
            if buffer_tokens + para_tokens > self.chunk_size:
                chunks.append("\n\n".join(buffer))
                buffer, buffer_tokens = [], 0

            buffer.append(para)
            buffer_tokens += para_tokens

        # Flush remaining buffer
        if buffer:
            chunks.append("\n\n".join(buffer))

        return chunks

    def _split_long_text(self, text: str) -> list[str]:
        """Split a too-large text by sentences, then hard-cut as fallback."""
        sentences = [s.strip() for s in RE_SENTENCE_END.split(text) if s.strip()]
        chunks: list[str] = []
        buffer: list[str] = []
        buffer_tokens = 0

        for sent in sentences:
            sent_tokens = self._count_tokens(sent)
            if sent_tokens > self.chunk_size:
                # Pathological case (huge code block, no spaces). Hard cut.
                if buffer:
                    chunks.append(" ".join(buffer))
                    buffer, buffer_tokens = [], 0
                chunks.extend(self._hard_cut(sent))
                continue

            if buffer_tokens + sent_tokens > self.chunk_size:
                chunks.append(" ".join(buffer))
                buffer, buffer_tokens = [], 0
            buffer.append(sent)
            buffer_tokens += sent_tokens

        if buffer:
            chunks.append(" ".join(buffer))
        return chunks

    def _hard_cut(self, text: str) -> list[str]:
        """Last-resort: split by token count regardless of meaning."""
        tokens = self.tokenizer.encode(text)
        chunks: list[str] = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[i : i + self.chunk_size]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Prepend the last `overlap` tokens of chunk N to chunk N+1."""
        if self.overlap == 0 or len(chunks) <= 1:
            return chunks

        with_overlap: list[str] = [chunks[0]]
        for prev, curr in zip(chunks, chunks[1:]):
            prev_tokens = self.tokenizer.encode(prev)
            tail = prev_tokens[-self.overlap :]
            tail_text = self.tokenizer.decode(tail)
            with_overlap.append(f"{tail_text} {curr}")
        return with_overlap

    def _make_chunk(self, doc: Document, idx: int, text: str) -> Chunk:
        """Build a Chunk with a deterministic ID."""
        content_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        chunk_id = f"{doc.id}::chunk-{idx}::{content_hash}"

        return Chunk(
            id=chunk_id,
            doc_id=doc.id,
            source=doc.source,
            content=text,
            metadata={
                **doc.metadata,
                "chunk_index": idx,
                "token_count": self._count_tokens(text),
            },
        )

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))