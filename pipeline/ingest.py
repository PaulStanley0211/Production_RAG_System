"""Ingest CLI — orchestrates the full offline data pipeline.

Usage:
    python -m pipeline.ingest <input_dir>
    python -m pipeline.ingest .\\test-docs
    python -m pipeline.ingest /path/to/docs --chunk-size 768 --overlap 80

Walks the input directory recursively, processes every file the format
detector recognizes, and indexes the result into Qdrant. Idempotent —
running again on the same folder upserts (no duplicates).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from app.models import Chunk
from pipeline.chunker import Chunker
from pipeline.deduplicator import Deduplicator
from pipeline.embedder import Embedder
from pipeline.format_detector import FormatDetector
from pipeline.indexer import Indexer
from pipeline.preprocessor import Preprocessor

log = logging.getLogger("ingest")


def run(input_dir: Path, chunk_size: int = 512, overlap: int = 50) -> None:
    """Run the full pipeline over input_dir."""
    if not input_dir.exists():
        log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)
    if not input_dir.is_dir():
        log.error("Input path is not a directory: %s", input_dir)
        sys.exit(1)

    start_time = time.time()

    # Initialize pipeline components
    detector = FormatDetector()
    preprocessor = Preprocessor()
    deduplicator = Deduplicator()
    chunker = Chunker(chunk_size=chunk_size, overlap=overlap)
    embedder = Embedder()
    indexer = Indexer()

    indexer.ensure_collection()

    # Stage 1: extract + preprocess + dedupe + chunk
    files = sorted(p for p in input_dir.rglob("*") if p.is_file())
    log.info("Found %d files in %s", len(files), input_dir)

    all_chunks: list[Chunk] = []
    skipped_unsupported = 0
    failed_extraction = 0

    for path in files:
        extractor = detector.detect(path)
        if extractor is None:
            log.debug("No extractor for %s, skipping", path.name)
            skipped_unsupported += 1
            continue

        try:
            doc = extractor.extract(path)
        except Exception as e:
            log.error("Extraction failed for %s: %s", path.name, e)
            failed_extraction += 1
            continue

        doc.content = preprocessor.clean(doc.content)
        if not doc.content.strip():
            log.warning("Empty content after preprocessing: %s", path.name)
            continue

        if deduplicator.is_duplicate(doc):
            continue

        chunks = chunker.chunk(doc)
        log.info("%s → %d chunks", path.name, len(chunks))
        all_chunks.extend(chunks)

    if not all_chunks:
        log.warning("No chunks produced. Nothing to index.")
        return

    # Stage 2: batch embedding
    log.info("Embedding %d chunks (this may take a moment)...", len(all_chunks))
    texts = [c.content for c in all_chunks]
    dense_vectors = embedder.embed_dense(texts)
    sparse_vectors = embedder.embed_sparse(texts)

    # Stage 3: indexing
    log.info("Indexing into Qdrant...")
    written = indexer.index(all_chunks, dense_vectors, sparse_vectors)

    elapsed = time.time() - start_time

    # Final report
    log.info("=" * 60)
    log.info("Ingest complete in %.1fs", elapsed)
    log.info("  Files found:              %d", len(files))
    log.info("  Skipped (unsupported):    %d", skipped_unsupported)
    log.info("  Failed extraction:        %d", failed_extraction)
    log.info("  Duplicates skipped:       %d", deduplicator.stats["duplicates_skipped"])
    log.info("  Unique documents:         %d", deduplicator.stats["unique_documents"])
    log.info("  Chunks generated:         %d", len(all_chunks))
    log.info("  Points written to Qdrant: %d", written)
    log.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG system",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing documents to ingest (walks recursively)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size in tokens (default: 512)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Token overlap between adjacent chunks (default: 50)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    run(args.input_dir, chunk_size=args.chunk_size, overlap=args.overlap)


if __name__ == "__main__":
    main()