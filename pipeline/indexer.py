"""Indexer — manages Qdrant collection and upserts chunks.

Sets up the collection with named vectors (dense + sparse) and payload
indexes for fast filtering. Indexing is idempotent: re-running on the same
chunks upserts in place via deterministic UUIDs.
"""

import logging
import uuid
from collections.abc import Iterable

from fastembed import SparseEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.config import settings
from app.models import Chunk

log = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 64

# UUID5 namespace — must be stable across runs so the same chunk.id
# always produces the same UUID. Any fixed UUID works; this one is arbitrary.
_NAMESPACE = uuid.UUID("6f0d1f1c-9f47-4b2a-8a3e-3c4f8b3e9c2d")


class Indexer:
    """Wraps Qdrant collection setup and chunk upserts."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        collection: str | None = None,
        vector_size: int | None = None,
    ):
        self.client = client or QdrantClient(url=settings.qdrant_url)
        self.collection = collection or settings.qdrant_collection
        self.vector_size = vector_size or settings.qdrant_vector_size

    def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist. Safe to call repeatedly."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            log.info("Collection %s already exists", self.collection)
            return

        log.info("Creating collection %s", self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )

        # Payload indexes for fast filtering by source / doc_id
        for field in ("source", "doc_id"):
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        log.info("Collection %s ready with payload indexes", self.collection)

    def index(
        self,
        chunks: list[Chunk],
        dense_vectors: list,
        sparse_vectors: list[SparseEmbedding],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> int:
        """Upsert chunks + their vectors. Returns the number of points written."""
        if not (len(chunks) == len(dense_vectors) == len(sparse_vectors)):
            raise ValueError(
                "chunks, dense_vectors, sparse_vectors must be the same length"
            )

        total = 0
        for batch in self._batched(chunks, dense_vectors, sparse_vectors, batch_size):
            points = [self._make_point(c, dv, sv) for c, dv, sv in batch]
            self.client.upsert(collection_name=self.collection, points=points)
            total += len(points)
            log.info("Upserted batch of %d (total: %d)", len(points), total)
        return total

    def _make_point(
        self,
        chunk: Chunk,
        dense_vec,
        sparse_vec: SparseEmbedding,
    ) -> PointStruct:
        """Build a Qdrant point from a chunk + its two vectors."""
        return PointStruct(
            id=str(uuid.uuid5(_NAMESPACE, chunk.id)),
            vector={
                "dense": dense_vec.tolist(),
                "sparse": SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
            },
            payload={
                "doc_id": chunk.doc_id,
                "source": chunk.source,
                "content": chunk.content,
                **chunk.metadata,
            },
        )

    @staticmethod
    def _batched(
        chunks: list[Chunk],
        dense_vectors: list,
        sparse_vectors: list[SparseEmbedding],
        size: int,
    ) -> Iterable[list[tuple[Chunk, object, SparseEmbedding]]]:
        """Yield batches of (chunk, dense, sparse) triples."""
        triples = list(zip(chunks, dense_vectors, sparse_vectors))
        for i in range(0, len(triples), size):
            yield triples[i : i + size]