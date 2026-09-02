"""ChromaDB vector store and the query-time retriever.

:class:`VectorStore` is the thin ChromaDB wrapper used by both the ingestion
pipeline (write path) and :class:`Retriever` (read path). :class:`Retriever`
turns a natural-language question into a
:class:`~camara_pgps.models.RetrievalResult` of scored
:class:`~camara_pgps.models.RetrievedChunk` objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field

from camara_pgps.config import Settings, get_settings
from camara_pgps.models import RetrievalQuery, RetrievalResult, RetrievedChunk
from camara_pgps.retrieval.documents import Chunk
from camara_pgps.retrieval.embeddings import Embedder, get_embedder


class VectorMatch(BaseModel):
    """A single hit from :meth:`VectorStore.query`."""

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    distance: float


class VectorStore:
    """ChromaDB-backed persistent collection of embedded chunks."""

    def __init__(
        self,
        *,
        persist_directory: str | Path,
        collection_name: str,
        distance: str = "cosine",
    ) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        self._distance = distance
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance},
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> VectorStore:
        settings = settings or get_settings()
        return cls(
            persist_directory=settings.vector_store.persist_directory,
            collection_name=settings.vector_store.collection_name,
            distance=settings.vector_store.distance,
        )

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
        """Upsert chunks and their vectors. Returns the number written."""

        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=cast("Any", [chunk.metadata.to_chroma() for chunk in chunks]),
            embeddings=cast("Any", [[float(v) for v in vector] for vector in embeddings]),
        )
        return len(chunks)

    def query(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 5,
        where: Mapping[str, Any] | None = None,
    ) -> list[VectorMatch]:
        result = self._collection.query(
            query_embeddings=cast("Any", [[float(v) for v in embedding]]),
            n_results=max(1, top_k),
            where=cast("Any", dict(where)) if where else None,
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        matches: list[VectorMatch] = []
        for idx, chunk_id in enumerate(ids):
            matches.append(
                VectorMatch(
                    id=chunk_id,
                    text=documents[idx] if idx < len(documents) else "",
                    metadata=metadatas[idx] if idx < len(metadatas) else {},
                    distance=float(distances[idx]) if idx < len(distances) else 0.0,
                )
            )
        return matches

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        """Drop and recreate the collection."""

        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self._distance},
        )


def _similarity(distance: float) -> float:
    """Map a distance to a 0..1 similarity score (cosine / L2-normalised)."""

    return max(0.0, min(1.0, 1.0 - distance))


class Retriever:
    """Embeds a question and returns the closest CAMARA knowledge chunks."""

    def __init__(
        self,
        *,
        vector_store: VectorStore | None = None,
        embedder: Embedder | None = None,
        settings: Settings | None = None,
    ) -> None:
        settings = settings or get_settings()
        self.vector_store = vector_store or VectorStore.from_settings(settings)
        self.embedder = embedder or get_embedder(settings)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Retriever:
        return cls(settings=settings or get_settings())

    def retrieve(self, query: str | RetrievalQuery) -> RetrievalResult:
        request = RetrievalQuery(text=query) if isinstance(query, str) else query
        vector = self.embedder.embed_query(request.text)
        matches = self.vector_store.query(
            vector,
            top_k=request.top_k,
            where=request.filters or None,
        )
        chunks = [
            RetrievedChunk(
                id=match.id,
                text=match.text,
                score=_similarity(match.distance),
                source=match.metadata.get("source"),
                api_name=match.metadata.get("api_name"),
                operation_id=match.metadata.get("operation_id"),
                metadata=match.metadata,
            )
            for match in matches
        ]
        return RetrievalResult(
            query=request.text,
            chunks=chunks,
            metadata={
                "top_k": request.top_k,
                "collection": self.vector_store.collection_name,
                "embedder": self.embedder.model_name,
            },
        )


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Return the process-wide retriever built from configured settings."""

    return Retriever.from_settings()


__all__ = [
    "VectorMatch",
    "VectorStore",
    "Retriever",
    "get_retriever",
]
