"""CAMARA knowledge ingestion pipeline.

    pre-chunked CAMARA JSONL
        -> load JSONL records          (retrieval.documents.load_jsonl)
        -> embed each record's text    (retrieval.embeddings)
        -> store in ChromaDB           (retrieval.retriever.VectorStore)

The JSONL corpus is already semantically chunked, so its records are embedded
directly. Each record: ``{"id", "text", "metadata", "structured"}`` (``id`` and
``structured`` optional).

``IngestionPipeline`` also exposes the generic ``ingest_chunks`` / ``ingest_documents``
primitives and an ``ingest_registry`` convenience (used by tests and anyone who
wants to embed the API registry instead of / alongside a JSONL corpus).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import Settings, get_settings
from camara_pgps.retrieval.documents import (
    Chunk,
    Document,
    chunk_documents,
    documents_from_registry,
    load_jsonl,
)
from camara_pgps.retrieval.embeddings import Embedder, get_embedder
from camara_pgps.retrieval.retriever import VectorStore

logger = logging.getLogger(__name__)


class IngestionResult(BaseModel):
    """Summary of one ingestion run."""

    documents: int = 0
    chunks: int = 0
    written: int = 0
    collection: str = ""
    persist_directory: str = ""

    def merge(self, other: IngestionResult) -> IngestionResult:
        return IngestionResult(
            documents=self.documents + other.documents,
            chunks=self.chunks + other.chunks,
            written=self.written + other.written,
            collection=other.collection or self.collection,
            persist_directory=other.persist_directory or self.persist_directory,
        )


class IngestionPipeline:
    """Load -> embed -> store CAMARA knowledge (JSONL corpus, or documents)."""

    def __init__(
        self,
        *,
        vector_store: VectorStore | None = None,
        embedder: Embedder | None = None,
        settings: Settings | None = None,
        max_chars: int = 1000,
        overlap: int = 150,
    ) -> None:
        self.settings = settings or get_settings()
        self.vector_store = vector_store or VectorStore.from_settings(self.settings)
        self.embedder = embedder or get_embedder(self.settings)
        self.max_chars = max_chars
        self.overlap = overlap

    def _empty(self) -> IngestionResult:
        return IngestionResult(
            collection=self.vector_store.collection_name,
            persist_directory=self.vector_store.persist_directory,
        )

    # -- generic primitives --------------------------------------------------

    def ingest_chunks(self, chunks: Sequence[Chunk]) -> IngestionResult:
        """Embed and store ready-made chunks. Every other path funnels here."""

        chunks = list(chunks)
        if not chunks:
            return self._empty()
        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        written = self.vector_store.add(chunks, vectors)
        logger.info(
            "ingested %d chunk(s) -> %d vector(s) into %s",
            len(chunks),
            written,
            self.vector_store.collection_name,
        )
        return IngestionResult(
            documents=len(chunks),
            chunks=len(chunks),
            written=written,
            collection=self.vector_store.collection_name,
            persist_directory=self.vector_store.persist_directory,
        )

    def ingest_documents(self, documents: Iterable[Document]) -> IngestionResult:
        """Chunk unstructured documents, then embed and store them."""

        documents = list(documents)
        chunks = chunk_documents(
            documents, max_chars=self.max_chars, overlap=self.overlap
        )
        result = self.ingest_chunks(chunks)
        return result.model_copy(update={"documents": len(documents)})

    # -- JSONL corpus (the primary path) --------------------------------------

    def ingest_jsonl(self, paths: Iterable[str | Path]) -> IngestionResult:
        """Ingest pre-chunked JSONL records directly (no re-chunking)."""

        chunks: list[Chunk] = []
        seen: set[str] = set()
        for path in paths:
            loaded = load_jsonl(path)
            for chunk in loaded:
                if chunk.id in seen:
                    continue
                seen.add(chunk.id)
                chunks.append(chunk)
            logger.info("loaded %d record(s) from %s", len(loaded), path)
        return self.ingest_chunks(chunks)

    def ingest_directory(self, directory: str | Path) -> IngestionResult:
        """Ingest every ``*.jsonl`` file found under ``directory`` (recursively)."""

        files = sorted(Path(directory).rglob("*.jsonl"))
        logger.info("found %d JSONL file(s) in %s", len(files), directory)
        return self.ingest_jsonl(files)

    # -- convenience -----------------------------------------------------------

    def ingest_registry(
        self, registry: CamaraRegistry | None = None
    ) -> IngestionResult:
        """Render the CAMARA registry to documents and ingest them."""

        registry = registry or CamaraRegistry.from_settings(self.settings)
        return self.ingest_documents(documents_from_registry(registry))

    def reset(self) -> None:
        self.vector_store.reset()


def run_ingestion(
    *,
    settings: Settings | None = None,
    jsonl_paths: Iterable[str | Path] | None = None,
    jsonl_dir: str | Path | None = None,
    reset: bool = False,
) -> IngestionResult:
    """Embed a pre-chunked JSONL corpus into ChromaDB.

    Uses ``jsonl_paths`` and/or ``jsonl_dir``; when neither is given it falls
    back to ``settings.retrieval.knowledge_dir`` if that directory exists.
    """

    settings = settings or get_settings()
    pipeline = IngestionPipeline(settings=settings)
    if reset:
        pipeline.reset()

    result = pipeline._empty()

    paths = list(jsonl_paths or [])
    if paths:
        result = result.merge(pipeline.ingest_jsonl(paths))

    directory = jsonl_dir if jsonl_dir is not None else settings.retrieval.knowledge_dir
    if directory is not None and Path(directory).is_dir():
        result = result.merge(pipeline.ingest_directory(directory))
    elif not paths:
        logger.warning(
            "no JSONL corpus to ingest (jsonl_paths empty, %s not a directory)",
            directory,
        )

    return result


__all__ = [
    "IngestionPipeline",
    "IngestionResult",
    "run_ingestion",
]
