"""The CAMARA knowledge ingestion pipeline.

    CAMARA documentation
        -> document loading / parsing   (retrieval.documents)
        -> chunking                     (retrieval.documents.chunk_documents)
        -> metadata extraction          (ChunkMetadata on every chunk)
        -> embeddings                   (retrieval.embeddings)
        -> ChromaDB                     (retrieval.retriever.VectorStore)

:class:`IngestionPipeline` wires those stages together. The metadata carried into
ChromaDB for every chunk is ``api_name, version, operation_id, source,
document_type, section``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from camara_pgps.camara.openapi_parser import parse_openapi
from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.camara.schemas import RegistryAPI
from camara_pgps.config import Settings, get_settings
from camara_pgps.retrieval.documents import (
    Document,
    chunk_documents,
    documents_from_api,
    documents_from_registry,
    load_markdown,
)
from camara_pgps.retrieval.embeddings import Embedder, get_embedder
from camara_pgps.retrieval.retriever import VectorStore

logger = logging.getLogger(__name__)

_MARKDOWN_SUFFIXES = {".md", ".markdown"}


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
    """Load -> parse -> chunk -> embed -> store CAMARA knowledge."""

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

    # -- stages -----------------------------------------------------------------

    def ingest_documents(self, documents: Iterable[Document]) -> IngestionResult:
        documents = list(documents)
        chunks = chunk_documents(
            documents, max_chars=self.max_chars, overlap=self.overlap
        )
        written = 0
        if chunks:
            vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
            written = self.vector_store.add(chunks, vectors)
        logger.info(
            "ingested %d document(s) -> %d chunk(s) into %s",
            len(documents),
            len(chunks),
            self.vector_store.collection_name,
        )
        return IngestionResult(
            documents=len(documents),
            chunks=len(chunks),
            written=written,
            collection=self.vector_store.collection_name,
            persist_directory=self.vector_store.persist_directory,
        )

    # -- sources --------------------------------------------------------------

    def ingest_registry(self, registry: CamaraRegistry | None = None) -> IngestionResult:
        registry = registry or CamaraRegistry.from_settings(self.settings)
        return self.ingest_documents(documents_from_registry(registry))

    def ingest_api(self, api: RegistryAPI) -> IngestionResult:
        return self.ingest_documents(documents_from_api(api))

    def ingest_openapi(
        self,
        specs: Iterable[str | Path],
        *,
        source_url: str | None = None,
    ) -> IngestionResult:
        documents: list[Document] = []
        for spec in specs:
            api = parse_openapi(spec, source_url=source_url)
            documents.extend(documents_from_api(api))
        return self.ingest_documents(documents)

    def ingest_markdown(self, paths: Iterable[str | Path]) -> IngestionResult:
        documents: list[Document] = []
        for path in paths:
            documents.extend(load_markdown(Path(path)))
        return self.ingest_documents(documents)

    def ingest_directory(self, directory: str | Path) -> IngestionResult:
        root = Path(directory)
        markdown_files = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in _MARKDOWN_SUFFIXES
        )
        return self.ingest_markdown(markdown_files)

    def reset(self) -> None:
        self.vector_store.reset()


def run_ingestion(
    *,
    settings: Settings | None = None,
    include_registry: bool = True,
    docs_dir: str | Path | None = None,
    reset: bool = False,
) -> IngestionResult:
    """Convenience entry point: ingest the local registry and an optional docs dir."""

    settings = settings or get_settings()
    pipeline = IngestionPipeline(settings=settings)
    if reset:
        pipeline.reset()

    result = IngestionResult(
        collection=pipeline.vector_store.collection_name,
        persist_directory=pipeline.vector_store.persist_directory,
    )
    if include_registry:
        result = result.merge(pipeline.ingest_registry())
    if docs_dir is not None:
        result = result.merge(pipeline.ingest_directory(docs_dir))
    return result


__all__ = [
    "IngestionPipeline",
    "IngestionResult",
    "run_ingestion",
]
