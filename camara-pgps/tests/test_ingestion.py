"""Tests for the CAMARA knowledge ingestion pipeline (milestone 5)."""

from __future__ import annotations

import math

import pytest

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import PROJECT_ROOT
from camara_pgps.models import RetrievalQuery, RetrievalResult
from camara_pgps.retrieval import (
    ChunkMetadata,
    DocumentType,
    HashingEmbedder,
    IngestionPipeline,
    Retriever,
    VectorStore,
    chunk_document,
    documents_from_registry,
    load_markdown,
)
from camara_pgps.retrieval.documents import Document

REGISTRY_FILE = PROJECT_ROOT / "data" / "registry" / "camara_apis.json"


@pytest.fixture(scope="module")
def registry() -> CamaraRegistry:
    return CamaraRegistry.from_file(REGISTRY_FILE)


@pytest.fixture
def store(tmp_path) -> VectorStore:
    return VectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_camara",
        distance="cosine",
    )


# --------------------------------------------------------------------------- #
# documents / metadata
# --------------------------------------------------------------------------- #

def test_documents_from_registry_carry_metadata(registry: CamaraRegistry) -> None:
    docs = documents_from_registry(registry)
    types = {d.metadata.document_type for d in docs}
    assert DocumentType.API_OVERVIEW in types
    assert DocumentType.OPERATION in types
    assert DocumentType.PARAMETERS in types

    create = next(
        d
        for d in docs
        if d.metadata.operation_id == "createSession"
        and d.metadata.document_type == DocumentType.OPERATION
    )
    assert create.metadata.api_name == "quality-on-demand"
    assert create.metadata.version == "0.11.0"
    assert create.metadata.section == "operations/createSession"
    assert create.metadata.source is not None


def test_markdown_is_split_into_sections() -> None:
    md = (
        "# Quality on Demand\n\n"
        "Intro text.\n\n"
        "## Creating a session\n\n"
        "Call createSession with a device and a QoS profile.\n\n"
        "## Deleting a session\n\n"
        "Call deleteSession with the session id.\n"
    )
    docs = load_markdown(md, source="qod.md")
    sections = {d.metadata.section for d in docs}
    assert sections == {
        "Quality on Demand",
        "Quality on Demand > Creating a session",
        "Quality on Demand > Deleting a session",
    }
    assert all(d.metadata.document_type == DocumentType.MARKDOWN for d in docs)
    assert all(d.metadata.source == "qod.md" for d in docs)


def test_chunk_metadata_to_and_from_chroma() -> None:
    meta = ChunkMetadata(
        document_type=DocumentType.OPERATION,
        api_name="quality-on-demand",
        version="0.11.0",
        operation_id="createSession",
        source="data/registry/camara_apis.json",
        section=None,
        extra={"weight": 3},
    )
    flat = meta.to_chroma()
    assert "section" not in flat  # None values are dropped
    assert flat["document_type"] == "operation"
    assert flat["weight"] == 3
    assert ChunkMetadata.from_chroma(flat).operation_id == "createSession"
    assert ChunkMetadata.from_chroma(flat).extra == {"weight": 3}


def test_chunking_is_deterministic_and_splits_long_text() -> None:
    long_text = "\n\n".join(f"Paragraph {i} about QoS sessions." for i in range(60))
    doc = Document(text=long_text, metadata=ChunkMetadata(source="s", section="body"))
    first = chunk_document(doc, max_chars=200, overlap=20)
    second = chunk_document(doc, max_chars=200, overlap=20)
    assert len(first) > 1
    assert [c.id for c in first] == [c.id for c in second]
    assert all(len(c.text) <= 200 for c in first)


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #

def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder(dimension=128)
    a = embedder.embed_query("create a quality on demand session")
    b = embedder.embed_query("create a quality on demand session")
    assert a == b
    assert len(a) == 128
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-6)
    docs = embedder.embed_documents(["one", "two"])
    assert len(docs) == 2


# --------------------------------------------------------------------------- #
# full pipeline + retrieval
# --------------------------------------------------------------------------- #

def test_pipeline_ingests_registry_and_is_idempotent(
    registry: CamaraRegistry, store: VectorStore
) -> None:
    pipeline = IngestionPipeline(vector_store=store, embedder=HashingEmbedder())
    result = pipeline.ingest_registry(registry)
    assert result.documents > 0
    assert result.chunks > 0
    assert result.written == result.chunks
    count_after_first = store.count()
    assert count_after_first > 0

    # re-ingesting the same content upserts, it does not duplicate
    pipeline.ingest_registry(registry)
    assert store.count() == count_after_first


def test_retriever_returns_scored_chunks_with_metadata(
    registry: CamaraRegistry, store: VectorStore
) -> None:
    embedder = HashingEmbedder()
    IngestionPipeline(vector_store=store, embedder=embedder).ingest_registry(registry)
    retriever = Retriever(vector_store=store, embedder=embedder)

    result = retriever.retrieve("how do I create a quality on demand session")
    assert isinstance(result, RetrievalResult)
    assert len(result) > 0
    top = result.chunks[0]
    assert 0.0 <= top.score <= 1.0
    assert top.metadata.get("document_type")
    # the best hit should be about the quality-on-demand API
    assert any(c.api_name == "quality-on-demand" for c in result.chunks)
    assert "createSession" in result.operation_ids()


def test_retriever_honours_metadata_filters(
    registry: CamaraRegistry, store: VectorStore
) -> None:
    embedder = HashingEmbedder()
    IngestionPipeline(vector_store=store, embedder=embedder).ingest_registry(registry)
    retriever = Retriever(vector_store=store, embedder=embedder)

    query = RetrievalQuery(
        text="verify the location of a device",
        top_k=5,
        filters={"api_name": "device-location"},
    )
    result = retriever.retrieve(query)
    assert result.chunks
    assert {c.api_name for c in result.chunks} == {"device-location"}
