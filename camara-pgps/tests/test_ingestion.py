"""Tests for the CAMARA knowledge ingestion pipeline (milestone 5)."""

from __future__ import annotations

import json
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


def test_ollama_embedder_calls_the_server_and_normalises() -> None:
    import httpx

    from camara_pgps.retrieval import OllamaEmbedder

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        seen.append(payload)
        count = len(payload["input"])
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0, 0.0]] * count})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedder = OllamaEmbedder(
        "nomic-embed-text", base_url="http://ollama:11434", client=client, batch_size=2
    )

    query = embedder.embed_query("hello")
    assert query == [0.6, 0.8, 0.0]  # L2-normalised
    assert embedder.dimension == 3
    assert seen[0] == {"model": "nomic-embed-text", "input": ["hello"]}

    docs = embedder.embed_documents(["a", "b", "c"])
    assert len(docs) == 3
    assert len(seen) == 3  # 1 query + 2 batches (batch_size=2)


def test_get_embedder_selects_ollama() -> None:
    from camara_pgps.config import Settings
    from camara_pgps.retrieval import OllamaEmbedder, get_embedder

    settings = Settings(embedding={"provider": "ollama", "model_name": "mxbai-embed-large"})  # type: ignore[arg-type]
    embedder = get_embedder(settings)
    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.model_name == "mxbai-embed-large"


# --------------------------------------------------------------------------- #
# JSONL corpus (the primary ingestion path)
# --------------------------------------------------------------------------- #

def _write_jsonl(path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )


def test_load_jsonl_parses_records(tmp_path) -> None:
    from camara_pgps.retrieval import load_jsonl

    corpus = tmp_path / "camara.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "id": "qod-1",
                "text": "createSession opens a QoS session.",
                "metadata": {
                    "api_name": "quality-on-demand",
                    "operation_id": "createSession",
                    "document_type": "operation",
                    "custom": "keep-me",
                },
                "structured": {"method": "POST", "path": "/sessions"},
            }
        )
        + "\n"
        + "\n"  # blank line ignored
        + json.dumps({"text": "A record with no id still loads."})
        + "\n",
        encoding="utf-8",
    )
    chunks = load_jsonl(corpus)
    assert [c.id for c in chunks][0] == "qod-1"
    assert len(chunks) == 2
    assert chunks[0].metadata.operation_id == "createSession"
    assert chunks[0].metadata.extra["custom"] == "keep-me"
    assert chunks[0].structured == {"method": "POST", "path": "/sessions"}
    assert chunks[1].id  # auto-generated
    assert chunks[1].metadata.source == str(corpus)


def test_structured_to_operation_maps_the_payload() -> None:
    from camara_pgps.models import CamaraOperation, HTTPMethod, ParameterLocation
    from camara_pgps.retrieval.documents import structured_to_operation

    op = structured_to_operation(
        {
            "operation_id": "createSession",
            "method": "post",
            "path": "/sessions",
            "summary": "Create a QoS session",
            "tags": ["QoS"],
            "parameters": [
                {"name": "qosProfile", "location": "body", "required": True,
                 "description": "the profile"},
                {"name": "x-correlator", "in": "header"},
            ],
            "request_body": {"required": True},
            "preconditions": [
                {"description": "authenticated", "requires": ["auth.token"]}
            ],
            "postconditions": [
                {"description": "session exists", "produces": ["qos.session.id"]}
            ],
        },
        api_name="quality-on-demand",
    )
    assert isinstance(op, CamaraOperation)
    assert op.method == HTTPMethod.POST
    assert op.path == "/sessions"
    assert op.api_name == "quality-on-demand"
    assert op.get_parameter("qosProfile").location == ParameterLocation.BODY
    assert op.get_parameter("x-correlator").location == ParameterLocation.HEADER
    assert op.request_schema == {"required": True}
    assert op.preconditions[0].requires == ["auth.token"]
    assert op.postconditions[0].produces == ["qos.session.id"]


def test_structured_to_operation_without_conditions_has_none() -> None:
    from camara_pgps.retrieval.documents import structured_to_operation

    op = structured_to_operation(
        {"operation_id": "getQosProfile", "method": "GET", "path": "/qos-profiles/{name}"}
    )
    assert op.preconditions == []
    assert op.postconditions == []


def test_structured_to_operation_rejects_incomplete_payload() -> None:
    from camara_pgps.retrieval.documents import structured_to_operation

    with pytest.raises(ValueError, match="method"):
        structured_to_operation({"operation_id": "x", "path": "/x"})


def test_load_jsonl_reports_bad_lines(tmp_path) -> None:
    from camara_pgps.retrieval import load_jsonl

    corpus = tmp_path / "broken.jsonl"
    corpus.write_text('{"text": "ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_jsonl(corpus)


def test_ingest_jsonl_end_to_end(tmp_path, store: VectorStore) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus,
        [
            {"id": "a", "text": "createSession opens a quality on demand session",
             "metadata": {"api_name": "quality-on-demand"}},
            {"id": "b", "text": "verifyLocation checks a device against an area",
             "metadata": {"api_name": "device-location"}},
        ],
    )
    pipeline = IngestionPipeline(vector_store=store, embedder=HashingEmbedder())
    result = pipeline.ingest_jsonl([corpus])
    assert result.chunks == 2 and result.written == 2
    assert store.count() == 2

    # idempotent (upsert by id)
    pipeline.ingest_jsonl([corpus])
    assert store.count() == 2

    retriever = Retriever(vector_store=store, embedder=HashingEmbedder())
    hits = retriever.retrieve(
        RetrievalQuery(
            text="quality on demand session",
            filters={"api_name": "quality-on-demand"},
        )
    )
    assert hits.chunks and hits.chunks[0].id == "a"


def test_run_ingestion_uses_the_knowledge_dir(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from camara_pgps.config import get_settings
    from camara_pgps.retrieval import run_ingestion

    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    _write_jsonl(
        knowledge / "part.jsonl",
        [{"id": "x", "text": "a chunk", "metadata": {"api_name": "quality-on-demand"}}],
    )
    monkeypatch.setenv("CAMARA_PGPS_EMBEDDING__PROVIDER", "hash")
    monkeypatch.setenv(
        "CAMARA_PGPS_VECTOR_STORE__PERSIST_DIRECTORY", str(tmp_path / "chroma")
    )
    monkeypatch.setenv("CAMARA_PGPS_VECTOR_STORE__COLLECTION_NAME", "kd_test")
    monkeypatch.setenv("CAMARA_PGPS_RETRIEVAL__KNOWLEDGE_DIR", str(knowledge))
    get_settings.cache_clear()

    result = run_ingestion(reset=True)
    assert result.chunks == 1 and result.written == 1

    get_settings.cache_clear()


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
