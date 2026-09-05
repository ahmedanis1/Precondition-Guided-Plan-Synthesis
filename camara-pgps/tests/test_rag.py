"""Tests for RAG retrieval (milestone 6)."""

from __future__ import annotations

import pytest

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import PROJECT_ROOT
from camara_pgps.models import (
    APIParameter,
    CamaraOperation,
    Condition,
    Intent,
    IntentEntity,
    RetrievalResult,
)
from camara_pgps.retrieval import (
    HashingEmbedder,
    IngestionPipeline,
    RagContext,
    RagRetriever,
    Retriever,
    VectorStore,
)

REGISTRY_FILE = PROJECT_ROOT / "data" / "registry" / "camara_apis.json"


@pytest.fixture(scope="module")
def registry() -> CamaraRegistry:
    return CamaraRegistry.from_file(REGISTRY_FILE)


@pytest.fixture
def rag(tmp_path, registry: CamaraRegistry) -> RagRetriever:
    embedder = HashingEmbedder()
    store = VectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_rag",
        distance="cosine",
    )
    IngestionPipeline(vector_store=store, embedder=embedder).ingest_registry(registry)
    return RagRetriever(
        retriever=Retriever(vector_store=store, embedder=embedder),
        registry=registry,
    )


def test_retrieve_accepts_intent_and_returns_four_evidence_kinds(rag: RagRetriever) -> None:
    intent = Intent(
        text="I need to set up quality on demand for a session",
        goal="create a quality on demand session",
        action="create_qos_session",
    )
    ctx = rag.retrieve(intent)

    assert isinstance(ctx, RagContext)
    # 1. documentation
    assert isinstance(ctx.documentation, RetrievalResult)
    assert len(ctx.documentation.chunks) > 0
    # 2. operations - resolved against the registry, not invented
    assert ctx.operations
    assert all(isinstance(item.operation, CamaraOperation) for item in ctx.operations)
    assert all(rag.registry.has_operation(oid) for oid in ctx.operation_ids())
    assert "quality-on-demand" in ctx.apis
    assert "createSession" in ctx.operation_ids()
    # 3. parameters
    assert ctx.parameters
    assert all(isinstance(item.parameter, APIParameter) for item in ctx.parameters)
    # 4. conditions
    assert ctx.conditions
    assert all(isinstance(item.condition, Condition) for item in ctx.conditions)


def test_retrieve_accepts_raw_text(rag: RagRetriever) -> None:
    ctx = rag.retrieve("verify the location of a device")
    assert ctx.query == "verify the location of a device"
    assert ctx.intent is not None and ctx.intent.goal == "verify the location of a device"
    assert "device-location" in ctx.apis


def test_operations_are_ranked_candidates_not_a_plan(rag: RagRetriever) -> None:
    ctx = rag.retrieve("quality on demand session management")

    # RAG returns several scored candidates, not one decision / ordering.
    assert len(ctx.operations) >= 2
    scores = [item.score for item in ctx.operations]
    assert scores == sorted(scores, reverse=True)
    assert all(item.score > 0 for item in ctx.operations)
    assert all(item.supporting_chunks for item in ctx.operations)

    # The bundle carries no plan / execution ordering.
    assert not hasattr(ctx, "plan")
    assert not hasattr(ctx, "steps")
    dumped = ctx.model_dump()
    assert "plan" not in dumped and "steps" not in dumped


def test_parameters_are_ordered_required_and_entity_matches_first(rag: RagRetriever) -> None:
    intent = Intent(
        text="create a quality on demand session",
        goal="create a quality on demand session",
        entities=[IntentEntity(name="qosProfile", value="QOS_L")],
    )
    ctx = rag.retrieve(intent)
    create_params = [
        p.parameter for p in ctx.parameters if p.operation_id == "createSession"
    ]
    assert create_params
    # all required params come before any optional one
    required_flags = [p.required for p in create_params]
    assert required_flags == sorted(required_flags, reverse=True)
    # the entity-named parameter is first among the required group
    assert create_params[0].name == "qosProfile"


def test_conditions_include_pre_and_post(rag: RagRetriever) -> None:
    ctx = rag.retrieve("create a quality on demand session")
    create_conditions = [
        c.condition for c in ctx.conditions if c.operation_id == "createSession"
    ]
    kinds = {c.kind for c in create_conditions}
    assert "precondition" in kinds
    assert "postcondition" in kinds
    assert any("auth.token" in c.requires for c in create_conditions)


def test_metadata_filter_scopes_retrieval(rag: RagRetriever) -> None:
    ctx = rag.retrieve(
        "how do I work with sessions and locations",
        filters={"api_name": "device-location"},
    )
    assert ctx.operations
    assert {item.api_name for item in ctx.operations} == {"device-location"}
    assert {c.metadata.get("api_name") for c in ctx.documentation.chunks} == {
        "device-location"
    }


def test_operation_chunk_without_structured_payload_is_noted_not_raised(
    tmp_path,
) -> None:
    """A chunk that names an operation but carries no ``structured`` payload
    cannot be rebuilt - it is noted and skipped, never a registry lookup."""

    import json

    corpus = tmp_path / "no-structured.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "id": "op-nostruct",
                "text": "createSession opens a quality on demand session.",
                "metadata": {
                    "api_name": "quality-on-demand",
                    "operation_id": "createSession",
                    "document_type": "operation",
                },
            }
        ),
        encoding="utf-8",
    )
    embedder = HashingEmbedder()
    store = VectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_rag_no_structured",
        distance="cosine",
    )
    IngestionPipeline(vector_store=store, embedder=embedder).ingest_jsonl([corpus])
    rag = RagRetriever(retriever=Retriever(vector_store=store, embedder=embedder))

    ctx = rag.retrieve("create a quality on demand session")
    assert "createSession" not in ctx.operation_ids()
    assert any(
        "createSession" in note and "no structured payload" in note
        for note in ctx.notes
    )


def test_rag_builds_operation_from_jsonl_structured_payload(
    tmp_path, registry: CamaraRegistry
) -> None:
    """JSONL is the source of truth: an operation absent from the registry is
    still rebuilt from the retrieved chunk's ``structured`` payload."""

    import json

    corpus = tmp_path / "extra.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "id": "op-x",
                "text": "provisionEdgeNode reserves an edge compute node near the device.",
                "metadata": {
                    "api_name": "edge-cloud",
                    "api_version": "2.1.0",
                    "source": "edge-cloud.yaml",
                    "operation_id": "provisionEdgeNode",
                    "document_type": "operation",
                },
                "structured": {
                    "operation_id": "provisionEdgeNode",
                    "method": "POST",
                    "path": "/edge-nodes",
                    "summary": "Provision an edge node",
                    "description": "Reserve an edge compute node.",
                    "tags": ["Edge"],
                    "deprecated": False,
                    "parameters": [
                        {"name": "region", "location": "query", "required": True}
                    ],
                    "request_body": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema_ref": "#/components/schemas/EdgeNodeRequest"
                            }
                        },
                    },
                    "responses": [
                        {"status_code": "201", "description": "created"}
                    ],
                    "relationships": [
                        {"relation": "request_body", "target": "EdgeNodeRequest"}
                    ],
                    "preconditions": [
                        {"description": "authenticated", "requires": ["auth.token"]}
                    ],
                    "postconditions": [
                        {"description": "edge node reserved", "produces": ["edge.node.id"]}
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    embedder = HashingEmbedder()
    store = VectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_rag_jsonl",
        distance="cosine",
    )
    IngestionPipeline(vector_store=store, embedder=embedder).ingest_jsonl([corpus])
    rag = RagRetriever(
        retriever=Retriever(vector_store=store, embedder=embedder),
        registry=registry,
    )
    assert not registry.has_operation("provisionEdgeNode")

    ctx = rag.retrieve("provision an edge compute node near my device")
    assert "provisionEdgeNode" in ctx.operation_ids()
    item = next(
        i for i in ctx.operations if i.operation_id == "provisionEdgeNode"
    )
    # raw structure preserved verbatim for the parameter resolver
    assert item.method == "POST"
    assert item.path == "/edge-nodes"
    assert item.api_name == "edge-cloud"
    assert item.api_version == "2.1.0"
    assert item.source == "edge-cloud.yaml"
    assert item.description == "Reserve an edge compute node."
    assert item.tags == ["Edge"]
    assert item.parameters == [
        {"name": "region", "location": "query", "required": True}
    ]
    assert item.request_body["content"]["application/json"]["schema_ref"] == (
        "#/components/schemas/EdgeNodeRequest"
    )
    assert item.responses == [{"status_code": "201", "description": "created"}]
    assert item.relationships == [
        {"relation": "request_body", "target": "EdgeNodeRequest"}
    ]
    # typed bridge still available for PGPS / validation
    assert item.operation.method == "POST"
    assert item.operation.get_parameter("region") is not None
    assert any("edge.node.id" in c.produces for c in item.operation.postconditions)
    assert any("auth.token" in c.requires for c in item.operation.preconditions)
    assert any(c.operation_id == "provisionEdgeNode" for c in ctx.conditions)

    # the debug rendering carries operation + version + source provenance
    rendered = ctx.as_prompt_context()
    assert "provisionEdgeNode [edge-cloud] POST /edge-nodes" in rendered
    assert "version 2.1.0" in rendered
    assert "source edge-cloud.yaml" in rendered


def test_build_queries_dedupes_and_expands(registry: CamaraRegistry) -> None:
    intent = Intent(
        text="my call keeps freezing",
        goal="my call keeps freezing",
        action="improve_call_quality",
        constraints=["low latency required"],
    )
    queries = RagRetriever._build_queries(intent)
    assert queries[0] == "my call keeps freezing"
    assert "improve call quality" in queries
    assert "low latency required" in queries  # IntentConstraint.raw
    assert len(queries) == len(set(q.lower() for q in queries))


def test_as_prompt_context_is_readonly_text(rag: RagRetriever) -> None:
    ctx = rag.retrieve("create a quality on demand session")
    text = ctx.as_prompt_context()
    assert "Candidate operations (ranked, NOT a plan):" in text
    assert "createSession" in text
