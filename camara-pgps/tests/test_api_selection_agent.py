"""Tests for the API Selection agent (milestone 8)."""

from __future__ import annotations

import pytest

from camara_pgps.agents import (
    APISelection,
    APISelectionAgent,
    CandidateAPI,
    SelectionSource,
)
from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import PROJECT_ROOT
from camara_pgps.models import CamaraAPI, Intent, RetrievalResult
from camara_pgps.retrieval import HashingEmbedder, IngestionPipeline, Retriever, VectorStore
from camara_pgps.retrieval.rag import RagContext, RagRetriever

REGISTRY_FILE = PROJECT_ROOT / "data" / "registry" / "camara_apis.json"


@pytest.fixture(scope="module")
def registry() -> CamaraRegistry:
    return CamaraRegistry.from_file(REGISTRY_FILE)


@pytest.fixture
def rag(tmp_path, registry: CamaraRegistry) -> RagRetriever:
    embedder = HashingEmbedder()
    store = VectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_selection",
        distance="cosine",
    )
    IngestionPipeline(vector_store=store, embedder=embedder).ingest_registry(registry)
    return RagRetriever(
        retriever=Retriever(vector_store=store, embedder=embedder),
        registry=registry,
    )


class FakeChatModel:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list = []

    def invoke(self, messages: list) -> object:
        self.calls.append(messages)

        class _Resp:
            content = self._response

        return _Resp()


QOS_INTENT = Intent(
    text="my video call keeps freezing, I want a better quality on demand session",
    goal="get a better quality on demand session",
    required_capabilities=["qos"],
    target_domains=["quality-on-demand"],
)


def test_selects_registry_verified_candidate_apis(
    registry: CamaraRegistry, rag: RagRetriever
) -> None:
    agent = APISelectionAgent(registry=registry, rag=rag)
    selection = agent.select(QOS_INTENT)

    assert isinstance(selection, APISelection)
    assert selection.apis
    assert all(isinstance(item, CandidateAPI) for item in selection.apis)
    assert all(isinstance(item.api, CamaraAPI) for item in selection.apis)
    # every candidate actually exists in the registry
    assert all(registry.has_api(name) for name in selection.api_names())
    assert "quality-on-demand" in selection.api_names()
    # candidate operations are all verified too
    assert selection.operations
    assert all(registry.has_operation(oid) for oid in selection.operation_ids())
    # operation ids listed on a candidate API really belong to it
    for candidate in selection.apis:
        assert set(candidate.operation_ids) <= set(registry.operation_ids(candidate.name))


def test_llm_hallucinations_are_rejected_not_returned(
    registry: CamaraRegistry, rag: RagRetriever
) -> None:
    payload = (
        '{"apis": ["Quality on Demand", "Teleportation API", "SuperLocation"],'
        ' "operations": ["createSession", "warpDevice"],'
        ' "rationale": "QoS fits a freezing call"}'
    )
    agent = APISelectionAgent(registry=registry, rag=rag, llm=FakeChatModel(payload))
    selection = agent.select(QOS_INTENT)

    assert agent.selector is not None
    names = selection.api_names()
    assert "quality-on-demand" in names
    assert "Teleportation API" not in names
    assert "SuperLocation" not in names

    rejected_by_name = {r.name: r for r in selection.rejected}
    assert "Teleportation API" in rejected_by_name
    assert rejected_by_name["Teleportation API"].kind == "api"
    assert rejected_by_name["Teleportation API"].source == SelectionSource.LLM
    assert "registry" in rejected_by_name["Teleportation API"].reason
    assert "warpDevice" in rejected_by_name
    assert rejected_by_name["warpDevice"].kind == "operation"

    # the good LLM suggestions still came through
    assert "createSession" in selection.operation_ids()
    assert selection.metadata["llm_rationale"] == "QoS fits a freezing call"
    assert selection.metadata["used_llm"] is True


def test_api_title_is_normalised_to_registered_name(
    registry: CamaraRegistry, rag: RagRetriever
) -> None:
    intent = Intent(
        text="check where my device is",
        goal="locate my device",
        target_domains=["Device Location"],  # human title, not the registered slug
    )
    agent = APISelectionAgent(registry=registry, rag=rag)
    selection = agent.select(intent)

    assert "device-location" in selection.api_names()
    assert "Device Location" not in {r.name for r in selection.rejected}


def test_selection_is_a_shortlist_not_a_plan(
    registry: CamaraRegistry, rag: RagRetriever
) -> None:
    agent = APISelectionAgent(registry=registry, rag=rag)
    selection = agent.select(QOS_INTENT)

    dumped = selection.model_dump()
    for planning_field in ("steps", "plan", "order", "sequence"):
        assert planning_field not in dumped

    scores = [item.score for item in selection.apis]
    assert scores == sorted(scores, reverse=True)


def test_capability_families_expand_to_apis_without_rag(
    registry: CamaraRegistry, rag: RagRetriever
) -> None:
    agent = APISelectionAgent(registry=registry, rag=rag)
    empty_rag = RagContext(
        query="x", documentation=RetrievalResult(query="x", chunks=[])
    )
    selection = agent.select(
        Intent(text="improve streaming", goal="improve streaming",
               required_capabilities=["qos"]),
        rag_context=empty_rag,
    )
    assert "quality-on-demand" in selection.api_names()
    assert "createSession" in selection.operation_ids()


def test_unknown_domain_from_intent_is_rejected(
    registry: CamaraRegistry, rag: RagRetriever
) -> None:
    agent = APISelectionAgent(registry=registry, rag=rag)
    empty_rag = RagContext(
        query="x", documentation=RetrievalResult(query="x", chunks=[])
    )
    selection = agent.select(
        Intent(text="do a thing", goal="do a thing", target_domains=["made-up-api"]),
        rag_context=empty_rag,
    )
    rejected = {r.name: r for r in selection.rejected}
    assert "made-up-api" in rejected
    assert rejected["made-up-api"].source == SelectionSource.INTENT
    assert "made-up-api" not in selection.api_names()


def test_accepts_raw_text(registry: CamaraRegistry, rag: RagRetriever) -> None:
    agent = APISelectionAgent(registry=registry, rag=rag)
    selection = agent.select("I need a quality on demand session for streaming")
    assert selection.intent is not None
    assert selection.apis
