"""Tests for the HTTP API (milestone 17)."""

from __future__ import annotations

import pytest
from conftest import stub_intent_agent
from fastapi.testclient import TestClient

from camara_pgps.agents import RepairAgent
from camara_pgps.api.main import app, workflow_dependency
from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import PROJECT_ROOT
from camara_pgps.execution import MockExecutor
from camara_pgps.retrieval import HashingEmbedder, IngestionPipeline, Retriever, VectorStore
from camara_pgps.retrieval.rag import RagRetriever
from camara_pgps.workflow import CamaraWorkflow, WorkflowNodes

REGISTRY_FILE = PROJECT_ROOT / "data" / "registry" / "camara_apis.json"

HAPPY = {
    "request": "Give my phone +12025550123 a QOS_L quality on demand session for 1 hour",
    "arguments": {
        "createSession": {
            "device": {"phoneNumber": "+12025550123"},
            "applicationServer": {"ipv4Address": "10.0.0.1"},
            "qosProfile": "QOS_L",
            "duration": 3600,
        }
    },
}


@pytest.fixture
def client(tmp_path) -> TestClient:
    registry = CamaraRegistry.from_file(REGISTRY_FILE)
    embedder = HashingEmbedder()
    store = VectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_api",
        distance="cosine",
    )
    IngestionPipeline(vector_store=store, embedder=embedder).ingest_registry(registry)
    rag = RagRetriever(
        retriever=Retriever(vector_store=store, embedder=embedder), registry=registry
    )
    workflow = CamaraWorkflow(
        nodes=WorkflowNodes(
            registry=registry,
            rag_retriever=rag,
            executor=MockExecutor(registry=registry),
            intent_agent=stub_intent_agent(),
            repair_agent=RepairAgent(registry=registry),
        )
    )
    app.dependency_overrides[workflow_dependency] = lambda: workflow
    yield TestClient(app)
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# registry / meta endpoints
# --------------------------------------------------------------------------- #

def test_health(client: TestClient) -> None:
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert set(body["apis"]) == {"device-location", "quality-on-demand", "network-insights"}
    assert body["execution_mode"] == "mock"


def test_list_and_get_apis(client: TestClient) -> None:
    listing = client.get("/api/v1/apis").json()["apis"]
    assert {a["name"] for a in listing} == {
        "device-location",
        "quality-on-demand",
        "network-insights",
    }

    detail = client.get("/api/v1/apis/quality-on-demand").json()
    assert detail["version"]
    assert "createSession" in detail["capabilities"]
    assert detail["capabilities"]["createSession"] == "qos.session.create"

    assert client.get("/api/v1/apis/does-not-exist").status_code == 404


# --------------------------------------------------------------------------- #
# POST /run
# --------------------------------------------------------------------------- #

def test_run_returns_every_section(client: TestClient) -> None:
    body = client.post("/api/v1/run", json=HAPPY).json()

    assert body["stage"] == "done"
    for section in (
        "intent",
        "retrieved",
        "selected_apis",
        "parameters",
        "plan",
        "validation",
        "execution",
        "explanation",
    ):
        assert body[section] is not None, section
    assert [s["operation_id"] for s in body["plan"]["steps"]] == ["createSession"]
    assert body["execution"]["succeeded"] is True
    assert "createSession" in body["explanation"]


def test_run_with_missing_information_returns_clarification(client: TestClient) -> None:
    body = client.post("/api/v1/run", json={"request": "Improve my network quality."}).json()
    assert body["stage"] == "clarification"
    assert body["execution"] is None
    assert body["clarification"]
    assert any(m["role"] == "assistant" for m in body["messages"])


def test_run_rejects_empty_request(client: TestClient) -> None:
    assert client.post("/api/v1/run", json={"request": ""}).status_code == 422


# --------------------------------------------------------------------------- #
# POST /plan, /validate, /execute
# --------------------------------------------------------------------------- #

def test_plan_stops_before_validation(client: TestClient) -> None:
    body = client.post("/api/v1/plan", json=HAPPY).json()
    assert body["stage"] == "planning"
    assert body["plan"]["status"] == "valid"
    assert body["validation"] is None
    assert body["execution"] is None


def test_validate_a_supplied_plan(client: TestClient) -> None:
    plan = client.post("/api/v1/plan", json=HAPPY).json()["plan"]
    body = client.post(
        "/api/v1/validate",
        json={"plan": plan, "arguments": HAPPY["arguments"]},
    ).json()
    assert body["stage"] == "validation"
    assert body["validation"] is not None
    assert body["validation"]["structural_ok"] is True
    assert body["execution"] is None


def test_execute_a_supplied_plan(client: TestClient) -> None:
    plan = client.post("/api/v1/plan", json=HAPPY).json()["plan"]
    body = client.post(
        "/api/v1/execute",
        json={"plan": plan, "arguments": HAPPY["arguments"]},
    ).json()
    assert body["stage"] == "done"
    assert body["execution"]["succeeded"] is True
    assert "qos.session.id" in body["execution"]["facts"]


def test_validate_needs_request_or_plan(client: TestClient) -> None:
    assert client.post("/api/v1/validate", json={}).status_code == 422
    assert client.post("/api/v1/execute", json={}).status_code == 422
