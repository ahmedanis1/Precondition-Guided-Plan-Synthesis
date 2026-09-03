"""Tests for the bundled mock CAMARA services (milestone 14)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import PROJECT_ROOT
from camara_pgps.execution import RESTExecutor
from camara_pgps.pgps import PGPSPlanner
from mock_camara.main import app

REGISTRY_FILE = PROJECT_ROOT / "data" / "registry" / "camara_apis.json"

CREATE_BODY = {
    "device": {"phoneNumber": "+12025550123"},
    "applicationServer": {"ipv4Address": "10.0.0.1"},
    "qosProfile": "QOS_L",
    "duration": 3600,
}


@pytest.fixture
def client() -> TestClient:
    c = TestClient(app)
    c.post("/_mock/reset")
    return c


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body["apis"]) == {"device-location", "quality-on-demand", "network-insights"}


# --------------------------------------------------------------------------- #
# Quality on Demand - stateful session lifecycle
# --------------------------------------------------------------------------- #

def test_qod_session_lifecycle(client: TestClient) -> None:
    created = client.post("/sessions", json=CREATE_BODY)
    assert created.status_code == 201
    session_id = created.json()["sessionId"]
    assert created.json()["qosStatus"] == "REQUESTED"

    fetched = client.get(f"/sessions/{session_id}")
    assert fetched.status_code == 200
    assert fetched.json()["qosStatus"] == "AVAILABLE"
    assert fetched.json()["qosProfile"] == "QOS_L"

    assert client.delete(f"/sessions/{session_id}").status_code == 204
    assert client.get(f"/sessions/{session_id}").status_code == 404


def test_qod_unknown_session_is_404(client: TestClient) -> None:
    body = client.get("/sessions/does-not-exist")
    assert body.status_code == 404
    assert body.json()["code"] == "NOT_FOUND"


def test_qod_validation_error(client: TestClient) -> None:
    bad = client.post("/sessions", json={"device": {}, "qosProfile": "X", "duration": -5})
    assert bad.status_code == 422  # duration must be > 0


def test_camara_style_prefix_also_works(client: TestClient) -> None:
    created = client.post("/quality-on-demand/v0/sessions", json=CREATE_BODY)
    assert created.status_code == 201
    session_id = created.json()["sessionId"]
    assert client.get(f"/quality-on-demand/v0/sessions/{session_id}").status_code == 200


# --------------------------------------------------------------------------- #
# Device Location
# --------------------------------------------------------------------------- #

def test_device_location(client: TestClient) -> None:
    verify = client.post(
        "/verify", json={"device": {"phoneNumber": "+1"}, "area": {"areaType": "CIRCLE"}}
    )
    assert verify.status_code == 200
    assert verify.json()["verificationResult"] in {"TRUE", "PARTIAL", "FALSE"}

    retrieve = client.post("/retrieve", json={"device": {"phoneNumber": "+1"}})
    assert retrieve.status_code == 200
    assert retrieve.json()["area"]["areaType"] == "CIRCLE"
    # the area is deterministic on the device value (only the timestamp varies)
    again = client.post("/retrieve", json={"device": {"phoneNumber": "+1"}})
    assert retrieve.json()["area"] == again.json()["area"]


# --------------------------------------------------------------------------- #
# Network Insights - stateful subscriptions
# --------------------------------------------------------------------------- #

def test_network_insights_subscription_lifecycle(client: TestClient) -> None:
    created = client.post(
        "/subscriptions",
        json={"device": {"phoneNumber": "+1"}, "types": ["NETWORK_QUALITY"], "sink": "http://cb"},
    )
    assert created.status_code == 201
    subscription_id = created.json()["subscriptionId"]

    fetched = client.get(f"/subscriptions/{subscription_id}")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "ACTIVE"

    assert client.get("/subscriptions/nope").status_code == 404


def test_reset_clears_state(client: TestClient) -> None:
    client.post("/sessions", json=CREATE_BODY)
    client.post(
        "/subscriptions",
        json={"device": {}, "types": ["X"], "sink": "http://cb"},
    )
    assert client.get("/sessions").json()
    client.post("/_mock/reset")
    assert client.get("/sessions").json() == []
    assert client.get("/subscriptions").json() == []


# --------------------------------------------------------------------------- #
# end-to-end: PGPS plan -> RESTExecutor -> mock CAMARA
# --------------------------------------------------------------------------- #

def test_rest_executor_runs_a_plan_against_the_mock(client: TestClient) -> None:
    registry = CamaraRegistry.from_file(REGISTRY_FILE)
    plan = PGPSPlanner(registry=registry).plan_for(
        goals=["qos.session.deleted"],
        initial_facts=["auth.token", "device.identifier", "qos.profile"],
        candidate_operations=["createSession", "deleteSession"],
    )
    result = RESTExecutor(registry=registry, client=client).execute(
        plan, arguments={"createSession": CREATE_BODY}
    )

    assert result.succeeded is True
    assert [s.status_code for s in result.steps] == [201, 204]
    session_id = result.facts["qos.session.id"]
    assert result.steps[1].request.url.endswith(session_id)
    # the session really was created and then removed in the mock's store
    assert client.get("/sessions").json() == []
