"""Tests for the execution layer (milestone 13)."""

from __future__ import annotations

import httpx
import pytest

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import PROJECT_ROOT, ExecutionSettings, Settings
from camara_pgps.execution import (
    APIExecutor,
    ExecutionError,
    MCPExecutor,
    MockExecutor,
    RESTExecutor,
    build_executor,
)
from camara_pgps.models import HTTPMethod
from camara_pgps.models.execution import ExecutionResult
from camara_pgps.models.plan import (
    BindingSource,
    ParameterBinding,
    PlanResult,
    PlanStatus,
    PlanStep,
)
from camara_pgps.pgps import PGPSPlanner

REGISTRY_FILE = PROJECT_ROOT / "data" / "registry" / "camara_apis.json"


def _step(index: int, operation_id: str, method: HTTPMethod, path: str, **kw: object) -> PlanStep:
    return PlanStep(index=index, operation_id=operation_id, method=method, path=path, **kw)

ARGS = {
    "createSession": {
        "device": {"phoneNumber": "+12025550123"},
        "applicationServer": {"ipv4Address": "10.0.0.1"},
        "qosProfile": "QOS_L",
        "duration": 3600,
    }
}


@pytest.fixture(scope="module")
def registry() -> CamaraRegistry:
    return CamaraRegistry.from_file(REGISTRY_FILE)


@pytest.fixture
def plan(registry: CamaraRegistry) -> PlanResult:
    return PGPSPlanner(registry=registry).plan_for(
        goals=["qos.session.deleted"],
        initial_facts=["auth.token", "device.identifier", "qos.profile"],
        candidate_operations=["createSession", "deleteSession"],
    )


# --------------------------------------------------------------------------- #
# mock executor
# --------------------------------------------------------------------------- #

def test_mock_execution_runs_the_plan_and_leaves_it_unchanged(
    registry: CamaraRegistry, plan: PlanResult
) -> None:
    assert plan.status is PlanStatus.VALID
    before = plan.model_dump()

    result = MockExecutor(registry=registry).execute(plan, arguments=ARGS)

    assert plan.model_dump() == before  # the executor must not change the plan
    assert isinstance(result, ExecutionResult)
    assert result.succeeded is True
    assert [s.operation_id for s in result.steps] == ["createSession", "deleteSession"]
    assert result.steps[0].status_code == 201
    assert result.steps[1].status_code == 204
    # postcondition facts were extracted and threaded forward
    session_id = result.steps[0].response_body["sessionId"]
    assert result.facts["qos.session.id"] == session_id
    assert result.facts["qos.session.deleted"] is True
    # the session id flowed into the delete step's path
    assert result.steps[1].request.url.endswith(session_id)


def test_prior_step_binding_is_resolved_from_the_fact_store(
    registry: CamaraRegistry,
) -> None:
    plan = PlanResult(
        status=PlanStatus.VALID,
        steps=[
            _step(0, "createSession", HTTPMethod.POST, "/sessions"),
            _step(
                1,
                "getSession",
                HTTPMethod.GET,
                "/sessions/{sessionId}",
                depends_on=[0],
                bindings=[
                    ParameterBinding(
                        parameter="sessionId", source=BindingSource.PRIOR_STEP, from_step=0
                    )
                ],
            ),
        ],
    )
    result = MockExecutor(registry=registry).execute(plan, arguments=ARGS)
    session_id = result.steps[0].response_body["sessionId"]
    assert result.steps[1].request.url.endswith(session_id)
    assert result.steps[1].response_body["sessionId"] == session_id


def test_stop_on_error_halts_the_run(registry: CamaraRegistry, plan: PlanResult) -> None:
    executor = MockExecutor(registry=registry, fail_operations={"createSession"})
    result = executor.execute(plan, arguments=ARGS)
    assert result.succeeded is False
    assert len(result.steps) == 1
    assert result.failed_step is not None
    assert result.failed_step.operation_id == "createSession"

    result = executor.execute(plan, arguments=ARGS, stop_on_error=False)
    assert len(result.steps) == 2


def test_executor_refuses_an_invalid_plan(registry: CamaraRegistry) -> None:
    plan = PlanResult(
        status=PlanStatus.INFEASIBLE,
        steps=[_step(0, "createSession", HTTPMethod.POST, "/sessions")],
    )
    result = MockExecutor(registry=registry).execute(plan, arguments=ARGS)
    assert result.succeeded is False
    assert result.steps == []
    assert "infeasible" in (result.error or "")


def test_dry_run_builds_requests_but_does_not_perform(
    registry: CamaraRegistry, plan: PlanResult
) -> None:
    settings = Settings(execution=ExecutionSettings(dry_run=True))
    executor = MockExecutor(registry=registry, settings=settings)
    result = executor.execute(plan, arguments=ARGS)
    assert result.succeeded is True
    assert all(s.response_body == {"dry_run": True} for s in result.steps)
    assert result.steps[0].request is not None
    assert result.steps[0].request.json_body["qosProfile"] == "QOS_L"


def test_request_construction(registry: CamaraRegistry) -> None:
    settings = Settings(camara={"token": "SECRET", "base_url": "https://gw.example"})  # type: ignore[arg-type]
    plan = PlanResult(
        status=PlanStatus.VALID,
        steps=[_step(0, "getSession", HTTPMethod.GET, "/sessions/{sessionId}")],
    )
    result = MockExecutor(registry=registry, settings=settings).execute(
        plan, arguments={"getSession": {"sessionId": "abc"}}
    )
    request = result.steps[0].request
    assert request.url == "https://gw.example/sessions/abc"
    assert request.headers["Authorization"] == "Bearer SECRET"


# --------------------------------------------------------------------------- #
# factory / mode selection
# --------------------------------------------------------------------------- #

def test_build_executor_selects_by_mode(registry: CamaraRegistry) -> None:
    assert isinstance(build_executor(mode="mock", registry=registry), MockExecutor)
    assert isinstance(build_executor(mode="rest", registry=registry), RESTExecutor)
    assert isinstance(build_executor(mode="mcp", registry=registry), MCPExecutor)
    with pytest.raises(ExecutionError):
        build_executor(mode="carrier-pigeon", registry=registry)


def test_default_mode_is_mock(registry: CamaraRegistry) -> None:
    assert Settings().execution.mode == "mock"
    assert isinstance(build_executor(registry=registry), APIExecutor)


# --------------------------------------------------------------------------- #
# REST executor (mocked transport)
# --------------------------------------------------------------------------- #

def test_rest_executor_issues_http_requests(registry: CamaraRegistry, plan: PlanResult) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/sessions":
            return httpx.Response(201, json={"sessionId": "sess-1", "qosStatus": "REQUESTED"})
        return httpx.Response(204)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = RESTExecutor(registry=registry, client=client).execute(plan, arguments=ARGS)

    assert result.succeeded is True
    assert [r.method for r in seen] == ["POST", "DELETE"]
    assert result.steps[0].status_code == 201
    assert result.facts["qos.session.id"] == "sess-1"
    assert seen[1].url.path == "/sessions/sess-1"


# --------------------------------------------------------------------------- #
# MCP executor
# --------------------------------------------------------------------------- #

def test_mcp_executor_without_session_fails_gracefully(
    registry: CamaraRegistry, plan: PlanResult
) -> None:
    before = plan.model_dump()
    result = MCPExecutor(registry=registry).execute(plan, arguments=ARGS)
    assert plan.model_dump() == before
    assert result.succeeded is False
    assert result.steps[0].succeeded is False
    assert "MCP session not configured" in (result.steps[0].error or "")


def test_mcp_executor_with_a_session(registry: CamaraRegistry, plan: PlanResult) -> None:
    calls: list[tuple[str, dict]] = []

    class FakeSession:
        def call_tool(self, name: str, arguments: dict) -> dict:
            calls.append((name, arguments))
            return {"sessionId": "mcp-sess", "isError": False}

    result = MCPExecutor(registry=registry, session=FakeSession()).execute(
        plan, arguments=ARGS
    )
    assert result.succeeded is True
    assert calls[0][0] == "createSession"
    assert calls[0][1]["qosProfile"] == "QOS_L"
    assert result.facts["qos.session.id"] == "mcp-sess"
