"""Tests for the LangGraph workflow (milestone 15)."""

from __future__ import annotations

import pytest

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import PROJECT_ROOT
from camara_pgps.execution import MockExecutor
from camara_pgps.models import HTTPMethod
from camara_pgps.models.plan import PlanResult, PlanStatus, PlanStep
from camara_pgps.models.state import AgentState, PipelineStage
from camara_pgps.retrieval import HashingEmbedder, IngestionPipeline, Retriever, VectorStore
from camara_pgps.retrieval.rag import RagRetriever
from camara_pgps.workflow import (
    CamaraWorkflow,
    WorkflowNodes,
    get_workflow,
    route_after_pgps,
    route_after_validation,
)

REGISTRY_FILE = PROJECT_ROOT / "data" / "registry" / "camara_apis.json"

HAPPY_REQUEST = (
    "Give my phone +12025550123 a QOS_L quality on demand session for 1 hour"
)
HAPPY_ARGS = {"createSession": {"applicationServer": {"ipv4Address": "10.0.0.1"}}}


@pytest.fixture(scope="module")
def registry() -> CamaraRegistry:
    return CamaraRegistry.from_file(REGISTRY_FILE)


@pytest.fixture
def nodes(tmp_path, registry: CamaraRegistry) -> WorkflowNodes:
    embedder = HashingEmbedder()
    store = VectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_workflow",
        distance="cosine",
    )
    IngestionPipeline(vector_store=store, embedder=embedder).ingest_registry(registry)
    rag = RagRetriever(
        retriever=Retriever(vector_store=store, embedder=embedder), registry=registry
    )
    return WorkflowNodes(
        registry=registry, rag_retriever=rag, executor=MockExecutor(registry=registry)
    )


@pytest.fixture
def workflow(nodes: WorkflowNodes) -> CamaraWorkflow:
    return CamaraWorkflow(nodes=nodes)


# --------------------------------------------------------------------------- #
# graph shape
# --------------------------------------------------------------------------- #

def test_graph_matches_the_pipeline(workflow: CamaraWorkflow) -> None:
    graph = workflow.app.get_graph()
    assert {
        "understand_intent",
        "rag_retrieval",
        "api_selection",
        "parameter_extraction",
        "assemble_plan_request",
        "pgps_solve",
        "clarify",
        "validate",
        "execute",
    } <= set(graph.nodes)
    mermaid = workflow.mermaid()
    assert "pgps_solve -.-> clarify" in mermaid
    assert "pgps_solve -.-> validate" in mermaid
    assert "validate -.-> execute" in mermaid
    assert "clarify --> __end__" in mermaid


# --------------------------------------------------------------------------- #
# full run - happy path reaches execution
# --------------------------------------------------------------------------- #

def test_full_pipeline_runs_to_execution(workflow: CamaraWorkflow) -> None:
    state = workflow.run(HAPPY_REQUEST, arguments=HAPPY_ARGS)

    assert isinstance(state, AgentState)
    assert state.stage is PipelineStage.DONE
    assert state.intent is not None
    assert state.rag is not None
    assert state.selection is not None
    assert state.parameters is not None
    assert state.plan is not None and state.plan.status is PlanStatus.VALID
    assert [s.operation_id for s in state.plan.steps] == ["createSession"]
    assert state.execution is not None and state.execution.succeeded is True
    assert "qos.session.id" in state.facts
    assert any(m["role"] == "assistant" for m in state.messages)


def test_execution_failure_marks_state_failed(nodes: WorkflowNodes) -> None:
    nodes._executor = MockExecutor(
        registry=nodes.registry, fail_operations={"createSession"}
    )
    state = CamaraWorkflow(nodes=nodes).run(HAPPY_REQUEST, arguments=HAPPY_ARGS)
    assert state.stage is PipelineStage.FAILED
    assert state.execution is not None and state.execution.succeeded is False
    assert state.errors


# --------------------------------------------------------------------------- #
# clarification branch
# --------------------------------------------------------------------------- #

def test_missing_information_branches_to_clarification(workflow: CamaraWorkflow) -> None:
    state = workflow.run("Improve my network quality.")

    assert state.stage is PipelineStage.CLARIFICATION
    assert state.execution is None
    assert state.clarification
    last = state.messages[-1]
    assert last["role"] == "assistant"
    assert "still need you to provide" in last["content"].lower()
    assert state.metadata["clarification"]["missing"]


def test_unrelated_request_does_not_execute(workflow: CamaraWorkflow) -> None:
    state = workflow.run("please recommend a good pizza topping")
    assert state.stage is not PipelineStage.DONE
    assert state.execution is None


# --------------------------------------------------------------------------- #
# routing (deterministic, unit)
# --------------------------------------------------------------------------- #

def _plan(
    status: PlanStatus, *, steps: bool = True, unmet: list[str] | None = None
) -> PlanResult:
    step = PlanStep(
        index=0, operation_id="createSession", method=HTTPMethod.POST, path="/sessions"
    )
    return PlanResult(
        status=status,
        steps=[step] if steps else [],
        unmet_preconditions=unmet or [],
    )


def _state(**kwargs: object) -> AgentState:
    return AgentState(request_text="x", **kwargs)


def test_route_after_pgps() -> None:
    assert route_after_pgps(_state(stage=PipelineStage.FAILED)) == "clarify"
    assert route_after_pgps(_state(plan=_plan(PlanStatus.INFEASIBLE))) == "clarify"
    assert route_after_pgps(_state(plan=_plan(PlanStatus.VALID, steps=False))) == "clarify"
    assert (
        route_after_pgps(_state(plan=_plan(PlanStatus.VALID, unmet=["auth.token"])))
        == "clarify"
    )
    assert route_after_pgps(_state(plan=_plan(PlanStatus.VALID))) == "validate"


def test_route_after_validation() -> None:
    assert route_after_validation(_state(plan=_plan(PlanStatus.VALID))) == "execute"
    assert route_after_validation(_state(plan=_plan(PlanStatus.INVALID))) == "clarify"
    assert route_after_validation(_state(stage=PipelineStage.FAILED)) == "clarify"


def test_get_workflow_is_cached() -> None:
    assert get_workflow() is get_workflow()
