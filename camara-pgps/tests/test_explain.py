"""Tests for plan explainability / provenance (milestone 19)."""

from __future__ import annotations

import pytest

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import PROJECT_ROOT
from camara_pgps.models import HTTPMethod
from camara_pgps.models.intent import Intent
from camara_pgps.models.plan import PlanResult, PlanStatus, PlanStep
from camara_pgps.models.state import AgentState
from camara_pgps.pgps import PGPSPlanner
from camara_pgps.pgps.conditions import (
    ConditionModel,
    OperationConditions,
    Postcondition,
    Precondition,
)
from camara_pgps.pgps.explain import EXPLANATION_SOURCE, explain_plan
from camara_pgps.pgps.solver import PGPSSolver, SolverRequest
from camara_pgps.workflow.report import state_report

REGISTRY_FILE = PROJECT_ROOT / "data" / "registry" / "camara_apis.json"


@pytest.fixture(scope="module")
def registry() -> CamaraRegistry:
    return CamaraRegistry.from_file(REGISTRY_FILE)


def _plan_from_model(model: ConditionModel, order: list[str]) -> PlanResult:
    steps: list[PlanStep] = []
    for index, operation_id in enumerate(order):
        conditions = model.for_operation(operation_id)
        depends = [
            j
            for j in range(index)
            if set(model.for_operation(order[j]).produces) & conditions.requires
        ]
        steps.append(
            PlanStep(
                index=index,
                operation_id=operation_id,
                method=HTTPMethod.POST,
                path="/x",
                requires=sorted(conditions.requires),
                produces=sorted(conditions.produces),
                depends_on=depends,
            )
        )
    return PlanResult(status=PlanStatus.VALID, steps=steps)


# --------------------------------------------------------------------------- #
# the task's flagship example
# --------------------------------------------------------------------------- #

def test_explanation_matches_the_task_example() -> None:
    model = ConditionModel(
        [
            OperationConditions(
                operation_id="DeviceLocation",
                preconditions=[
                    Precondition(description="", requires=["device_id_available"])
                ],
                postconditions=[
                    Postcondition(description="", produces=["location_verified"])
                ],
            ),
            OperationConditions(
                operation_id="QualityOnDemand",
                preconditions=[
                    Precondition(
                        description="",
                        requires=["device_id_available", "location_verified"],
                    )
                ],
                postconditions=[
                    Postcondition(description="", produces=["qos_session_created"])
                ],
            ),
        ]
    )
    order = PGPSSolver(model).solve(
        SolverRequest(
            goals={"qos_session_created"}, initial_facts={"device_id_available"}
        )
    ).order
    plan = _plan_from_model(model, order)
    plan.metadata["goal_facts"] = ["qos_session_created"]

    explanation = explain_plan(
        plan,
        intent=Intent(text="Improve network quality", goal="Improve network quality"),
        initial_facts={"device_id_available"},
    )

    # provenance: NOT an LLM
    assert explanation.source == EXPLANATION_SOURCE
    assert explanation.source != "llm"

    assert explanation.goal == "Improve network quality"
    assert [s.operation_id for s in explanation.execution_order] == [
        "DeviceLocation",
        "QualityOnDemand",
    ]

    verified = next(f for f in explanation.required_facts if f.fact == "location_verified")
    assert verified.external is False
    assert verified.produced_by == ["DeviceLocation"]

    device_id = next(
        f for f in explanation.required_facts if f.fact == "device_id_available"
    )
    assert device_id.external is True
    assert device_id.source == "initial facts"

    assert any(
        d.producer == "DeviceLocation"
        and d.consumer == "QualityOnDemand"
        and d.fact == "location_verified"
        for d in explanation.dependencies
    )

    for fragment in (
        "Goal: Improve network quality",
        "Required facts:",
        "Dependencies:",
        "Execution order (from the PGPS dependency graph):",
        "1. DeviceLocation",
        "2. QualityOnDemand",
    ):
        assert fragment in explanation.narrative


# --------------------------------------------------------------------------- #
# real registry chain
# --------------------------------------------------------------------------- #

def test_explanation_over_the_real_registry(registry: CamaraRegistry) -> None:
    plan = PGPSPlanner(registry=registry).plan_for(
        goals=["qos.session.deleted"],
        initial_facts=["auth.token", "device.identifier", "qos.profile"],
        candidate_operations=["createSession", "deleteSession"],
    )
    explanation = explain_plan(
        plan,
        registry=registry,
        initial_facts={"auth.token", "device.identifier", "qos.profile"},
    )

    assert explanation.selected_apis == ["quality-on-demand"]
    assert [s.operation_id for s in explanation.execution_order] == [
        "createSession",
        "deleteSession",
    ]
    assert any(
        d.producer == "createSession"
        and d.consumer == "deleteSession"
        and d.fact == "qos.session.id"
        for d in explanation.dependencies
    )

    first, second = explanation.execution_order
    assert "runs first" in first.reason
    assert "runs after createSession" in second.reason
    assert second.depends_on == ["createSession"]


def test_single_step_plan_has_no_dependencies(registry: CamaraRegistry) -> None:
    plan = PGPSPlanner(registry=registry).plan_for(
        goals=["qos.session"],
        initial_facts=["auth.token", "device.identifier", "qos.profile"],
        candidate_operations=["createSession"],
    )
    explanation = explain_plan(plan, registry=registry, initial_facts=["auth.token"])
    assert len(explanation.execution_order) == 1
    assert explanation.dependencies == []


def test_empty_plan_is_handled() -> None:
    explanation = explain_plan(PlanResult(status=PlanStatus.INFEASIBLE))
    assert explanation.execution_order == []
    assert explanation.source == EXPLANATION_SOURCE


# --------------------------------------------------------------------------- #
# surfaced in the run report
# --------------------------------------------------------------------------- #

def test_state_report_exposes_provenance() -> None:
    state = AgentState(
        request_text="x",
        metadata={"provenance": {"source": EXPLANATION_SOURCE, "narrative": "..."}},
    )
    report = state_report(state)
    assert report["provenance"]["source"] == EXPLANATION_SOURCE
