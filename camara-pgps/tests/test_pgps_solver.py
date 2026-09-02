"""Tests for the PGPS solver and planner (milestone 11)."""

from __future__ import annotations

import pytest

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import PROJECT_ROOT
from camara_pgps.models import HTTPMethod
from camara_pgps.models.plan import PlanStatus
from camara_pgps.pgps import (
    ConditionModel,
    PGPSPlanner,
    PGPSSolver,
    SolverRequest,
    SolverStatus,
)
from camara_pgps.pgps.conditions import OperationConditions, Postcondition, Precondition

REGISTRY_FILE = PROJECT_ROOT / "data" / "registry" / "camara_apis.json"


@pytest.fixture(scope="module")
def registry() -> CamaraRegistry:
    return CamaraRegistry.from_file(REGISTRY_FILE)


def _op(operation_id: str, requires: list[str], produces: list[str]) -> OperationConditions:
    return OperationConditions(
        operation_id=operation_id,
        preconditions=[Precondition(description="", requires=requires)] if requires else [],
        postconditions=[Postcondition(description="", produces=produces)] if produces else [],
    )


@pytest.fixture
def example_solver() -> PGPSSolver:
    # the task's flagship example
    device_location = _op("DeviceLocation", ["device_id_available"], ["location_verified"])
    quality_on_demand = _op(
        "QualityOnDemand",
        ["device_id_available", "location_verified"],
        ["qos_session_created"],
    )
    return PGPSSolver(ConditionModel([device_location, quality_on_demand]))


# --------------------------------------------------------------------------- #
# solver - deterministic backward chaining
# --------------------------------------------------------------------------- #

def test_solver_produces_the_expected_two_step_plan(example_solver: PGPSSolver) -> None:
    result = example_solver.solve(
        SolverRequest(
            goals={"qos_session_created"},
            initial_facts={"device_id_available"},
        )
    )
    assert result.status is SolverStatus.SOLVED
    assert result.order == ["DeviceLocation", "QualityOnDemand"]
    assert result.selected == {"DeviceLocation", "QualityOnDemand"}
    assert "qos_session_created" in result.achieved_facts
    # the invariant holds along the order
    assert result.reasoning[-1] == "order: DeviceLocation -> QualityOnDemand"


def test_solver_is_deterministic(example_solver: PGPSSolver) -> None:
    request = SolverRequest(
        goals={"qos_session_created"}, initial_facts={"device_id_available"}
    )
    first = example_solver.solve(request)
    second = example_solver.solve(request)
    assert first.model_dump() == second.model_dump()


def test_goal_already_satisfied_is_empty(example_solver: PGPSSolver) -> None:
    result = example_solver.solve(
        SolverRequest(goals={"device_id_available"}, initial_facts={"device_id_available"})
    )
    assert result.status is SolverStatus.EMPTY
    assert result.order == []


def test_impossible_dependency_is_detected(example_solver: PGPSSolver) -> None:
    result = example_solver.solve(SolverRequest(goals={"qos_session_created"}))
    assert result.status is SolverStatus.INFEASIBLE
    assert "device_id_available" in result.unresolved_goals
    assert result.missing_facts  # per-operation unmet preconditions
    assert result.order == []


def test_cycle_is_detected() -> None:
    a = _op("A", ["y"], ["x"])
    b = _op("B", ["x"], ["y"])
    solver = PGPSSolver(ConditionModel([a, b]))
    result = solver.solve(SolverRequest(goals={"x"}))
    assert result.status is SolverStatus.CYCLIC
    assert result.cycle and result.cycle[0] == result.cycle[-1]


def test_solver_over_the_registry_prefers_known_facts(registry: CamaraRegistry) -> None:
    solver = PGPSSolver.from_registry(registry)
    # session id already known -> getSession, not createSession
    with_id = solver.solve(
        SolverRequest(
            goals={"qos.session"},
            initial_facts={"auth.token", "qos.session.id"},
        )
    )
    assert with_id.status is SolverStatus.SOLVED
    assert with_id.order == ["getSession"]

    # session id not known, but createSession's inputs are -> createSession
    without_id = solver.solve(
        SolverRequest(
            goals={"qos.session"},
            initial_facts={"auth.token", "device.identifier", "qos.profile"},
        )
    )
    assert without_id.order == ["createSession"]


def test_allowed_operations_bounds_the_search(registry: CamaraRegistry) -> None:
    solver = PGPSSolver.from_registry(registry)
    result = solver.solve(
        SolverRequest(
            goals={"qos.session"},
            initial_facts={"auth.token", "qos.session.id"},
            allowed_operations={"createSession", "deleteSession"},
        )
    )
    # getSession would have solved it, but it is not allowed; createSession needs
    # facts nobody in the allowed set can supply
    assert result.status is SolverStatus.INFEASIBLE
    assert "device.identifier" in {
        fact for facts in result.missing_facts.values() for fact in facts
    }


# --------------------------------------------------------------------------- #
# planner - PlanRequest -> PlanResult
# --------------------------------------------------------------------------- #

def test_planner_single_step_plan(registry: CamaraRegistry) -> None:
    planner = PGPSPlanner(registry=registry)
    result = planner.plan_for(
        goals=["qos.session"],
        initial_facts=["auth.token", "device.identifier", "qos.profile"],
        candidate_operations=["createSession", "getSession", "deleteSession"],
    )
    assert result.status is PlanStatus.VALID
    assert [step.operation_id for step in result.steps] == ["createSession"]
    step = result.steps[0]
    assert step.method is HTTPMethod.POST
    assert step.path == "/sessions"
    assert "qos.session.id" in step.produces
    assert step.depends_on == []


def test_planner_orders_a_dependency_chain(registry: CamaraRegistry) -> None:
    planner = PGPSPlanner(registry=registry)
    result = planner.plan_for(
        goals=["qos.session.deleted"],
        initial_facts=["auth.token", "device.identifier", "qos.profile"],
        candidate_operations=["createSession", "deleteSession"],
    )
    assert result.status is PlanStatus.VALID
    assert [step.operation_id for step in result.steps] == [
        "createSession",
        "deleteSession",
    ]
    assert result.steps[1].depends_on == [0]
    assert result.steps[0].index == 0 and result.steps[1].index == 1


def test_planner_reports_infeasible_with_unmet_preconditions(
    registry: CamaraRegistry,
) -> None:
    planner = PGPSPlanner(registry=registry)
    result = planner.plan_for(
        goals=["qos.session"],
        initial_facts=["auth.token"],  # missing device.identifier, qos.profile
        candidate_operations=["createSession"],
    )
    assert result.status is PlanStatus.INFEASIBLE
    assert set(result.unmet_preconditions) >= {"device.identifier", "qos.profile"}
    assert result.steps == []


def test_planner_no_goal_facts_is_infeasible(registry: CamaraRegistry) -> None:
    planner = PGPSPlanner(registry=registry)
    result = planner.plan_for(goals=[], candidate_operations=[])
    assert result.status is PlanStatus.INFEASIBLE
    assert result.validation_errors
