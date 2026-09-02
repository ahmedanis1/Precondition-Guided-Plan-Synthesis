"""Tests for the PGPS condition model and dependency graph (milestone 10)."""

from __future__ import annotations

import pytest

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import PROJECT_ROOT
from camara_pgps.models import Condition, ConditionKind
from camara_pgps.pgps import (
    ConditionModel,
    CyclicDependencyError,
    Dependency,
    DependencyGraph,
    FactSet,
    OperationConditions,
    Postcondition,
    Precondition,
    UnsatisfiedPreconditionError,
)

REGISTRY_FILE = PROJECT_ROOT / "data" / "registry" / "camara_apis.json"


@pytest.fixture(scope="module")
def registry() -> CamaraRegistry:
    return CamaraRegistry.from_file(REGISTRY_FILE)


@pytest.fixture(scope="module")
def model(registry: CamaraRegistry) -> ConditionModel:
    return ConditionModel.from_registry(registry)


@pytest.fixture(scope="module")
def graph(model: ConditionModel) -> DependencyGraph:
    return DependencyGraph.from_condition_model(model)


# --------------------------------------------------------------------------- #
# FactSet
# --------------------------------------------------------------------------- #

def test_factset_operations() -> None:
    facts = FactSet(["auth.token"])
    facts.add("device.identifier")
    assert "device.identifier" in facts
    assert facts.satisfies(["auth.token", "device.identifier"])
    assert facts.missing(["auth.token", "qos.profile"]) == {"qos.profile"}
    grown = facts.apply(["qos.session.id"])
    assert "qos.session.id" in grown and "qos.session.id" not in facts  # apply is pure
    assert (facts | ["x"]).as_set() == {"auth.token", "device.identifier", "x"}


# --------------------------------------------------------------------------- #
# ConditionModel over the registry
# --------------------------------------------------------------------------- #

def test_precondition_postcondition_wrappers() -> None:
    raw = Condition(kind=ConditionKind.POSTCONDITION, description="x", requires=["a"])
    pre = Precondition.from_condition(raw)
    assert isinstance(pre, Condition)
    assert pre.kind is ConditionKind.PRECONDITION
    assert pre.required_facts == {"a"}
    post = Postcondition(description="y", produces=["b"])
    assert post.kind is ConditionKind.POSTCONDITION
    assert post.produced_facts == {"b"}


def test_fact_flow_from_registry(model: ConditionModel) -> None:
    create = model.for_operation("createSession")
    assert create.requires == {"auth.token", "device.identifier", "qos.profile"}
    assert create.produces == {"qos.session.id", "qos.session"}

    assert model.producers_of("qos.session.id") == ["createSession"]
    assert set(model.consumers_of("qos.session.id")) == {"getSession", "deleteSession"}
    assert model.external_facts() == {
        "auth.token",
        "device.identifier",
        "location.area",
        "qos.profile",
    }


def test_operation_conditions_applicability(model: ConditionModel) -> None:
    get_session = model.for_operation("getSession")
    assert get_session.is_applicable({"auth.token"}) is False
    assert get_session.unmet({"auth.token"}) == {"qos.session.id"}
    assert get_session.is_applicable({"auth.token", "qos.session.id"}) is True
    grown = get_session.apply({"auth.token", "qos.session.id"})
    assert "qos.session" in grown


# --------------------------------------------------------------------------- #
# DependencyGraph over the registry
# --------------------------------------------------------------------------- #

def test_graph_is_bipartite_operation_fact(graph: DependencyGraph) -> None:
    assert set(graph.operations()) == {
        "verifyLocation",
        "retrieveLocation",
        "createSession",
        "getSession",
        "deleteSession",
        "createInsightsSubscription",
        "getInsightsSubscription",
    }
    fact_node = "fact::qos.session.id"
    assert graph.graph.nodes[fact_node]["kind"] == "fact"
    assert graph.graph.edges["createSession", fact_node]["relation"] == "produces"
    assert graph.graph.edges[fact_node, "getSession"]["relation"] == "requires"


def test_dependencies_follow_the_facts(graph: DependencyGraph) -> None:
    deps = graph.dependencies()
    assert Dependency(
        producer="createSession", consumer="getSession", fact="qos.session.id"
    ) in deps
    assert Dependency(
        producer="createSession", consumer="deleteSession", fact="qos.session.id"
    ) in deps
    assert Dependency(
        producer="createInsightsSubscription",
        consumer="getInsightsSubscription",
        fact="network-insights.subscription.id",
    ) in deps

    assert {d.producer for d in graph.dependencies_of("getSession")} == {"createSession"}
    assert {d.consumer for d in graph.dependents_of("createSession")} == {
        "getSession",
        "deleteSession",
    }
    assert graph.has_cycle() is False


def test_topological_order_puts_producers_first(graph: DependencyGraph) -> None:
    order = graph.topological_order(["getSession", "deleteSession", "createSession"])
    assert order.index("createSession") < order.index("getSession")
    assert order.index("createSession") < order.index("deleteSession")


def test_resolve_order_pulls_in_prerequisites(graph: DependencyGraph) -> None:
    order = graph.resolve_order(
        ["deleteSession"],
        available={"auth.token", "device.identifier", "qos.profile"},
    )
    assert order == ["createSession", "deleteSession"]

    # session id already known -> no prerequisite needed
    assert graph.resolve_order(
        ["getSession"], available={"auth.token", "qos.session.id"}
    ) == ["getSession"]


def test_resolve_order_raises_on_unsatisfiable_external_fact(
    graph: DependencyGraph,
) -> None:
    with pytest.raises(UnsatisfiedPreconditionError) as excinfo:
        graph.resolve_order(["createSession"], available={"auth.token"})
    assert excinfo.value.missing["createSession"] == {"device.identifier", "qos.profile"}


def test_required_inputs(graph: DependencyGraph) -> None:
    needed = graph.required_inputs(
        ["createSession", "deleteSession"], available={"auth.token"}
    )
    # qos.session.id is produced within the pair, so it is not an external input
    assert needed == {"device.identifier", "qos.profile"}


# --------------------------------------------------------------------------- #
# the task's flagship example, built explicitly
# --------------------------------------------------------------------------- #

def test_pgps_foundation_example() -> None:
    device_location = OperationConditions(
        operation_id="DeviceLocation",
        preconditions=[
            Precondition(description="device id known", requires=["device_id_available"])
        ],
        postconditions=[
            Postcondition(description="location verified", produces=["location_verified"])
        ],
    )
    quality_on_demand = OperationConditions(
        operation_id="QualityOnDemand",
        preconditions=[
            Precondition(
                description="device id and verified location",
                requires=["device_id_available", "location_verified"],
            )
        ],
        postconditions=[
            Postcondition(
                description="session created", produces=["qos_session_created"]
            )
        ],
    )
    graph = DependencyGraph.from_operation_conditions(
        [device_location, quality_on_demand]
    )

    assert Dependency(
        producer="DeviceLocation",
        consumer="QualityOnDemand",
        fact="location_verified",
    ) in graph.dependencies()
    assert graph.topological_order() == ["DeviceLocation", "QualityOnDemand"]

    order = graph.resolve_order(
        ["QualityOnDemand"], available={"device_id_available"}
    )
    assert order == ["DeviceLocation", "QualityOnDemand"]
    assert graph.model.external_facts() == {"device_id_available"}


def test_cycle_is_detected() -> None:
    a = OperationConditions(
        operation_id="A",
        preconditions=[Precondition(description="needs y", requires=["y"])],
        postconditions=[Postcondition(description="makes x", produces=["x"])],
    )
    b = OperationConditions(
        operation_id="B",
        preconditions=[Precondition(description="needs x", requires=["x"])],
        postconditions=[Postcondition(description="makes y", produces=["y"])],
    )
    graph = DependencyGraph.from_operation_conditions([a, b])
    assert graph.has_cycle() is True
    with pytest.raises(CyclicDependencyError):
        graph.topological_order()
