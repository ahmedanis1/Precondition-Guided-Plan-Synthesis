"""PGPS dependency graph.

Builds a directed graph from the :class:`ConditionModel`: operations linked
through the facts that flow between them. An edge ``A -> B`` means *B requires a
fact that A produces*::

    createSession --(qos.session.id)--> getSession
    createSession --(qos.session.id)--> deleteSession

The graph is bipartite (operation nodes + fact nodes) so the raw fact flow is
inspectable; :meth:`DependencyGraph.operation_graph` projects it to a plain
operation-to-operation DAG for ordering.
"""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
from pydantic import BaseModel

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.pgps.conditions import ConditionModel, FactSet, OperationConditions


class Dependency(BaseModel):
    """A single fact-carried dependency edge between two operations."""

    producer: str
    consumer: str
    fact: str


class CyclicDependencyError(RuntimeError):
    """Raised when the operation dependency graph contains a cycle."""

    def __init__(self, cycle: list[str]) -> None:
        super().__init__("cyclic operation dependency: " + " -> ".join(cycle))
        self.cycle = cycle


class UnsatisfiedPreconditionError(RuntimeError):
    """Raised when an ordering cannot satisfy every operation's preconditions."""

    def __init__(self, missing: dict[str, set[str]]) -> None:
        detail = "; ".join(
            f"{operation} still needs {sorted(facts)}"
            for operation, facts in missing.items()
        )
        super().__init__(f"unsatisfiable preconditions: {detail}")
        self.missing = missing


class DependencyGraph:
    """Fact-flow graph over a set of operations."""

    OPERATION = "operation"
    FACT = "fact"

    def __init__(self, model: ConditionModel) -> None:
        self.model = model
        self.graph: nx.DiGraph = nx.DiGraph()

        for operation_id in model.operation_ids():
            conditions = model.for_operation(operation_id)
            self.graph.add_node(
                operation_id,
                kind=self.OPERATION,
                api_name=conditions.api_name,
                requires=sorted(conditions.requires),
                produces=sorted(conditions.produces),
            )
        for fact in sorted(model.all_facts()):
            self.graph.add_node(self._fact_node(fact), kind=self.FACT, fact=fact)

        for operation_id in model.operation_ids():
            conditions = model.for_operation(operation_id)
            for fact in sorted(conditions.produces):
                self.graph.add_edge(
                    operation_id, self._fact_node(fact), relation="produces", fact=fact
                )
            for fact in sorted(conditions.requires):
                self.graph.add_edge(
                    self._fact_node(fact), operation_id, relation="requires", fact=fact
                )

    # -- construction ----------------------------------------------------------

    @staticmethod
    def _fact_node(fact: str) -> str:
        return f"fact::{fact}"

    @classmethod
    def from_registry(cls, registry: CamaraRegistry) -> DependencyGraph:
        return cls(ConditionModel.from_registry(registry))

    @classmethod
    def from_condition_model(cls, model: ConditionModel) -> DependencyGraph:
        return cls(model)

    @classmethod
    def from_operation_conditions(
        cls, operations: Iterable[OperationConditions]
    ) -> DependencyGraph:
        return cls(ConditionModel(operations))

    # -- structure -----------------------------------------------------------

    def operations(self) -> list[str]:
        return [
            node
            for node, data in self.graph.nodes(data=True)
            if data["kind"] == self.OPERATION
        ]

    def facts(self) -> list[str]:
        return sorted(
            data["fact"]
            for _, data in self.graph.nodes(data=True)
            if data["kind"] == self.FACT
        )

    def producers_of(self, fact: str) -> list[str]:
        return self.model.producers_of(fact)

    def consumers_of(self, fact: str) -> list[str]:
        return self.model.consumers_of(fact)

    def external_facts(self) -> set[str]:
        return self.model.external_facts()

    def dependencies(self) -> list[Dependency]:
        deps: list[Dependency] = []
        for consumer in self.operations():
            for fact in sorted(self.model.for_operation(consumer).requires):
                for producer in self.model.producers_of(fact):
                    if producer != consumer:
                        deps.append(
                            Dependency(producer=producer, consumer=consumer, fact=fact)
                        )
        return deps

    def dependencies_of(self, operation_id: str) -> list[Dependency]:
        """Edges into ``operation_id`` - what it depends on."""

        return [d for d in self.dependencies() if d.consumer == operation_id]

    def dependents_of(self, operation_id: str) -> list[Dependency]:
        """Edges out of ``operation_id`` - what depends on it."""

        return [d for d in self.dependencies() if d.producer == operation_id]

    def operation_graph(self, operations: Iterable[str] | None = None) -> nx.DiGraph:
        """Projected operation-to-operation DAG (edge data: ``facts``)."""

        selected = set(operations) if operations is not None else set(self.operations())
        graph: nx.DiGraph = nx.DiGraph()
        graph.add_nodes_from(sorted(selected))
        for dep in self.dependencies():
            if dep.producer not in selected or dep.consumer not in selected:
                continue
            if graph.has_edge(dep.producer, dep.consumer):
                graph[dep.producer][dep.consumer]["facts"].append(dep.fact)
            else:
                graph.add_edge(dep.producer, dep.consumer, facts=[dep.fact])
        return graph

    # -- ordering -----------------------------------------------------------

    def has_cycle(self) -> bool:
        return not nx.is_directed_acyclic_graph(self.operation_graph())

    def topological_order(self, operations: Iterable[str] | None = None) -> list[str]:
        graph = self.operation_graph(operations)
        try:
            return list(nx.lexicographical_topological_sort(graph))
        except nx.NetworkXUnfeasible:
            cycle_edges = nx.find_cycle(graph)
            path = [u for u, _ in cycle_edges] + [cycle_edges[0][0]]
            raise CyclicDependencyError(path) from None

    def required_inputs(
        self, operations: Iterable[str], *, available: Iterable[str] = ()
    ) -> set[str]:
        """Facts the given operations need that no operation among them produces."""

        selected = set(operations)
        produced_within: set[str] = set()
        needed: set[str] = set()
        for operation_id in selected:
            conditions = self.model.for_operation(operation_id)
            produced_within |= conditions.produces
            needed |= conditions.requires
        return needed - produced_within - FactSet(available).as_set()

    def resolve_order(
        self,
        targets: Iterable[str],
        *,
        available: Iterable[str] = (),
        include_prerequisites: bool = True,
    ) -> list[str]:
        """Return the targets (plus in-graph prerequisites) in a precondition-safe order.

        ``available`` is the set of facts already known (from the intent /
        parameter extraction). Raises :class:`UnsatisfiedPreconditionError` if
        some precondition can never be met, or :class:`CyclicDependencyError`.
        """

        known = FactSet(available)
        wanted: set[str] = {str(t) for t in targets}
        for target in wanted:
            self.model.for_operation(target)  # KeyError if unknown

        if include_prerequisites:
            frontier = list(wanted)
            while frontier:
                operation_id = frontier.pop()
                for fact in sorted(self.model.for_operation(operation_id).requires):
                    if fact in known:
                        continue
                    producers = [
                        p for p in self.model.producers_of(fact) if p != operation_id
                    ]
                    if not producers:
                        continue  # external fact - caught by the feasibility check
                    chosen = producers[0]
                    if chosen not in wanted:
                        wanted.add(chosen)
                        frontier.append(chosen)

        order = self.topological_order(wanted)

        state = known.copy()
        missing: dict[str, set[str]] = {}
        for operation_id in order:
            conditions = self.model.for_operation(operation_id)
            unmet = conditions.unmet(state)
            if unmet:
                missing[operation_id] = unmet
            state = conditions.apply(state)
        if missing:
            raise UnsatisfiedPreconditionError(missing)
        return order


__all__ = [
    "Dependency",
    "DependencyGraph",
    "CyclicDependencyError",
    "UnsatisfiedPreconditionError",
]
