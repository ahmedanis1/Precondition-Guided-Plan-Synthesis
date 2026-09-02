"""The PGPS solver: deterministic, goal-directed plan synthesis.

No LLM is involved. Given a set of goal facts and the facts known at the start,
the solver performs backward chaining over the :class:`ConditionModel`:

1. take an outstanding goal fact
2. find an operation that produces it (deterministic tie-break: alphabetical,
   preferring an already-selected operation, then one whose preconditions are
   already satisfiable)
3. add that operation's preconditions as new outstanding facts
4. repeat until nothing is outstanding

then it checks the invariant for every selected operation::

    preconditions(op)  subset of  initial_facts  union  postconditions(earlier ops)

builds the dependency graph, detects cycles and impossible dependencies, and
topologically orders the operations.
"""

from __future__ import annotations

from collections import deque
from enum import StrEnum

import networkx as nx
from pydantic import BaseModel, Field

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.pgps.conditions import ConditionModel
from camara_pgps.pgps.dependency_graph import Dependency, DependencyGraph


class SolverStatus(StrEnum):
    """Outcome of a solve."""

    SOLVED = "solved"
    EMPTY = "empty"  # goals already satisfied by the initial facts
    INFEASIBLE = "infeasible"  # a required fact has no producer / goal unreachable
    CYCLIC = "cyclic"  # the selected operations form a dependency cycle


class SolverRequest(BaseModel):
    """Input to :meth:`PGPSSolver.solve`."""

    goals: set[str]
    initial_facts: set[str] = Field(default_factory=set)
    allowed_operations: set[str] | None = None
    max_operations: int = 20


class SolverResult(BaseModel):
    """Deterministic output of the solver."""

    status: SolverStatus
    order: list[str] = Field(default_factory=list)
    selected: set[str] = Field(default_factory=set)
    goal_facts: set[str] = Field(default_factory=set)
    initial_facts: set[str] = Field(default_factory=set)
    achieved_facts: set[str] = Field(default_factory=set)
    unresolved_goals: set[str] = Field(default_factory=set)
    missing_facts: dict[str, list[str]] = Field(default_factory=dict)
    dependencies: list[Dependency] = Field(default_factory=list)
    cycle: list[str] | None = None
    reasoning: list[str] = Field(default_factory=list)

    @property
    def solved(self) -> bool:
        return self.status in (SolverStatus.SOLVED, SolverStatus.EMPTY)


class PGPSSolver:
    """Deterministic backward-chaining planner over a :class:`ConditionModel`."""

    def __init__(self, model: ConditionModel) -> None:
        self.model = model
        self.graph = DependencyGraph(model)

    @classmethod
    def from_registry(cls, registry: CamaraRegistry) -> PGPSSolver:
        return cls(ConditionModel.from_registry(registry))

    @classmethod
    def from_condition_model(cls, model: ConditionModel) -> PGPSSolver:
        return cls(model)

    # -- solve -----------------------------------------------------------------

    def solve(self, request: SolverRequest) -> SolverResult:
        initial = set(request.initial_facts)
        goals = set(request.goals)
        allowed = request.allowed_operations
        reasoning: list[str] = []

        outstanding = goals - initial
        if not outstanding:
            reasoning.append("all goal facts are already in the initial facts")
            return SolverResult(
                status=SolverStatus.EMPTY,
                goal_facts=goals,
                initial_facts=initial,
                achieved_facts=set(initial),
                reasoning=reasoning,
            )

        selected: set[str] = set()
        producible: set[str] = set(initial)
        queue: deque[str] = deque(sorted(outstanding))
        impossible_facts: set[str] = set()

        guard = (len(self.model.operation_ids()) + 1) * (len(self.model.all_facts()) + 1) + 50
        steps = 0
        while queue:
            steps += 1
            if steps > guard:  # pragma: no cover - defensive
                reasoning.append("solver did not converge; aborting")
                break
            fact = queue.popleft()
            if fact in producible:
                continue
            producers = sorted(
                p
                for p in self.model.producers_of(fact)
                if allowed is None or p in allowed
            )
            if not producers:
                impossible_facts.add(fact)
                reasoning.append(f"no allowed operation produces {fact!r}")
                continue
            chosen = self._choose(producers, selected, producible)
            reasoning.append(
                f"{fact!r} <- {chosen}"
                + (f" (alternatives: {producers})" if len(producers) > 1 else "")
            )
            selected.add(chosen)
            chosen_conditions = self.model.for_operation(chosen)
            producible |= chosen_conditions.produces
            for requirement in sorted(chosen_conditions.requires):
                if requirement not in producible:
                    queue.append(requirement)

        return self._finalise(
            goals, initial, selected, impossible_facts, reasoning, request.max_operations
        )

    # -- helpers -------------------------------------------------------------

    def _choose(
        self, producers: list[str], selected: set[str], producible: set[str]
    ) -> str:
        """Deterministically pick a producer (``producers`` is already sorted)."""

        already = [p for p in producers if p in selected]
        if already:
            return already[0]
        ready = [
            p
            for p in producers
            if self.model.for_operation(p).requires <= producible
        ]
        if ready:
            return ready[0]
        return producers[0]

    def _finalise(
        self,
        goals: set[str],
        initial: set[str],
        selected: set[str],
        impossible_facts: set[str],
        reasoning: list[str],
        max_operations: int,
    ) -> SolverResult:
        producible = set(initial)
        for operation_id in selected:
            producible |= self.model.for_operation(operation_id).produces

        missing_facts: dict[str, list[str]] = {}
        for operation_id in sorted(selected):
            unmet = self.model.for_operation(operation_id).requires - producible
            if unmet:
                missing_facts[operation_id] = sorted(unmet)

        unresolved_goals = {g for g in goals if g not in producible} | impossible_facts

        dependencies = [
            dep
            for dep in self.graph.dependencies()
            if dep.producer in selected and dep.consumer in selected
        ]

        if unresolved_goals or missing_facts:
            return SolverResult(
                status=SolverStatus.INFEASIBLE,
                order=[],
                selected=selected,
                goal_facts=goals,
                initial_facts=initial,
                achieved_facts=producible,
                unresolved_goals=unresolved_goals,
                missing_facts=missing_facts,
                dependencies=dependencies,
                reasoning=reasoning,
            )

        subgraph = self.graph.operation_graph(selected)
        if not nx.is_directed_acyclic_graph(subgraph):
            cycle_edges = nx.find_cycle(subgraph)
            cycle = [u for u, _ in cycle_edges] + [cycle_edges[0][0]]
            reasoning.append("cycle detected: " + " -> ".join(cycle))
            return SolverResult(
                status=SolverStatus.CYCLIC,
                selected=selected,
                goal_facts=goals,
                initial_facts=initial,
                achieved_facts=producible,
                dependencies=dependencies,
                cycle=cycle,
                reasoning=reasoning,
            )

        order = self.graph.topological_order(selected)

        state = set(initial)
        for operation_id in order:
            conditions = self.model.for_operation(operation_id)
            unmet = conditions.requires - state
            if unmet:
                missing_facts[operation_id] = sorted(unmet)
            state |= conditions.produces
        if missing_facts:
            return SolverResult(
                status=SolverStatus.INFEASIBLE,
                order=order,
                selected=selected,
                goal_facts=goals,
                initial_facts=initial,
                achieved_facts=state,
                missing_facts=missing_facts,
                dependencies=dependencies,
                reasoning=reasoning,
            )

        if len(order) > max_operations:
            reasoning.append(
                f"plan needs {len(order)} operations, over the limit of {max_operations}"
            )
            return SolverResult(
                status=SolverStatus.INFEASIBLE,
                order=order,
                selected=selected,
                goal_facts=goals,
                initial_facts=initial,
                achieved_facts=state,
                missing_facts={"__plan__": [f"exceeds max_operations={max_operations}"]},
                dependencies=dependencies,
                reasoning=reasoning,
            )

        reasoning.append("order: " + " -> ".join(order))
        return SolverResult(
            status=SolverStatus.SOLVED,
            order=order,
            selected=selected,
            goal_facts=goals,
            initial_facts=initial,
            achieved_facts=state,
            dependencies=dependencies,
            reasoning=reasoning,
        )


__all__ = [
    "PGPSSolver",
    "SolverRequest",
    "SolverResult",
    "SolverStatus",
]
