"""Structural plan validation - the PGPS layer.

Answers: do the referenced operations exist, is the step ordering / indexing
consistent, are ``depends_on`` edges sane, is the dependency graph acyclic, and
is every step's preconditions satisfied by the initial facts plus the
postconditions of earlier steps?

This does NOT look at parameter values - that is :mod:`parameters`.
"""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.models.plan import PlanResult
from camara_pgps.pgps.conditions import ConditionModel
from camara_pgps.validator._types import (
    Severity,
    ValidationCategory,
    ValidationIssue,
    ValidationLayer,
)


def _issue(
    category: ValidationCategory,
    severity: Severity,
    message: str,
    *,
    step_index: int | None = None,
    operation_id: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        layer=ValidationLayer.STRUCTURAL,
        category=category,
        severity=severity,
        message=message,
        step_index=step_index,
        operation_id=operation_id,
    )


def check_structure(
    plan: PlanResult,
    registry: CamaraRegistry,
    *,
    condition_model: ConditionModel | None = None,
    known_facts: Iterable[str] = (),
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    steps = plan.steps
    if not steps:
        return issues

    model = condition_model or ConditionModel.from_registry(registry)

    # -- 1. existence + method/path consistency ---------------------------
    for step in steps:
        operation = registry.get_operation(step.operation_id)
        if operation is None:
            issues.append(
                _issue(
                    ValidationCategory.OPERATION_EXISTS,
                    Severity.ERROR,
                    f"operation {step.operation_id!r} is not in the CAMARA registry",
                    step_index=step.index,
                    operation_id=step.operation_id,
                )
            )
            continue

        api_name = operation.api_name or registry.api_of(step.operation_id)
        if not registry.has_api(api_name):
            issues.append(
                _issue(
                    ValidationCategory.API_EXISTS,
                    Severity.ERROR,
                    f"API {api_name!r} for operation {step.operation_id!r} is not registered",
                    step_index=step.index,
                    operation_id=step.operation_id,
                )
            )
        if step.method != operation.method or step.path != operation.path:
            issues.append(
                _issue(
                    ValidationCategory.OPERATION_EXISTS,
                    Severity.WARNING,
                    f"step {step.method} {step.path} does not match the registry "
                    f"({operation.method} {operation.path}); plan may be stale",
                    step_index=step.index,
                    operation_id=step.operation_id,
                )
            )

    # -- 2. index / ordering consistency ---------------------------------
    indices = [step.index for step in steps]
    if indices != list(range(len(steps))):
        issues.append(
            _issue(
                ValidationCategory.ORDERING,
                Severity.ERROR,
                f"step indices {indices} are not a 0-based contiguous sequence",
            )
        )

    known_indices = set(indices)
    for step in steps:
        for dep in step.depends_on:
            if dep not in known_indices:
                issues.append(
                    _issue(
                        ValidationCategory.DEPENDENCY,
                        Severity.ERROR,
                        f"depends_on {dep} refers to a step that does not exist",
                        step_index=step.index,
                        operation_id=step.operation_id,
                    )
                )
            elif dep >= step.index:
                issues.append(
                    _issue(
                        ValidationCategory.ORDERING,
                        Severity.ERROR,
                        f"depends_on {dep} is not before this step (index {step.index})",
                        step_index=step.index,
                        operation_id=step.operation_id,
                    )
                )

    # -- 3. cycle detection --------------------------------------------------
    graph: nx.DiGraph = nx.DiGraph()
    graph.add_nodes_from(indices)
    for step in steps:
        for dep in step.depends_on:
            if dep in known_indices:
                graph.add_edge(dep, step.index)
    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph)
        path = " -> ".join(str(u) for u, _ in cycle) + f" -> {cycle[0][0]}"
        issues.append(
            _issue(
                ValidationCategory.CYCLE,
                Severity.ERROR,
                f"the plan's depends_on edges form a cycle: {path}",
            )
        )

    # -- 4. precondition satisfaction along the order --------------------
    #   preconditions(step) subset of  known_facts + produces(earlier steps)
    state: set[str] = set(known_facts)
    producer_of: dict[str, int] = {}
    for step in steps:
        conditions = model.get(step.operation_id)
        requires = set(conditions.requires) if conditions else set(step.requires)
        produces = set(conditions.produces) if conditions else set(step.produces)

        unmet = requires - state
        if unmet:
            issues.append(
                _issue(
                    ValidationCategory.PRECONDITION,
                    Severity.ERROR,
                    f"unmet preconditions {sorted(unmet)}: not in the initial facts "
                    "nor produced by an earlier step",
                    step_index=step.index,
                    operation_id=step.operation_id,
                )
            )

        # depends_on should reference the steps that actually supply the facts
        expected_deps = {
            producer_of[fact]
            for fact in requires
            if fact in producer_of
        }
        missing_deps = expected_deps - set(step.depends_on)
        if missing_deps:
            issues.append(
                _issue(
                    ValidationCategory.DEPENDENCY,
                    Severity.WARNING,
                    f"depends_on is missing {sorted(missing_deps)} which produce "
                    "facts this step requires",
                    step_index=step.index,
                    operation_id=step.operation_id,
                )
            )

        state |= produces
        for fact in produces:
            producer_of.setdefault(fact, step.index)

    return issues


__all__ = ["check_structure"]
