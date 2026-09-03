"""Plan explainability - derived from the PGPS dependency graph, never an LLM.

Turns a :class:`~camara_pgps.models.plan.PlanResult` into a structured account of
*why* the plan has the shape it does::

    Goal -> Selected APIs -> Required facts -> Produced facts
         -> Dependencies -> Execution order

Every field comes from the operations' pre/postconditions and the fact-flow edges
between them; :attr:`PlanExplanation.source` is fixed to ``pgps-dependency-graph``.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.models.intent import Intent
from camara_pgps.models.plan import PlanResult

EXPLANATION_SOURCE = "pgps-dependency-graph"


class FactProvenance(BaseModel):
    """Where one fact comes from and who needs it."""

    fact: str
    external: bool  # not produced by any step -> supplied before the plan runs
    source: str  # "initial facts" | "step N (<operation>)"
    required_by: list[str] = Field(default_factory=list)
    produced_by: list[str] = Field(default_factory=list)


class DependencyExplanation(BaseModel):
    """One fact-carried edge between two plan steps."""

    producer: str
    consumer: str
    fact: str
    text: str


class StepExplanation(BaseModel):
    """One ordered step and the reason it sits where it does."""

    order: int
    operation_id: str
    api_name: str | None = None
    requires: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    reason: str


class PlanExplanation(BaseModel):
    """A graph-derived explanation of a plan."""

    source: str = EXPLANATION_SOURCE
    goal: str | None = None
    goal_facts: list[str] = Field(default_factory=list)
    selected_apis: list[str] = Field(default_factory=list)
    initial_facts: list[str] = Field(default_factory=list)
    required_facts: list[FactProvenance] = Field(default_factory=list)
    produced_facts: list[FactProvenance] = Field(default_factory=list)
    dependencies: list[DependencyExplanation] = Field(default_factory=list)
    execution_order: list[StepExplanation] = Field(default_factory=list)
    narrative: str = ""


def explain_plan(
    plan: PlanResult,
    *,
    registry: CamaraRegistry | None = None,
    intent: Intent | None = None,
    initial_facts: Iterable[str] = (),
) -> PlanExplanation:
    """Build a :class:`PlanExplanation` from the plan's fact flow."""

    steps = plan.steps
    known_initial = sorted({str(f) for f in initial_facts})
    known_set = set(known_initial)

    api_of: dict[str, str | None] = {}
    if registry is not None:
        for step in steps:
            operation = registry.get_operation(step.operation_id)
            api_of[step.operation_id] = (
                operation.api_name
                if operation is not None and operation.api_name
                else (
                    registry.api_of(step.operation_id)
                    if registry.has_operation(step.operation_id)
                    else None
                )
            )

    selected_apis = sorted({api for api in api_of.values() if api}) or []

    # producer index: fact -> earliest step (order, operation_id) that produces it
    producer_of: dict[str, tuple[int, str]] = {}
    for step in steps:
        for fact in step.produces:
            producer_of.setdefault(fact, (step.index, step.operation_id))

    # -- required / produced fact provenance ------------------------------
    required_names: list[str] = []
    for step in steps:
        for fact in step.requires:
            if fact not in required_names:
                required_names.append(fact)
    produced_names: list[str] = []
    for step in steps:
        for fact in step.produces:
            if fact not in produced_names:
                produced_names.append(fact)

    def _requirers(fact: str) -> list[str]:
        return [s.operation_id for s in steps if fact in s.requires]

    def _producers(fact: str) -> list[str]:
        return [s.operation_id for s in steps if fact in s.produces]

    required_facts: list[FactProvenance] = []
    for fact in required_names:
        producers = _producers(fact)
        external = not producers
        if external:
            source = "initial facts" if fact in known_set else "must be supplied"
        else:
            order, op_id = producer_of[fact]
            source = f"step {order} ({op_id})"
        required_facts.append(
            FactProvenance(
                fact=fact,
                external=external,
                source=source,
                required_by=_requirers(fact),
                produced_by=producers,
            )
        )

    produced_facts = [
        FactProvenance(
            fact=fact,
            external=False,
            source=f"step {producer_of[fact][0]} ({producer_of[fact][1]})",
            required_by=_requirers(fact),
            produced_by=_producers(fact),
        )
        for fact in produced_names
    ]

    # -- dependencies (producer -> fact -> consumer) --------------------------
    dependencies: list[DependencyExplanation] = []
    for step in steps:
        for fact in sorted(step.requires):
            producer = producer_of.get(fact)
            if producer is None or producer[0] >= step.index:
                continue
            dependencies.append(
                DependencyExplanation(
                    producer=producer[1],
                    consumer=step.operation_id,
                    fact=fact,
                    text=(
                        f"{producer[1]} produces {fact!r}, which "
                        f"{step.operation_id} requires"
                    ),
                )
            )

    # -- execution order + reasons -----------------------------------------
    execution_order: list[StepExplanation] = []
    for step in steps:
        unmet_by_initial = [f for f in step.requires if f not in known_set]
        depends_ops = [
            steps[j].operation_id
            for j in step.depends_on
            if 0 <= j < len(steps)
        ]
        if not step.requires:
            reason = "no preconditions"
        elif not unmet_by_initial:
            reason = "runs first; needs only facts known at the start"
        else:
            suppliers: dict[str, list[str]] = {}
            for fact in unmet_by_initial:
                producer = producer_of.get(fact)
                if producer is not None and producer[0] < step.index:
                    suppliers.setdefault(producer[1], []).append(fact)
            if suppliers:
                joined = ", ".join(
                    f"{op} (provides {facts})" for op, facts in suppliers.items()
                )
                reason = f"runs after {joined}"
            else:
                reason = f"needs {unmet_by_initial}"
        execution_order.append(
            StepExplanation(
                order=step.index + 1,
                operation_id=step.operation_id,
                api_name=api_of.get(step.operation_id),
                requires=sorted(step.requires),
                produces=sorted(step.produces),
                depends_on=depends_ops,
                reason=reason,
            )
        )

    goal_facts = [str(f) for f in (plan.metadata.get("goal_facts") or [])]

    explanation = PlanExplanation(
        goal=intent.goal if intent is not None else None,
        goal_facts=goal_facts,
        selected_apis=selected_apis,
        initial_facts=known_initial,
        required_facts=required_facts,
        produced_facts=produced_facts,
        dependencies=dependencies,
        execution_order=execution_order,
    )
    explanation.narrative = render_narrative(explanation)
    return explanation


def render_narrative(explanation: PlanExplanation) -> str:
    """Render a :class:`PlanExplanation` as the Goal -> ... -> order text block."""

    lines: list[str] = []
    if explanation.goal:
        lines.append(f"Goal: {explanation.goal}")
    if explanation.goal_facts:
        lines.append(f"Goal fact(s): {', '.join(explanation.goal_facts)}")
    if explanation.selected_apis:
        lines.append(f"Selected APIs: {', '.join(explanation.selected_apis)}")

    if explanation.required_facts:
        lines.append("\nRequired facts:")
        for fact_item in explanation.required_facts:
            origin = (
                fact_item.source
                if fact_item.external
                else f"produced by {', '.join(fact_item.produced_by)}"
            )
            lines.append(f"  - {fact_item.fact}  ({origin})")

    if explanation.dependencies:
        lines.append("\nDependencies:")
        for dep in explanation.dependencies:
            lines.append(f"  - {dep.text}")

    lines.append("\nExecution order (from the PGPS dependency graph):")
    for step_item in explanation.execution_order:
        api = f" [{step_item.api_name}]" if step_item.api_name else ""
        lines.append(
            f"  {step_item.order}. {step_item.operation_id}{api} - {step_item.reason}"
        )

    return "\n".join(lines)


__all__ = [
    "EXPLANATION_SOURCE",
    "PlanExplanation",
    "FactProvenance",
    "DependencyExplanation",
    "StepExplanation",
    "explain_plan",
    "render_narrative",
]
