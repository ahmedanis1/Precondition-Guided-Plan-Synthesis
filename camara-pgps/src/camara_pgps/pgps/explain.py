"""Plan explainability - derived from the PGPS dependency graph, never an LLM.

Turns a :class:`~camara_pgps.models.plan.PlanResult` into a structured account of
*why* the plan has the shape it does::

    Goal -> Selected APIs -> Required facts -> Produced facts
         -> Dependencies -> Execution order -> Conditional branch(es)

Every field comes from the operations' pre/postconditions and the fact-flow edges
between them; :attr:`PlanExplanation.source` is fixed to ``pgps-dependency-graph``.

The last part - conditional branches - answers a question this module used
to leave entirely to hindsight: *before* any execution happens, how many
check-then-branch decisions did PGPS build into this plan, what triggers
each one, and what would run on either side of it? A plan step tagged with
``precondition_branch`` (see ``pgps/planner.py::_plan_precondition_branch``)
carries exactly that - which operation resolves the check, whether it's an
``"alternative"`` (two real continuations) or a ``"gate"`` (one, or
nothing), and, for an alternative, the *other* continuation's own steps
serialized as ``alt_steps`` - so this module can show both sides of a
decision before either one has actually run, not just narrate whichever
side execution happened to pick afterward.
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


class BranchExplanation(BaseModel):
    """One check-then-branch decision PGPS built into the plan, and what's
    pre-validated on either side of it - known before any execution happens.
    """

    resolver_operation: str
    mode: str  # "alternative" | "gate"
    reason: str | None = None
    default_path: list[str] = Field(default_factory=list)
    """Operations that run if the check confirms the assumed default - these
    are already spliced into `execution_order` above, not held separately."""
    alternate_path: list[str] = Field(default_factory=list)
    """For `mode="alternative"`: the *other* fully-planned continuation,
    pre-validated but not spliced into the main plan unless execution picks
    it. Empty for `mode="gate"` (see `alternate_note` instead)."""
    alternate_note: str | None = None
    """For `mode="gate"`: what happens instead of `default_path` when the
    check isn't confirmed (the gated operation(s) are skipped entirely, not
    replaced by another operation)."""


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
    branches: list[BranchExplanation] = Field(default_factory=list)
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

    # -- conditional branches ------------------------------------------------
    # A branch point is identified by its `precondition_branch` payload
    # (planner.py tags every spliced default-branch step with the *same*
    # dict), not by step position - grouping by content rather than object
    # identity means this still works after a JSON round-trip (e.g. a plan
    # that arrived over the HTTP API rather than in-process).
    branches: list[BranchExplanation] = []
    seen_branch_keys: set[tuple[str | None, ...]] = set()
    for step in steps:
        branch = step.precondition_branch
        if not branch:
            continue
        key = (
            branch.get("mode"),
            branch.get("resolver_operation"),
            branch.get("reason"),
        )
        if key in seen_branch_keys:
            continue
        seen_branch_keys.add(key)

        default_path = [
            s.operation_id
            for s in steps
            if s.precondition_branch is not None
            and (
                s.precondition_branch.get("mode"),
                s.precondition_branch.get("resolver_operation"),
                s.precondition_branch.get("reason"),
            )
            == key
        ]

        mode = branch.get("mode")
        if mode == "alternative":
            alt_steps = branch.get("alt_steps") or []
            alternate_path = [
                s.get("operation_id") for s in alt_steps if s.get("operation_id")
            ]
            alternate_note = None
        else:  # "gate" - there is no other operation, only "don't run it"
            alternate_path = []
            gated = branch.get("gated_operation") or (default_path[0] if default_path else None)
            alternate_note = (
                f"{gated} is skipped entirely - not replaced by anything else"
                if gated
                else "the gated operation(s) are skipped entirely"
            )

        branches.append(
            BranchExplanation(
                resolver_operation=branch.get("resolver_operation") or "",
                mode=mode or "",
                reason=branch.get("reason"),
                default_path=default_path,
                alternate_path=alternate_path,
                alternate_note=alternate_note,
            )
        )

    explanation = PlanExplanation(
        goal=intent.goal if intent is not None else None,
        goal_facts=goal_facts,
        selected_apis=selected_apis,
        initial_facts=known_initial,
        required_facts=required_facts,
        produced_facts=produced_facts,
        dependencies=dependencies,
        execution_order=execution_order,
        branches=branches,
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

    # Conditional branches are synthesized and fully planned *before* any
    # execution happens (Section 4.7/4.8 of docs/design-decisions.md) - this
    # is where that becomes visible pre-execution, not just afterward in the
    # post-run narrative (workflow/report.py), which can only ever describe
    # the one side execution actually picked.
    lines.append(f"\nConditional branch(es): {len(explanation.branches)}")
    for i, branch in enumerate(explanation.branches, start=1):
        header = f"  Branch {i} - check via {branch.resolver_operation} ({branch.mode})"
        if branch.reason:
            header += f": {branch.reason}"
        lines.append(header)
        default_chain = " -> ".join(branch.default_path) or "(nothing)"
        lines.append(f"    IF confirmed      -> {default_chain}")
        if branch.mode == "alternative":
            alt_chain = " -> ".join(branch.alternate_path) or "(nothing)"
            lines.append(f"    IF NOT confirmed  -> {alt_chain}")
        else:
            lines.append(f"    IF NOT confirmed  -> {branch.alternate_note}")

    return "\n".join(lines)


__all__ = [
    "EXPLANATION_SOURCE",
    "PlanExplanation",
    "FactProvenance",
    "DependencyExplanation",
    "StepExplanation",
    "BranchExplanation",
    "explain_plan",
    "render_narrative",
]
