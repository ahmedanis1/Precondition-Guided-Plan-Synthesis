"""Planning models.

:class:`PlanRequest` is the input to the PGPS planner (milestone 11); it bundles
the interpreted intent with the candidate operations chosen by API selection and
the facts already known. :class:`PlanResult` is the planner's output: an ordered,
dependency-checked sequence of :class:`PlanStep` objects that the plan validator
(milestone 12) and execution layer (milestone 13) act on.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from camara_pgps.models.api import CamaraOperation, HTTPMethod
from camara_pgps.models.intent import Intent
from camara_pgps.models.retrieval import RetrievalResult


class BindingSource(StrEnum):
    """Where a step argument's value comes from."""

    INTENT = "intent"
    CONSTANT = "constant"
    PRIOR_STEP = "prior_step"
    USER = "user"
    DEFAULT = "default"
    UNRESOLVED = "unresolved"


class PlanStatus(StrEnum):
    """Lifecycle state of a :class:`PlanResult`."""

    DRAFT = "draft"
    VALID = "valid"
    INVALID = "invalid"
    INFEASIBLE = "infeasible"


class ParameterBinding(BaseModel):
    """How a single operation parameter is filled for a step.

    When ``source`` is :attr:`BindingSource.PRIOR_STEP`, ``from_step`` is the
    producing step's index and ``from_pointer`` is a JSON Pointer into that
    step's response body.
    """

    parameter: str
    source: BindingSource
    value: Any | None = None
    from_step: int | None = None
    from_pointer: str | None = None
    required: bool = True


class PlanStep(BaseModel):
    """One CAMARA call in an ordered plan."""

    index: int
    operation_id: str
    method: HTTPMethod
    path: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    bindings: list[ParameterBinding] = Field(default_factory=list)
    depends_on: list[int] = Field(default_factory=list)
    requires: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    rationale: str | None = None
    # Set only on a step whose real operation isn't known until execution -
    # a check-then-branch or a gate (see agents.precondition_agent). Carries
    # everything the executor needs to resolve the real operation once the
    # resolver step (referenced by resolver_step_index) has actually run:
    # mode, resolver_operation, resolver_produces, if_resolved_operation /
    # if_unresolved_operation (alternative mode) or gated_operation (gate
    # mode), reason, and - for alternative mode - alt_steps: the fully
    # planned step sequence for the branch NOT shown here, serialised via
    # PlanStep.model_dump(), so the executor can splice it in wholesale if
    # that turns out to be the real branch instead of re-planning at
    # execution time.
    precondition_branch: dict[str, Any] | None = None

    def unresolved_bindings(self) -> list[ParameterBinding]:
        """Return required bindings that still have no value source."""

        return [
            b
            for b in self.bindings
            if b.required and b.source == BindingSource.UNRESOLVED
        ]


class PlanRequest(BaseModel):
    """Everything the planner needs to synthesise a plan."""

    intent: Intent
    candidate_operations: list[CamaraOperation] = Field(default_factory=list)
    retrieval: RetrievalResult | None = None
    known_facts: dict[str, Any] = Field(default_factory=dict)
    max_steps: int = 20
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanResult(BaseModel):
    """An ordered plan plus the planner's / validator's assessment of it."""

    status: PlanStatus = PlanStatus.DRAFT
    steps: list[PlanStep] = Field(default_factory=list)
    unresolved_parameters: list[str] = Field(default_factory=list)
    unmet_preconditions: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rationale: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_executable(self) -> bool:
        """True when the plan is valid and has at least one step."""

        return self.status == PlanStatus.VALID and bool(self.steps)

    def get_step(self, index: int) -> PlanStep | None:
        """Return the step at ``index`` in execution order, if present."""

        return next((s for s in self.steps if s.index == index), None)
