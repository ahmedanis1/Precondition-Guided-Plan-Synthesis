"""PGPS planner: turns a :class:`PlanRequest` into a :class:`PlanResult`.

Thin, deterministic bridge between the fact-level :class:`PGPSSolver` and the
shared plan models. It:

* derives the goal facts (explicit ``metadata['goal_facts']`` or the terminal
  postconditions of the candidate operations),
* takes the initial facts from ``known_facts`` and ``intent.known_facts``,
* restricts the solver to the candidate operations from API Selection,
* maps the ordered operation ids to :class:`PlanStep` objects (method, path,
  ``requires`` / ``produces``, ``depends_on``).

Parameter binding is not done here - that is milestones 9 / 12.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import Settings, get_settings
from camara_pgps.models import CamaraOperation, Intent
from camara_pgps.models.plan import PlanRequest, PlanResult, PlanStatus, PlanStep
from camara_pgps.pgps.conditions import ConditionModel
from camara_pgps.pgps.solver import PGPSSolver, SolverRequest, SolverResult, SolverStatus


class PGPSPlanner:
    """Deterministic PGPS planner."""

    def __init__(
        self,
        *,
        registry: CamaraRegistry | None = None,
        condition_model: ConditionModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or CamaraRegistry.from_settings(self.settings)
        self.model = condition_model or ConditionModel.from_registry(self.registry)
        self.solver = PGPSSolver(self.model)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> PGPSPlanner:
        return cls(settings=settings or get_settings())

    # -- public -----------------------------------------------------------

    def plan(self, request: PlanRequest) -> PlanResult:
        candidate_ids = [op.operation_id for op in request.candidate_operations]
        allowed = set(candidate_ids) or None

        initial_facts = set(request.known_facts) | set(request.intent.known_facts)
        goal_facts = self._goal_facts(request)

        if not goal_facts:
            return PlanResult(
                status=PlanStatus.INFEASIBLE,
                validation_errors=[
                    "no goal facts: the request has neither metadata['goal_facts'] "
                    "nor candidate operations with postconditions"
                ],
                metadata={"initial_facts": sorted(initial_facts)},
            )

        solver_result = self.solver.solve(
            SolverRequest(
                goals=goal_facts,
                initial_facts=initial_facts,
                allowed_operations=allowed,
                max_operations=request.max_steps,
            )
        )
        return self._to_plan_result(solver_result, request)

    def plan_for(
        self,
        *,
        goals: Iterable[str],
        initial_facts: Iterable[str] = (),
        candidate_operations: Sequence[CamaraOperation | str] | None = None,
        intent: Intent | None = None,
        max_steps: int = 20,
    ) -> PlanResult:
        """Convenience wrapper that builds the :class:`PlanRequest` for you."""

        operations = [
            op
            if isinstance(op, CamaraOperation)
            else self.registry.require_operation(str(op))
            for op in (candidate_operations or [])
        ]
        request = PlanRequest(
            intent=intent or Intent(text="", goal=""),
            candidate_operations=operations,
            known_facts=dict.fromkeys(initial_facts, True),
            max_steps=max_steps,
            metadata={"goal_facts": list(goals)},
        )
        return self.plan(request)

    # -- goal derivation -----------------------------------------------------

    @staticmethod
    def _goal_facts(request: PlanRequest) -> set[str]:
        explicit = request.metadata.get("goal_facts")
        if explicit:
            return {str(fact) for fact in explicit}

        produced: set[str] = set()
        required: set[str] = set()
        for operation in request.candidate_operations:
            for condition in operation.postconditions:
                produced.update(condition.produces)
            for condition in operation.preconditions:
                required.update(condition.requires)
        terminal = produced - required
        return terminal or produced

    # -- mapping to PlanResult ---------------------------------------------

    def _to_plan_result(
        self, result: SolverResult, request: PlanRequest
    ) -> PlanResult:
        metadata = {
            "solver_status": result.status.value,
            "goal_facts": sorted(result.goal_facts),
            "initial_facts": sorted(result.initial_facts),
            "selected_operations": sorted(result.selected),
        }
        rationale = "; ".join(result.reasoning) or None

        if result.status is SolverStatus.EMPTY:
            return PlanResult(
                status=PlanStatus.VALID,
                steps=[],
                warnings=["goal facts are already satisfied by the known facts"],
                rationale=rationale,
                metadata=metadata,
            )

        if result.status is SolverStatus.CYCLIC:
            return PlanResult(
                status=PlanStatus.INVALID,
                validation_errors=[
                    "cyclic dependency: " + " -> ".join(result.cycle or [])
                ],
                rationale=rationale,
                metadata=metadata,
            )

        if result.status is SolverStatus.INFEASIBLE:
            unmet = sorted(
                {fact for facts in result.missing_facts.values() for fact in facts}
            )
            return PlanResult(
                status=PlanStatus.INFEASIBLE,
                unmet_preconditions=unmet,
                validation_errors=(
                    [f"cannot achieve goal fact(s): {sorted(result.unresolved_goals)}"]
                    if result.unresolved_goals
                    else []
                ),
                rationale=rationale,
                metadata={**metadata, "missing_facts": result.missing_facts},
            )

        steps = self._build_steps(result)
        return PlanResult(
            status=PlanStatus.VALID,
            steps=steps,
            rationale=rationale,
            metadata={
                **metadata,
                "note": "dependency-valid; parameter binding pending (milestones 9/12)",
            },
        )

    def _build_steps(self, result: SolverResult) -> list[PlanStep]:
        order = result.order
        produced_by_index: list[set[str]] = []
        steps: list[PlanStep] = []
        for index, operation_id in enumerate(order):
            operation = self.registry.require_operation(operation_id)
            conditions = self.model.for_operation(operation_id)
            requires = sorted(conditions.requires)
            produces = sorted(conditions.produces)
            depends_on = [
                earlier
                for earlier in range(index)
                if produced_by_index[earlier] & conditions.requires
            ]
            steps.append(
                PlanStep(
                    index=index,
                    operation_id=operation_id,
                    method=operation.method,
                    path=operation.path,
                    depends_on=depends_on,
                    requires=requires,
                    produces=produces,
                    rationale=f"produces {produces}" if produces else None,
                )
            )
            produced_by_index.append(set(produces))
        return steps


@lru_cache(maxsize=1)
def get_planner() -> PGPSPlanner:
    """Return the process-wide PGPS planner built from configured settings."""

    return PGPSPlanner.from_settings()


__all__ = ["PGPSPlanner", "get_planner"]
