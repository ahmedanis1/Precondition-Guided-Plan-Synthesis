"""PGPS planner: turns a :class:`PlanRequest` into a :class:`PlanResult`.

Thin, deterministic bridge between the fact-level :class:`PGPSSolver` and the
shared plan models. It:

* derives the goal facts (explicit ``metadata['goal_facts']``, or - the normal
  case - every ``requires`` fact of the candidate operations themselves, so a
  pure "sink" operation that produces nothing, like ``extendQosSessionDuration``,
  still drives backward chaining even though it has no postcondition of its
  own to serve as a goal),
* takes the initial facts from ``request.known_facts``,
* lets the solver search the *whole* condition model for a producer of a
  missing fact - not just the operations API Selection happened to pick, so
  PGPS can genuinely discover on its own that, say, ``extendQosSessionDuration``
  needs a session id, and that ``retrieveSessionsByDevice`` (or whichever real
  operation the CAMARA spec names) is what supplies it. This is the core
  planning behaviour the thesis describes: API Selection narrows candidates by
  intent, PGPS alone decides the dependency-ordered sequence by checking each
  operation's pre/postconditions,
* always includes the candidate operations themselves as (final) steps once
  their preconditions are satisfiable - a sink operation is never "selected"
  by the solver's own produces-a-fact mechanism (it has nothing to produce),
  so it has to be added explicitly rather than left to fall out of the search,
* maps the ordered operation ids to :class:`PlanStep` objects (method, path,
  ``requires`` / ``produces``, ``depends_on``).

Decoupling: PGPS's dependency knowledge (the :class:`ConditionModel` - what
each operation requires/produces) always comes from ``data/conditions/``
(see ``conditions_loader.py``) or an explicitly-injected ``condition_model``,
never from :class:`CamaraRegistry`. The registry's only role here is
structural: when the solver pulls in an operation that wasn't itself an
API-Selection candidate, this planner looks up its ``method``/``path`` from a
registry (built lazily, once, only when neither an explicit ``registry`` nor
``condition_model`` was supplied) purely to build a runnable :class:`PlanStep`
- it plays no part in *which* operations get chained or in what order.

Parameter binding is not done here - that is milestones 9 / 12.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import Settings, get_settings
from camara_pgps.models import CamaraOperation, Intent
from camara_pgps.models.plan import (
    BindingSource,
    ParameterBinding,
    PlanRequest,
    PlanResult,
    PlanStatus,
    PlanStep,
)
from camara_pgps.pgps.conditions import ConditionModel
from camara_pgps.pgps.conditions_loader import build_condition_model
from camara_pgps.pgps.parameter_facts import parameter_for_fact
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
        self.registry = registry

        if condition_model is not None:
            self.model = condition_model
        elif registry is not None:
            # Explicit backward-compat path: caller wants conditions sourced
            # from their own registry document rather than data/conditions/.
            self.model = ConditionModel.from_registry(registry)
        else:
            self.model = build_condition_model(settings=self.settings)
            # Conditions still come from data/conditions/, not the registry -
            # but backward chaining can now pull in an operation that was
            # never handed to this planner as a candidate (see module
            # docstring), and building its PlanStep needs a real
            # method/path from somewhere. Built once (this planner instance
            # is itself cached process-wide by get_planner()), purely for
            # that structural lookup.
            self.registry = CamaraRegistry.from_settings(self.settings)

        self.solver = PGPSSolver(self.model)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> PGPSPlanner:
        return cls(settings=settings or get_settings())

    # -- public -----------------------------------------------------------

    def plan(self, request: PlanRequest) -> PlanResult:
        operations_by_id = {
            op.operation_id: op for op in request.candidate_operations
        }

        initial_facts = set(request.known_facts)

        brief = request.metadata.get("precondition_brief")
        if brief:
            branch_plan = self._plan_precondition_branch(
                request, brief, operations_by_id, initial_facts
            )
            if branch_plan is not None:
                return branch_plan
            # Brief didn't ground (an unknown operation id, or its branch(es)
            # don't plan at all) - fall through to ordinary planning exactly
            # as if no brief had been given, rather than failing the request
            # over a check that turned out not to be groundable.
        # An *explicit* metadata['goal_facts'] (the plan_for()/legacy path,
        # and every existing hand-written test) means "here is a pool of
        # candidate operations - find the minimal subset that reaches this
        # goal", so only whatever the solver actually needs belongs in the
        # plan. When goal facts are derived instead (the real, RAG-driven
        # workflow - build_plan_request_node never has a postcondition to
        # name one explicitly), candidate_operations are the actions API
        # Selection actually chose to satisfy the user's intent, not a pool
        # to pick from - they must always end up in the plan once whatever
        # they depend on has been chained in front of them.
        explicit_goal = bool(request.metadata.get("goal_facts"))
        goal_facts = self._goal_facts(request)

        if not goal_facts:
            if not request.candidate_operations:
                # Genuinely nothing to plan: no explicit goal and no
                # candidates to fall back to.
                return PlanResult(
                    status=PlanStatus.INFEASIBLE,
                    validation_errors=[
                        "no goal facts: the request has neither "
                        "metadata['goal_facts'] nor candidate operations "
                        "with conditions to derive requirements from"
                    ],
                    metadata={"initial_facts": sorted(initial_facts)},
                )
            # Candidates exist but the condition model has nothing for any
            # of them (e.g. an operation with no data/conditions entry at
            # all) - nothing for the solver to reason about, so fall back to
            # including them directly and surfacing unmet requires as
            # warnings rather than failures.
            return self._plan_direct(request, operations_by_id, initial_facts)

        solver_result = self.solver.solve(
            SolverRequest(
                goals=goal_facts,
                initial_facts=initial_facts,
                # Deliberately unrestricted: PGPS reasons over every
                # operation the condition model knows about, not just the
                # candidates API Selection handed it - that's what lets it
                # discover a prerequisite (e.g. a "list sessions" call) on
                # its own from pre/postconditions alone.
                allowed_operations=None,
                max_operations=request.max_steps,
            )
        )
        return self._to_plan_result(
            solver_result,
            request,
            operations_by_id,
            initial_facts,
            force_include_candidates=not explicit_goal,
        )

    def plan_for(
        self,
        *,
        goals: Iterable[str],
        initial_facts: Iterable[str] = (),
        candidate_operations: Sequence[CamaraOperation | str] | None = None,
        intent: Intent | None = None,
        max_steps: int = 20,
    ) -> PlanResult:
        """Convenience wrapper that builds the :class:`PlanRequest` for you.

        A bare operation-id string can only be resolved when this planner
        was constructed with an explicit ``registry=``, or (the normal
        production path) built one lazily in ``__init__`` because neither a
        ``registry`` nor a ``condition_model`` was supplied.
        """

        operations: list[CamaraOperation] = []
        for op in candidate_operations or ():
            if isinstance(op, CamaraOperation):
                operations.append(op)
            elif self.registry is not None:
                operations.append(self.registry.require_operation(str(op)))
            else:
                raise ValueError(
                    f"cannot resolve bare operation id {op!r}: this planner "
                    "has no registry. Pass a CamaraOperation instance - as "
                    "RAG / API selection provides - or construct "
                    "PGPSPlanner(registry=...) / PGPSPlanner(condition_model=...) "
                    "if you need this legacy string-id convenience without "
                    "the default lazily-built registry."
                )

        request = PlanRequest(
            intent=intent or Intent(text="", goal=""),
            candidate_operations=operations,
            known_facts=dict.fromkeys(initial_facts, True),
            max_steps=max_steps,
            metadata={"goal_facts": list(goals)},
        )
        return self.plan(request)

    # -- precondition branch (check-then-branch / gate) ---------------------

    def _find_observation_replacement(
        self,
        facts: list[str],
        operations_by_id: dict[str, CamaraOperation],
        initial_facts: set[str],
        exclude: set[str],
        extra_pool_ids: list[str] | None = None,
    ) -> tuple[str, CamaraOperation] | None:
        """Find a real, already-satisfiable producer of ``facts`` to use as
        the resolver instead, when the LLM's chosen one turned out circular.

        A genuine observation operation for this purpose is one that (a) is
        already a candidate operation - not something invented, mirroring
        the grounding discipline everywhere else in this planner, (b) is
        not one of the branch destinations themselves (a branch's own
        effect is not a precondition check for entering that branch), and
        (c) needs nothing beyond what's already known, so it can still be
        an unconditional first step exactly like the LLM's original pick
        was meant to be.

        ``operations_by_id`` alone is often too narrow here: it's built
        from `state.selected_operations`, and `precondition_node`
        deliberately only widens that to the resolver, the branch
        operations and whatever was originally selected - not every
        sibling it looked at while grounding the brief (that would bloat
        parameter extraction and the final narrative with irrelevant
        operations). ``extra_pool_ids`` is that fuller sibling list
        (`precondition_brief["candidate_pool"]`), resolved against the
        registry directly, so a real observation operation sitting right
        next to a bad resolver choice (e.g. retrieveSessionsByDevice next
        to getSession, both from the same API) is still visible here even
        though it never made it into `selected_operations`.

        Iteration order follows ``operations_by_id`` first, then
        ``extra_pool_ids``, so the choice is deterministic for a given
        request.
        """

        needed = set(facts)

        def _consider(operation_id: str, operation: CamaraOperation | None):
            if operation is None or operation_id in exclude:
                return None
            conditions = self.model.get(operation_id)
            if conditions is None:
                return None
            if not (needed & set(conditions.produces)):
                return None
            if set(conditions.requires) - initial_facts:
                return None
            return operation_id, operation

        for operation_id, operation in operations_by_id.items():
            found = _consider(operation_id, operation)
            if found is not None:
                return found

        for operation_id in extra_pool_ids or []:
            if operation_id in operations_by_id:
                continue  # already tried above
            found = _consider(operation_id, self.registry.get_operation(operation_id))
            if found is not None:
                return found

        return None

    def _plan_precondition_branch(
        self,
        request: PlanRequest,
        brief: dict,
        operations_by_id: dict[str, CamaraOperation],
        initial_facts: set[str],
    ) -> PlanResult | None:
        """Build a plan whose step 0 is a real precondition check.

        This is what puts the check literally *in* the plan PGPS reports and
        the executor runs, instead of resolving it somewhere else before
        planning even starts: step 0 is always the resolver operation
        (forced in - the whole point is to run it for real, not skip it
        because some fact it produces already looks "known"), and the
        remaining steps are a normal, fully backward-chained sub-plan for
        the branch operation applies-optimistically-assumes, tagged so the
        executor can tell, once the resolver has actually run, whether that
        assumption held - and if not, splice in the other branch's own
        (also fully chained) sub-plan instead. Neither branch is executed
        here; this method only plans, exactly like every other path through
        this planner.
        """

        mode = brief.get("mode")
        resolver_id = brief.get("resolver_operation")
        resolver_op = operations_by_id.get(resolver_id)
        if resolver_op is None:
            return None

        if mode == "alternative":
            default_id = brief.get("if_resolved_operation")
            other_id = brief.get("if_unresolved_operation")
        elif mode == "gate":
            default_id = brief.get("gated_operation")
            other_id = None
        else:
            return None

        default_op = operations_by_id.get(default_id) if default_id else None
        if default_op is None:
            return None
        if other_id is not None and other_id not in operations_by_id:
            return None

        resolver_conditions = self.model.get(resolver_id)
        resolver_requires = sorted(resolver_conditions.requires) if resolver_conditions else []
        resolver_produces = sorted(resolver_conditions.produces) if resolver_conditions else []

        warnings: list[str] = []
        unmet_resolver = sorted(set(resolver_requires) - initial_facts)
        if unmet_resolver:
            # An unmet requirement with NO producer anywhere is just a
            # genuine gap (typically something that belongs in initial
            # facts, e.g. an auth token) - forcing the resolver in as step 0
            # anyway and surfacing a warning is the existing, harmless
            # behaviour for that case.
            #
            # But when the missing fact *does* have a real producer, this
            # resolver is circular (e.g. the LLM picking getSession - which
            # itself requires qos.session.id - as the check for *whether a
            # session exists at all*: the very thing that fact would come
            # from). It can never legitimately run first. Rather than give
            # up on the whole conditional plan and fall back to ordinary
            # planning - which has no notion of "these are alternatives,
            # only run one" and ends up demanding parameters for every
            # branch at once - look for a real observation operation
            # already among the candidates that can serve as the resolver
            # instead: one that actually produces the missing fact and
            # needs nothing beyond what's already known, so it can still be
            # a clean, unconditional first step. This is exactly what
            # api_selection's own rejected siblings exist for - e.g.
            # retrieveSessionsByDevice sitting right next to getSession.
            circular = [fact for fact in unmet_resolver if self.model.producers_of(fact)]
            if circular:
                replacement = self._find_observation_replacement(
                    circular,
                    operations_by_id,
                    initial_facts,
                    exclude={resolver_id, default_id, other_id} - {None},
                    extra_pool_ids=brief.get("candidate_pool"),
                )
                if replacement is None:
                    return None
                original_resolver_id = resolver_id
                resolver_id, resolver_op = replacement
                warnings.append(
                    f"{original_resolver_id}: circular precondition check "
                    f"(needs {circular} to discover it) - "
                    f"using {resolver_id} as the resolver instead"
                )
                resolver_conditions = self.model.get(resolver_id)
                resolver_requires = (
                    sorted(resolver_conditions.requires) if resolver_conditions else []
                )
                resolver_produces = (
                    sorted(resolver_conditions.produces) if resolver_conditions else []
                )
                unmet_resolver = sorted(set(resolver_requires) - initial_facts)
                if unmet_resolver:
                    # The replacement turned out not to be clean either -
                    # don't chain replacements recursively, just bail.
                    return None
            else:
                warnings.append(f"{resolver_id}: unmet precondition(s) {unmet_resolver}")

        # Two sanity checks specific to "alternative" mode, both found from
        # real (Ollama-backed) runs where the LLM's own reasoning, not
        # anything mechanical, was the source of the mistake - grounding a
        # brief against real operation ids (as this planner always has)
        # says nothing about whether the *pairing* of resolver and branches
        # actually makes sense together.
        if mode == "alternative" and other_id is not None:
            other_op = operations_by_id[other_id]
            default_conditions = self.model.get(default_id)
            other_conditions = self.model.get(other_id)
            default_produces = set(default_conditions.produces) if default_conditions else set()
            other_produces = set(other_conditions.produces) if other_conditions else set()
            other_requires = set(other_conditions.requires) if other_conditions else set()
            default_requires = set(default_conditions.requires) if default_conditions else set()
            resolver_fact_set = set(resolver_produces)

            # (a) Backwards assignment: the "confirmed" (if_resolved) branch
            # re-produces the very resource the resolver just confirmed
            # already exists (e.g. resolver confirms a subscription exists,
            # but if_resolved_operation is the *create* operation - which
            # would create a duplicate of something just confirmed present)
            # while the "not confirmed" (if_unresolved) branch does not.
            # That is exactly backwards: creating something is what belongs
            # on the *not-confirmed* side. Swap them rather than reject the
            # whole branch - the check itself is still meaningful, only the
            # LLM's mapping of which side is which was inverted.
            if (resolver_fact_set & default_produces) and not (resolver_fact_set & other_produces):
                warnings.append(
                    f"{default_id}/{other_id}: swapped - {default_id} would "
                    f"re-produce {sorted(resolver_fact_set & default_produces)}, "
                    "which the check just confirmed already exists; that "
                    f"belongs on the not-confirmed side, not {other_id}"
                )
                default_id, other_id = other_id, default_id
                default_op, other_op = other_op, default_op
                default_conditions, other_conditions = other_conditions, default_conditions
                default_produces, other_produces = other_produces, default_produces
                default_requires, other_requires = other_requires, default_requires

            # (b) No structural relationship at all: neither branch requires
            # or produces anything the resolver itself produces. A
            # legitimate check-then-branch means the branches' own shape
            # (usually: one of them needs the fact the check discovers)
            # actually depends on what was checked - if it doesn't, the
            # "check" and the "branch" are about two unrelated things the
            # LLM only linked in its own narrative, not in the fact graph
            # (seen for real: a SIM-swap-recency check gating whether to
            # create vs. extend a completely unrelated QoS session). Reject
            # the whole branch and fall back to ordinary planning rather
            # than executing a plan whose two halves have no real
            # dependency on each other.
            related = resolver_fact_set & (
                default_requires | default_produces | other_requires | other_produces
            )
            if not related:
                return None

        step0 = PlanStep(
            index=0,
            operation_id=resolver_id,
            method=resolver_op.method,
            path=resolver_op.path,
            requires=resolver_requires,
            produces=resolver_produces,
            rationale=f"precondition check - {brief.get('reason')}".rstrip(" -")
            if brief.get("reason")
            else "precondition check",
        )

        known_after_resolver = initial_facts | set(resolver_produces)

        default_plan = self.plan(
            PlanRequest(
                intent=request.intent,
                candidate_operations=[default_op],
                known_facts=dict.fromkeys(known_after_resolver, True),
                max_steps=request.max_steps,
            )
        )
        if default_plan.status not in (PlanStatus.VALID, PlanStatus.DRAFT) or not default_plan.steps:
            return None

        other_steps_dump: list[dict] | None = None
        if other_id is not None:
            other_op = operations_by_id[other_id]
            other_plan = self.plan(
                PlanRequest(
                    intent=request.intent,
                    candidate_operations=[other_op],
                    known_facts=dict.fromkeys(initial_facts, True),
                    max_steps=request.max_steps,
                )
            )
            if other_plan.status in (PlanStatus.VALID, PlanStatus.DRAFT) and other_plan.steps:
                other_steps_dump = [s.model_dump(mode="json") for s in other_plan.steps]
            else:
                # The fallback branch doesn't plan at all - still proceed with
                # the default branch (better than refusing the whole request
                # over a branch that may never actually be needed), but the
                # executor will have nothing to splice in if it is needed.
                warnings.append(
                    f"{other_id}: could not be planned as the fallback branch "
                    f"({other_plan.status.value}); it will fail at execution "
                    "time if the check ever selects it"
                )

        branch_meta = {
            "mode": mode,
            "resolver_operation": resolver_id,
            "resolver_step_index": 0,
            "resolver_produces": resolver_produces,
            # Reported from `default_id`/`other_id`, not re-read from the
            # original `brief` - if the backwards-assignment check above
            # swapped them, the brief's own fields are stale and would
            # describe the plan this method decided *against* building.
            "if_resolved_operation": default_id if mode == "alternative" else None,
            "if_unresolved_operation": other_id if mode == "alternative" else None,
            "gated_operation": brief.get("gated_operation"),
            "reason": brief.get("reason"),
            "alt_steps": other_steps_dump,
        }

        # `default_plan` was solved in isolation (known_after_resolver just
        # says "qos.session.id is available", not "step 0 produces it") -
        # its own steps therefore carry no PRIOR_STEP binding back to the
        # resolver, exactly the "fully-chained plan still stops for
        # clarification" gap _build_steps's own binding pass exists to
        # avoid. Redo that same pass here across the spliced-in sequence, so
        # a parameter step 0 actually supplies (sessionId, from
        # qos.session.id) is recognised as bound, not re-requested.
        fact_producer: dict[str, int] = dict.fromkeys(resolver_produces, 0)
        steps = [step0]
        for offset, sub_step in enumerate(default_plan.steps):
            depends_on = [d + 1 for d in sub_step.depends_on]
            index = offset + 1
            if offset == 0:
                depends_on = sorted({*depends_on, 0})

            bindings = list(sub_step.bindings)
            bound_params = {b.parameter for b in bindings}
            for fact in sub_step.requires:
                producer_index = fact_producer.get(fact)
                if producer_index is None:
                    continue
                parameter = parameter_for_fact(fact)
                if parameter is None or parameter in bound_params:
                    continue
                bindings.append(
                    ParameterBinding(
                        parameter=parameter,
                        source=BindingSource.PRIOR_STEP,
                        from_step=producer_index,
                    )
                )
                bound_params.add(parameter)
                depends_on = sorted({*depends_on, producer_index})

            steps.append(
                sub_step.model_copy(
                    update={"index": index, "depends_on": depends_on, "bindings": bindings}
                )
            )
            for fact in sub_step.produces:
                fact_producer.setdefault(fact, index)

        # Tag every spliced-in step of the default branch, not just the
        # first: whichever one the executor reaches first still carries
        # enough information to splice in the alternate branch from there.
        steps[1:] = [s.model_copy(update={"precondition_branch": branch_meta}) for s in steps[1:]]

        return PlanResult(
            status=PlanStatus.VALID,
            steps=steps,
            warnings=[*warnings, *default_plan.warnings],
            rationale=brief.get("reason"),
            metadata={
                "initial_facts": sorted(initial_facts),
                "precondition_branch": branch_meta,
            },
        )

    # -- goal derivation -----------------------------------------------------

    def _goal_facts(self, request: PlanRequest) -> set[str]:
        explicit = request.metadata.get("goal_facts")
        if explicit:
            return {str(fact) for fact in explicit}

        # The goal is "make every candidate operation executable", not
        # "whatever fact nothing else consumes" - the latter is invisible to
        # a pure sink operation (produces nothing), which is exactly the
        # shape of the terminal action a user's request usually names
        # (extend/delete/revoke/...). Asking for its own requires directly is
        # what lets the solver discover a prerequisite on its own.
        required: set[str] = set()
        for operation in request.candidate_operations:
            conditions = self.model.get(operation.operation_id)
            if conditions is None:
                continue
            required.update(conditions.requires)

        # Only hand the solver a requirement at least one operation can
        # actually produce. A requirement with no producer anywhere (a
        # duration, a profile name, ...) isn't a missing API call to chain
        # in - it's a plain value parameter extraction is responsible for.
        # Asking the solver to "solve" it would report the whole plan
        # infeasible over something PGPS was never meant to resolve; it
        # still surfaces per-step as an unmet-precondition warning (see
        # _build_steps) and, if genuinely never grounded, via parameter
        # extraction's own missing_information / clarification.
        return {fact for fact in required if self.model.producers_of(fact)}

    # -- direct fallback (no conditions to reason about at all) -------------

    def _plan_direct(
        self,
        request: PlanRequest,
        operations_by_id: dict[str, CamaraOperation],
        initial_facts: set[str],
    ) -> PlanResult:
        """Handle candidates the condition model has nothing for.

        Some operations genuinely have no ``data/conditions`` entry (not yet
        authored, or a one-off custom operation). With no fact-level
        information at all there is nothing for the solver to reason about,
        so this returns the candidates as a single-pass plan in the order API
        Selection ranked them, surfacing any unmet ``requires`` as warnings
        rather than failures, since parameter binding is a later stage
        (milestones 9 / 12), not this planner's job.
        """

        steps: list[PlanStep] = []
        warnings: list[str] = []
        known = set(initial_facts)

        for index, operation in enumerate(request.candidate_operations):
            conditions = self.model.get(operation.operation_id)
            requires = sorted(conditions.requires) if conditions else []
            produces = sorted(conditions.produces) if conditions else []
            missing = sorted(set(requires) - known)
            if missing:
                warnings.append(
                    f"{operation.operation_id}: unmet precondition(s) {missing} "
                    "(expected to be filled by parameter extraction)"
                )
            steps.append(
                PlanStep(
                    index=index,
                    operation_id=operation.operation_id,
                    method=operation.method,
                    path=operation.path,
                    requires=requires,
                    produces=produces,
                    rationale="direct: no conditions to chain against another step",
                )
            )
            known |= set(produces)

        return PlanResult(
            status=PlanStatus.VALID,
            steps=steps,
            warnings=warnings
            or ["direct plan: candidate operations have no conditions to chain"],
            metadata={
                "initial_facts": sorted(initial_facts),
                "selected_operations": [
                    op.operation_id for op in request.candidate_operations
                ],
                "note": "no goal facts derived; steps kept in candidate order",
            },
        )

    # -- mapping to PlanResult ---------------------------------------------

    def _to_plan_result(
        self,
        result: SolverResult,
        request: PlanRequest,
        operations_by_id: dict[str, CamaraOperation],
        initial_facts: set[str],
        *,
        force_include_candidates: bool,
    ) -> PlanResult:
        metadata = {
            "solver_status": result.status.value,
            "goal_facts": sorted(result.goal_facts),
            "initial_facts": sorted(result.initial_facts),
            "selected_operations": sorted(result.selected),
        }
        rationale = "; ".join(result.reasoning) or None

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

        # SOLVED or EMPTY (goal facts already satisfied by initial facts):
        # either way, still append the candidate operations themselves - a
        # sink operation is never "selected" by the solver (it produces
        # nothing to be selected *for*), so EMPTY in particular must not
        # collapse to an empty plan when candidates exist.
        prereq_order = result.order if result.status is SolverStatus.SOLVED else []
        must_include = request.candidate_operations if force_include_candidates else []
        steps, warnings = self._build_steps(
            prereq_order, operations_by_id, must_include, initial_facts
        )

        return PlanResult(
            status=PlanStatus.VALID,
            steps=steps,
            warnings=warnings,
            rationale=rationale,
            metadata={
                **metadata,
                "note": "dependency-valid; parameter binding pending (milestones 9/12)",
            },
        )

    def _build_steps(
        self,
        prereq_order: list[str],
        operations_by_id: dict[str, CamaraOperation],
        candidate_operations: list[CamaraOperation],
        initial_facts: set[str],
    ) -> tuple[list[PlanStep], list[str]]:
        """Build steps for the solved prerequisite chain, then the candidates.

        ``prereq_order`` comes from the solver and is already
        dependency-correct. The candidate operations (the ones API Selection
        actually picked) are appended afterwards in their given order,
        skipping any already present (the solver *can* select a candidate
        directly, when that candidate also happens to produce a needed
        fact - e.g. ``createSession`` chained in front of ``deleteSession``).

        An unmet requirement on an appended candidate is reported as a
        warning, not a planning failure: some facts (a duration, a profile
        name) are only ever meant to come from parameter extraction, not
        from another operation's response, and binding those is a later
        stage's job (milestones 9 / 12) - PGPS only owns *inter-operation*
        dependencies.

        Also records, generically for any operation/fact pair via the shared
        ``parameter_for_fact`` vocabulary, a :class:`ParameterBinding` on any
        step whose requirement is satisfied by an *earlier step in this same
        plan* (as opposed to an initial fact). This is what lets the
        validator's payload layer and the executor recognise a parameter like
        ``sessionId`` as already resolved when PGPS itself chained in the
        operation that supplies it - instead of both independently treating
        it as something a person still has to provide, which is exactly the
        gap that let a fully-chained plan still stop for clarification.
        """

        order = list(prereq_order)
        for operation in candidate_operations:
            if operation.operation_id not in order:
                order.append(operation.operation_id)

        steps: list[PlanStep] = []
        warnings: list[str] = []
        produced_by_index: list[set[str]] = []
        known = set(initial_facts)
        # fact -> earliest step index that produces it (within this plan only;
        # a fact already in `initial_facts` never gets an entry here, since
        # that isn't a *prior-step* binding - parameter extraction already
        # had a chance to bind it directly).
        fact_producer: dict[str, int] = {}

        for index, operation_id in enumerate(order):
            operation = operations_by_id.get(operation_id)
            if operation is None and self.registry is not None:
                operation = self.registry.get_operation(operation_id)

            conditions = self.model.for_operation(operation_id)
            requires = sorted(conditions.requires)
            produces = sorted(conditions.produces)

            unmet = sorted(set(requires) - known)
            if unmet:
                warnings.append(f"{operation_id}: unmet precondition(s) {unmet}")

            depends_on = [
                earlier
                for earlier in range(index)
                if produced_by_index[earlier] & conditions.requires
            ]

            bindings: list[ParameterBinding] = []
            for fact in requires:
                producer_index = fact_producer.get(fact)
                if producer_index is None:
                    continue
                parameter = parameter_for_fact(fact)
                if parameter is None:
                    continue
                bindings.append(
                    ParameterBinding(
                        parameter=parameter,
                        source=BindingSource.PRIOR_STEP,
                        from_step=producer_index,
                    )
                )

            steps.append(
                PlanStep(
                    index=index,
                    operation_id=operation_id,
                    method=operation.method if operation else "GET",
                    path=operation.path if operation else "",
                    bindings=bindings,
                    depends_on=depends_on,
                    requires=requires,
                    produces=produces,
                    rationale=f"produces {produces}" if produces else None,
                )
            )
            produced_by_index.append(set(produces))
            known |= set(produces)
            for fact in produces:
                fact_producer.setdefault(fact, index)

        return steps, warnings


@lru_cache(maxsize=1)
def get_planner() -> PGPSPlanner:
    """Return the process-wide PGPS planner built from configured settings."""

    return PGPSPlanner.from_settings()


__all__ = ["PGPSPlanner", "get_planner"]
