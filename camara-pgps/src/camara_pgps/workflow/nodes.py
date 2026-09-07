"""LangGraph node functions for the end-to-end CAMARA-PGPS pipeline.

Each node takes the :class:`AgentState` and returns a partial update. Every node
short-circuits when a previous node has set ``stage = FAILED`` so one failure
carries cleanly to the clarification step.
"""

from __future__ import annotations

from typing import Any

from camara_pgps.agents.api_selection_agent import APISelectionAgent
from camara_pgps.agents.intent_agent import IntentAgent
from camara_pgps.agents.parameter_agent import ParameterExtractionAgent, fact_for_parameter
from camara_pgps.agents.precondition_agent import PreconditionAgent, PreconditionBrief
from camara_pgps.agents.repair_agent import RepairAgent
from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import Settings, get_settings
from camara_pgps.execution import APIExecutor, build_executor
from camara_pgps.models.plan import PlanRequest, PlanStatus
from camara_pgps.models.state import AgentState, PipelineStage
from camara_pgps.pgps.conditions_loader import build_condition_model
from camara_pgps.pgps.explain import explain_plan
from camara_pgps.pgps.planner import PGPSPlanner
from camara_pgps.retrieval.rag import RagRetriever
from camara_pgps.validator.validator import PlanValidator

_Update = dict[str, Any]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


# supplied-argument name -> fact key(s) it makes available (for callers that
# skip the parameter-extraction node, e.g. a hand-supplied plan)
_ARG_FACTS: dict[str, tuple[str, ...]] = {
    "device": ("device.identifier",),
    "qosProfile": ("qos.profile",),
    "sessionId": ("qos.session.id",),
    "subscriptionId": ("network-insights.subscription.id",),
    "area": ("location.area",),
}


def facts_from_arguments(
    arguments: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Derive known fact keys from supplied ``{operation: {param: value}}`` arguments."""

    facts: dict[str, Any] = {}
    for operation_arguments in (arguments or {}).values():
        for name, value in operation_arguments.items():
            for fact_key in _ARG_FACTS.get(name, ()):
                facts.setdefault(fact_key, value)
    return facts


def plan_missing_parameters(state: AgentState) -> list[str]:
    """Required parameters of the *plan's* operations that are still unbound.

    Parameter extraction runs *before* PGPS planning, against only the
    operations API Selection picked - it has no idea PGPS might chain in an
    extra step (e.g. "list sessions" ahead of "extend session") purely to
    produce a fact a later step needs. So a name in ``missing_required`` is
    only genuinely missing if nothing already known, and nothing an earlier
    step in *this* plan produces, can supply it. Walking the plan in order
    and accumulating facts (exactly as the planner itself does when building
    steps) is what lets this recognise "PGPS's own plan will fetch this from
    the API" and stop asking the person for something they never have to
    provide - generically, for any operation/parameter, via the same
    parameter<->fact convention parameter extraction itself binds from.
    """

    if state.plan is None or state.parameters is None:
        return []
    supplied = state.supplied_arguments
    known_facts = set(state.facts)
    missing: list[str] = []
    for step in state.plan.steps:
        op_params = state.parameters.for_operation(step.operation_id)
        if op_params is not None:
            covered = set(supplied.get(step.operation_id, {}))
            for name in op_params.missing_required:
                if name in covered or name in missing:
                    continue
                fact_key = fact_for_parameter(name)
                if fact_key is not None and fact_key in known_facts:
                    continue
                missing.append(name)
        known_facts |= set(step.produces)
    return missing


class WorkflowNodes:
    """Holds the (injectable) components and exposes them as graph nodes."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        registry: CamaraRegistry | None = None,
        intent_agent: IntentAgent | None = None,
        rag_retriever: RagRetriever | None = None,
        selection_agent: APISelectionAgent | None = None,
        parameter_agent: ParameterExtractionAgent | None = None,
        planner: PGPSPlanner | None = None,
        validator: PlanValidator | None = None,
        executor: APIExecutor | None = None,
        repair_agent: RepairAgent | None = None,
        precondition_agent: PreconditionAgent | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or CamaraRegistry.from_settings(self.settings)
        self._intent_agent = intent_agent  # LLM-only; built lazily so the graph
        self._rag = rag_retriever          # can be compiled without an LLM key
        self.selection_agent = selection_agent or APISelectionAgent(
            settings=self.settings,
            rag=self._rag,
        )
        self.parameter_agent = parameter_agent or ParameterExtractionAgent(
            registry=self.registry, settings=self.settings
        )
        # Built once and shared by the planner, validator and the
        # precondition agent below - all three must reason about the exact
        # same dependency graph, and building it three times separately is
        # how it would drift out of sync if one call site's settings/loader
        # path ever changed without the others following.
        self.condition_model = build_condition_model(settings=self.settings)
        self.planner = planner or PGPSPlanner(
            registry=self.registry,
            # Explicit, because passing only `registry=` here would make
            # PGPSPlanner take its *legacy* from_registry() branch - building
            # the condition model from CamaraOperation.preconditions /
            # .postconditions on the registry object, which are empty for
            # every RAG-reconstructed real CAMARA operation. That silently
            # produced a PGPS that could never see any requires/produces at
            # all (hence plans that never chained anything, and "no
            # preconditions" in every explanation). PGPS's dependency
            # knowledge must always come from data/conditions/ - `registry=`
            # is kept alongside purely so the planner can still resolve
            # method/path for an operation the solver discovers that wasn't
            # itself a candidate.
            condition_model=self.condition_model,
            settings=self.settings,
        )
        self.validator = validator or PlanValidator(
            registry=self.registry,
            # Same reasoning as the planner above: without an explicit
            # condition_model, PlanValidator falls back to building one from
            # the registry's own (empty, for real operations) pre/post
            # conditions, which silently turns its structural precondition
            # check into a no-op instead of an independent check against the
            # real data/conditions/ dependency graph.
            condition_model=self.condition_model,
            settings=self.settings,
        )
        self._repair_agent = repair_agent
        self._precondition_agent = precondition_agent
        self._executor = executor

    # -- lazily built (may need an LLM / touch external state) --------------

    @property
    def intent_agent(self) -> IntentAgent:
        if self._intent_agent is None:
            self._intent_agent = IntentAgent.from_settings(self.settings)
        return self._intent_agent

    @property
    def repair_agent(self) -> RepairAgent:
        if self._repair_agent is None:
            self._repair_agent = RepairAgent.from_settings(self.settings)
        return self._repair_agent

    @property
    def precondition_agent(self) -> PreconditionAgent:
        if self._precondition_agent is None:
            self._precondition_agent = PreconditionAgent.from_settings(
                condition_model=self.condition_model, settings=self.settings
            )
        return self._precondition_agent

    @property
    def rag(self) -> RagRetriever:
        if self._rag is None:
            self._rag = RagRetriever.from_settings(self.settings)
        return self._rag

    @property
    def executor(self) -> APIExecutor:
        if self._executor is None:
            self._executor = build_executor(self.settings, registry=self.registry)
        return self._executor

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _fail(stage_label: str, exc: Exception) -> _Update:
        return {
            "stage": PipelineStage.FAILED,
            "errors": [f"{stage_label}: {type(exc).__name__}: {exc}"],
        }

    @staticmethod
    def _effective_arguments(state: AgentState) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        extraction = state.parameters
        if extraction is not None:
            for item in getattr(extraction, "operations", []):
                if item.arguments:
                    merged[item.operation_id] = dict(item.arguments)
        for operation_id, values in state.supplied_arguments.items():
            merged.setdefault(operation_id, {}).update(values)
        return merged

    @staticmethod
    def _family(capability: str | None) -> str:
        return (capability or "").split(".", 1)[0]

    def _fallback_missing(self, state: AgentState) -> list[str]:
        """Missing params when there is no plan: scope to the relevant operation(s)."""

        extraction = state.parameters
        if extraction is None:
            return []
        targets: set[str] = set()
        if state.selection is not None and state.selection.operations:
            targets = {state.selection.operations[0].operation_id}

        missing: list[str] = []
        for op_params in extraction.operations:
            if not targets or op_params.operation_id in targets:
                missing.extend(op_params.missing_required)
        return missing

    # -- nodes -----------------------------------------------------------------

    def intent_node(self, state: AgentState) -> _Update:
        if state.stage is PipelineStage.FAILED:
            return {}
        try:
            intent = self.intent_agent.understand(state.request_text)
        except Exception as exc:  # noqa: BLE001 - surface as a pipeline failure
            return self._fail("intent", exc)
        return {
            "intent": intent,
            "stage": PipelineStage.INTENT,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"intent: goal={intent.goal!r} action={intent.action!r} "
                        f"entities={list(intent.entity_values())}"
                    ),
                }
            ],
        }

    def rag_node(self, state: AgentState) -> _Update:
        if state.stage is PipelineStage.FAILED or state.intent is None:
            return {}
        try:
            context = self.rag.retrieve(state.intent)
        except Exception as exc:  # noqa: BLE001
            return self._fail("rag", exc)
        return {
            "rag": context,
            "retrieval": context.documentation,
            "stage": PipelineStage.RETRIEVAL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"retrieved {len(context.documentation.chunks)} chunk(s); "
                        f"candidate apis={context.apis}"
                    ),
                }
            ],
        }

    def selection_node(self, state: AgentState) -> _Update:
        if state.stage is PipelineStage.FAILED or state.intent is None:
            return {}
        try:
            selection = self.selection_agent.select(
                query=state.request_text,
                intent=state.intent,
            )
        except Exception as exc:  # noqa: BLE001
            return self._fail("selection", exc)
        operations = [candidate.operation for candidate in selection.operations]
        return {
            "selection": selection,
            "selected_operations": operations,
            "stage": PipelineStage.SELECTION,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"selected apis={[api.name for api in selection.apis]} "
                        f"operations={[op.operation_id for op in selection.operations]}"
                    ),
                }
            ],
        }

    def precondition_node(self, state: AgentState) -> _Update:
        """Declare - never execute - any check-then-branch or gate that applies.

        ``PreconditionAgent`` makes exactly one grounded judgement call and
        never calls anything itself. This node used to also perform the
        actual check right here, in a side plan/validate/execute cycle of
        its own - which worked, but hid a real API call from the plan and
        execution trace PGPS reports, since that call happened before
        planning even started. It does not do that anymore: this node only
        widens the candidate set (so parameter extraction grounds arguments
        for the resolver and every branch operation too) and hands the
        brief to PGPS as plain data. PGPS (`_plan_precondition_branch`) is
        what puts the resolver in as a real, first step of the one plan it
        returns, and the executor is what actually runs it and picks the
        real branch from its real result - so the check is now a visible
        part of the same plan and the same execution trace as everything
        else, not a hidden pre-flight step.

        Any failure (agent unavailable, brief doesn't ground) degrades to
        doing nothing: the pipeline continues exactly as it would without
        this node.
        """

        if state.stage is PipelineStage.FAILED or not state.selected_operations:
            return {}

        candidates_by_id: dict[str, Any] = {
            op.operation_id: op for op in state.selected_operations
        }
        selection = state.selection
        for operation_id in getattr(selection, "rejected", []) or []:
            if operation_id in candidates_by_id:
                continue
            sibling = self.registry.get_operation(operation_id)
            if sibling is not None:
                candidates_by_id[operation_id] = sibling

        try:
            brief = self.precondition_agent.assess(
                intent=state.intent, candidates=list(candidates_by_id.values())
            )
        except Exception:  # noqa: BLE001 - reasoning must never block the pipeline
            return {}

        if not brief.applies or brief.resolver_operation is None:
            return {}

        resolver_op = candidates_by_id.get(brief.resolver_operation)
        if resolver_op is None:
            return {}

        if brief.mode == "alternative":
            branch_ids = [brief.if_resolved_operation, brief.if_unresolved_operation]
        elif brief.mode == "gate":
            branch_ids = [brief.gated_operation]
        else:
            return {}
        branch_ops = [candidates_by_id.get(bid) for bid in branch_ids if bid]
        if len(branch_ops) != len(branch_ids) or any(op is None for op in branch_ops):
            return {}

        # Widen the working candidate set - resolver, then each branch
        # operation, then whatever was already selected - deduped by
        # operation id so parameter extraction (the next node) grounds
        # arguments for all of them, whichever one PGPS/the executor ends
        # up actually running.
        ordered: dict[str, Any] = {resolver_op.operation_id: resolver_op}
        for op in branch_ops:
            ordered.setdefault(op.operation_id, op)
        for op in state.selected_operations:
            ordered.setdefault(op.operation_id, op)

        precondition_brief = {
            "mode": brief.mode,
            "resolver_operation": resolver_op.operation_id,
            "if_resolved_operation": brief.if_resolved_operation,
            "if_unresolved_operation": brief.if_unresolved_operation,
            "gated_operation": brief.gated_operation,
            "reason": brief.reason,
            # Every sibling this node looked at while grounding the brief -
            # not just the resolver/branch operations that made it into
            # `selected_operations` below. Kept narrow there so parameter
            # extraction and the final narrative aren't bloated with
            # irrelevant operations (verifyLocation, checkSimSwap, ...) -
            # but the planner needs the full pool too: if the LLM's chosen
            # resolver turns out circular (e.g. getSession, which itself
            # needs the very session id it's meant to help discover), it
            # can look here for a real observation operation to swap in
            # instead (e.g. retrieveSessionsByDevice), rather than
            # abandoning the whole conditional plan.
            "candidate_pool": sorted(candidates_by_id.keys()),
        }

        return {
            "selected_operations": list(ordered.values()),
            "metadata": {**state.metadata, "precondition_brief": precondition_brief},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"precondition declared ({brief.mode}): check "
                        f"{resolver_op.operation_id} first - {brief.reason}"
                    ),
                }
            ],
        }

    def parameters_node(self, state: AgentState) -> _Update:
        if state.stage is PipelineStage.FAILED or state.intent is None:
            return {}
        if not state.selected_operations:
            return {
                "clarification": ["No registered CAMARA operation matched the request."],
                "stage": PipelineStage.PARAMETERS,
            }
        try:
            extraction = self.parameter_agent.extract(
                state.intent, state.selected_operations, known_facts=state.facts
            )
        except Exception as exc:  # noqa: BLE001
            return self._fail("parameters", exc)

        return {
            "parameters": extraction,
            "facts": {
                **state.facts,
                **extraction.known_facts,
                **facts_from_arguments(state.supplied_arguments),
            },
            "stage": PipelineStage.PARAMETERS,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"parameters extracted; missing overall="
                        f"{extraction.missing_information}"
                    ),
                }
            ],
        }

    def build_plan_request_node(self, state: AgentState) -> _Update:
        if state.stage is PipelineStage.FAILED or state.intent is None:
            return {}

        # When `precondition_node` declared a check-then-branch or a gate,
        # hand its brief straight to PGPS as plain data - PGPS is what turns
        # it into a real plan (resolver as a real first step, the branch
        # operation's own fully chained sub-plan after it), not this node.
        # Only derive `goal_facts` the ordinary way when there is no brief:
        # that derivation reads `selected_operations[0]` as *the* goal, which
        # is meaningless once precondition_node has widened that list to
        # include the resolver and every branch candidate too.
        precondition_brief = (state.metadata or {}).get("precondition_brief")

        goal_facts: list[str] = []
        if not precondition_brief and state.selected_operations:
            top = state.selected_operations[0]
            goal_facts = sorted(
                {fact for cond in top.postconditions for fact in cond.produces}
            )

        metadata: dict[str, Any] = {}
        if precondition_brief:
            metadata["precondition_brief"] = precondition_brief
        elif goal_facts:
            metadata["goal_facts"] = goal_facts

        request = PlanRequest(
            intent=state.intent,
            candidate_operations=state.selected_operations,
            retrieval=state.retrieval,
            known_facts=dict(state.facts),
            metadata=metadata,
        )
        return {"plan_request": request, "stage": PipelineStage.PLANNING}

    def pgps_node(self, state: AgentState) -> _Update:
        if state.stage is PipelineStage.FAILED or state.plan_request is None:
            return {}
        try:
            result = self.planner.plan(state.plan_request)
        except Exception as exc:  # noqa: BLE001
            return self._fail("pgps", exc)

        update: _Update = {
            "plan": result,
            "stage": PipelineStage.PLANNING,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"plan status={result.status.value} "
                        f"steps={[step.operation_id for step in result.steps]}"
                    ),
                }
            ],
        }
        if result.steps:
            explanation = explain_plan(
                result,
                registry=self.registry,
                intent=state.intent,
                initial_facts=set(state.facts),
            )
            update["metadata"] = {
                **state.metadata,
                "provenance": explanation.model_dump(mode="json"),
            }
        return update

    def clarification_node(self, state: AgentState) -> _Update:
        # what values are still needed
        missing = plan_missing_parameters(state)
        if not missing and state.parameters is not None and (
            state.plan is None or not state.plan.steps
        ):
            missing = self._fallback_missing(state)
        if state.plan is not None:
            missing += list(state.plan.unmet_preconditions)
        missing = _dedupe(missing)

        operations = (
            [step.operation_id for step in state.plan.steps]
            if state.plan is not None and state.plan.steps
            else [op.operation_id for op in state.selected_operations]
        )
        request = self.repair_agent.clarify(
            missing,
            intent=state.intent,
            plan=state.plan,
            operations=operations,
            request_text=state.request_text,
        )

        extra: list[str] = []
        if not state.selected_operations:
            extra.append("No registered CAMARA operation matched the request.")
        if state.plan is not None:
            extra += state.plan.validation_errors
        extra += state.errors

        body = request.question
        if extra:
            body += "\n(" + "; ".join(_dedupe(extra)) + ")"

        return {
            "clarification": request.missing_names(),
            "stage": PipelineStage.CLARIFICATION,
            "metadata": {**state.metadata, "clarification": request.model_dump()},
            "messages": [{"role": "assistant", "content": body}],
        }

    def validation_node(self, state: AgentState) -> _Update:
        if state.stage is PipelineStage.FAILED or state.plan is None:
            return {}
        try:
            updated, report = self.validator.validate_and_update(
                state.plan,
                known_facts=set(state.facts),
                parameters=self._effective_arguments(state),
            )
        except Exception as exc:  # noqa: BLE001
            return self._fail("validation", exc)
        return {
            "plan": updated,
            "stage": PipelineStage.VALIDATION,
            "metadata": {
                **state.metadata,
                "validation_report": report.model_dump(mode="json"),
            },
            "messages": [{"role": "system", "content": report.summary()}],
        }

    def execution_node(self, state: AgentState) -> _Update:
        if state.stage is PipelineStage.FAILED or state.plan is None:
            return {}
        try:
            result = self.executor.execute(
                state.plan,
                arguments=self._effective_arguments(state),
                initial_facts=state.facts,
                query=state.request_text,
            )
        except Exception as exc:  # noqa: BLE001
            return self._fail("execution", exc)
        update: _Update = {
            "execution": result,
            "facts": {**state.facts, **result.facts},
            "stage": PipelineStage.DONE if result.succeeded else PipelineStage.FAILED,
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        "execution succeeded"
                        if result.succeeded
                        else f"execution failed: {result.error or 'a step failed'}"
                    ),
                }
            ],
        }
        if not result.succeeded:
            update["errors"] = [f"execution failed: {result.error or 'a step failed'}"]
        return update


# --------------------------------------------------------------------------- #
# routing (deterministic, read-only)
# --------------------------------------------------------------------------- #

def route_after_pgps(state: AgentState) -> str:
    """After PGPS: branch to clarification on missing info, else validation."""

    if state.stage is PipelineStage.FAILED or state.clarification:
        return "clarify"
    plan = state.plan
    if (
        plan is None
        or plan.status not in (PlanStatus.VALID, PlanStatus.DRAFT)
        or plan.unmet_preconditions
        or not plan.steps
        or plan_missing_parameters(state)
    ):
        return "clarify"
    return "validate"


def route_after_validation(state: AgentState) -> str:
    if (
        state.stage is not PipelineStage.FAILED
        and state.plan is not None
        and state.plan.status is PlanStatus.VALID
    ):
        return "execute"
    return "clarify"


__all__ = [
    "WorkflowNodes",
    "facts_from_arguments",
    "plan_missing_parameters",
    "route_after_pgps",
    "route_after_validation",
]