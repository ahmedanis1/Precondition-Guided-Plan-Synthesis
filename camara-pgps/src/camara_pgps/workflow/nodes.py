"""LangGraph node functions for the end-to-end CAMARA-PGPS pipeline.

Each node takes the :class:`AgentState` and returns a partial update. Every node
short-circuits when a previous node has set ``stage = FAILED`` so one failure
carries cleanly to the clarification step.
"""

from __future__ import annotations

from typing import Any

from camara_pgps.agents.api_selection_agent import APISelectionAgent
from camara_pgps.agents.intent_agent import IntentAgent
from camara_pgps.agents.parameter_agent import ParameterExtractionAgent
from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import Settings, get_settings
from camara_pgps.execution import APIExecutor, build_executor
from camara_pgps.models.plan import PlanRequest, PlanStatus
from camara_pgps.models.state import AgentState, PipelineStage
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


def plan_missing_parameters(state: AgentState) -> list[str]:
    """Required parameters of the *plan's* operations that are still unbound."""

    if state.plan is None or state.parameters is None:
        return []
    supplied = state.supplied_arguments
    missing: list[str] = []
    for step in state.plan.steps:
        op_params = state.parameters.for_operation(step.operation_id)
        if op_params is None:
            continue
        covered = set(supplied.get(step.operation_id, {}))
        for name in op_params.missing_required:
            if name not in covered and name not in missing:
                missing.append(name)
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
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or CamaraRegistry.from_settings(self.settings)
        self.intent_agent = intent_agent or IntentAgent.from_settings(self.settings)
        self._rag = rag_retriever
        self.selection_agent = selection_agent or APISelectionAgent(
            registry=self.registry, settings=self.settings, rag=self._rag
        )
        self.parameter_agent = parameter_agent or ParameterExtractionAgent(
            registry=self.registry, settings=self.settings
        )
        self.planner = planner or PGPSPlanner(
            registry=self.registry, settings=self.settings
        )
        self.validator = validator or PlanValidator(
            registry=self.registry, settings=self.settings
        )
        self._executor = executor

    # -- lazily built (touch external state) -------------------------------

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

    # -- nodes -----------------------------------------------------------------

    def intent_node(self, state: AgentState) -> _Update:
        if state.stage is PipelineStage.FAILED:
            return {}
        try:
            intent = self.intent_agent.understand(state.request_text)
        except Exception as exc:  # noqa: BLE001 - surface as a pipeline failure
            return self._fail("intent", exc)
        facts = {**state.facts, **dict.fromkeys(intent.known_facts, True)}
        return {
            "intent": intent,
            "facts": facts,
            "stage": PipelineStage.INTENT,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"intent: goal={intent.goal!r} "
                        f"capabilities={intent.required_capabilities}"
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
            selection = self.selection_agent.select(state.intent, rag_context=state.rag)
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
                        f"selected apis={selection.api_names()} "
                        f"operations={selection.operation_ids()}"
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
            "facts": {**state.facts, **extraction.known_facts},
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

        # scope the candidate operations to the intent's capability families so the
        # solver stays focused
        operations = state.selected_operations
        families = set(state.intent.required_capabilities)
        if families:
            scoped = [
                op
                for op in operations
                if self._family(self.registry.capability_of(op.operation_id)) in families
            ]
            operations = scoped or operations

        # the goal is what the top-ranked selected operation produces
        goal_facts: list[str] = []
        selection = state.selection
        if selection is not None and selection.operations:
            top = selection.operations[0].operation
            goal_facts = sorted(
                {fact for cond in top.postconditions for fact in cond.produces}
            )

        request = PlanRequest(
            intent=state.intent,
            candidate_operations=operations,
            retrieval=state.retrieval,
            known_facts=dict(state.facts),
            metadata={"goal_facts": goal_facts} if goal_facts else {},
        )
        return {"plan_request": request, "stage": PipelineStage.PLANNING}

    def pgps_node(self, state: AgentState) -> _Update:
        if state.stage is PipelineStage.FAILED or state.plan_request is None:
            return {}
        try:
            result = self.planner.plan(state.plan_request)
        except Exception as exc:  # noqa: BLE001
            return self._fail("pgps", exc)
        return {
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

    def clarification_node(self, state: AgentState) -> _Update:
        reasons = list(state.clarification)
        if not state.selected_operations:
            reasons.append("No registered CAMARA operation matched the request.")

        plan_gaps = plan_missing_parameters(state)
        if not plan_gaps and state.parameters is not None and (
            state.plan is None or not state.plan.steps
        ):
            # no executable plan yet - fall back to the overall missing parameters
            plan_gaps = list(state.parameters.missing_information)
        reasons += [f"provide a value for {name!r}" for name in plan_gaps]

        if state.plan is not None:
            reasons += [
                f"unmet precondition: {fact}" for fact in state.plan.unmet_preconditions
            ]
            reasons += state.plan.validation_errors
        reasons += state.errors
        reasons = _dedupe(reasons)
        body = (
            "I need more information before I can proceed:\n- " + "\n- ".join(reasons)
            if reasons
            else "I could not build an executable plan for this request."
        )
        return {
            "clarification": reasons,
            "stage": PipelineStage.CLARIFICATION,
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
    "plan_missing_parameters",
    "route_after_pgps",
    "route_after_validation",
]
