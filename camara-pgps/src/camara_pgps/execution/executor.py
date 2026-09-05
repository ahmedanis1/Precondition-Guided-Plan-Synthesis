"""The execution layer interface.

    validated PlanResult  ->  APIExecutor  ->  CAMARA API

``APIExecutor`` is the template: it resolves each :class:`PlanStep` into an
:class:`HTTPRequestSpec`, hands it to the backend-specific :meth:`_perform`,
records a :class:`StepExecution`, and threads the extracted postcondition facts
into the next step. Subclasses only implement :meth:`_perform`.

The executor never mutates the plan.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from camara_pgps.agents.item_selection_agent import ItemSelectionAgent
from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import Settings, get_settings
from camara_pgps.models import CamaraOperation, ParameterLocation
from camara_pgps.models.execution import ExecutionResult, HTTPRequestSpec, StepExecution
from camara_pgps.models.plan import BindingSource, PlanResult, PlanStatus, PlanStep
from camara_pgps.pgps.conditions import ConditionModel
from camara_pgps.pgps.conditions_loader import build_condition_model
from camara_pgps.pgps.parameter_facts import FACT_TO_PARAMETER as _FACT_ALIASES

_PATH_PLACEHOLDER = re.compile(r"\{([^}]+)\}")

# _FACT_ALIASES (fact key -> a request-friendly alias also stored in the fact
# store, so _build_request's raw path/query lookups - which use the
# parameter's literal name, not the dotted fact-key convention - can find the
# same value) is now imported above from pgps.parameter_facts: the single,
# shared table also used by parameter extraction (to bind a value to every
# fact key it satisfies) and the planner (to record cross-step bindings), so
# a fact recognised in one place is always recognised in all of them. Keyed
# by the shared fact-key *vocabulary*, not by operation, so this rarely needs
# a new entry when a new API is added - only when a genuinely new fact
# category is introduced that a path/query parameter also needs under a
# different name.

# Legacy fallback only: fact key -> response body field (same direction as
# ``produces_from``), for operations that predate data/conditions (e.g.
# hand-authored milestone-3 registry fixtures) and therefore have no
# ConditionModel entry - and hence no declarative ``produces_from`` - to read
# from. Every real CAMARA operation should declare its own ``produces_from``
# in its data/conditions/*.conditions.json file instead of growing this
# table; see _extract_facts below.
_RESPONSE_FACT_FIELDS: dict[str, str] = {
    "qos.session.id": "sessionId",
    "network-insights.subscription.id": "subscriptionId",
}
_OPERATION_RESPONSE_FACTS: dict[str, dict[str, str]] = {}


class ExecutionError(RuntimeError):
    """Raised for configuration problems that prevent execution starting."""


class APIExecutor(ABC):
    """Abstract executor. Runs a validated plan step by step."""

    mode: str = "abstract"

    def __init__(
        self,
        *,
        registry: CamaraRegistry | None = None,
        condition_model: ConditionModel | None = None,
        item_selector: ItemSelectionAgent | None = None,
        settings: Settings | None = None,
        base_url: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or CamaraRegistry.from_settings(self.settings)
        self.condition_model = condition_model or build_condition_model(settings=self.settings)
        self.item_selector = item_selector or ItemSelectionAgent.from_settings(self.settings)
        self.base_url = (base_url or self.settings.camara.base_url).rstrip("/")

    # -- public template --------------------------------------------------

    def execute(
        self,
        plan: PlanResult,
        *,
        arguments: Any | None = None,
        initial_facts: Mapping[str, Any] | None = None,
        stop_on_error: bool | None = None,
        query: str | None = None,
    ) -> ExecutionResult:
        stop = self.settings.execution.stop_on_error if stop_on_error is None else stop_on_error
        started = datetime.now(tz=UTC)
        metadata: dict[str, Any] = {"mode": self.mode, "base_url": self.base_url}

        if plan.status in (PlanStatus.INVALID, PlanStatus.INFEASIBLE):
            return ExecutionResult(
                succeeded=False,
                error=f"refusing to execute a plan with status {plan.status.value!r}",
                started_at=started,
                finished_at=datetime.now(tz=UTC),
                metadata=metadata,
            )
        if plan.status is not PlanStatus.VALID:
            metadata["warning"] = f"plan status is {plan.status.value!r}, not 'valid'"

        supplied = self._merge_arguments(arguments)
        facts: dict[str, Any] = dict(initial_facts or {})
        step_results: list[StepExecution] = []
        succeeded = True

        for step in plan.steps:
            operation = self.registry.get_operation(step.operation_id)
            if operation is None:
                step_results.append(
                    StepExecution(
                        step_index=step.index,
                        operation_id=step.operation_id,
                        succeeded=False,
                        error="operation is not in the CAMARA registry",
                    )
                )
                succeeded = False
                if stop:
                    break
                continue

            request = self._build_request(step, operation, supplied, facts)
            step_exec = self._run_step(request, step, operation)
            if step_exec.succeeded:
                extracted = self._extract_facts(
                    step.operation_id, operation, step_exec.response_body, query=query
                )
                step_exec.extracted_facts = extracted
                facts.update(extracted)
            else:
                succeeded = False
            step_results.append(step_exec)
            if not step_exec.succeeded and stop:
                break

        return ExecutionResult(
            succeeded=succeeded and len(step_results) == len(plan.steps),
            steps=step_results,
            facts=facts,
            started_at=started,
            finished_at=datetime.now(tz=UTC),
            metadata=metadata,
        )

    # -- subclass hook -----------------------------------------------------

    @abstractmethod
    def _perform(
        self, request: HTTPRequestSpec, step: PlanStep, operation: CamaraOperation
    ) -> StepExecution:
        """Issue the request (or simulate it) and return the recorded outcome."""

    # -- shared machinery ------------------------------------------------------

    def _run_step(
        self, request: HTTPRequestSpec, step: PlanStep, operation: CamaraOperation
    ) -> StepExecution:
        started = datetime.now(tz=UTC)
        if self.settings.execution.dry_run:
            return StepExecution(
                step_index=step.index,
                operation_id=step.operation_id,
                request=request,
                succeeded=True,
                status_code=None,
                response_body={"dry_run": True},
                started_at=started,
                finished_at=started,
                duration_ms=0.0,
            )
        try:
            result = self._perform(request, step, operation)
        except Exception as exc:  # noqa: BLE001 - a backend failure must not crash the run
            return StepExecution(
                step_index=step.index,
                operation_id=step.operation_id,
                request=request,
                succeeded=False,
                error=f"{type(exc).__name__}: {exc}",
                started_at=started,
                finished_at=datetime.now(tz=UTC),
            )
        result.request = result.request or request
        if result.started_at is None:
            result.started_at = started
        if result.finished_at is None:
            result.finished_at = datetime.now(tz=UTC)
        if result.duration_ms is None and result.started_at and result.finished_at:
            result.duration_ms = (
                result.finished_at - result.started_at
            ).total_seconds() * 1000
        return result

    def _resolve_arguments(
        self, step: PlanStep, supplied: Mapping[str, Mapping[str, Any]], facts: Mapping[str, Any]
    ) -> dict[str, Any]:
        resolved: dict[str, Any] = dict(step.arguments)
        resolved.update(supplied.get(step.operation_id, {}))
        for binding in step.bindings:
            if binding.parameter in resolved:
                continue
            if binding.value is not None:
                resolved[binding.parameter] = binding.value
            elif binding.source is BindingSource.PRIOR_STEP:
                value = facts.get(binding.parameter)
                if value is None and binding.from_pointer:
                    value = facts.get(binding.from_pointer.lstrip("/"))
                if value is not None:
                    resolved[binding.parameter] = value
        return resolved

    def _build_request(
        self,
        step: PlanStep,
        operation: CamaraOperation,
        supplied: Mapping[str, Mapping[str, Any]],
        facts: Mapping[str, Any],
    ) -> HTTPRequestSpec:
        arguments = self._resolve_arguments(step, supplied, facts)
        by_name = {param.name: param for param in operation.parameters}

        path = operation.path
        for match in _PATH_PLACEHOLDER.finditer(operation.path):
            name = match.group(1)
            value = arguments.get(name, facts.get(name, ""))
            path = path.replace(match.group(0), str(value))

        query: dict[str, Any] = {}
        body: dict[str, Any] = {}
        for name, value in arguments.items():
            location = by_name[name].location if name in by_name else ParameterLocation.BODY
            if location is ParameterLocation.QUERY:
                query[name] = value
            elif location is ParameterLocation.BODY:
                body[name] = value

        headers: dict[str, str] = {}
        if self.settings.camara.token:
            headers["Authorization"] = f"Bearer {self.settings.camara.token}"
        if body:
            headers["Content-Type"] = "application/json"

        return HTTPRequestSpec(
            method=operation.method,
            url=f"{self.base_url}{path}",
            headers=headers,
            query_params=query,
            json_body=body or None,
        )

    def _produced_facts(
        self, operation_id: str, operation: CamaraOperation
    ) -> set[str]:
        """Facts this operation produces once it succeeds.

        Prefers the real PGPS condition data (``data/conditions``) - the
        hand-reviewed source of truth PGPS itself plans against. Falls back
        to the registry operation's own ``.postconditions`` only when the
        condition model has nothing for this operation (e.g. an older,
        hand-authored registry fixture that embeds pre/postconditions
        directly, predating the registry/conditions split).
        """

        conditions = self.condition_model.get(operation_id)
        if conditions and conditions.produces:
            return set(conditions.produces)
        return {
            fact
            for condition in operation.postconditions
            for fact in condition.produces
        }

    def _extract_facts(
        self,
        operation_id: str,
        operation: CamaraOperation,
        response_body: Any,
        *,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Turn a step's response body into the facts it produces.

        Generic across every operation: the fact -> response-field mapping
        comes from that operation's own ``produces_from`` declaration in
        ``data/conditions/*.conditions.json`` (via the condition model), never
        from a hardcoded Python table. Only operations with no condition-model
        entry at all (pre-dating the registry/conditions split) fall back to
        the legacy ``_OPERATION_RESPONSE_FACTS`` / ``_RESPONSE_FACT_FIELDS``
        tables below, purely for backward compatibility with those fixtures.
        """

        produced = self._produced_facts(operation_id, operation)
        if not produced:
            return {}
        facts: dict[str, Any] = {}
        body = self._resolve_response_body(response_body, query=query)

        conditions = self.condition_model.get(operation_id)
        if conditions is not None:
            field_map = conditions.produces_from
        else:
            field_map = _OPERATION_RESPONSE_FACTS.get(operation_id) or _RESPONSE_FACT_FIELDS

        for fact_key, field in field_map.items():
            if fact_key in produced and field in body:
                facts[fact_key] = body[field]
        for fact_key in produced:
            if fact_key not in facts:
                facts[fact_key] = True
            alias = _FACT_ALIASES.get(fact_key)
            if alias:
                facts.setdefault(alias, facts[fact_key])
        return facts

    def _resolve_response_body(
        self, response_body: Any, *, query: str | None = None
    ) -> Mapping[str, Any]:
        """Normalise a step's response into the single dict fact-extraction reads.

        Most CAMARA operations return one resource (a dict) - that's passed
        through unchanged. A "discovery" operation such as
        ``retrieveSessionsByDevice`` returns a *list* of resources instead,
        and there is no way to know which one a chained fact (e.g.
        ``qos.session.id``) should come from without looking at what the
        user actually asked for - "extend my gaming session" needs the
        session whose profile matches "gaming", not just index 0. That
        disambiguation is delegated to :class:`ItemSelectionAgent`, which
        always returns a valid index (falling back to the first item when
        there's nothing to disambiguate or the LLM is unavailable) so this
        never blocks execution.
        """

        if isinstance(response_body, Mapping):
            return response_body
        if isinstance(response_body, list) and response_body:
            if all(isinstance(item, Mapping) for item in response_body):
                index = self.item_selector.select(query=query or "", items=response_body)
                return response_body[index]
        return {}

    @staticmethod
    def _merge_arguments(arguments: Any | None) -> dict[str, dict[str, Any]]:
        if arguments is None:
            return {}
        operations = getattr(arguments, "operations", None)
        if operations is not None:  # a ParameterExtraction
            return {
                item.operation_id: dict(item.arguments)
                for item in operations
                if getattr(item, "arguments", None)
            }
        if isinstance(arguments, Mapping):
            return {str(k): dict(v) for k, v in arguments.items()}
        raise TypeError("arguments must be a ParameterExtraction or a nested mapping")


__all__ = ["APIExecutor", "ExecutionError"]
