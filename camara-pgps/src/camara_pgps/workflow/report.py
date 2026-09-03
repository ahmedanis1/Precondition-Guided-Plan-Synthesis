"""Human- and JSON-friendly views of a finished :class:`AgentState`.

Shared by the HTTP API (:mod:`camara_pgps.api.main`) and the CLI
(:mod:`camara_pgps.cli`) so both present a run the same way.
"""

from __future__ import annotations

from typing import Any

from camara_pgps.models.state import AgentState


def _dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def explain(state: AgentState) -> str:
    """A one-paragraph plain-language summary of the run."""

    parts: list[str] = []
    if state.intent is not None:
        caps = state.intent.required_capabilities
        parts.append(
            f"Understood the request as {state.intent.goal!r}"
            + (f" (capabilities: {caps})" if caps else "")
            + "."
        )
    if state.selection is not None:
        parts.append(f"Candidate API(s): {state.selection.api_names()}.")
    if state.plan is not None and state.plan.steps:
        sequence = " -> ".join(step.operation_id for step in state.plan.steps)
        parts.append(f"PGPS produced a {state.plan.status.value} plan: {sequence}.")
    elif state.plan is not None:
        parts.append(
            f"PGPS could not produce an executable plan "
            f"(status: {state.plan.status.value})."
        )
    if state.clarification:
        parts.append(f"Clarification needed: {state.clarification}.")
    if state.execution is not None:
        outcome = "succeeded" if state.execution.succeeded else "failed"
        parts.append(
            f"Execution {outcome} across {len(state.execution.steps)} step(s)."
        )
    parts.append(f"Final stage: {state.stage.value}.")
    return " ".join(parts)


def state_report(state: AgentState) -> dict[str, Any]:
    """The full run as a JSON-serialisable dict (every pipeline stage)."""

    return {
        "request": state.request_text,
        "stage": state.stage.value,
        "intent": _dump(state.intent),
        "retrieved": _dump(state.rag) or _dump(state.retrieval),
        "selected_apis": _dump(state.selection),
        "parameters": _dump(state.parameters),
        "plan": _dump(state.plan),
        "validation": state.metadata.get("validation_report"),
        "execution": _dump(state.execution),
        "clarification": state.clarification,
        "provenance": state.metadata.get("provenance"),
        "explanation": explain(state),
        "messages": state.messages,
        "errors": state.errors,
    }


__all__ = ["explain", "state_report"]
