"""Human- and JSON-friendly views of a finished :class:`AgentState`.

Shared by the HTTP API (:mod:`camara_pgps.api.main`) and the CLI
(:mod:`camara_pgps.cli`) so both present a run the same way.

This module is intentionally read-only. It consumes the current workflow
models directly and does not depend on legacy API-selection helper methods
such as ``api_names()`` or ``operation_ids()``.
"""

from __future__ import annotations

from typing import Any

from camara_pgps.models.state import AgentState


def _dump(obj: Any) -> Any:
    """Convert a Pydantic model into a JSON-compatible representation."""
    if obj is None:
        return None

    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")

    if isinstance(obj, list):
        return [_dump(item) for item in obj]

    if isinstance(obj, dict):
        return {key: _dump(value) for key, value in obj.items()}

    return obj


def _selected_api_names(selection: Any) -> list[str]:
    """Return selected API names from the current APISelection model."""

    if selection is None:
        return []

    return [
        api.name
        for api in getattr(selection, "apis", [])
        if getattr(api, "name", None)
    ]


def _selected_operation_ids(selection: Any) -> list[str]:
    """Return selected operation IDs from the current APISelection model."""

    if selection is None:
        return []

    return [
        operation.operation_id
        for operation in getattr(selection, "operations", [])
        if getattr(operation, "operation_id", None)
    ]


def explain(state: AgentState) -> str:
    """Create a plain-language summary of the completed workflow."""

    parts: list[str] = []

    # ------------------------------------------------------------------
    # Intent
    # ------------------------------------------------------------------
    if state.intent is not None:
        action = state.intent.action

        parts.append(
            f"Understood the request as {state.intent.goal!r}"
            + (f" (action: {action})" if action else "")
            + "."
        )

    # ------------------------------------------------------------------
    # RAG / API selection
    # ------------------------------------------------------------------
    if state.selection is not None:
        api_names = _selected_api_names(state.selection)
        operation_ids = _selected_operation_ids(state.selection)

        if api_names:
            parts.append(
                f"Selected API(s): {', '.join(api_names)}."
            )
        else:
            parts.append("No API was selected.")

        if operation_ids:
            parts.append(
                f"Selected operation(s): {', '.join(operation_ids)}."
            )

        rejected = getattr(state.selection, "rejected", [])
        if rejected:
            parts.append(
                f"Rejected candidate(s): {', '.join(rejected)}."
            )

        notes = getattr(state.selection, "notes", [])
        if notes:
            parts.append(
                f"Selection notes: {' '.join(str(note) for note in notes)}."
            )

    # ------------------------------------------------------------------
    # PGPS plan
    # ------------------------------------------------------------------
    if state.plan is not None and state.plan.steps:
        sequence = " -> ".join(
            step.operation_id
            for step in state.plan.steps
            if getattr(step, "operation_id", None)
        )

        if sequence:
            parts.append(
                f"PGPS produced a {state.plan.status.value} plan: "
                f"{sequence}."
            )
        else:
            parts.append(
                f"PGPS produced a {state.plan.status.value} plan."
            )

    elif state.plan is not None:
        parts.append(
            "PGPS could not produce an executable plan "
            f"(status: {state.plan.status.value})."
        )

    # ------------------------------------------------------------------
    # Clarification
    # ------------------------------------------------------------------
    if state.clarification:
        parts.append(
            f"Clarification needed: {state.clarification}."
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    if state.execution is not None:
        outcome = (
            "succeeded"
            if state.execution.succeeded
            else "failed"
        )

        parts.append(
            f"Execution {outcome} across "
            f"{len(state.execution.steps)} step(s)."
        )

    # ------------------------------------------------------------------
    # Final stage
    # ------------------------------------------------------------------
    parts.append(
        f"Final stage: {state.stage.value}."
    )

    return " ".join(parts)


def state_report(state: AgentState) -> dict[str, Any]:
    """Return the complete workflow state as a JSON-serialisable dict.

    The report exposes the output of every major pipeline stage:

        request
            ↓
        intent
            ↓
        retrieval
            ↓
        API selection
            ↓
        parameters
            ↓
        PGPS plan
            ↓
        validation
            ↓
        execution
    """

    selection = _dump(state.selection)

    return {
        # Original request
        "request": state.request_text,

        # Workflow state
        "stage": state.stage.value,

        # Intent understanding
        "intent": _dump(state.intent),

        # RAG retrieval
        "retrieved": (
            _dump(state.rag)
            or _dump(state.retrieval)
        ),

        # API selection
        "selected_apis": selection,

        # Extracted/resolved parameters
        "parameters": _dump(state.parameters),

        # PGPS plan
        "plan": _dump(state.plan),

        # Plan validation
        "validation": state.metadata.get("validation_report"),

        # Execution result
        "execution": _dump(state.execution),

        # User clarification
        "clarification": state.clarification,

        # Provenance
        "provenance": state.metadata.get("provenance"),

        # Human-readable explanation
        "explanation": explain(state),

        # Workflow messages
        "messages": state.messages,

        # Workflow errors
        "errors": state.errors,
    }


__all__ = [
    "explain",
    "state_report",
]