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


def _dedupe_case_insensitive(names: Any) -> list[str]:
    """Collapse names that differ only by case, keeping the first spelling seen.

    ``api_name`` can reach this report from two different provenances - the
    registry's own parsed name (e.g. "quality-on-demand") for an operation
    ``precondition_node`` widened in directly, versus RAG/API-Selection's
    copy of the same name (e.g. "Quality-On-Demand") for one that arrived
    through the ordinary selection path - and a straight ``in`` check on the
    exact string does not catch that they are the same API. This does not
    fix the underlying provenance mismatch (that lives further upstream, in
    how RAG-chunk metadata and the registry each capture an API's name), but
    it keeps the user-facing narrative from listing one API twice.
    """

    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(name)
    return result


def _selected_api_names(selection: Any) -> list[str]:
    """Return selected API names from the current APISelection model."""

    if selection is None:
        return []

    return _dedupe_case_insensitive(
        getattr(api, "name", None) for api in getattr(selection, "apis", [])
    )


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
        # `state.selection` is a frozen snapshot of what API Selection chose
        # *before* the precondition check ran. `state.selected_operations`
        # reflects `precondition_node` widening the candidate set (resolver +
        # every branch operation) but is STILL only a plan-time guess - the
        # one thing that cannot lie about what really happened is
        # `state.execution`: the executor records every step's REAL
        # operation_id, already resolved from the branch's actual outcome
        # (see APIExecutor.execute / PlanStep.precondition_branch). Prefer
        # that truth whenever execution has actually run; fall back to the
        # plan, then the widened candidate set, then the original selection.
        outcomes = (
            (state.execution.metadata.get("precondition_outcomes") if state.execution else None)
            or []
        )

        if state.execution is not None and state.execution.steps:
            operation_ids = [
                step.operation_id
                for step in state.execution.steps
                if not getattr(step, "skipped", False)
            ]
        elif state.plan is not None and state.plan.steps:
            operation_ids = [step.operation_id for step in state.plan.steps]
        elif state.selected_operations:
            operation_ids = [
                op.operation_id
                for op in state.selected_operations
                if getattr(op, "operation_id", None)
            ]
        else:
            operation_ids = _selected_operation_ids(state.selection)

        if operation_ids:
            op_lookup = {op.operation_id: op for op in state.selected_operations}
            api_names = _dedupe_case_insensitive(
                getattr(op_lookup.get(op_id), "api_name", None) for op_id in operation_ids
            )
            if not api_names:
                api_names = _selected_api_names(state.selection)
        else:
            api_names = _selected_api_names(state.selection)

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

        consulted: set[str] = set()
        for outcome in outcomes:
            resolver = outcome.get("resolver_operation")
            consulted.add(resolver)
            resolved = outcome.get("resolved")
            reason = (outcome.get("reason") or "").strip()
            sentence = (
                f"Checked {resolver} first "
                f"({'confirmed' if resolved else 'not confirmed'})"
            )
            if reason:
                sentence += f": {reason}"
            parts.append(sentence + ".")

            skipped_op = outcome.get("skipped_operation")
            if skipped_op:
                parts.append(
                    f"{skipped_op} was not run because the condition was not confirmed."
                )

        # The alternate branch (Section 4.7/4.8: both branches of a
        # check-then-branch are fully planned and validated *before*
        # execution, not invented afterward) was never actually run, but it
        # was never "rejected" either - conflating the two here would make
        # a validated, ready-to-go fallback (e.g. createSession, held in
        # reserve behind a session-exists check) look identical to an
        # operation that was never relevant (checkSimSwap, verifyLocation),
        # undermining the entire point of pre-validating it. Recover
        # whichever branch didn't run: `branch_meta["alt_steps"]` if the
        # default branch is the one that actually ran (`resolved` True),
        # or the plan's own original default-branch steps (still sitting in
        # `state.plan.steps` even though the executor spliced them out of
        # `state.execution.steps`) if the alternate branch ran instead.
        not_taken_ids: set[str] = set()
        branch_meta = (state.plan.metadata or {}).get("precondition_branch") if state.plan else None
        if branch_meta and outcomes:
            resolved = outcomes[0].get("resolved")
            if resolved is True:
                not_taken_ids = {
                    step.get("operation_id")
                    for step in branch_meta.get("alt_steps") or []
                    if step.get("operation_id")
                }
            elif resolved is False and state.plan is not None:
                not_taken_ids = {
                    step.operation_id for step in state.plan.steps[1:]
                }

        if not_taken_ids:
            parts.append(
                f"Also pre-validated as the alternate branch, not needed here: "
                f"{', '.join(sorted(not_taken_ids))}."
            )

        # A resolver operation was consulted, not discarded, and the
        # alternate branch was pre-validated, not discarded either - drop
        # both from the plain "rejected" list so it doesn't read as if
        # either was never considered.
        rejected = [
            op_id
            for op_id in getattr(state.selection, "rejected", []) or []
            if op_id not in consulted
            and op_id not in operation_ids
            and op_id not in not_taken_ids
        ]
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

        # Precondition / alternative-path check (createSession vs.
        # extendQosSessionDuration, verifyLocation-as-gate, etc.): the brief
        # PreconditionAgent declared, plus - once execution has actually run
        # the resolver - the real, grounded outcome(s) from
        # ExecutionResult.metadata['precondition_outcomes'].
        "precondition": {
            "brief": state.metadata.get("precondition_brief"),
            "outcomes": (
                state.execution.metadata.get("precondition_outcomes")
                if state.execution
                else None
            ),
        },

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