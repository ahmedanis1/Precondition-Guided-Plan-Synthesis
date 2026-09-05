"""Loads PGPS pre/postcondition data from the hand-authored condition files.

This is the decoupled counterpart to :meth:`ConditionModel.from_registry`.
RAG / API Selection never touch condition data - they only pick *operations*
(method, path, parameters) and hand them to the planner as
``PlanRequest.candidate_operations``. The planner in turn does not need RAG
or the (stale, unrelated) ``CamaraRegistry`` to plan - it loads its own
condition data from ``data/conditions/*.conditions.json`` and joins the two
worlds purely by matching ``operation_id`` strings.

File format - one JSON file per CAMARA API, keyed by ``operation_id`` exactly
as it appears in that API's ``*.dataset.jsonl`` (copied verbatim from the
OpenAPI ``operationId``, never retyped)::

    {
      "createSession": {
        "requires": ["auth.token", "qos.profile.name", ...],
        "produces": ["qos.session.id", "qos.session"],
        "produces_from": {"qos.session.id": "sessionId"}
      },
      "getSession": {
        "requires": ["auth.token", "qos.session.id"],
        "produces": []
      }
    }

``produces_from`` is optional and declares, per produced fact, which field of
the operation's JSON response body carries its value (``{fact_key:
response_field}``). This is what the execution layer reads to extract real
values from *any* operation's response generically - see
:meth:`camara_pgps.execution.executor.APIExecutor._extract_facts` - instead of
needing a hand-maintained Python table per new API. A fact with no entry here
(e.g. ``qos.session`` above, a "this now exists" marker rather than a scalar)
is simply recorded as ``True`` once produced.

Each entry is wrapped into a single :class:`Precondition` / :class:`Postcondition`
pair (only added when the corresponding list is non-empty) to build an
:class:`OperationConditions`, matching the shape :class:`ConditionModel` expects.
"""

from __future__ import annotations

import json
from pathlib import Path

from camara_pgps.config import Settings, get_settings
from camara_pgps.pgps.conditions import (
    ConditionModel,
    OperationConditions,
    Postcondition,
    Precondition,
)


class DuplicateConditionError(ValueError):
    """Raised when the same ``operation_id`` appears in more than one condition file."""


def _default_directory(settings: Settings | None) -> Path:
    settings = settings or get_settings()
    return Path(settings.conditions.directory)


def load_operation_conditions(directory: Path | str) -> list[OperationConditions]:
    """Read every ``*.conditions.json`` file in ``directory`` into ``OperationConditions``.

    Raises :class:`DuplicateConditionError` if the same ``operation_id`` is
    defined in more than one file - mirroring the safeguard ``CamaraRegistry``
    applies within a single registry document, extended across files since
    each CAMARA API now has its own condition file.
    """

    directory = Path(directory)
    seen: dict[str, Path] = {}
    operations: list[OperationConditions] = []

    for path in sorted(directory.glob("*.conditions.json")):
        api_name = path.name.removesuffix(".conditions.json")
        raw = json.loads(path.read_text(encoding="utf-8"))

        for operation_id, spec in raw.items():
            if operation_id in seen:
                raise DuplicateConditionError(
                    f"operation_id {operation_id!r} defined in both "
                    f"{seen[operation_id]} and {path}"
                )
            seen[operation_id] = path

            requires = list(spec.get("requires", []))
            produces = list(spec.get("produces", []))
            produces_from = dict(spec.get("produces_from", {}))

            preconditions = (
                [
                    Precondition(
                        description=f"Facts required before calling {operation_id}",
                        requires=requires,
                    )
                ]
                if requires
                else []
            )
            postconditions = (
                [
                    Postcondition(
                        description=f"Facts established after calling {operation_id}",
                        produces=produces,
                        produces_from=produces_from,
                    )
                ]
                if produces
                else []
            )

            operations.append(
                OperationConditions(
                    operation_id=operation_id,
                    api_name=api_name,
                    preconditions=preconditions,
                    postconditions=postconditions,
                )
            )

    return operations


def build_condition_model(
    directory: Path | str | None = None,
    *,
    settings: Settings | None = None,
) -> ConditionModel:
    """Build the :class:`ConditionModel` real planning code should use.

    No dependency on ``CamaraRegistry`` or RAG - only on the hand-authored
    condition files, keyed by the same ``operation_id`` RAG hands back in
    ``PlanRequest.candidate_operations``.
    """

    resolved_dir = (
        Path(directory) if directory is not None else _default_directory(settings)
    )
    return ConditionModel(load_operation_conditions(resolved_dir))


__all__ = [
    "DuplicateConditionError",
    "build_condition_model",
    "load_operation_conditions",
]
