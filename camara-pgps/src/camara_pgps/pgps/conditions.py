"""PGPS condition model: facts, preconditions, postconditions.

Every CAMARA operation is described by the *facts* it needs before it can run
(preconditions) and the *facts* it makes true afterwards (postconditions). A
fact is just a stable key in the shared convention, e.g. ``device.identifier``,
``qos.session.id``, ``device.location.verification``.

    verifyLocation      requires {auth.token, device.identifier, location.area}
                        produces {device.location.verification}

    createSession       requires {auth.token, device.identifier, qos.profile}
                        produces {qos.session.id, qos.session}

    getSession          requires {auth.token, qos.session.id}
                        produces {qos.session}

The dependency graph (``dependency_graph.py``) is built on top of this model:
an edge A -> B exists whenever B requires a fact that A produces. This is the
foundation of Precondition-Guided Plan Synthesis.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.models import CamaraOperation, Condition, ConditionKind


class Fact(BaseModel):
    """A named piece of state that can be known / true, optionally bound to a value."""

    key: str
    description: str | None = None
    value: Any | None = None

    @property
    def category(self) -> str:
        return self.key.split(".", 1)[0]

    @property
    def is_bound(self) -> bool:
        return self.value is not None


class FactSet:
    """A mutable set of fact keys - the "known state" during planning."""

    __slots__ = ("_keys",)

    def __init__(self, keys: Iterable[str] = ()) -> None:
        self._keys: set[str] = {str(key) for key in keys}

    # -- container protocol -------------------------------------------------

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    def __iter__(self) -> Iterable[str]:
        return iter(sorted(self._keys))

    def __len__(self) -> int:
        return len(self._keys)

    def __bool__(self) -> bool:
        return bool(self._keys)

    def __repr__(self) -> str:
        return f"FactSet({sorted(self._keys)!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FactSet):
            return self._keys == other._keys
        if isinstance(other, set | frozenset):
            return self._keys == set(other)
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def __or__(self, other: Iterable[str]) -> FactSet:
        return FactSet(self._keys | self._coerce(other))

    def __ior__(self, other: Iterable[str]) -> FactSet:
        self._keys |= self._coerce(other)
        return self

    # -- operations ------------------------------------------------------------

    @staticmethod
    def _coerce(other: Iterable[str]) -> set[str]:
        if isinstance(other, FactSet):
            return set(other._keys)
        return {str(key) for key in other}

    def add(self, key: str) -> None:
        self._keys.add(str(key))

    def update(self, keys: Iterable[str]) -> None:
        self._keys |= self._coerce(keys)

    def copy(self) -> FactSet:
        return FactSet(self._keys)

    def as_set(self) -> set[str]:
        return set(self._keys)

    def satisfies(self, requirements: Iterable[str]) -> bool:
        return set(requirements) <= self._keys

    def missing(self, requirements: Iterable[str]) -> set[str]:
        return set(requirements) - self._keys

    def apply(self, produces: Iterable[str]) -> FactSet:
        return FactSet(self._keys | {str(fact) for fact in produces})


class Precondition(Condition):
    """A :class:`Condition` that must hold *before* an operation runs."""

    kind: ConditionKind = ConditionKind.PRECONDITION

    @classmethod
    def from_condition(cls, condition: Condition) -> Precondition:
        data = condition.model_dump()
        data["kind"] = ConditionKind.PRECONDITION
        return cls.model_validate(data)

    @property
    def required_facts(self) -> set[str]:
        return set(self.requires)


class Postcondition(Condition):
    """A :class:`Condition` that holds *after* an operation runs."""

    kind: ConditionKind = ConditionKind.POSTCONDITION

    @classmethod
    def from_condition(cls, condition: Condition) -> Postcondition:
        data = condition.model_dump()
        data["kind"] = ConditionKind.POSTCONDITION
        return cls.model_validate(data)

    @property
    def produced_facts(self) -> set[str]:
        return set(self.produces)


class OperationConditions(BaseModel):
    """The pre/postconditions of a single operation, normalised for PGPS."""

    operation_id: str
    api_name: str | None = None
    preconditions: list[Precondition] = Field(default_factory=list)
    postconditions: list[Postcondition] = Field(default_factory=list)

    @classmethod
    def from_operation(cls, operation: CamaraOperation) -> OperationConditions:
        return cls(
            operation_id=operation.operation_id,
            api_name=operation.api_name,
            preconditions=[
                Precondition.from_condition(c) for c in operation.preconditions
            ],
            postconditions=[
                Postcondition.from_condition(c) for c in operation.postconditions
            ],
        )

    @property
    def requires(self) -> set[str]:
        return {fact for c in self.preconditions for fact in c.requires}

    @property
    def produces(self) -> set[str]:
        return {fact for c in self.postconditions for fact in c.produces}

    @property
    def produces_from(self) -> dict[str, str]:
        """``{fact_key: response_body_field}`` merged across all postconditions.

        Purely declarative - sourced from ``data/conditions/*.conditions.json``
        (via ``Postcondition.produces_from``), never hardcoded per operation in
        Python. Empty when a fact has no known response field (e.g. it is a
        synthetic "we now know this happened" marker), in which case the
        executor falls back to recording ``True`` for that fact.
        """

        mapping: dict[str, str] = {}
        for condition in self.postconditions:
            mapping.update(condition.produces_from)
        return mapping

    def _known(self, state: FactSet | Iterable[str]) -> FactSet:
        return state if isinstance(state, FactSet) else FactSet(state)

    def is_applicable(self, state: FactSet | Iterable[str]) -> bool:
        return self._known(state).satisfies(self.requires)

    def unmet(self, state: FactSet | Iterable[str]) -> set[str]:
        return self._known(state).missing(self.requires)

    def apply(self, state: FactSet | Iterable[str]) -> FactSet:
        return self._known(state).apply(self.produces)


class ConditionModel:
    """A registry-wide index of operation conditions and fact flow."""

    def __init__(self, operations: Iterable[OperationConditions]) -> None:
        self._by_id: dict[str, OperationConditions] = {
            oc.operation_id: oc for oc in operations
        }
        self._producers: dict[str, list[str]] = {}
        self._consumers: dict[str, list[str]] = {}
        for operation_id, oc in self._by_id.items():
            for fact in sorted(oc.produces):
                self._producers.setdefault(fact, []).append(operation_id)
            for fact in sorted(oc.requires):
                self._consumers.setdefault(fact, []).append(operation_id)

    @classmethod
    def from_registry(cls, registry: CamaraRegistry) -> ConditionModel:
        return cls(
            OperationConditions.from_operation(op) for op in registry.list_operations()
        )

    # -- lookups ------------------------------------------------------------

    def operation_ids(self) -> list[str]:
        return list(self._by_id)

    def for_operation(self, operation_id: str) -> OperationConditions:
        try:
            return self._by_id[operation_id]
        except KeyError:
            raise KeyError(
                f"no condition model for operation {operation_id!r}"
            ) from None

    def get(self, operation_id: str) -> OperationConditions | None:
        return self._by_id.get(operation_id)

    def producers_of(self, fact: str) -> list[str]:
        return list(self._producers.get(fact, []))

    def consumers_of(self, fact: str) -> list[str]:
        return list(self._consumers.get(fact, []))

    # -- fact sets --------------------------------------------------------

    def produced_facts(self) -> set[str]:
        return set(self._producers)

    def required_facts(self) -> set[str]:
        return set(self._consumers)

    def all_facts(self) -> set[str]:
        return set(self._producers) | set(self._consumers)

    def external_facts(self) -> set[str]:
        """Facts required somewhere but produced by no operation.

        These must be supplied by the intent / parameter extraction
        (``auth.token``, ``device.identifier``, ``location.area``, ``qos.profile``).
        """

        return set(self._consumers) - set(self._producers)

    def facts(self) -> list[Fact]:
        return [Fact(key=key) for key in sorted(self.all_facts())]


__all__ = [
    "Fact",
    "FactSet",
    "Precondition",
    "Postcondition",
    "OperationConditions",
    "ConditionModel",
]
