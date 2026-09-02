"""Capability <-> operation mapping.

A *capability* is a stable, dotted semantic slug (``"<domain>.<resource>.<action>"``,
e.g. ``qos.session.create``) describing what an operation lets you do, independent
of its HTTP shape. Later components (API Selection, milestone 8) let the LLM pick
a capability from this **closed vocabulary** and then resolve it to concrete,
registered operations - the LLM never names an API or operation directly.
"""

from __future__ import annotations

import re
from collections import defaultdict

from camara_pgps.camara.schemas import RegistryDocument

CAPABILITY_PATTERN = re.compile(
    r"^[a-z0-9]+(?:-[a-z0-9]+)*(?:\.[a-z0-9]+(?:-[a-z0-9]+)*){1,}$"
)


class CapabilityError(Exception):
    """Base class for capability-mapping errors."""


class InvalidCapabilityError(CapabilityError):
    """Raised when a capability string does not match the required format."""


class UnknownCapabilityError(CapabilityError):
    """Raised when a capability is not provided by any registered operation."""


class CapabilityMapper:
    """Bidirectional index between operation ids and capability slugs."""

    def __init__(self, operation_capabilities: dict[str, str]) -> None:
        self._forward: dict[str, str] = {}
        self._reverse: dict[str, list[str]] = defaultdict(list)
        for operation_id, capability in operation_capabilities.items():
            normalised = self.validate(capability)
            self._forward[operation_id] = normalised
            self._reverse[normalised].append(operation_id)

    @staticmethod
    def validate(capability: str) -> str:
        """Normalise and format-check a capability slug."""

        normalised = capability.strip().lower()
        if not CAPABILITY_PATTERN.match(normalised):
            raise InvalidCapabilityError(
                f"Invalid capability {capability!r}; expected a dotted slug like "
                "'qos.session.create'"
            )
        return normalised

    @classmethod
    def from_document(cls, document: RegistryDocument) -> CapabilityMapper:
        """Build a mapper from every operation in a registry document that has a capability."""

        mapping = {
            op.operation_id: op.capability
            for _, op in document.iter_operations()
            if op.capability
        }
        return cls(mapping)

    def capability_for(self, operation_id: str) -> str | None:
        """Return the capability an operation provides, or ``None`` if untagged."""

        return self._forward.get(operation_id)

    def operations_with(self, capability: str) -> list[str]:
        """Return the operation ids that provide a capability (empty if none)."""

        return list(self._reverse.get(self.validate(capability), []))

    def require_operations_with(self, capability: str) -> list[str]:
        """Like :meth:`operations_with`, but raise if nothing provides the capability."""

        operations = self.operations_with(capability)
        if not operations:
            raise UnknownCapabilityError(
                f"No registered operation provides capability {capability!r}. "
                f"Known capabilities: {self.capabilities()}"
            )
        return operations

    def has_capability(self, capability: str) -> bool:
        try:
            return self.validate(capability) in self._reverse
        except InvalidCapabilityError:
            return False

    def capabilities(self) -> list[str]:
        """Return the sorted closed vocabulary of known capabilities."""

        return sorted(self._reverse)

    def as_dict(self) -> dict[str, str]:
        """Return a copy of the ``{operation_id: capability}`` mapping."""

        return dict(self._forward)
