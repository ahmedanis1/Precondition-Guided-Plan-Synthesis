"""User-intent models.

Represents the semantic interpretation of a user's natural-language request.

The Intent model describes WHAT the user wants. It does not describe which
CAMARA APIs are required, which parameters are missing, or how the request
should be executed. Those concerns belong to downstream API selection,
parameter resolution, planning, and execution components.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class IntentEntity(BaseModel):
    """A typed value extracted from the user's request.

    Examples:
        name="activity", value="gaming"
        name="phoneNumber", value="+123456789"
        name="ipAddress", value="192.168.1.10"
    """

    name: str
    value: Any
    type: str | None = None
    raw: str | None = None
    confidence: float | None = None


class IntentConstraint(BaseModel):
    """A requirement or preference expressed by the user."""

    type: str
    value: Any
    unit: str | None = None
    raw: str | None = None
    confidence: float | None = None


class Intent(BaseModel):
    """Semantic interpretation of what the user wants."""

    text: str

    goal: str

    action: str | None = None

    entities: list[IntentEntity] = Field(default_factory=list)

    constraints: list[IntentConstraint] = Field(default_factory=list)

    confidence: float | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("constraints", mode="before")
    @classmethod
    def _coerce_constraints(cls, value: Any) -> Any:
        """Accept a bare string constraint as a generic ``requirement``."""

        if not isinstance(value, list):
            return value
        return [
            {"type": "requirement", "value": item, "raw": item}
            if isinstance(item, str)
            else item
            for item in value
        ]

    def get_entity(self, name: str) -> IntentEntity | None:
        """Return the first entity with the given name, if any."""

        return next(
            (entity for entity in self.entities if entity.name == name),
            None,
        )

    def entity_values(self) -> dict[str, Any]:
        """Return a {name: value} mapping of extracted entities."""

        return {
            entity.name: entity.value
            for entity in self.entities
        }

    def get_constraint(self, constraint_type: str) -> IntentConstraint | None:
        """Return the first constraint of the requested type."""

        return next(
            (
                constraint
                for constraint in self.constraints
                if constraint.type == constraint_type
            ),
            None,
        )

    def to_understanding(self) -> dict[str, Any]:
        """Return the semantic understanding produced by the Intent Agent."""

        return {
            "goal": self.goal,
            "action": self.action,
            "entities": self.entity_values(),
            "constraints": [
                {
                    "type": constraint.type,
                    "value": constraint.value,
                    **(
                        {"unit": constraint.unit}
                        if constraint.unit is not None
                        else {}
                    ),
                }
                for constraint in self.constraints
            ],
            "confidence": self.confidence,
        }
