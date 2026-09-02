"""User-intent models.

Produced by the Intent Understanding component (milestone 7) from a raw
natural-language request and consumed by API selection, parameter extraction
and planning.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IntentEntity(BaseModel):
    """A typed value extracted from the user's request.

    Example: ``name="phoneNumber"``, ``value="+123456789"``, ``raw="my phone"``.
    """

    name: str
    value: Any
    type: str | None = None
    raw: str | None = None
    confidence: float | None = None


class Intent(BaseModel):
    """A structured interpretation of what the user wants to achieve.

    ``required_capabilities`` / ``known_facts`` / ``missing_information`` are
    populated by the Intent Understanding agent (milestone 7). They describe *what*
    is needed, not *how* or *in what order* - the agent never decides an execution
    plan. ``required_capabilities`` come from a closed vocabulary (the CAMARA
    capability families); ``known_facts`` / ``missing_information`` use the shared
    fact-key convention (e.g. ``device.identifier``, ``qos.session.id``).
    """

    text: str
    goal: str
    action: str | None = None
    entities: list[IntentEntity] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    target_domains: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    known_facts: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_entity(self, name: str) -> IntentEntity | None:
        """Return the first entity with the given name, if any."""

        return next((e for e in self.entities if e.name == name), None)

    def entity_values(self) -> dict[str, Any]:
        """Return a ``{name: value}`` mapping of all extracted entities."""

        return {e.name: e.value for e in self.entities}

    def to_understanding(self) -> dict[str, Any]:
        """Return the structured-intent view produced by the milestone-7 agent."""

        return {
            "goal": self.goal,
            "entities": self.entity_values(),
            "constraints": self.constraints,
            "required_capabilities": self.required_capabilities,
            "known_facts": self.known_facts,
            "missing_information": self.missing_information,
        }
