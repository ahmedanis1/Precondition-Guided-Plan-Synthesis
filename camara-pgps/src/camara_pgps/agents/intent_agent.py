"""LLM-based Intent Understanding Agent.

The Intent Agent is responsible only for understanding the user's
natural-language request.

It does NOT:
- select CAMARA APIs
- select QoS profiles
- resolve API parameters
- inspect operation preconditions
- determine missing API parameters
- plan API calls
- execute API calls

Those responsibilities belong to later stages of the pipeline:
RAG -> API Selection -> Parameter Resolution -> Planning -> Execution.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from camara_pgps.agents._llm import build_chat_model, chat_content, extract_json
from camara_pgps.config import Settings, get_settings
from camara_pgps.models import Intent, IntentEntity

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# LLM prompt
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """\
You are the Intent Understanding Agent for a CAMARA API orchestration system.

Your ONLY responsibility is to understand what the user wants.

Convert the user's natural-language request into a structured intent.

Extract ONLY information explicitly stated or strongly implied by the user:

1. goal
   - A short, normalized description of what the user wants.
   - Do not mention CAMARA APIs or implementation details.

2. action
   - The requested high-level action.
   - Examples: create, enable, disable, retrieve, check, monitor, modify.
   - Use null when the action cannot be determined.

3. entities
   - Concrete entities or concepts mentioned by the user.
   - Examples:
     activity = gaming
     phoneNumber = +49123456789
     ipAddress = 192.168.1.10
     sessionId = ...
   - Do not invent values.

4. constraints
   - User requirements, preferences, limits, or conditions.
   - Preserve their semantic meaning.
   - Normalize units where useful.
   - Example:
       "next 2 hours" -> duration = 7200 seconds
       "30 minutes" -> duration = 1800 seconds
       "low latency" -> quality = low_latency
       "best quality" -> quality = best

IMPORTANT:
- "best quality" is a user requirement. Do NOT map it to a specific
  CAMARA QoS profile.
- "gaming" should be extracted as an activity/entity.
- Durations must be normalized to seconds.
- Do NOT invent device identifiers, authentication tokens, session IDs,
  application server addresses, QoS profiles, or other API parameters.
- Do NOT choose a CAMARA API.
- Do NOT choose an API operation.
- Do NOT determine required API parameters.
- Do NOT determine missing API parameters.
- Do NOT create an execution plan.
- Do NOT retrieve or infer information from external documentation.

Return ONLY a JSON object matching this schema:

{
  "goal": "string",
  "action": "string or null",
  "entities": {
    "entity_name": "entity_value"
  },
  "constraints": [
    "string"
  ]
}

No markdown.
No explanation.
No commentary.
"""


# --------------------------------------------------------------------------- #
# Structured LLM output
# --------------------------------------------------------------------------- #


class IntentDraft(BaseModel):
    """LLM output before conversion into the domain Intent model."""

    goal: str = ""

    action: str | None = None

    entities: dict[str, Any] = Field(default_factory=dict)

    constraints: list[str] = Field(default_factory=list)


@runtime_checkable
class IntentParser(Protocol):
    """Interface for an intent parser."""

    def parse(self, text: str) -> IntentDraft:
        ...


# --------------------------------------------------------------------------- #
# LLM parser
# --------------------------------------------------------------------------- #


class LLMIntentParser:
    """Use an LLM to extract semantic intent from natural language."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def parse(self, text: str) -> IntentDraft:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self._model.invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=text),
            ]
        )

        raw = chat_content(response)
        data = extract_json(raw)

        return IntentDraft.model_validate(data)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


class IntentAgent:
    """Natural language -> structured Intent using an LLM only."""

    def __init__(
        self,
        *,
        llm: Any | None = None,
        parser: IntentParser | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        if parser is not None:
            self.parser = parser
        elif llm is not None:
            self.parser = LLMIntentParser(llm)
        else:
            raise ValueError(
                "IntentAgent requires an LLM or an IntentParser. "
                "Configure an LLM through the application settings."
            )

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
    ) -> IntentAgent:
        settings = settings or get_settings()

        return cls(
            llm=build_chat_model(
                settings,
                purpose="intent agent",
            ),
            settings=settings,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def understand(self, text: str) -> Intent:
        """Understand a user's request and return a structured Intent."""

        if not text or not text.strip():
            raise ValueError("intent text must not be empty")

        cleaned = text.strip()
        draft = self.parser.parse(cleaned)

        return self._finalise(cleaned, draft)

    def understand_raw(self, text: str) -> IntentDraft:
        """Return the raw structured LLM output."""

        if not text or not text.strip():
            raise ValueError("intent text must not be empty")

        return self.parser.parse(text.strip())

    # ------------------------------------------------------------------ #
    # Domain mapping
    # ------------------------------------------------------------------ #

    def _finalise(
        self,
        text: str,
        draft: IntentDraft,
    ) -> Intent:
        """Convert the LLM draft into the application Intent model."""

        entities = [
            IntentEntity(
                name=str(name),
                value=value,
            )
            for name, value in draft.entities.items()
        ]

        return Intent(
            text=text,
            goal=(draft.goal or text).strip(),
            action=draft.action,
            entities=entities,
            constraints=[
                str(constraint).strip()
                for constraint in draft.constraints
                if str(constraint).strip()
            ],
            metadata={
                "parser": type(self.parser).__name__,
                "stage": "intent-understanding",
            },
        )


# --------------------------------------------------------------------------- #
# Process-wide agent
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def get_intent_agent() -> IntentAgent:
    """Return the process-wide LLM intent agent."""

    return IntentAgent.from_settings()


__all__ = [
    "IntentAgent",
    "IntentDraft",
    "IntentParser",
    "LLMIntentParser",
    "get_intent_agent",
]