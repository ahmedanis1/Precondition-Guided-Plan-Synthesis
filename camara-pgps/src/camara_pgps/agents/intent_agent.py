"""Intent Understanding agent (milestone 7) - the first LLM agent.

Converts a natural-language request into a structured
:class:`~camara_pgps.models.Intent`. It answers *what the user wants*; it never
chooses APIs and never orders API calls - that is the job of API Selection
(milestone 8) and the PGPS planner (milestone 11).

Two parsers:

* :class:`LLMIntentParser` - prompts a LangChain chat model for JSON and validates
  it against the registry's capability vocabulary. Falls back to the rule-based
  parser on any error.
* :class:`RuleBasedIntentParser` - deterministic keyword / regex extraction, used
  when no LLM is configured (offline, tests). Grounded in the registry: capability
  keywords and precondition fact keys come from the registered operations.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from camara_pgps.agents._llm import build_chat_model, chat_content, extract_json
from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import Settings, get_settings
from camara_pgps.models import Intent, IntentEntity

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PHONE_RE = re.compile(r"\+\d[\d\s\-]{6,}\d")
_IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_DURATION_RE = re.compile(r"(\d+)\s*(second|minute|hour|day)s?", re.IGNORECASE)

_DURATION_UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}

_GOAL_MARKERS = (
    "i want to ",
    "i want ",
    "i need to ",
    "i need ",
    "i would like to ",
    "i'd like to ",
    "so that ",
    "so i can ",
    "trying to ",
    "help me ",
)

_CONSTRAINT_MARKERS = (
    "better",
    "faster",
    "improve",
    "low latency",
    "high quality",
    "at least",
    "no more than",
    "within",
    "reliable",
    "stable",
    "guarantee",
    "priorit",
    "minimum",
    "maximum",
    "as soon as",
    "urgent",
)

_FAMILY_SYNONYMS: dict[str, set[str]] = {
    "qos": {
        "quality", "qos", "latency", "bandwidth", "throughput", "speed", "slow",
        "freezing", "freeze", "buffering", "buffer", "lag", "laggy", "stutter",
        "jitter", "stable", "reliable", "prioritise", "prioritize", "guarantee",
        "guaranteed", "streaming", "stream", "call", "video", "voip", "gaming",
        "session", "connection",
    },
    "device-location": {
        "location", "locate", "where", "geofence", "geofencing", "area",
        "position", "positioned", "located", "region", "coordinates", "gps",
        "whereabouts", "nearby", "boundary", "verify",
    },
    "network-insights": {
        "insight", "insights", "analytics", "report", "reporting", "reachability",
        "reachable", "subscribe", "subscription", "notify", "notification",
        "monitor", "monitoring", "status",
    },
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "for", "in",
    "on", "at", "is", "are", "was", "were", "be", "been", "being", "it", "its",
    "this", "that", "these", "those", "my", "your", "our", "their", "his", "her",
    "we", "you", "they", "me", "us", "as", "by", "with", "from", "out", "so",
    "can", "will", "would", "should", "could", "do", "does", "did", "have", "has",
    "had", "not", "get", "got", "give", "want", "need", "like", "keep", "keeps",
    "kept", "now", "just", "please", "help", "some", "any", "all", "about",
    "within", "given", "whether", "into", "per", "via", "also", "up",
}

# how many distinct keyword hits a capability family needs before it is reported
_FAMILY_MATCH_MIN = 2

_ENTITY_FACTS: dict[str, tuple[str, ...]] = {
    "phoneNumber": ("device.phoneNumber", "device.identifier"),
    "ipAddress": ("device.ipAddress", "device.identifier"),
    "sessionId": ("qos.session.id",),
}

_SYSTEM_PROMPT = """You convert a user's natural-language request into a STRUCTURED INTENT.
You do NOT choose APIs. You do NOT order or plan API calls. You only describe what
the user wants.

Return ONLY a JSON object with these keys:
  goal:                  short imperative paraphrase of the user's objective
  entities:              object of concrete values mentioned (phone number, ids, durations)
  constraints:           array of qualitative / quantitative requirements
  required_capabilities: array; choose ONLY from this closed list: {families}
  known_facts:           array of fact keys already satisfied (e.g. "device.identifier")
  missing_information:   array of fact keys still needed from the user

No commentary, no markdown fences."""


class IntentDraft(BaseModel):
    """Raw structured-intent output, before validation against the registry.

    Matches the milestone-7 output shape one-to-one.
    """

    goal: str = ""
    entities: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    known_facts: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


@runtime_checkable
class IntentParser(Protocol):
    """Turns text into an :class:`IntentDraft`."""

    def parse(self, text: str) -> IntentDraft: ...


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def _family_of(capability: str) -> str:
    return capability.split(".", 1)[0]


# Re-exported for callers/tests that import it from this module.
_extract_json = extract_json


# --------------------------------------------------------------------------- #
# rule-based parser
# --------------------------------------------------------------------------- #

class RuleBasedIntentParser:
    """Deterministic intent extraction grounded in the CAMARA registry."""

    def __init__(self, registry: CamaraRegistry) -> None:
        self._registry = registry
        self._family_keywords = self._build_family_keywords(registry)

    @staticmethod
    def _build_family_keywords(registry: CamaraRegistry) -> dict[str, set[str]]:
        keywords: dict[str, set[str]] = {}
        for capability in registry.capabilities():
            family = _family_of(capability)
            bucket = keywords.setdefault(family, set())
            bucket.update(_FAMILY_SYNONYMS.get(family, set()))
            bucket.update(_TOKEN_RE.findall(capability))
            for operation in registry.operations_for_capability(capability):
                bucket.update(_tokens(operation.summary or ""))
                bucket.update(t.lower() for t in operation.tags)
            keywords[family] = {
                word for word in bucket if len(word) > 2 and word not in _STOPWORDS
            }
        return keywords

    def parse(self, text: str) -> IntentDraft:
        tokens = _tokens(text)

        entities = self._entities(text)
        known_facts = self._known_facts(entities)
        families = self._match_families(tokens)

        return IntentDraft(
            goal=self._goal(text),
            entities=entities,
            constraints=self._constraints(text),
            required_capabilities=families,
            known_facts=known_facts,
            missing_information=self._missing_information(families, known_facts),
        )

    # -- pieces ------------------------------------------------------------

    @staticmethod
    def _goal(text: str) -> str:
        cleaned = " ".join(text.split())
        lowered = cleaned.lower()
        for marker in _GOAL_MARKERS:
            index = lowered.find(marker)
            if index != -1:
                tail = cleaned[index + len(marker) :].strip(" .")
                if tail:
                    return tail[0].lower() + tail[1:]
        return cleaned.rstrip(".")

    @staticmethod
    def _entities(text: str) -> dict[str, Any]:
        entities: dict[str, Any] = {}
        phone = _PHONE_RE.search(text)
        if phone:
            entities["phoneNumber"] = re.sub(r"[\s\-]", "", phone.group(0))
        uuid = _UUID_RE.search(text)
        if uuid:
            entities["sessionId"] = uuid.group(0)
        ipv4 = _IPV4_RE.search(text)
        if ipv4 and not uuid:
            entities["ipAddress"] = ipv4.group(0)
        duration = _DURATION_RE.search(text)
        if duration:
            amount = int(duration.group(1))
            unit = duration.group(2).lower()
            entities["duration"] = amount * _DURATION_UNIT_SECONDS[unit]
        return entities

    @staticmethod
    def _known_facts(entities: dict[str, Any]) -> list[str]:
        facts: list[str] = []
        for name in entities:
            facts.extend(_ENTITY_FACTS.get(name, ()))
        return _dedupe(facts)

    @staticmethod
    def _constraints(text: str) -> list[str]:
        clauses = re.split(r"[,.;]| and | but ", text, flags=re.IGNORECASE)
        found: list[str] = []
        for clause in clauses:
            stripped = clause.strip()
            if stripped and any(m in stripped.lower() for m in _CONSTRAINT_MARKERS):
                found.append(stripped)
        return _dedupe(found)

    def _match_families(self, tokens: set[str]) -> list[str]:
        scored: list[tuple[int, str]] = []
        for family, keywords in self._family_keywords.items():
            overlap = len(tokens & keywords)
            if overlap >= _FAMILY_MATCH_MIN:
                scored.append((overlap, family))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [family for _, family in scored]

    def _missing_information(
        self, families: list[str], known_facts: list[str]
    ) -> list[str]:
        needed: set[str] = set()
        for capability in self._registry.capabilities():
            if _family_of(capability) not in families:
                continue
            for operation in self._registry.operations_for_capability(capability):
                for condition in operation.preconditions:
                    needed.update(condition.requires)
        return sorted(needed - set(known_facts))


# --------------------------------------------------------------------------- #
# LLM parser
# --------------------------------------------------------------------------- #

class LLMIntentParser:
    """Prompts a LangChain chat model for a structured intent (JSON)."""

    def __init__(
        self,
        model: Any,
        registry: CamaraRegistry,
        *,
        fallback: IntentParser | None = None,
    ) -> None:
        self._model = model
        self._registry = registry
        self._fallback: IntentParser = fallback or RuleBasedIntentParser(registry)

    def parse(self, text: str) -> IntentDraft:
        try:
            raw = self._invoke(text)
            return IntentDraft.model_validate(extract_json(raw))
        except Exception as exc:  # noqa: BLE001 - any failure must degrade gracefully
            logger.warning("LLM intent parse failed (%s); using rule-based parser", exc)
            return self._fallback.parse(text)

    def _invoke(self, text: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        families = sorted({_family_of(c) for c in self._registry.capabilities()})
        system = _SYSTEM_PROMPT.format(families=", ".join(families))
        response = self._model.invoke(
            [SystemMessage(content=system), HumanMessage(content=text)]
        )
        return chat_content(response)


# --------------------------------------------------------------------------- #
# agent
# --------------------------------------------------------------------------- #

class IntentAgent:
    """Natural language -> structured :class:`Intent`."""

    def __init__(
        self,
        *,
        parser: IntentParser | None = None,
        llm: Any | None = None,
        registry: CamaraRegistry | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or CamaraRegistry.from_settings(self.settings)
        if parser is not None:
            self.parser: IntentParser = parser
        elif llm is not None:
            self.parser = LLMIntentParser(llm, self.registry)
        else:
            self.parser = RuleBasedIntentParser(self.registry)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> IntentAgent:
        settings = settings or get_settings()
        return cls(
            llm=build_chat_model(settings, purpose="intent agent"),
            settings=settings,
        )

    # -- public --------------------------------------------------------------

    def understand(self, text: str) -> Intent:
        if not text or not text.strip():
            raise ValueError("intent text must not be empty")
        cleaned = text.strip()
        return self._finalise(cleaned, self.parser.parse(cleaned))

    def understand_raw(self, text: str) -> IntentDraft:
        if not text or not text.strip():
            raise ValueError("intent text must not be empty")
        return self.parser.parse(text.strip())

    # -- validation / mapping to the domain model ---------------------------

    def _finalise(self, text: str, draft: IntentDraft) -> Intent:
        families = self._known_families()
        capabilities = [c for c in _dedupe(draft.required_capabilities) if c in families]
        dropped = [c for c in draft.required_capabilities if c not in families]
        known_facts = _dedupe([str(f) for f in draft.known_facts])
        missing = [
            m
            for m in _dedupe([str(m) for m in draft.missing_information])
            if m not in known_facts
        ]

        metadata: dict[str, Any] = {"parser": type(self.parser).__name__}
        if dropped:
            metadata["dropped_capabilities"] = dropped

        return Intent(
            text=text,
            goal=(draft.goal or text).strip() or text,
            entities=[
                IntentEntity(name=str(name), value=value)
                for name, value in draft.entities.items()
            ],
            constraints=_dedupe([str(c) for c in draft.constraints]),
            target_domains=self._domains_for(capabilities),
            required_capabilities=capabilities,
            known_facts=known_facts,
            missing_information=missing,
            metadata=metadata,
        )

    def _known_families(self) -> set[str]:
        return {_family_of(c) for c in self.registry.capabilities()}

    def _domains_for(self, families: list[str]) -> list[str]:
        apis: set[str] = set()
        for capability in self.registry.capabilities():
            if _family_of(capability) not in families:
                continue
            for operation in self.registry.operations_for_capability(capability):
                if operation.api_name:
                    apis.add(operation.api_name)
        return sorted(apis)


@lru_cache(maxsize=1)
def get_intent_agent() -> IntentAgent:
    """Return the process-wide intent agent built from configured settings."""

    return IntentAgent.from_settings()


__all__ = [
    "IntentAgent",
    "IntentDraft",
    "IntentParser",
    "LLMIntentParser",
    "RuleBasedIntentParser",
    "get_intent_agent",
]
