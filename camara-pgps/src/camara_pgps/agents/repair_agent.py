"""Clarification / Repair agent (milestone 16).

When PGPS (or parameter extraction, or validation) reports missing information,
this turns the bare list of missing names into a single, polite natural-language
question for the user::

    missing: [device_id, duration]
        ->  "To improve your network quality, I still need you to provide the
             device identifier and the session duration."

It **never** fills in a missing value. :class:`ClarificationRequest` has no field
for a value on a missing item, and no code path assigns one; the LLM phraser is
only allowed to produce prose and is instructed not to assume anything.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

from camara_pgps.agents._llm import build_chat_model, chat_content
from camara_pgps.agents.parameter_agent import ParameterExtraction
from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import Settings, get_settings
from camara_pgps.models import CamaraOperation, Intent
from camara_pgps.models.plan import PlanResult
from camara_pgps.models.state import AgentState

_LABELS: dict[str, str] = {
    "device": "device identifier",
    "deviceid": "device identifier",
    "deviceidentifier": "device identifier",
    "device.identifier": "device identifier",
    "device.phonenumber": "device phone number",
    "phonenumber": "phone number",
    "duration": "session duration",
    "qosprofile": "QoS profile",
    "qos.profile": "QoS profile",
    "applicationserver": "application server address",
    "area": "location area",
    "location.area": "location area",
    "sessionid": "session identifier",
    "qos.session.id": "QoS session identifier",
    "subscriptionid": "subscription identifier",
    "sink": "notification callback URL",
    "types": "event types to subscribe to",
    "maxage": "maximum acceptable location age",
    "auth.token": "API access token",
}

# if a model reply looks like it is answering for the user, discard it
_ANSWERING_RE = re.compile(
    r"\b(i'?ll use|i will use|i'?ll assume|i am assuming|assuming|let'?s say|"
    r"i'?ve set|i have set|defaulting to|use the default|i'?ll default)\b",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = (
    "You write ONE short, polite sentence asking a user to provide the listed "
    "pieces of information so a task can continue.\n"
    "Rules:\n"
    "- Do NOT invent, assume, guess or fill in any value.\n"
    "- Do NOT answer on the user's behalf.\n"
    "- Only ask for what is listed. No preamble, no bullet points."
)


class MissingItem(BaseModel):
    """One thing the user still has to supply. Intentionally has no ``value``."""

    name: str
    label: str
    kind: str  # "parameter" | "fact"
    operation_id: str | None = None
    description: str | None = None
    enum: list[Any] = Field(default_factory=list)


class ClarificationRequest(BaseModel):
    """A single natural-language question plus the structured list behind it."""

    question: str
    missing: list[MissingItem] = Field(default_factory=list)
    intent_goal: str | None = None
    provided: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def missing_names(self) -> list[str]:
        return [item.name for item in self.missing]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", name.lower())


def _humanize(name: str) -> str:
    key = _norm(name)
    if key in _LABELS:
        return _LABELS[key]
    parts = re.split(r"[._\s]+|(?<=[a-z0-9])(?=[A-Z])", name)
    return " ".join(part.lower() for part in parts if part) or name


def _oxford(items: Sequence[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return "the required information"
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


# --------------------------------------------------------------------------- #
# phrasers
# --------------------------------------------------------------------------- #

class RuleBasedPhraser:
    """Deterministic template phrasing - no LLM."""

    def phrase(
        self, request_text: str, goal: str | None, items: Sequence[MissingItem]
    ) -> str:
        fragments: list[str] = []
        for item in items:
            label = item.label
            if item.enum:
                label += f" (one of: {', '.join(str(v) for v in item.enum)})"
            fragments.append(label)
        need = _oxford(fragments)
        lead = f"To {goal.rstrip('.').strip().lower()}, I" if goal else "I"
        return f"{lead} still need you to provide {need}."


class LLMPhraser:
    """Polished phrasing from a chat model; falls back to the rule-based phraser."""

    def __init__(self, model: Any, *, fallback: RuleBasedPhraser | None = None) -> None:
        self._model = model
        self._fallback = fallback or RuleBasedPhraser()

    def phrase(
        self, request_text: str, goal: str | None, items: Sequence[MissingItem]
    ) -> str:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            listing = "\n".join(
                f"- {item.label}"
                + (f" (one of: {', '.join(map(str, item.enum))})" if item.enum else "")
                for item in items
            )
            human = f"User's request: {request_text!r}\nStill needed:\n{listing}"
            response = self._model.invoke(
                [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human)]
            )
            text = chat_content(response).strip().splitlines()
            sentence = next((line.strip() for line in text if line.strip()), "")
            if sentence and not _ANSWERING_RE.search(sentence):
                return sentence
        except Exception:  # noqa: BLE001 - phrasing must always succeed
            pass
        return self._fallback.phrase(request_text, goal, items)


# --------------------------------------------------------------------------- #
# agent
# --------------------------------------------------------------------------- #

class RepairAgent:
    """Missing information -> one natural-language clarification question."""

    def __init__(
        self,
        *,
        registry: CamaraRegistry | None = None,
        llm: Any | None = None,
        phraser: RuleBasedPhraser | LLMPhraser | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or CamaraRegistry.from_settings(self.settings)
        if phraser is not None:
            self.phraser: RuleBasedPhraser | LLMPhraser = phraser
        elif llm is not None:
            self.phraser = LLMPhraser(llm)
        else:
            self.phraser = RuleBasedPhraser()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RepairAgent:
        settings = settings or get_settings()
        return cls(
            llm=build_chat_model(settings, purpose="repair agent"),
            settings=settings,
        )

    # -- public --------------------------------------------------------------

    def clarify(
        self,
        missing: Iterable[str] | AgentState | ParameterExtraction,
        *,
        intent: Intent | None = None,
        plan: PlanResult | None = None,
        operations: Sequence[str | CamaraOperation] | None = None,
        request_text: str = "",
    ) -> ClarificationRequest:
        names, intent, plan, operations, request_text, provided = self._unpack(
            missing, intent, plan, operations, request_text
        )
        goal = intent.goal if intent is not None else None

        param_index = self._parameter_index(plan, operations)
        items: list[MissingItem] = []
        seen_labels: set[str] = set()
        for name in _dedupe(names):
            item = self._item(name, param_index)
            if item.label in seen_labels:
                continue
            seen_labels.add(item.label)
            items.append(item)

        question = (
            self.phraser.phrase(request_text, goal, items)
            if items
            else "I could not build an executable plan for this request. "
            "Could you rephrase or add more detail?"
        )
        return ClarificationRequest(
            question=question,
            missing=items,
            intent_goal=goal,
            provided=provided,
            metadata={"phraser": type(self.phraser).__name__, "count": len(items)},
        )

    # -- internals -----------------------------------------------------------

    def _unpack(
        self,
        missing: Iterable[str] | AgentState | ParameterExtraction,
        intent: Intent | None,
        plan: PlanResult | None,
        operations: Sequence[str | CamaraOperation] | None,
        request_text: str,
    ) -> tuple[
        list[str],
        Intent | None,
        PlanResult | None,
        Sequence[str | CamaraOperation] | None,
        str,
        dict[str, Any],
    ]:
        provided: dict[str, Any] = {}
        if isinstance(missing, AgentState):
            state = missing
            names = list(state.clarification)
            if not names and state.parameters is not None:
                names = list(state.parameters.missing_information)
            intent = intent or state.intent
            plan = plan or state.plan
            request_text = request_text or state.request_text
            provided = {k: v for k, v in state.facts.items() if v is not True}
            if not operations and state.plan is not None:
                operations = [s.operation_id for s in state.plan.steps]
            return names, intent, plan, operations, request_text, provided
        if isinstance(missing, ParameterExtraction):
            extraction = missing
            names = list(extraction.missing_information)
            intent = intent or extraction.intent
            request_text = request_text or extraction.query
            provided = dict(extraction.known_facts)
            if not operations:
                operations = [op.operation_id for op in extraction.operations]
            return names, intent, plan, operations, request_text, provided
        return list(missing), intent, plan, operations, request_text, provided

    def _parameter_index(
        self,
        plan: PlanResult | None,
        operations: Sequence[str | CamaraOperation] | None,
    ) -> dict[str, tuple[str, Any]]:
        index: dict[str, tuple[str, Any]] = {}
        op_ids: list[str] = []
        if operations:
            for entry in operations:
                op_ids.append(
                    entry.operation_id
                    if isinstance(entry, CamaraOperation)
                    else str(entry)
                )
        if plan is not None:
            op_ids.extend(step.operation_id for step in plan.steps)
        for op_id in _dedupe(op_ids):
            operation = self.registry.get_operation(op_id)
            if operation is None:
                continue
            for param in operation.parameters:
                index.setdefault(param.name, (op_id, param))
        return index

    @staticmethod
    def _item(name: str, param_index: dict[str, tuple[str, Any]]) -> MissingItem:
        entry = param_index.get(name)
        if entry is not None:
            op_id, param = entry
            return MissingItem(
                name=name,
                label=_humanize(name),
                kind="parameter",
                operation_id=op_id,
                description=param.description,
                enum=list(param.enum),
            )
        kind = "fact" if "." in name else "parameter"
        return MissingItem(name=name, label=_humanize(name), kind=kind)


@lru_cache(maxsize=1)
def get_repair_agent() -> RepairAgent:
    """Return the process-wide repair agent from configured settings."""

    return RepairAgent.from_settings()


__all__ = [
    "RepairAgent",
    "ClarificationRequest",
    "MissingItem",
    "RuleBasedPhraser",
    "LLMPhraser",
    "get_repair_agent",
]
