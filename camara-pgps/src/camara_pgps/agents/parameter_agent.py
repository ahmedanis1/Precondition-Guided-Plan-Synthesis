"""Parameter Extraction agent (milestone 9).

Given the user's request and the candidate operations from API Selection
(milestone 8), this pulls concrete parameter values out of the request -
device identifier, location, QoS profile, application server, duration,
priority, and so on.

Hard rule: **never fabricate a missing parameter.** A value is only bound if it
is grounded in the request (an intent entity, a regex match in the text, an enum
literal mentioned by the user) or in the facts carried in from earlier steps. A
required parameter with no grounded value is reported in ``missing_information``,
not guessed.

    LLM / rule extraction -> grounding check -> bound value  (else -> missing_information)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from camara_pgps.agents._llm import build_chat_model, chat_content, extract_json
from camara_pgps.agents.api_selection_agent import APISelection
from camara_pgps.pgps.parameter_facts import PARAMETER_TO_FACTS as _PARAM_FACTS
from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import Settings, get_settings
from camara_pgps.models import (
    APIParameter,
    BindingSource,
    CamaraOperation,
    Intent,
    ParameterBinding,
)

logger = logging.getLogger(__name__)

_PHONE_RE = re.compile(r"\+\d[\d\s\-]{6,}\d")
_IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_DURATION_RE = re.compile(
    r"(\d+|[a-z]+)\s*(?:of\s+)?(second|minute|hour|day)s?", re.IGNORECASE
)
_DURATION_UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
_HALF_HOUR_RE = re.compile(r"half\s+an?\s+hour", re.IGNORECASE)
# Spelled-out quantities ("two hours", "an hour", "a couple of hours") - a
# user typing a duration in words is at least as common as typing digits, and
# without this the regex above never fires for a request like "next two
# hours", so no duration ever reaches the pool and the pipeline asks the
# user to re-state something they already said.
_NUMBER_WORDS: dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "couple": 2,
}
_PRIORITY_RE = re.compile(r"\b(high|medium|normal|low)\s+priority\b", re.IGNORECASE)

# parameter name (normalised) -> accepted synonyms in the value pool
_PARAM_ALIASES: dict[str, set[str]] = {
    "device": {
        "device", "deviceid", "phonenumber", "phone", "msisdn", "subscriber",
        "identifier", "networkaccessidentifier", "nai", "ipaddress", "ip",
    },
    "applicationserver": {
        "applicationserver", "application", "app", "appserver", "server",
        "endpoint", "destination",
    },
    "qosprofile": {"qosprofile", "profile", "qos", "qosclass"},
    "duration": {
        "duration", "time", "length", "period",
        # Real CAMARA operations name this field differently per API - the
        # regex in _build_pool only ever populates the generic "duration"
        # pool key, so every real name it can bind to has to be aliased here
        # or a value like "30 minutes" grounds in the request text but never
        # reaches the operation's actual parameter (e.g. extendQosSessionDuration's
        # requestedAdditionalDuration).
        "requestedadditionalduration", "additionalduration",
        "requestedduration", "sessionduration", "durationseconds",
    },
    # Deliberately narrow: "area" is a structured geofencing circle (center coordinates + radius), not a place name - unlike "device" below, there is no way to derive one from a bare word like "office" or "downtown", so words that only describe an informal place ("location", "place", "region", "zone") are excluded on purpose. Widening this back to include them reintroduces the exact bug this comment is guarding against: a place-name string silently bound into an object-typed parameter, which then fails payload validation with a confusing schema error instead of asking the user for real coordinates. "coordinates" itself stays, since that word actually implies the right kind of data.
    "area": {"area", "geofence", "coordinates"},
    "sessionid": {"sessionid", "session"},
    "subscriptionid": {"subscriptionid", "subscription"},
    "priority": {"priority", "importance", "urgency"},
    "maxage": {"maxage", "freshness", "age"},
    "sink": {"sink", "callback", "webhook", "notificationurl", "callbackurl"},
    "types": {"types", "eventtypes", "kinds"},
}

# _PARAM_FACTS (resolved parameter, normalised -> fact keys it satisfies) is
# now imported above from pgps.parameter_facts - the single, shared table
# also used by the planner (to record cross-step bindings) and the executor
# (to substitute a fact into a raw path/query parameter), so a fact
# recognised in one place is always recognised in all of them.

# reverse: fact key -> the parameter (normalised) it can fill
_FACT_PARAM: dict[str, str] = {
    fact: param for param, facts in _PARAM_FACTS.items() for fact in facts
}

_SYSTEM_PROMPT = """You extract parameter VALUES for CAMARA API operations from a user request.

Rules:
- Only output a value that is explicitly present in the user request.
- NEVER invent, guess, or default a value. If it is not stated, leave it out.
- Use the exact parameter names given for each operation.

Return ONLY JSON: {{"<operationId>": {{"<param>": <value>, ...}}, ...}}.
Operations and their parameters:
{operations}"""


# --------------------------------------------------------------------------- #
# result models
# --------------------------------------------------------------------------- #

class OperationParameters(BaseModel):
    """Extracted parameters for one candidate operation."""

    operation_id: str
    api_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    bindings: list[ParameterBinding] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    unknown_optional: list[str] = Field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not self.missing_required


class ParameterExtraction(BaseModel):
    """Parameter extraction result across all candidate operations."""

    query: str
    intent: Intent | None = None
    operations: list[OperationParameters] = Field(default_factory=list)
    known_facts: dict[str, Any] = Field(default_factory=dict)
    missing_information: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def for_operation(self, operation_id: str) -> OperationParameters | None:
        return next(
            (item for item in self.operations if item.operation_id == operation_id),
            None,
        )

    def missing_for(self, operation_id: str) -> list[str]:
        item = self.for_operation(operation_id)
        return list(item.missing_required) if item else []


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def fact_for_parameter(name: str) -> str | None:
    """The fact key a raw parameter name resolves to, if any.

    Thin re-export of ``pgps.parameter_facts.fact_for_parameter`` under this
    module's existing name/normalisation, so callers that already imported
    it from here (PGPS plan routing, in ``workflow/nodes.py``) don't need to
    change - it now resolves from the single shared table.
    """

    facts = _PARAM_FACTS.get(_norm(name))
    return facts[0] if facts else None


def _alias_group(param_name: str) -> set[str]:
    key = _norm(param_name)
    for group in _PARAM_ALIASES.values():
        if key in group:
            return group | {key}
    return {key}


def _enum_literal(value: Any, enum: list[Any]) -> Any | None:
    lowered = str(value).strip().lower()
    return next((e for e in enum if str(e).lower() == lowered), None)


def _is_grounded(value: Any, text: str, pool: dict[str, Any]) -> bool:
    if isinstance(value, int | float | bool):
        return str(value) in text or value in pool.values()
    if isinstance(value, str):
        needle = value.strip().lower()
        if not needle:
            return False
        if needle in text.lower():
            return True
        return any(needle == str(v).strip().lower() for v in pool.values())
    if isinstance(value, dict):
        return any(_is_grounded(v, text, pool) for v in value.values())
    if isinstance(value, list):
        return any(_is_grounded(v, text, pool) for v in value)
    return False


# --------------------------------------------------------------------------- #
# extractors
# --------------------------------------------------------------------------- #

@runtime_checkable
class ParameterExtractor(Protocol):
    """Propose ``{operation_id: {param_name: value}}`` - grounded values only."""

    def propose(
        self,
        text: str,
        intent: Intent,
        operations: Sequence[CamaraOperation],
        pool: dict[str, Any],
    ) -> dict[str, dict[str, Any]]: ...


class RuleBasedParameterExtractor:
    """Deterministic alias / regex / enum matching against the value pool."""

    def propose(
        self,
        text: str,
        intent: Intent,
        operations: Sequence[CamaraOperation],
        pool: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        proposals: dict[str, dict[str, Any]] = {}
        for operation in operations:
            found: dict[str, Any] = {}
            for param in operation.parameters:
                value = self._match(param, text, pool)
                if value is not None:
                    found[param.name] = value
            if found:
                proposals[operation.operation_id] = found
        return proposals

    @staticmethod
    def _match(param: APIParameter, text: str, pool: dict[str, Any]) -> Any | None:
        for key in _alias_group(param.name):
            if key in pool:
                candidate = pool[key]
                if param.enum:
                    literal = _enum_literal(candidate, param.enum)
                    return literal
                return candidate
        if param.enum:
            for literal in param.enum:
                if re.search(re.escape(str(literal)), text, re.IGNORECASE):
                    return literal
        return None


class LLMParameterExtractor:
    """Asks a chat model for parameter values; the agent still grounding-checks them."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._fallback = RuleBasedParameterExtractor()

    def propose(
        self,
        text: str,
        intent: Intent,
        operations: Sequence[CamaraOperation],
        pool: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        try:
            data = extract_json(self._invoke(text, operations))
        except Exception as exc:  # noqa: BLE001 - degrade to rule-based
            logger.warning("LLM parameter extraction failed (%s); using rules", exc)
            return self._fallback.propose(text, intent, operations, pool)

        valid_ids = {op.operation_id for op in operations}
        proposals: dict[str, dict[str, Any]] = {}
        for operation_id, params in data.items():
            if operation_id in valid_ids and isinstance(params, dict):
                proposals[operation_id] = dict(params)
        return proposals

    def _invoke(self, text: str, operations: Sequence[CamaraOperation]) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        lines = []
        for operation in operations:
            names = ", ".join(
                f"{p.name}{'*' if p.required else ''}" for p in operation.parameters
            )
            lines.append(f"- {operation.operation_id}: {names or '(no parameters)'}")
        system = _SYSTEM_PROMPT.format(operations="\n".join(lines))
        response = self._model.invoke(
            [SystemMessage(content=system), HumanMessage(content=text)]
        )
        return chat_content(response)


# --------------------------------------------------------------------------- #
# agent
# --------------------------------------------------------------------------- #

class ParameterExtractionAgent:
    """User request (+ candidate operations) -> grounded parameter bindings."""

    def __init__(
        self,
        *,
        registry: CamaraRegistry | None = None,
        llm: Any | None = None,
        extractor: ParameterExtractor | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or CamaraRegistry.from_settings(self.settings)
        if extractor is not None:
            self.extractor: ParameterExtractor = extractor
        elif llm is not None:
            self.extractor = LLMParameterExtractor(llm)
        else:
            self.extractor = RuleBasedParameterExtractor()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> ParameterExtractionAgent:
        settings = settings or get_settings()
        return cls(
            llm=build_chat_model(settings, purpose="parameter extraction agent"),
            settings=settings,
        )

    # -- public -----------------------------------------------------------

    def extract(
        self,
        intent: Intent | str,
        operations: APISelection | Sequence[CamaraOperation | str],
        *,
        known_facts: dict[str, Any] | None = None,
    ) -> ParameterExtraction:
        resolved_intent = self._as_intent(intent)
        ops = self._resolve_operations(operations)
        text = resolved_intent.text or resolved_intent.goal

        prior_facts = dict(known_facts or {})
        pool = self._build_pool(resolved_intent, prior_facts)
        notes: list[str] = []

        proposals = self.extractor.propose(text, resolved_intent, ops, pool)
        if isinstance(self.extractor, LLMParameterExtractor):
            proposals = self._ground_filter(proposals, text, pool, notes)

        facts: dict[str, Any] = dict(prior_facts)
        op_results: list[OperationParameters] = []
        all_missing: list[str] = []

        for operation in ops:
            result = self._resolve_operation(
                operation, proposals.get(operation.operation_id, {}), pool, prior_facts,
                text, notes, facts,
            )
            op_results.append(result)
            for name in result.missing_required:
                if name not in all_missing:
                    all_missing.append(name)

        return ParameterExtraction(
            query=text,
            intent=resolved_intent,
            operations=op_results,
            known_facts=facts,
            missing_information=all_missing,
            notes=notes,
            metadata={
                "extractor": type(self.extractor).__name__,
                "operations": [op.operation_id for op in ops],
            },
        )

    # -- pieces ------------------------------------------------------------

    @staticmethod
    def _as_intent(intent: Intent | str) -> Intent:
        if isinstance(intent, Intent):
            return intent
        text = str(intent)
        return Intent(text=text, goal=text)

    def _resolve_operations(
        self, operations: APISelection | Sequence[CamaraOperation | str]
    ) -> list[CamaraOperation]:
        if isinstance(operations, APISelection):
            candidates = [item.operation for item in operations.operations]
        else:
            candidates = [
                entry
                if isinstance(entry, CamaraOperation)
                else self.registry.require_operation(str(entry))
                for entry in operations
            ]
        return [self._prefer_registry_shape(op) for op in candidates]

    def _prefer_registry_shape(self, operation: CamaraOperation) -> CamaraOperation:
        """Use the registry's parameter list instead of RAG's, when available.

        RAG deliberately reconstructs operations from JSONL alone (see
        ``RagRetriever``'s docstring) so it can surface an operation that
        isn't in the registry yet - but the JSONL corpus's structured payload
        never resolves a request body's ``schema_ref`` into individual
        fields, so a RAG-reconstructed operation can be missing required body
        parameters entirely (e.g. ``createQosAssignment`` shows up with only
        its header parameter, never ``qosProfile``). The registry - built
        fresh from the real OpenAPI specs - always has the true, complete
        parameter list, so binding/missing-detection should trust it over
        RAG whenever the operation is registered. Anything RAG found that
        isn't in the registry yet falls back to RAG's own shape unchanged,
        preserving the same "JSONL as source of truth" guarantee RAG itself
        provides.
        """

        if self.registry.has_operation(operation.operation_id):
            return self.registry.get_operation(operation.operation_id) or operation
        return operation

    def _build_pool(
        self, intent: Intent, prior_facts: dict[str, Any]
    ) -> dict[str, Any]:
        pool: dict[str, Any] = {}

        for entity in intent.entities:
            if entity.value is not None:
                pool[_norm(entity.name)] = entity.value

        for key, value in prior_facts.items():
            pool[_norm(key)] = value
            pool[_norm(key.split(".")[-1])] = value
            mapped = _FACT_PARAM.get(key)
            if mapped:
                pool[mapped] = value

        text = intent.text or intent.goal
        phone = _PHONE_RE.search(text)
        if phone:
            pool.setdefault("phonenumber", re.sub(r"[\s\-]", "", phone.group(0)))
        uuid = _UUID_RE.search(text)
        if uuid:
            pool.setdefault("sessionid", uuid.group(0))
            pool.setdefault("subscriptionid", uuid.group(0))
        ipv4 = _IPV4_RE.search(text)
        if ipv4 and not uuid:
            pool.setdefault("ipaddress", ipv4.group(0))
        for duration in _DURATION_RE.finditer(text):
            raw_amount, unit = duration.group(1), duration.group(2).lower()
            amount = (
                int(raw_amount)
                if raw_amount.isdigit()
                else _NUMBER_WORDS.get(raw_amount.lower())
            )
            if amount:
                pool.setdefault("duration", amount * _DURATION_UNIT_SECONDS[unit])
                break
        else:
            if _HALF_HOUR_RE.search(text):
                pool.setdefault("duration", 30 * _DURATION_UNIT_SECONDS["minute"])
        priority = _PRIORITY_RE.search(text)
        if priority:
            pool.setdefault("priority", priority.group(1).upper())
        return pool

    def _ground_filter(
        self,
        proposals: dict[str, dict[str, Any]],
        text: str,
        pool: dict[str, Any],
        notes: list[str],
    ) -> dict[str, dict[str, Any]]:
        filtered: dict[str, dict[str, Any]] = {}
        for operation_id, params in proposals.items():
            kept: dict[str, Any] = {}
            for name, value in params.items():
                if _is_grounded(value, text, pool):
                    kept[name] = value
                else:
                    notes.append(
                        f"dropped ungrounded value for {operation_id}.{name} "
                        "(not present in the request)"
                    )
            if kept:
                filtered[operation_id] = kept
        return filtered

    def _resolve_operation(
        self,
        operation: CamaraOperation,
        proposed: dict[str, Any],
        pool: dict[str, Any],
        prior_facts: dict[str, Any],
        text: str,
        notes: list[str],
        facts: dict[str, Any],
    ) -> OperationParameters:
        api_name = operation.api_name or self.registry.api_of(operation.operation_id)
        bindings: list[ParameterBinding] = []
        arguments: dict[str, Any] = {}
        missing_required: list[str] = []
        unknown_optional: list[str] = []
        prior_norm = (
            {_norm(k) for k in prior_facts}
            | {_norm(k.split(".")[-1]) for k in prior_facts}
            | {_FACT_PARAM[k] for k in prior_facts if k in _FACT_PARAM}
        )

        for param in operation.parameters:
            value, source = self._value_for(
                param, proposed, pool, prior_norm, text, notes, operation.operation_id
            )
            if value is not None:
                bindings.append(
                    ParameterBinding(
                        parameter=param.name,
                        source=source,
                        value=value,
                        required=param.required,
                    )
                )
                arguments[param.name] = value
                for fact_key in _PARAM_FACTS.get(_norm(param.name), ()):
                    facts.setdefault(fact_key, value)
            else:
                bindings.append(
                    ParameterBinding(
                        parameter=param.name,
                        source=BindingSource.UNRESOLVED,
                        value=None,
                        required=param.required,
                    )
                )
                if param.required:
                    missing_required.append(param.name)
                else:
                    unknown_optional.append(param.name)

        return OperationParameters(
            operation_id=operation.operation_id,
            api_name=api_name,
            arguments=arguments,
            bindings=bindings,
            missing_required=missing_required,
            unknown_optional=unknown_optional,
        )

    @staticmethod
    def _value_for(
        param: APIParameter,
        proposed: dict[str, Any],
        pool: dict[str, Any],
        prior_norm: set[str],
        text: str,
        notes: list[str],
        operation_id: str,
    ) -> tuple[Any | None, BindingSource]:
        raw = proposed.get(param.name)
        if raw is None:
            for key in _alias_group(param.name):
                if key in pool:
                    raw = pool[key]
                    break
        if raw is None and param.enum:
            for literal in param.enum:
                if re.search(re.escape(str(literal)), text, re.IGNORECASE):
                    raw = literal
                    break

        if raw is None:
            if not param.required and param.default is not None:
                return param.default, BindingSource.DEFAULT
            return None, BindingSource.UNRESOLVED

        if param.enum:
            literal = _enum_literal(raw, param.enum)
            if literal is None:
                notes.append(
                    f"{operation_id}.{param.name}: value {raw!r} is not one of "
                    f"{param.enum}; left unresolved"
                )
                return None, BindingSource.UNRESOLVED
            raw = literal

        source = (
            BindingSource.PRIOR_STEP
            if _norm(param.name) in prior_norm or _alias_group(param.name) & prior_norm
            else BindingSource.INTENT
        )
        return raw, source


@lru_cache(maxsize=1)
def get_parameter_agent() -> ParameterExtractionAgent:
    """Return the process-wide parameter extraction agent from configured settings."""

    return ParameterExtractionAgent.from_settings()


__all__ = [
    "ParameterExtractionAgent",
    "ParameterExtraction",
    "OperationParameters",
    "ParameterExtractor",
    "RuleBasedParameterExtractor",
    "LLMParameterExtractor",
    "get_parameter_agent",
    "fact_for_parameter",
]
