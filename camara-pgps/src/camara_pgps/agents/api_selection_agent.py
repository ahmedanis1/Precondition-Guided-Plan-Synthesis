"""API Selection agent (milestone 8).

Position in the pipeline::

    Intent agent -> RAG -> API Selection agent -> CAMARA registry

Given an :class:`~camara_pgps.models.Intent` (and the RAG evidence for it) this
identifies **candidate** CAMARA APIs and operations. It narrows; it does not
order them and does not build a plan.

Safety boundary: every name - whether it came from the intent, from RAG, or from
an LLM - is verified against the registry before it can become a candidate::

    suggestion -> registry verification -> valid CAMARA API

Anything the registry does not recognise is recorded in
:attr:`APISelection.rejected`, never returned as a candidate.
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

from camara_pgps.agents._llm import build_chat_model, chat_content, extract_json
from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import Settings, get_settings
from camara_pgps.models import CamaraAPI, CamaraOperation, Intent
from camara_pgps.retrieval.rag import RagContext, RagRetriever

logger = logging.getLogger(__name__)


class SelectionSource(StrEnum):
    """Where a candidate suggestion originated."""

    INTENT = "intent"
    RAG = "rag"
    LLM = "llm"


_SOURCE_WEIGHT = {
    SelectionSource.RAG: 1.0,
    SelectionSource.INTENT: 0.9,
    SelectionSource.LLM: 0.7,
}
_OPERATION_API_BOOST = 0.25

_SELECTION_PROMPT = """You narrow a user request down to CANDIDATE CAMARA APIs and operations.
You do NOT order them. You do NOT build an execution plan. You only shortlist.

Choose ONLY from the registered names provided. Prefer fewer, well-justified
candidates. Return ONLY a JSON object:
  {{"apis": ["<registered api name>", ...],
    "operations": ["<registered operationId>", ...],
    "rationale": "<one sentence>"}}

Registered APIs: {apis}
Registered operations: {operations}"""


# --------------------------------------------------------------------------- #
# result models
# --------------------------------------------------------------------------- #

class CandidateAPI(BaseModel):
    """A registry-verified candidate API."""

    name: str
    api: CamaraAPI
    score: float
    sources: list[SelectionSource] = Field(default_factory=list)
    operation_ids: list[str] = Field(default_factory=list)
    reason: str | None = None


class CandidateOperation(BaseModel):
    """A registry-verified candidate operation."""

    operation_id: str
    api_name: str
    operation: CamaraOperation
    capability: str | None = None
    score: float = 0.0
    sources: list[SelectionSource] = Field(default_factory=list)


class RejectedSuggestion(BaseModel):
    """A suggested name that the registry did not recognise."""

    name: str
    kind: str  # "api" | "operation" | "capability"
    source: SelectionSource
    reason: str = "not found in CAMARA registry"


class APISelection(BaseModel):
    """Candidate APIs / operations for one intent - a shortlist, not a plan."""

    query: str
    intent: Intent | None = None
    apis: list[CandidateAPI] = Field(default_factory=list)
    operations: list[CandidateOperation] = Field(default_factory=list)
    rejected: list[RejectedSuggestion] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def api_names(self) -> list[str]:
        return [item.name for item in self.apis]

    def operation_ids(self) -> list[str]:
        return [item.operation_id for item in self.operations]

    def get_api(self, name: str) -> CandidateAPI | None:
        return next((item for item in self.apis if item.name == name), None)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")


def _family_of(capability: str) -> str:
    return capability.split(".", 1)[0]


def _apis_for_family(registry: CamaraRegistry, family: str) -> set[str]:
    apis: set[str] = set()
    for capability in registry.capabilities():
        if _family_of(capability) != family:
            continue
        for operation in registry.operations_for_capability(capability):
            if operation.api_name:
                apis.add(operation.api_name)
    return apis


def _canonical_api(registry: CamaraRegistry, text: str) -> str | None:
    """Resolve an arbitrary API reference to a registered API name, or ``None``."""

    candidate = text.strip()
    if not candidate:
        return None
    if registry.has_api(candidate):
        return candidate
    slug = _slugify(candidate)
    if slug and registry.has_api(slug):
        return slug
    lowered = candidate.lower()
    for name in registry.api_names():
        api = registry.get_api(name)
        if name.lower() == lowered or name.replace("-", " ").lower() == lowered:
            return name
        if api is not None and api.title and api.title.lower() == lowered:
            return name
    return None


def _canonical_operation(registry: CamaraRegistry, text: str) -> str | None:
    candidate = text.strip()
    if not candidate:
        return None
    if registry.has_operation(candidate):
        return candidate
    lowered = candidate.lower()
    return next(
        (oid for oid in registry.operation_ids() if oid.lower() == lowered),
        None,
    )


# --------------------------------------------------------------------------- #
# optional LLM selector
# --------------------------------------------------------------------------- #

class LLMApiSelector:
    """Asks a chat model to shortlist APIs / operations from the registered names."""

    def __init__(self, model: Any, registry: CamaraRegistry) -> None:
        self._model = model
        self._registry = registry

    def propose(
        self, intent: Intent, rag_context: RagContext
    ) -> tuple[list[str], list[str], str]:
        try:
            data = extract_json(self._invoke(intent, rag_context))
            apis = [str(x) for x in data.get("apis", []) if str(x).strip()]
            operations = [str(x) for x in data.get("operations", []) if str(x).strip()]
            return apis, operations, str(data.get("rationale", "")).strip()
        except Exception as exc:  # noqa: BLE001 - degrade to intent + RAG only
            logger.warning("LLM API selection failed (%s); using intent + RAG only", exc)
            return [], [], ""

    def _invoke(self, intent: Intent, rag_context: RagContext) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        system = _SELECTION_PROMPT.format(
            apis=", ".join(self._registry.api_names()),
            operations=", ".join(self._registry.operation_ids()),
        )
        human = (
            f"Goal: {intent.goal}\n"
            f"Required capabilities: {intent.required_capabilities}\n"
            f"Constraints: {intent.constraints}\n\n"
            f"{rag_context.as_prompt_context()}"
        )
        response = self._model.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        return chat_content(response)


# --------------------------------------------------------------------------- #
# agent
# --------------------------------------------------------------------------- #

class _Accumulator:
    __slots__ = ("score", "sources")

    def __init__(self) -> None:
        self.score = 0.0
        self.sources: set[SelectionSource] = set()

    def add(self, source: SelectionSource, weight: float) -> None:
        self.score += weight
        self.sources.add(source)


class APISelectionAgent:
    """Intent (+ RAG) -> registry-verified :class:`APISelection`."""

    def __init__(
        self,
        *,
        registry: CamaraRegistry | None = None,
        rag: RagRetriever | None = None,
        llm: Any | None = None,
        selector: LLMApiSelector | None = None,
        settings: Settings | None = None,
        max_apis: int = 5,
        max_operations: int = 12,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or CamaraRegistry.from_settings(self.settings)
        self._rag = rag
        self.max_apis = max_apis
        self.max_operations = max_operations
        if selector is not None:
            self.selector: LLMApiSelector | None = selector
        elif llm is not None:
            self.selector = LLMApiSelector(llm, self.registry)
        else:
            self.selector = None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> APISelectionAgent:
        settings = settings or get_settings()
        return cls(
            llm=build_chat_model(settings, purpose="API selection agent"),
            settings=settings,
        )

    # -- public -----------------------------------------------------------

    def select(
        self,
        intent: Intent | str,
        *,
        rag_context: RagContext | None = None,
    ) -> APISelection:
        resolved_intent = self._as_intent(intent)
        rag = rag_context or self._get_rag().retrieve(resolved_intent)

        rejected: list[RejectedSuggestion] = []
        notes: list[str] = []

        api_suggestions = self._api_suggestions(resolved_intent, rag)
        op_suggestions = self._operation_suggestions(resolved_intent, rag)

        rationale = ""
        if self.selector is not None:
            llm_apis, llm_ops, rationale = self.selector.propose(resolved_intent, rag)
            api_suggestions.extend((name, SelectionSource.LLM) for name in llm_apis)
            op_suggestions.extend((oid, SelectionSource.LLM) for oid in llm_ops)

        # ---- registry verification (the safety boundary) ----
        api_acc = self._verify_apis(api_suggestions, rejected)
        op_acc = self._verify_operations(op_suggestions, rejected)

        # operations of a shortlisted API whose capability the intent asked for
        self._add_capability_operations(resolved_intent, api_acc, op_acc)

        operations = self._rank_operations(op_acc)
        for candidate in operations:
            api_acc.setdefault(candidate.api_name, _Accumulator()).score += (
                _OPERATION_API_BOOST
            )

        apis = self._rank_apis(api_acc, operations, rationale)

        return APISelection(
            query=resolved_intent.text or resolved_intent.goal,
            intent=resolved_intent,
            apis=apis,
            operations=operations,
            rejected=rejected,
            notes=notes,
            metadata={
                "llm_rationale": rationale,
                "used_llm": self.selector is not None,
                "candidate_api_count": len(apis),
            },
        )

    # -- suggestion gathering -------------------------------------------------

    @staticmethod
    def _as_intent(intent: Intent | str) -> Intent:
        if isinstance(intent, Intent):
            return intent
        text = str(intent)
        return Intent(text=text, goal=text)

    def _get_rag(self) -> RagRetriever:
        if self._rag is None:
            self._rag = RagRetriever.from_settings(self.settings)
        return self._rag

    def _api_suggestions(
        self, intent: Intent, rag: RagContext
    ) -> list[tuple[str, SelectionSource]]:
        suggestions: list[tuple[str, SelectionSource]] = []
        for domain in intent.target_domains:
            suggestions.append((domain, SelectionSource.INTENT))
        for family in intent.required_capabilities:
            for api_name in _apis_for_family(self.registry, family):
                suggestions.append((api_name, SelectionSource.INTENT))
        for api_name in rag.apis:
            suggestions.append((api_name, SelectionSource.RAG))
        for item in rag.operations:
            if item.api_name:
                suggestions.append((item.api_name, SelectionSource.RAG))
        return suggestions

    @staticmethod
    def _operation_suggestions(
        intent: Intent, rag: RagContext
    ) -> list[tuple[str, SelectionSource]]:
        return [
            (item.operation.operation_id, SelectionSource.RAG) for item in rag.operations
        ]

    # -- verification ------------------------------------------------------

    def _verify_apis(
        self,
        suggestions: list[tuple[str, SelectionSource]],
        rejected: list[RejectedSuggestion],
    ) -> dict[str, _Accumulator]:
        accumulators: dict[str, _Accumulator] = {}
        seen_rejects: set[tuple[str, SelectionSource]] = set()
        for raw_name, source in suggestions:
            canonical = _canonical_api(self.registry, raw_name)
            if canonical is None:
                key = (raw_name.strip().lower(), source)
                if key not in seen_rejects and raw_name.strip():
                    seen_rejects.add(key)
                    rejected.append(
                        RejectedSuggestion(name=raw_name, kind="api", source=source)
                    )
                continue
            accumulators.setdefault(canonical, _Accumulator()).add(
                source, _SOURCE_WEIGHT[source]
            )
        return accumulators

    def _verify_operations(
        self,
        suggestions: list[tuple[str, SelectionSource]],
        rejected: list[RejectedSuggestion],
    ) -> dict[str, _Accumulator]:
        accumulators: dict[str, _Accumulator] = {}
        seen_rejects: set[tuple[str, SelectionSource]] = set()
        for raw_id, source in suggestions:
            canonical = _canonical_operation(self.registry, raw_id)
            if canonical is None:
                key = (raw_id.strip().lower(), source)
                if key not in seen_rejects and raw_id.strip():
                    seen_rejects.add(key)
                    rejected.append(
                        RejectedSuggestion(
                            name=raw_id, kind="operation", source=source
                        )
                    )
                continue
            accumulators.setdefault(canonical, _Accumulator()).add(
                source, _SOURCE_WEIGHT[source]
            )
        return accumulators

    def _add_capability_operations(
        self,
        intent: Intent,
        api_acc: dict[str, _Accumulator],
        op_acc: dict[str, _Accumulator],
    ) -> None:
        if not intent.required_capabilities:
            return
        for api_name in list(api_acc):
            for operation in self.registry.list_operations(api_name):
                capability = self.registry.capability_of(operation.operation_id)
                if capability and _family_of(capability) in intent.required_capabilities:
                    op_acc.setdefault(operation.operation_id, _Accumulator()).add(
                        SelectionSource.INTENT, _SOURCE_WEIGHT[SelectionSource.INTENT]
                    )

    # -- ranking ---------------------------------------------------------------

    def _rank_operations(
        self, op_acc: dict[str, _Accumulator]
    ) -> list[CandidateOperation]:
        ranked = sorted(
            op_acc.items(), key=lambda kv: (-kv[1].score, kv[0])
        )[: self.max_operations]
        operations: list[CandidateOperation] = []
        for operation_id, acc in ranked:
            operation = self.registry.require_operation(operation_id)
            operations.append(
                CandidateOperation(
                    operation_id=operation_id,
                    api_name=operation.api_name or self.registry.api_of(operation_id),
                    operation=operation,
                    capability=self.registry.capability_of(operation_id),
                    score=round(acc.score, 6),
                    sources=sorted(acc.sources),
                )
            )
        return operations

    def _rank_apis(
        self,
        api_acc: dict[str, _Accumulator],
        operations: list[CandidateOperation],
        rationale: str,
    ) -> list[CandidateAPI]:
        ops_by_api: dict[str, list[str]] = {}
        for candidate in operations:
            ops_by_api.setdefault(candidate.api_name, []).append(candidate.operation_id)

        ranked = sorted(api_acc.items(), key=lambda kv: (-kv[1].score, kv[0]))
        ranked = ranked[: self.max_apis]
        result: list[CandidateAPI] = []
        for name, acc in ranked:
            api = self.registry.require_api(name)
            result.append(
                CandidateAPI(
                    name=name,
                    api=api,
                    score=round(acc.score, 6),
                    sources=sorted(acc.sources),
                    operation_ids=ops_by_api.get(name, []),
                    reason=rationale or None,
                )
            )
        return result


@lru_cache(maxsize=1)
def get_api_selection_agent() -> APISelectionAgent:
    """Return the process-wide API selection agent built from configured settings."""

    return APISelectionAgent.from_settings()


__all__ = [
    "APISelectionAgent",
    "APISelection",
    "CandidateAPI",
    "CandidateOperation",
    "RejectedSuggestion",
    "SelectionSource",
    "LLMApiSelector",
    "get_api_selection_agent",
]
