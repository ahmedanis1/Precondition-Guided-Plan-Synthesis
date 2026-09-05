"""RAG-grounded CAMARA API selection agent.

Architecture:

    Intent + RagContext
        -> candidate extraction from RAG
        -> deterministic relevance filtering
        -> LLM selection from RAG candidates only
        -> structured APISelection
        -> Parameter Resolver

The agent does NOT:
    - use CamaraRegistry for runtime validation
    - invent APIs or operations
    - resolve API parameters
    - select QoS profiles
    - create execution plans
    - execute API calls
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from camara_pgps.agents._llm import build_chat_model, chat_content, extract_json
from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import Settings, get_settings
from camara_pgps.models import CamaraAPI, CamaraOperation, Intent
from camara_pgps.retrieval.rag import RagContext, RagRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Selection source
# ---------------------------------------------------------------------------


class SelectionSource(str, Enum):
    RAG = "rag"
    LLM = "llm"


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------


class CandidateOperation(BaseModel):
    """An operation retrieved from the CAMARA knowledge base."""

    operation_id: str
    api_name: str
    operation: CamaraOperation
    score: float = 0.0
    sources: list[SelectionSource] = Field(default_factory=list)
    capability: str | None = None
    reason: str | None = None


class CandidateAPI(BaseModel):
    """An API that contains one or more relevant operations."""

    name: str
    score: float = 0.0
    sources: list[SelectionSource] = Field(default_factory=list)
    operation_ids: list[str] = Field(default_factory=list)
    reason: str | None = None


class APISelection(BaseModel):
    """Result returned by the API Selection Agent."""

    query: str
    apis: list[CandidateAPI] = Field(default_factory=list)
    operations: list[CandidateOperation] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM output schema
# ---------------------------------------------------------------------------


class LLMOperationChoice(BaseModel):
    """Structured output expected from the LLM."""

    selected_operation_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    confidence: float | None = None


# ---------------------------------------------------------------------------
# Internal candidate representation
# ---------------------------------------------------------------------------


@dataclass
class _RagCandidate:
    operation_id: str
    api_name: str
    operation: CamaraOperation
    score: float
    capability: str | None = None


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


_SELECTION_PROMPT = """
You are the CAMARA API Selection Agent.

Your ONLY responsibility is to select the CAMARA operation(s) that best
match the user's intent.

You are given:
1. The user's intent.
2. Operations retrieved from the CAMARA RAG knowledge base.

IMPORTANT RULES:

- You MUST select ONLY from the provided candidate operations.
- NEVER invent an API name.
- NEVER invent an operation ID.
- NEVER select an operation that is not in the candidate list.
- Do NOT resolve API parameters.
- Do NOT create parameter values.
- Do NOT select a QoS profile.
- Do NOT create an execution plan.
- Do NOT execute an API.
- Do NOT use outside CAMARA knowledge.
- Use the operation descriptions, HTTP methods, paths, and parameters
  provided in the candidates.
- Prefer the operation that directly satisfies the user's request.
- If the user asks to discover/list available resources, prefer a
  retrieval/list operation over an operation that retrieves one specific
  resource.
- If an operation requires a specific identifier that the user has not
  provided, do not prefer it over an operation that can discover resources.
- Return multiple operations ONLY when the user's intent clearly requires
  multiple distinct operations.

Return JSON only:

{
  "selected_operation_ids": ["operation-id"],
  "reason": "short explanation",
  "confidence": 0.0
}

The selected operation IDs MUST exactly match IDs from the candidate list.
"""


# ---------------------------------------------------------------------------
# LLM selector
# ---------------------------------------------------------------------------


# Selection only has to pick among candidates - it does not need the full
# operation text. Parameter extraction (registry-shape) and execution
# (conditions + real registry) already carry the complete detail later, so
# capping what goes into this prompt keeps it well clear of the model's
# context window regardless of how many verbose candidates RAG returns.
_MAX_CANDIDATE_TEXT_CHARS = 160


def _truncate(text: str | None, limit: int = _MAX_CANDIDATE_TEXT_CHARS) -> str:
    if not text:
        return ""
    text = " ".join(text.split())  # collapse whitespace/newlines
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut}..."


class LLMApiSelector:
    """Select operations using an LLM constrained to RAG candidates."""

    def __init__(self, model: Any):
        self._model = model

    def select(
        self,
        intent: Intent,
        candidates: list[_RagCandidate],
    ) -> LLMOperationChoice:
        if not candidates:
            return LLMOperationChoice(
                selected_operation_ids=[],
                reason="No relevant RAG candidates were available.",
                confidence=0.0,
            )

        candidate_text = self._format_candidates(candidates)

        prompt_body = f"""
USER INTENT
-----------
Goal: {getattr(intent, "goal", "")}
Action: {getattr(intent, "action", "")}

Entities:
{self._format_entities(intent)}

Constraints:
{self._format_constraints(intent)}

RAG CANDIDATE OPERATIONS
------------------------
{candidate_text}
"""

        try:
            # NOTE: OllamaChatModel.invoke() expects a list of messages, not a
            # raw string - passing a plain string here used to make it iterate
            # the prompt *character by character* (str is iterable), sending
            # one single-character "user" message per character instead of
            # one real message. That silently multiplied the prompt several
            # times over once Ollama's chat template added per-turn framing
            # to every one of those messages, which is what was actually
            # blowing prompts up to (and past) the model's context window.
            from langchain_core.messages import HumanMessage, SystemMessage

            response = self._model.invoke(
                [
                    SystemMessage(content=_SELECTION_PROMPT),
                    HumanMessage(content=prompt_body),
                ]
            )
            content = chat_content(response)
        except Exception as exc:
            # A slow/unreachable LLM (e.g. Ollama cold-loading a model, or a
            # timeout) must not sink the whole request when RAG already
            # found grounded candidates - fall back to the top-ranked one.
            logger.warning(
                "API selection LLM call failed (%s: %s); falling back to "
                "top-ranked RAG candidate(s)",
                type(exc).__name__,
                exc,
            )
            return self._fallback_choice(
                candidates,
                reason=(
                    f"API selection model was unavailable ({type(exc).__name__}); "
                    "used the top-ranked RAG result instead."
                ),
            )

        try:
            data = extract_json(content)
            return LLMOperationChoice.model_validate(data)
        except Exception as exc:
            logger.warning("Failed to parse API selection LLM output: %s", exc)

            return self._fallback_choice(
                candidates,
                reason=(
                    "API selection model returned an invalid response; "
                    "used the top-ranked RAG result instead."
                ),
            )

    @staticmethod
    def _fallback_choice(
        candidates: list[_RagCandidate],
        *,
        reason: str,
    ) -> LLMOperationChoice:
        """Deterministic fallback: trust the RAG ranking rather than failing outright.

        Never invents an operation - only ever returns an operation_id that was
        already in the RAG candidate list, same grounding guarantee as the LLM path.
        """

        top = sorted(candidates, key=lambda c: c.score, reverse=True)[:1]

        return LLMOperationChoice(
            selected_operation_ids=[c.operation_id for c in top],
            reason=reason,
            confidence=0.0,
        )

    @staticmethod
    def _format_candidates(candidates: list[_RagCandidate]) -> str:
        blocks: list[str] = []

        for index, candidate in enumerate(candidates, start=1):
            operation = candidate.operation

            method = getattr(operation, "method", None)
            path = getattr(operation, "path", None)
            description = _truncate(getattr(operation, "description", None))

            # `capability` falls back to the full description when the RAG
            # item has no distinct capability tag (see _get_rag_candidates) -
            # in that case it would just repeat the description line, so
            # skip it rather than doubling this candidate's prompt cost.
            capability = _truncate(candidate.capability)
            capability_line = (
                f"capability: {capability}\n" if capability and capability != description else ""
            )

            blocks.append(
                f"""
Candidate {index}
operation_id: {candidate.operation_id}
api_name: {candidate.api_name}
{capability_line}method: {method or ""}
path: {path or ""}
description: {description}
rag_score: {candidate.score:.4f}
"""
            )

        return "\n".join(blocks)

    @staticmethod
    def _format_entities(intent: Intent) -> str:
        entities = getattr(intent, "entities", [])

        if not entities:
            return "None"

        lines = []

        for entity in entities:
            if hasattr(entity, "name"):
                name = entity.name
                value = getattr(entity, "value", None)
            else:
                name = entity.get("name", "")
                value = entity.get("value")

            lines.append(f"- {name}: {value}")

        return "\n".join(lines)

    @staticmethod
    def _format_constraints(intent: Intent) -> str:
        constraints = getattr(intent, "constraints", [])

        if not constraints:
            return "None"

        lines = []

        for constraint in constraints:
            if hasattr(constraint, "type"):
                constraint_type = constraint.type
                value = getattr(constraint, "value", None)
                unit = getattr(constraint, "unit", None)
            else:
                constraint_type = constraint.get("type", "")
                value = constraint.get("value")
                unit = constraint.get("unit")

            unit_text = f" {unit}" if unit else ""

            lines.append(
                f"- {constraint_type}: {value}{unit_text}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# API Selection Agent
# ---------------------------------------------------------------------------


class APISelectionAgent:
    """Select CAMARA operations using RAG-grounded candidate selection."""

    def __init__(
        self,
        *,
        rag: RagRetriever | None = None,
        llm: Any | None = None,
        selector: LLMApiSelector | None = None,
        settings: Settings | None = None,
        max_candidates: int = 12,
        max_selected: int = 5,
    ):
        self.settings = settings or Settings()
        self.rag = rag or RagRetriever.from_settings(self.settings)

        self.llm = llm or build_chat_model(
            self.settings,
            purpose="API selection agent",
        )

        self.selector = selector or LLMApiSelector(self.llm)

        self.max_candidates = max_candidates
        self.max_selected = max_selected

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
    ) -> "APISelectionAgent":
        settings = settings or Settings()

        return cls(
            settings=settings,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        query: str,
        intent: Intent | None = None,
    ) -> APISelection:
        """Select the most relevant CAMARA operation(s).

        Flow:

            Intent
                +
            RAG
                ↓
            candidate operations
                ↓
            relevance filtering
                ↓
            LLM selection
                ↓
            structured APISelection
        """

        if intent is None:
            raise ValueError(
                "APISelectionAgent requires an Intent. "
                "Intent extraction must happen before API selection."
            )

        # --------------------------------------------------------------
        # 1. Get candidates from RAG
        # --------------------------------------------------------------

        rag_context = self.rag.retrieve(
            intent or query,
        )

        candidates = self._get_rag_candidates(rag_context)

        if not candidates:
            return APISelection(
                query=query,
                notes=["RAG returned no candidate operations."],
                metadata={
                    "stage": "api-selection",
                    "candidate_count": 0,
                },
            )

        # --------------------------------------------------------------
        # 2. Remove obviously irrelevant candidates
        # --------------------------------------------------------------

        filtered = self._filter_candidates(
            intent=intent,
            candidates=candidates,
        )

        # Never leave the LLM with an empty candidate set if RAG found
        # something. This fallback preserves recall.
        if not filtered:
            filtered = candidates[: self.max_candidates]

        # --------------------------------------------------------------
        # 3. Give ONLY filtered RAG candidates to LLM
        # --------------------------------------------------------------

        llm_result = self.selector.select(
            intent=intent,
            candidates=filtered,
        )

        # --------------------------------------------------------------
        # 4. Validate LLM selections
        # --------------------------------------------------------------

        candidate_by_id = {
            candidate.operation_id: candidate
            for candidate in filtered
        }

        selected_ids = []

        for operation_id in llm_result.selected_operation_ids:
            if operation_id not in candidate_by_id:
                logger.warning(
                    "LLM selected operation not present in RAG candidates: %s",
                    operation_id,
                )
                continue

            if operation_id not in selected_ids:
                selected_ids.append(operation_id)

            if len(selected_ids) >= self.max_selected:
                break

        # --------------------------------------------------------------
        # 5. Convert to structured CandidateOperation
        # --------------------------------------------------------------

        operations: list[CandidateOperation] = []

        for operation_id in selected_ids:
            candidate = candidate_by_id[operation_id]

            operations.append(
                CandidateOperation(
                    operation_id=candidate.operation_id,
                    api_name=candidate.api_name,
                    operation=candidate.operation,
                    score=candidate.score,
                    sources=[
                        SelectionSource.RAG,
                        SelectionSource.LLM,
                    ],
                    capability=candidate.capability,
                    reason=llm_result.reason,
                )
            )

        # --------------------------------------------------------------
        # Build API-level candidates from selected operations
        # --------------------------------------------------------------

        apis = self._build_api_candidates(operations)

        rejected = [
            candidate.operation_id
            for candidate in filtered
            if candidate.operation_id not in selected_ids
        ]

        return APISelection(
            query=query,
            apis=apis,
            operations=operations,
            rejected=rejected,
            notes=[
                llm_result.reason
                if llm_result.reason
                else "Operations selected from RAG candidates."
            ],
            metadata={
                "stage": "api-selection",
                "candidate_count": len(candidates),
                "filtered_candidate_count": len(filtered),
                "selected_operation_count": len(operations),
                "selection_source": "rag+llm",
                "registry_used": False,
            },
        )

    # ------------------------------------------------------------------
    # RAG candidate extraction
    # ------------------------------------------------------------------

    def _get_rag_candidates(
        self,
        rag_context: RagContext,
    ) -> list[_RagCandidate]:
        """Extract operation candidates from RagContext."""

        candidates: list[_RagCandidate] = []

        # --------------------------------------------------------------
        # Direct operation results
        # --------------------------------------------------------------

        for item in getattr(rag_context, "operations", []) or []:
            operation = getattr(item, "operation", None)

            if operation is None:
                continue

            operation_id = (
                getattr(item, "operation_id", None)
                or getattr(operation, "operation_id", None)
                or getattr(operation, "id", None)
            )

            api_name = (
                getattr(item, "api_name", None)
                or getattr(operation, "api_name", None)
                or getattr(operation, "api", None)
            )

            if not operation_id or not api_name:
                continue

            score = float(
                getattr(item, "score", None)
                or getattr(item, "similarity", None)
                or 0.0
            )

            capability = getattr(
                item,
                "capability",
                None,
            ) or getattr(
                operation,
                "description",
                None,
            )

            candidates.append(
                _RagCandidate(
                    operation_id=operation_id,
                    api_name=api_name,
                    operation=operation,
                    score=score,
                    capability=capability,
                )
            )

        # --------------------------------------------------------------
        # Deduplicate
        # --------------------------------------------------------------

        unique: dict[str, _RagCandidate] = {}

        for candidate in candidates:
            existing = unique.get(candidate.operation_id)

            if existing is None or candidate.score > existing.score:
                unique[candidate.operation_id] = candidate

        candidates = list(unique.values())

        candidates.sort(
            key=lambda candidate: candidate.score,
            reverse=True,
        )

        return candidates[: self.max_candidates]

    # ------------------------------------------------------------------
    # Candidate filtering
    # ------------------------------------------------------------------

    def _filter_candidates(
        self,
        *,
        intent: Intent,
        candidates: list[_RagCandidate],
    ) -> list[_RagCandidate]:
        """Remove candidates that are obviously incompatible.

        This is intentionally conservative.

        The filter should remove obvious mismatches, not try to perform
        the complete semantic selection. The LLM remains responsible for
        choosing among plausible candidates.
        """

        filtered: list[_RagCandidate] = []

        action = (
            str(getattr(intent, "action", "") or "")
            .strip()
            .lower()
        )

        goal = (
            str(getattr(intent, "goal", "") or "")
            .strip()
            .lower()
        )

        query_text = f"{goal} {action}"

        for candidate in candidates:
            operation = candidate.operation

            operation_id = candidate.operation_id.lower()

            description = str(
                getattr(operation, "description", "") or ""
            ).lower()

            method = str(
                getattr(operation, "method", "") or ""
            ).upper()

            path = str(
                getattr(operation, "path", "") or ""
            ).lower()

            text = (
                f"{operation_id} "
                f"{candidate.api_name.lower()} "
                f"{description} "
                f"{path}"
            )

            # ----------------------------------------------------------
            # Obvious retrieval/list intent
            # ----------------------------------------------------------

            discovery_intent = any(
                word in query_text
                for word in (
                    "show",
                    "list",
                    "available",
                    "discover",
                    "retrieve",
                    "find",
                    "get available",
                )
            )

            if discovery_intent:
                # If the operation clearly requires a resource identifier
                # and looks like a single-resource lookup, deprioritize it.
                has_identifier_path = any(
                    token in path
                    for token in (
                        "{name}",
                        "{id}",
                        "{sessionid}",
                        "{assignmentid}",
                    )
                )

                if has_identifier_path and method == "GET":
                    continue

            # ----------------------------------------------------------
            # Obvious create/update/delete mismatch
            # ----------------------------------------------------------

            if action == "create":
                # Keep creation operations and discovery operations that
                # may be required before creation.
                if method == "DELETE":
                    continue

            elif action == "delete":
                if method != "DELETE":
                    continue

            elif action == "update":
                if method not in ("PATCH", "PUT"):
                    continue

            elif action in ("retrieve", "check"):
                if method == "DELETE":
                    continue

            # ----------------------------------------------------------
            # Basic activity relevance
            # ----------------------------------------------------------

            entities = getattr(intent, "entities", []) or []

            entity_values: list[str] = []

            for entity in entities:
                if hasattr(entity, "value"):
                    value = getattr(entity, "value", None)
                elif isinstance(entity, dict):
                    value = entity.get("value")
                else:
                    value = None

                if value is not None:
                    entity_values.append(str(value).lower())

            # If an entity is explicitly present and the candidate has
            # completely unrelated content, only reject when we have a
            # strong obvious mismatch.
            #
            # We deliberately do not require entity words to occur in the
            # operation because CAMARA APIs often use technical terminology
            # rather than the user's wording.
            _ = entity_values
            _ = text

            filtered.append(candidate)

        return filtered

    # ------------------------------------------------------------------
    # API aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_api_candidates(
        operations: list[CandidateOperation],
    ) -> list[CandidateAPI]:
        grouped: dict[str, CandidateAPI] = {}

        for operation in operations:
            api = grouped.get(operation.api_name)

            if api is None:
                api = CandidateAPI(
                    name=operation.api_name,
                    score=operation.score,
                    sources=operation.sources.copy(),
                    operation_ids=[operation.operation_id],
                    reason=operation.reason,
                )
                grouped[operation.api_name] = api
                continue

            api.operation_ids.append(operation.operation_id)

            if operation.score > api.score:
                api.score = operation.score

            for source in operation.sources:
                if source not in api.sources:
                    api.sources.append(source)

        return sorted(
            grouped.values(),
            key=lambda api: api.score,
            reverse=True,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_agent: APISelectionAgent | None = None


def get_api_selection_agent(
    settings: Settings | None = None,
) -> APISelectionAgent:
    """Return the cached API Selection Agent."""

    global _agent

    if _agent is None:
        _agent = APISelectionAgent.from_settings(
            settings=settings,
        )

    return _agent