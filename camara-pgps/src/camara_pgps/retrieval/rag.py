"""RAG retrieval: turn a user intent into a bundle of grounded CAMARA evidence.

Given an :class:`~camara_pgps.models.Intent` (or raw text) this returns the
material later stages need in order to *decide* what to do:

* relevant CAMARA documentation chunks (with provenance)
* relevant API operations - resolved against the registry, never invented
* relevant parameters
* relevant pre/postconditions

Architectural rule: **RAG retrieves information, it does not decide the execution
plan.** :class:`RagContext` therefore holds ranked *candidates* with scores and
supporting evidence - no ordering, no single selection, no plan. API Selection
(milestone 8) and the PGPS planner (milestone 11) consume this bundle.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import Settings, get_settings
from camara_pgps.models import (
    APIParameter,
    CamaraOperation,
    Condition,
    Intent,
    RetrievalQuery,
    RetrievalResult,
    RetrievedChunk,
)
from camara_pgps.retrieval.retriever import Retriever

_OPERATION_DOCTYPES = {"operation", "parameters", "responses"}


class RetrievedOperation(BaseModel):
    """A candidate operation backed by retrieved documentation."""

    operation: CamaraOperation
    api_name: str
    score: float
    supporting_chunks: list[str] = Field(default_factory=list)


class RetrievedParameter(BaseModel):
    """A parameter of a candidate operation."""

    api_name: str
    operation_id: str
    parameter: APIParameter


class RetrievedCondition(BaseModel):
    """A pre- or post-condition of a candidate operation."""

    api_name: str
    operation_id: str
    condition: Condition


class RagContext(BaseModel):
    """Everything RAG retrieved for one intent - evidence, not a decision."""

    query: str
    intent: Intent | None = None
    documentation: RetrievalResult
    apis: list[str] = Field(default_factory=list)
    operations: list[RetrievedOperation] = Field(default_factory=list)
    parameters: list[RetrievedParameter] = Field(default_factory=list)
    conditions: list[RetrievedCondition] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def operation_ids(self) -> list[str]:
        return [item.operation.operation_id for item in self.operations]

    def as_prompt_context(self, *, max_chunks: int = 6) -> str:
        """Render the bundle as plain text for an LLM prompt (read-only helper)."""

        lines: list[str] = [f"Query: {self.query}"]
        if self.apis:
            lines.append(f"Candidate APIs: {', '.join(self.apis)}")

        if self.operations:
            lines.append("\nCandidate operations (ranked, NOT a plan):")
            for op_item in self.operations:
                op = op_item.operation
                lines.append(
                    f"- {op.operation_id} [{op_item.api_name}] "
                    f"{op.method} {op.path} (score {op_item.score:.3f})"
                )
                if op.summary:
                    lines.append(f"    {op.summary}")

        if self.parameters:
            lines.append("\nParameters:")
            for param_item in self.parameters:
                param = param_item.parameter
                flag = "required" if param.required else "optional"
                lines.append(
                    f"- {param_item.operation_id}.{param.name} "
                    f"({param.location}, {flag})"
                )

        if self.conditions:
            lines.append("\nConditions:")
            for cond_item in self.conditions:
                cond = cond_item.condition
                lines.append(
                    f"- {cond_item.operation_id} [{cond.kind}] {cond.description}"
                )

        if self.documentation.chunks:
            lines.append("\nDocumentation excerpts:")
            for chunk in self.documentation.chunks[:max_chunks]:
                snippet = chunk.text.strip().replace("\n", " ")
                if len(snippet) > 240:
                    snippet = snippet[:240] + "…"
                lines.append(f"- ({chunk.score:.3f}) {snippet}")

        return "\n".join(lines)


class RagRetriever:
    """Intent -> :class:`RagContext`. Wraps the vector :class:`Retriever` and the registry."""

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        registry: CamaraRegistry | None = None,
        settings: Settings | None = None,
        top_k: int = 8,
        max_chunks: int = 20,
        max_operations: int = 6,
    ) -> None:
        settings = settings or get_settings()
        self.retriever = retriever or Retriever.from_settings(settings)
        self.registry = registry or CamaraRegistry.from_settings(settings)
        self.top_k = top_k
        self.max_chunks = max_chunks
        self.max_operations = max_operations

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RagRetriever:
        return cls(settings=settings or get_settings())

    # -- public --------------------------------------------------------------

    def retrieve(
        self,
        intent: Intent | str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RagContext:
        intent = self._as_intent(intent)
        k = top_k or self.top_k
        queries = self._build_queries(intent)

        chunks = self._retrieve_chunks(queries, k, filters)
        documentation = RetrievalResult(
            query=intent.text or intent.goal,
            chunks=chunks,
            metadata={"queries": queries, "top_k": k},
        )

        notes: list[str] = []
        op_score: dict[str, float] = {}
        op_support: dict[str, list[str]] = {}
        api_score: dict[str, float] = {}

        for chunk in chunks:
            doctype = str(chunk.metadata.get("document_type", ""))
            weight = 1.0 if doctype in _OPERATION_DOCTYPES else 0.5
            if chunk.api_name:
                api_score[chunk.api_name] = (
                    api_score.get(chunk.api_name, 0.0) + chunk.score * weight
                )
            operation_id = chunk.operation_id
            if not operation_id:
                continue
            if self.registry.get_operation(operation_id) is None:
                notes.append(
                    f"chunk {chunk.id} references unknown operation "
                    f"{operation_id!r}; skipped"
                )
                continue
            op_score[operation_id] = (
                op_score.get(operation_id, 0.0) + chunk.score * weight
            )
            op_support.setdefault(operation_id, []).append(chunk.id)

        ranked = sorted(op_score, key=lambda o: op_score[o], reverse=True)
        ranked = ranked[: self.max_operations]

        operations = [
            RetrievedOperation(
                operation=self.registry.require_operation(operation_id),
                api_name=self._api_of(operation_id),
                score=round(op_score[operation_id], 6),
                supporting_chunks=op_support.get(operation_id, []),
            )
            for operation_id in ranked
        ]

        parameters = self._collect_parameters(operations, intent)
        conditions = self._collect_conditions(operations)
        apis = self._rank_apis(api_score, operations)

        return RagContext(
            query=intent.text or intent.goal,
            intent=intent,
            documentation=documentation,
            apis=apis,
            operations=operations,
            parameters=parameters,
            conditions=conditions,
            notes=notes,
            metadata={"queries": queries, "chunk_count": len(chunks)},
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _as_intent(intent: Intent | str) -> Intent:
        if isinstance(intent, Intent):
            return intent
        text = str(intent)
        return Intent(text=text, goal=text)

    @staticmethod
    def _build_queries(intent: Intent) -> list[str]:
        candidates: list[str] = []
        if intent.goal:
            candidates.append(intent.goal)
        if intent.text and intent.text != intent.goal:
            candidates.append(intent.text)
        if intent.action:
            candidates.append(intent.action.replace("_", " "))
        candidates.extend(intent.constraints)
        candidates.extend(intent.target_domains)
        entity_values = [str(v) for v in intent.entity_values().values() if v is not None]
        if entity_values and intent.goal:
            candidates.append(f"{intent.goal} {' '.join(entity_values)}")

        seen: set[str] = set()
        queries: list[str] = []
        for candidate in candidates:
            cleaned = candidate.strip()
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                queries.append(cleaned)
        return queries[:6] or [intent.text or intent.goal or ""]

    def _retrieve_chunks(
        self,
        queries: Sequence[str],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        merged: dict[str, RetrievedChunk] = {}
        for query in queries:
            result = self.retriever.retrieve(
                RetrievalQuery(text=query, top_k=top_k, filters=filters or {})
            )
            for chunk in result.chunks:
                current = merged.get(chunk.id)
                if current is None or chunk.score > current.score:
                    merged[chunk.id] = chunk
        ordered = sorted(merged.values(), key=lambda c: c.score, reverse=True)
        return ordered[: self.max_chunks]

    def _api_of(self, operation_id: str) -> str:
        operation = self.registry.get_operation(operation_id)
        if operation is not None and operation.api_name:
            return operation.api_name
        return self.registry.api_of(operation_id)

    @staticmethod
    def _order_parameters(
        parameters: Sequence[APIParameter], intent: Intent
    ) -> list[APIParameter]:
        entity_names = {entity.name.lower() for entity in intent.entities}

        def sort_key(param: APIParameter) -> tuple[bool, bool, str]:
            return (
                not param.required,
                param.name.lower() not in entity_names,
                param.name.lower(),
            )

        return sorted(parameters, key=sort_key)

    def _collect_parameters(
        self, operations: Sequence[RetrievedOperation], intent: Intent
    ) -> list[RetrievedParameter]:
        seen: set[tuple[str, str]] = set()
        collected: list[RetrievedParameter] = []
        for item in operations:
            operation = item.operation
            for param in self._order_parameters(operation.parameters, intent):
                key = (operation.operation_id, param.name)
                if key in seen:
                    continue
                seen.add(key)
                collected.append(
                    RetrievedParameter(
                        api_name=item.api_name,
                        operation_id=operation.operation_id,
                        parameter=param,
                    )
                )
        return collected

    @staticmethod
    def _collect_conditions(
        operations: Sequence[RetrievedOperation],
    ) -> list[RetrievedCondition]:
        collected: list[RetrievedCondition] = []
        for item in operations:
            operation = item.operation
            for condition in (*operation.preconditions, *operation.postconditions):
                collected.append(
                    RetrievedCondition(
                        api_name=item.api_name,
                        operation_id=operation.operation_id,
                        condition=condition,
                    )
                )
        return collected

    @staticmethod
    def _rank_apis(
        api_score: dict[str, float], operations: Sequence[RetrievedOperation]
    ) -> list[str]:
        ranked = sorted(api_score, key=lambda a: api_score[a], reverse=True)
        for item in operations:
            if item.api_name and item.api_name not in ranked:
                ranked.append(item.api_name)
        return ranked


@lru_cache(maxsize=1)
def get_rag_retriever() -> RagRetriever:
    """Return the process-wide RAG retriever built from configured settings."""

    return RagRetriever.from_settings()


__all__ = [
    "RagRetriever",
    "RagContext",
    "RetrievedOperation",
    "RetrievedParameter",
    "RetrievedCondition",
    "get_rag_retriever",
]
