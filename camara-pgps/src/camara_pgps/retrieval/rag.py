"""RAG retrieval for grounded CAMARA API evidence.

Given an :class:`~camara_pgps.models.Intent` (or raw text), this module
retrieves the CAMARA knowledge needed by downstream decision-making stages:

* relevant CAMARA documentation chunks
* relevant API candidates
* relevant API operations
* relevant parameters
* relevant pre/postconditions

Architectural rule:
    RAG retrieves evidence; it does not make execution decisions.

JSONL is the single source of truth for CAMARA API structure at runtime.
Operations are reconstructed directly from the structured JSONL payload
stored with the retrieved Chroma chunks.

The retrieval strategy is structure-aware:

    semantic retrieval
        -> identify relevant API(s)
        -> retrieve operation chunks for those API(s)
        -> reconstruct operations from JSONL
        -> return ranked candidates

API Selection and PGPS consume the resulting RagContext later.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

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
from camara_pgps.retrieval.documents import structured_to_operation
from camara_pgps.retrieval.retriever import Retriever


logger = logging.getLogger(__name__)


# Operation-level chunks contain information that is more directly useful
# for identifying executable CAMARA operations.
_OPERATION_DOCTYPES = {
    "operation",
    "parameters",
    "responses",
}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    """Coerce a JSONL list-of-objects field into plain dictionaries."""

    if not isinstance(value, list):
        return []

    return [
        dict(item)
        for item in value
        if isinstance(item, Mapping)
    ]


def _build_operation_filter(
    api_name: str,
    filters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ChromaDB filter for operation chunks belonging to one API.

    ChromaDB requires multiple metadata conditions to be combined using
    an explicit logical operator such as ``$and``.

    The ``api_name`` and ``chunk_type`` constraints are controlled by the
    structure-aware retrieval stage and therefore cannot be overridden by
    caller-provided filters.
    """

    conditions: list[dict[str, Any]] = [
        {"api_name": api_name},
        {"chunk_type": "operation"},
    ]

    for key, value in (filters or {}).items():
        if key in {"api_name", "chunk_type"}:
            continue

        conditions.append(
            {key: value}
        )

    return {
        "$and": conditions
    }


class RetrievedOperation(BaseModel):
    """A candidate CAMARA operation reconstructed from JSONL.

    JSONL is the runtime source of truth.

    The raw structured operation data is preserved so downstream stages such
    as parameter resolution can access:

    * method
    * path
    * parameters
    * request body
    * responses
    * schema references
    * relationships
    * required fields

    ``operation`` is also retained as a typed :class:`CamaraOperation` for
    downstream validation and PGPS compatibility.
    """

    operation_id: str
    api_name: str
    api_version: str | None = None

    method: str
    path: str

    summary: str | None = None
    description: str | None = None

    parameters: list[dict[str, Any]] = Field(
        default_factory=list
    )

    request_body: dict[str, Any] | None = None

    responses: list[dict[str, Any]] = Field(
        default_factory=list
    )

    relationships: list[dict[str, Any]] = Field(
        default_factory=list
    )

    tags: list[str] = Field(
        default_factory=list
    )

    deprecated: bool = False

    operation: CamaraOperation

    source: str | None = None
    score: float

    supporting_chunks: list[str] = Field(
        default_factory=list
    )


class RetrievedParameter(BaseModel):
    """A parameter belonging to a candidate operation."""

    api_name: str
    operation_id: str
    parameter: APIParameter


class RetrievedCondition(BaseModel):
    """A pre- or post-condition of a candidate operation."""

    api_name: str
    operation_id: str
    condition: Condition


class RagContext(BaseModel):
    """Everything RAG retrieved for one intent.

    This object contains evidence and candidates, not decisions.

    RAG does not:

    * select one API
    * select one operation
    * resolve parameter values
    * select a QoS profile
    * create an execution plan
    """

    query: str

    intent: Intent | None = None

    documentation: RetrievalResult

    apis: list[str] = Field(
        default_factory=list
    )

    operations: list[RetrievedOperation] = Field(
        default_factory=list
    )

    parameters: list[RetrievedParameter] = Field(
        default_factory=list
    )

    conditions: list[RetrievedCondition] = Field(
        default_factory=list
    )

    notes: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    def operation_ids(self) -> list[str]:
        """Return IDs of reconstructed candidate operations."""

        return [
            item.operation_id
            for item in self.operations
        ]

    def as_prompt_context(
        self,
        *,
        max_chunks: int = 6,
    ) -> str:
        """Render retrieved evidence as read-only LLM prompt context."""

        lines: list[str] = [
            f"Query: {self.query}"
        ]

        if self.apis:
            lines.append(
                f"Candidate APIs: {', '.join(self.apis)}"
            )

        if self.operations:
            lines.append(
                "\nCandidate operations (ranked, NOT a plan):"
            )

            for op_item in self.operations:
                lines.append(
                    f"- {op_item.operation_id} "
                    f"[{op_item.api_name}] "
                    f"{op_item.method} "
                    f"{op_item.path} "
                    f"(score {op_item.score:.3f})"
                )

                provenance = [
                    bit
                    for bit in (
                        (
                            f"version {op_item.api_version}"
                            if op_item.api_version
                            else ""
                        ),
                        (
                            f"source {op_item.source}"
                            if op_item.source
                            else ""
                        ),
                    )
                    if bit
                ]

                if provenance:
                    lines.append(
                        f"    ({', '.join(provenance)})"
                    )

                if op_item.summary:
                    lines.append(
                        f"    {op_item.summary}"
                    )

        if self.parameters:
            lines.append(
                "\nParameters:"
            )

            for param_item in self.parameters:
                param = param_item.parameter

                flag = (
                    "required"
                    if param.required
                    else "optional"
                )

                lines.append(
                    f"- {param_item.operation_id}."
                    f"{param.name} "
                    f"({param.location}, {flag})"
                )

        if self.conditions:
            lines.append(
                "\nConditions:"
            )

            for cond_item in self.conditions:
                cond = cond_item.condition

                lines.append(
                    f"- {cond_item.operation_id} "
                    f"[{cond.kind}] "
                    f"{cond.description}"
                )

        if self.documentation.chunks:
            lines.append(
                "\nDocumentation excerpts:"
            )

            for chunk in self.documentation.chunks[
                :max_chunks
            ]:
                snippet = (
                    chunk.text
                    .strip()
                    .replace("\n", " ")
                )

                if len(snippet) > 240:
                    snippet = (
                        snippet[:240]
                        + "…"
                    )

                lines.append(
                    f"- ({chunk.score:.3f}) {snippet}"
                )

        return "\n".join(lines)


class RagRetriever:
    """Retrieve grounded CAMARA evidence for an intent.

    The retrieval pipeline is:

        intent
            -> semantic retrieval
            -> relevant API candidates
            -> API-constrained operation retrieval
            -> JSONL operation reconstruction
            -> candidate ranking
            -> RagContext

    JSONL is the single source of truth for CAMARA API structure.

    JSONL is the only runtime source of truth for API structure. The RAG
    retriever does not use a runtime CAMARA registry for operation lookup,
    validation, or rejection.
    """

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        settings: Settings | None = None,
        top_k: int = 8,
        # NOTE: max_chunks/max_operations used to be 20/6. Once structure-aware
        # retrieval (_retrieve_operation_chunks) legitimately widens to
        # max_candidate_apis APIs and pulls every one of their operation
        # chunks, those operations all compete for these same two global
        # caps *before* APISelectionAgent's own LLM selector - the component
        # actually designed to discriminate between close sibling operations
        # (e.g. createSession vs. extendQosSessionDuration under the same
        # Quality-On-Demand API) - ever gets to look at more than one of
        # them. A session observed exactly this: extendQosSessionDuration's
        # chunk consistently outscored createSession's for "ensure best
        # quality" - style phrasing, and createSession was cut before the
        # LLM ever saw it, even though the LLM's own selection prompt
        # already says not to prefer an operation that needs an identifier
        # (qos.session.id here) nothing has provided. Raising these two caps
        # to match/comfortably exceed APISelectionAgent's own max_candidates
        # (12) removes this project's small knowledge base as a source of
        # false negatives while leaving that agent's own cap as the actual,
        # intentional recall/precision tradeoff point.
        max_chunks: int = 40,
        max_operations: int = 12,
        max_candidate_apis: int = 5,
    ) -> None:
        settings = settings or get_settings()

        self.retriever = (
            retriever
            or Retriever.from_settings(settings)
        )

        self.top_k = top_k
        self.max_chunks = max_chunks
        self.max_operations = max_operations
        self.max_candidate_apis = max_candidate_apis

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
    ) -> RagRetriever:
        """Build a RAG retriever from application settings."""

        return cls(
            settings=settings or get_settings()
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def retrieve(
        self,
        intent: Intent | str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RagContext:
        """Retrieve grounded CAMARA evidence for an intent."""

        intent = self._as_intent(intent)

        k = top_k or self.top_k

        queries = self._build_queries(intent)

        # --------------------------------------------------------------
        # Stage 1: semantic retrieval
        # --------------------------------------------------------------

        semantic_chunks = self._retrieve_chunks(
            queries,
            k,
            filters,
        )

        # --------------------------------------------------------------
        # Stage 2: structure-aware operation expansion
        # --------------------------------------------------------------

        operation_chunks = self._retrieve_operation_chunks(
            queries=queries,
            semantic_chunks=semantic_chunks,
            top_k=k,
            filters=filters,
        )

        # --------------------------------------------------------------
        # Combine semantic evidence and operation evidence.
        # --------------------------------------------------------------

        chunks = self._merge_chunks(
            semantic_chunks,
            operation_chunks,
            max_chunks=self.max_chunks,
        )

        documentation = RetrievalResult(
            query=intent.text or intent.goal,
            chunks=chunks,
            metadata={
                "queries": queries,
                "top_k": k,
                "semantic_chunk_count": len(
                    semantic_chunks
                ),
                "operation_chunk_count": len(
                    operation_chunks
                ),
            },
        )

        notes: list[str] = []

        # Operation scoring.
        op_score: dict[str, float] = {}

        # Which chunks support each operation.
        op_support: dict[str, list[str]] = {}

        # Typed operation objects.
        op_cache: dict[
            str,
            CamaraOperation,
        ] = {}

        # Original structured JSONL payloads.
        op_structured: dict[
            str,
            dict[str, Any],
        ] = {}

        # Operation provenance.
        op_provenance: dict[
            str,
            dict[str, str | None],
        ] = {}

        # API relevance scores.
        api_score: dict[str, float] = {}

        # --------------------------------------------------------------
        # Convert retrieved chunks into structured operations
        # --------------------------------------------------------------

        for chunk in chunks:

            doctype = str(
                chunk.metadata.get(
                    "document_type"
                )
                or chunk.metadata.get(
                    "chunk_type"
                )
                or ""
            )

            # Operation-level chunks receive full weight.
            # Overview/schema/etc. receive lower weight.
            weight = (
                1.0
                if doctype in _OPERATION_DOCTYPES
                else 0.5
            )

            # ----------------------------------------------------------
            # API scoring
            # ----------------------------------------------------------

            if chunk.api_name:
                api_score[chunk.api_name] = (
                    api_score.get(
                        chunk.api_name,
                        0.0,
                    )
                    + chunk.score * weight
                )

            operation_id = chunk.operation_id

            if not operation_id:
                continue

            # ----------------------------------------------------------
            # Reconstruct each operation only once.
            # ----------------------------------------------------------

            if operation_id not in op_cache:

                data = self._parse_structured(
                    chunk
                )

                operation = (
                    self._operation_from_data(
                        data,
                        chunk.api_name,
                    )
                    if data
                    else None
                )

                if (
                    data is None
                    or operation is None
                ):
                    if (
                        chunk.metadata.get(
                            "structured"
                        )
                        is not None
                    ):
                        reason = (
                            "could not parse "
                            "its structured payload"
                        )
                    else:
                        reason = (
                            "carries no structured "
                            "payload"
                        )

                    notes.append(
                        f"chunk {chunk.id} "
                        f"for operation "
                        f"{operation_id!r} "
                        f"{reason}; skipped"
                    )

                    logger.warning(
                        "Skipping operation chunk %s: %s",
                        chunk.id,
                        reason,
                    )

                    continue

                op_cache[
                    operation_id
                ] = operation

                op_structured[
                    operation_id
                ] = data

                op_provenance[
                    operation_id
                ] = {
                    "api_name": (
                        operation.api_name
                        or chunk.api_name
                    ),
                    "version": (
                        chunk.metadata.get(
                            "version"
                        )
                        or chunk.metadata.get(
                            "api_version"
                        )
                        or data.get(
                            "api_version"
                        )
                    ),
                    "source": (
                        chunk.metadata.get(
                            "source"
                        )
                    ),
                }

            # ----------------------------------------------------------
            # Operation score
            # ----------------------------------------------------------

            op_score[operation_id] = (
                op_score.get(
                    operation_id,
                    0.0,
                )
                + chunk.score * weight
            )

            op_support.setdefault(
                operation_id,
                [],
            ).append(chunk.id)

        # --------------------------------------------------------------
        # Rank reconstructed operations
        # --------------------------------------------------------------

        ranked = sorted(
            op_score,
            key=lambda operation_id: (
                op_score[operation_id]
            ),
            reverse=True,
        )

        ranked = ranked[
            : self.max_operations
        ]

        operations: list[
            RetrievedOperation
        ] = []

        for operation_id in ranked:

            # Defensive check.
            if (
                operation_id not in op_cache
                or operation_id not in op_structured
                or operation_id not in op_provenance
            ):
                continue

            operations.append(
                self._build_retrieved_operation(
                    operation_id,
                    operation=op_cache[
                        operation_id
                    ],
                    structured=op_structured[
                        operation_id
                    ],
                    provenance=op_provenance[
                        operation_id
                    ],
                    score=round(
                        op_score[
                            operation_id
                        ],
                        6,
                    ),
                    supporting_chunks=(
                        op_support.get(
                            operation_id,
                            [],
                        )
                    ),
                )
            )

        # --------------------------------------------------------------
        # Extract downstream evidence
        # --------------------------------------------------------------

        parameters = self._collect_parameters(
            operations,
            intent,
        )

        conditions = self._collect_conditions(
            operations
        )

        apis = self._rank_apis(
            api_score,
            operations,
        )

        return RagContext(
            query=(
                intent.text
                or intent.goal
            ),
            intent=intent,
            documentation=documentation,
            apis=apis,
            operations=operations,
            parameters=parameters,
            conditions=conditions,
            notes=notes,
            metadata={
                "queries": queries,
                "chunk_count": len(chunks),
                "semantic_chunk_count": len(
                    semantic_chunks
                ),
                "operation_chunk_count": len(
                    operation_chunks
                ),
                "operation_count": len(
                    operations
                ),
                "api_count": len(
                    apis
                ),
                "candidate_api_limit": (
                    self.max_candidate_apis
                ),
            },
        )

    # ==================================================================
    # Intent helpers
    # ==================================================================

    @staticmethod
    def _as_intent(
        intent: Intent | str,
    ) -> Intent:
        """Convert raw text to a minimal Intent."""

        if isinstance(intent, Intent):
            return intent

        text = str(intent)

        return Intent(
            text=text,
            goal=text,
        )

    @staticmethod
    def _build_queries(
        intent: Intent,
    ) -> list[str]:
        """Build semantic retrieval queries from an intent."""

        candidates: list[str] = []

        if intent.goal:
            candidates.append(
                intent.goal
            )

        if (
            intent.text
            and intent.text != intent.goal
        ):
            candidates.append(
                intent.text
            )

        if intent.action:
            candidates.append(
                intent.action.replace(
                    "_",
                    " ",
                )
            )

        for constraint in intent.constraints:
            candidates.append(
                constraint.raw
                or (
                    f"{constraint.type} "
                    f"{constraint.value}"
                )
            )

        entity_values = [
            str(value)
            for value in (
                intent.entity_values()
                .values()
            )
            if value is not None
        ]

        if (
            entity_values
            and intent.goal
        ):
            candidates.append(
                f"{intent.goal} "
                f"{' '.join(entity_values)}"
            )

        seen: set[str] = set()

        queries: list[str] = []

        for candidate in candidates:

            cleaned = candidate.strip()

            if (
                cleaned
                and cleaned.lower()
                not in seen
            ):
                seen.add(
                    cleaned.lower()
                )

                queries.append(
                    cleaned
                )

        return (
            queries[:6]
            or [
                intent.text
                or intent.goal
                or ""
            ]
        )

    # ==================================================================
    # Vector retrieval
    # ==================================================================

    def _retrieve_chunks(
        self,
        queries: Sequence[str],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        """Perform normal semantic retrieval across the knowledge base."""

        merged: dict[
            str,
            RetrievedChunk,
        ] = {}

        for query in queries:

            result = self.retriever.retrieve(
                RetrievalQuery(
                    text=query,
                    top_k=top_k,
                    filters=filters or {},
                )
            )

            for chunk in result.chunks:

                current = merged.get(
                    chunk.id
                )

                if (
                    current is None
                    or chunk.score
                    > current.score
                ):
                    merged[
                        chunk.id
                    ] = chunk

        # NOTE: deliberately NOT truncated to ``self.max_chunks`` here. A
        # multi-query Intent (see ``_build_queries``) fans a single request
        # out into up to 6 separate retrieval queries - the paraphrased
        # goal, the raw text, the action, each constraint, and a
        # goal+entities combination. Each query's own top-``top_k`` results
        # are legitimate evidence, but merging all of them and cutting to a
        # single, small ``max_chunks`` window *before* choosing which APIs
        # get structure-aware operation expansion (below) let a handful of
        # near-duplicate, location-heavy queries (e.g. a goal restating an
        # entity, or a lone constraint) crowd out a real but modestly-scored
        # match from a single differently-worded query - concretely, a
        # session observed "increase my internet speed if i am at
        # frankfurt" retrieving Quality-On-Demand's createSession as a
        # top-3 candidate when queried with the raw text alone, but zero
        # operations at all once IntentAgent's structured Intent (goal
        # "Increase internet speed in Frankfurt", action "modify", entity
        # location=Frankfurt, constraint "improve connectivity") fanned
        # that into 5 queries - the createSession evidence chunk simply
        # fell out of the merged top-``max_chunks`` before its API ever got
        # a chance to be scored as a structure-aware candidate.
        #
        # Returning the full per-query union here lets every query
        # contribute to API relevance scoring with full breadth; the
        # eventual chunk list actually kept for documentation/evidence is
        # still capped by ``_merge_chunks(..., max_chunks=self.max_chunks)``
        # at the end of ``retrieve()``, after structure-aware operation
        # chunks have had a chance to add back the operations that
        # mattered.
        return sorted(
            merged.values(),
            key=lambda chunk: chunk.score,
            reverse=True,
        )

    def _retrieve_operation_chunks(
        self,
        *,
        queries: Sequence[str],
        semantic_chunks: Sequence[RetrievedChunk],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        """Retrieve operation chunks for semantically relevant APIs.

        This is the structure-aware second retrieval stage.

        First, semantic retrieval identifies candidate APIs.

        Then, for each candidate API, Chroma is queried with a structural
        filter equivalent to:

            api_name=<candidate API>
            AND
            chunk_type=operation

        ChromaDB requires multiple metadata conditions to be wrapped in
        ``$and``.

        JSONL remains the source of truth because the operation is later
        reconstructed from the chunk's ``structured`` payload.
        """

        # --------------------------------------------------------------
        # Calculate API relevance from semantic retrieval.
        #
        # This prevents operation expansion for every API appearing in
        # the semantic result set.
        # --------------------------------------------------------------

        api_scores: dict[str, float] = {}

        for chunk in semantic_chunks:

            if not chunk.api_name:
                continue

            doctype = str(
                chunk.metadata.get(
                    "document_type"
                )
                or chunk.metadata.get(
                    "chunk_type"
                )
                or ""
            )

            weight = (
                1.0
                if doctype in _OPERATION_DOCTYPES
                else 0.5
            )

            api_scores[chunk.api_name] = (
                api_scores.get(
                    chunk.api_name,
                    0.0,
                )
                + chunk.score * weight
            )

        candidate_apis = [
            api_name
            for api_name, _score in sorted(
                api_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )[
                : self.max_candidate_apis
            ]
        ]

        if not candidate_apis:
            return []

        logger.debug(
            "Structure-aware RAG candidate APIs: %s",
            candidate_apis,
        )

        merged: dict[
            str,
            RetrievedChunk,
        ] = {}

        # --------------------------------------------------------------
        # Retrieve operation chunks for each candidate API.
        # --------------------------------------------------------------

        for api_name in candidate_apis:

            operation_filters = (
                _build_operation_filter(
                    api_name,
                    filters,
                )
            )

            for query in queries:

                result = self.retriever.retrieve(
                    RetrievalQuery(
                        text=query,

                        # Operation retrieval should have enough candidates
                        # even when the normal semantic top-k is small.
                        top_k=max(
                            top_k,
                            10,
                        ),

                        filters=operation_filters,
                    )
                )

                for chunk in result.chunks:

                    # Structural filtering should already guarantee this,
                    # but keep this defensive check.
                    if not chunk.operation_id:
                        continue

                    if (
                        chunk.api_name
                        != api_name
                    ):
                        continue

                    if (
                        chunk.metadata.get(
                            "chunk_type"
                        )
                        != "operation"
                    ):
                        continue

                    current = merged.get(
                        chunk.id
                    )

                    if (
                        current is None
                        or chunk.score
                        > current.score
                    ):
                        merged[
                            chunk.id
                        ] = chunk

        return sorted(
            merged.values(),
            key=lambda chunk: chunk.score,
            reverse=True,
        )

    @staticmethod
    def _merge_chunks(
        semantic_chunks: Sequence[RetrievedChunk],
        operation_chunks: Sequence[RetrievedChunk],
        *,
        max_chunks: int,
    ) -> list[RetrievedChunk]:
        """Merge semantic and structure-aware retrieval results."""

        merged: dict[
            str,
            RetrievedChunk,
        ] = {}

        for chunk in semantic_chunks:
            merged[
                chunk.id
            ] = chunk

        for chunk in operation_chunks:

            current = merged.get(
                chunk.id
            )

            if (
                current is None
                or chunk.score
                > current.score
            ):
                merged[
                    chunk.id
                ] = chunk

        return sorted(
            merged.values(),
            key=lambda chunk: chunk.score,
            reverse=True,
        )[
            :max_chunks
        ]

    # ==================================================================
    # Structured JSONL reconstruction
    # ==================================================================

    @staticmethod
    def _parse_structured(
        chunk: RetrievedChunk,
    ) -> dict[str, Any] | None:
        """Extract the structured JSONL payload from a retrieved chunk.

        Expected primary location:

            chunk.metadata["structured"]

        Compatibility locations are also supported.
        """

        raw = chunk.metadata.get(
            "structured"
        )

        # Compatibility fallback.
        if raw is None:
            raw = chunk.metadata.get(
                "structured_data"
            )

        # Compatibility fallback for nested metadata.
        if raw is None:

            structured_container = (
                chunk.metadata.get(
                    "data"
                )
            )

            if isinstance(
                structured_container,
                Mapping,
            ):
                raw = structured_container.get(
                    "structured"
                )

        if raw is None:
            logger.debug(
                "No structured payload for chunk %s",
                chunk.id,
            )
            return None

        # Chroma normally returns the JSON payload as a string because
        # VectorStore.add() serializes structured data for metadata storage.
        if isinstance(raw, str):

            try:
                data = json.loads(
                    raw
                )

            except (
                ValueError,
                TypeError,
            ) as exc:

                logger.warning(
                    "Could not parse structured JSON for chunk %s: %s",
                    chunk.id,
                    exc,
                )

                return None

        else:
            data = raw

        if not isinstance(
            data,
            Mapping,
        ):
            logger.warning(
                "Structured payload for chunk %s is not a mapping",
                chunk.id,
            )

            return None

        data = dict(data)

        if not data.get(
            "operation_id"
        ):
            logger.warning(
                "Structured payload for chunk %s has no operation_id",
                chunk.id,
            )

            return None

        return data

    @staticmethod
    def _operation_from_data(
        data: Mapping[str, Any],
        api_name: str | None,
    ) -> CamaraOperation | None:
        """Convert structured JSONL data into CamaraOperation."""

        try:

            return structured_to_operation(
                data,
                api_name=api_name,
            )

        except ValueError as exc:

            logger.warning(
                "Could not reconstruct operation %s for API %s: %s",
                data.get("operation_id"),
                api_name,
                exc,
            )

            return None

    # ==================================================================
    # Retrieved operation construction
    # ==================================================================

    @staticmethod
    def _build_retrieved_operation(
        operation_id: str,
        *,
        operation: CamaraOperation,
        structured: Mapping[str, Any],
        provenance: Mapping[
            str,
            str | None,
        ],
        score: float,
        supporting_chunks: list[str],
    ) -> RetrievedOperation:
        """Build a RetrievedOperation preserving JSONL structure."""

        request_body = structured.get(
            "request_body"
        )

        return RetrievedOperation(
            operation_id=operation_id,

            api_name=(
                provenance.get(
                    "api_name"
                )
                or operation.api_name
                or ""
            ),

            api_version=provenance.get(
                "version"
            ),

            method=str(
                structured.get(
                    "method"
                )
                or operation.method
            ),

            path=str(
                structured.get(
                    "path"
                )
                or operation.path
            ),

            summary=(
                structured.get(
                    "summary"
                )
                or operation.summary
            ),

            description=(
                structured.get(
                    "description"
                )
                or operation.description
            ),

            parameters=_dict_list(
                structured.get(
                    "parameters"
                )
            ),

            request_body=(
                dict(request_body)
                if isinstance(
                    request_body,
                    Mapping,
                )
                else None
            ),

            responses=_dict_list(
                structured.get(
                    "responses"
                )
            ),

            relationships=_dict_list(
                structured.get(
                    "relationships"
                )
            ),

            tags=[
                str(tag)
                for tag in (
                    structured.get(
                        "tags"
                    )
                    or []
                )
            ],

            deprecated=bool(
                structured.get(
                    "deprecated",
                    False,
                )
            ),

            operation=operation,

            source=provenance.get(
                "source"
            ),

            score=score,

            supporting_chunks=(
                supporting_chunks
            ),
        )

    # ==================================================================
    # Parameters
    # ==================================================================

    @staticmethod
    def _order_parameters(
        parameters: Sequence[
            APIParameter
        ],
        intent: Intent,
    ) -> list[APIParameter]:
        """Order parameters by requiredness and intent relevance."""

        entity_names = {
            entity.name.lower()
            for entity in intent.entities
        }

        def sort_key(
            param: APIParameter,
        ) -> tuple[
            bool,
            bool,
            str,
        ]:
            return (
                not param.required,
                param.name.lower()
                not in entity_names,
                param.name.lower(),
            )

        return sorted(
            parameters,
            key=sort_key,
        )

    def _collect_parameters(
        self,
        operations: Sequence[
            RetrievedOperation
        ],
        intent: Intent,
    ) -> list[
        RetrievedParameter
    ]:
        """Collect parameters from reconstructed operations."""

        seen: set[
            tuple[str, str]
        ] = set()

        collected: list[
            RetrievedParameter
        ] = []

        for item in operations:

            operation = item.operation

            for param in self._order_parameters(
                operation.parameters,
                intent,
            ):

                key = (
                    operation.operation_id,
                    param.name,
                )

                if key in seen:
                    continue

                seen.add(
                    key
                )

                collected.append(
                    RetrievedParameter(
                        api_name=item.api_name,
                        operation_id=(
                            operation.operation_id
                        ),
                        parameter=param,
                    )
                )

        return collected

    # ==================================================================
    # Conditions
    # ==================================================================

    @staticmethod
    def _collect_conditions(
        operations: Sequence[
            RetrievedOperation
        ],
    ) -> list[
        RetrievedCondition
    ]:
        """Collect pre/postconditions from reconstructed operations."""

        collected: list[
            RetrievedCondition
        ] = []

        for item in operations:

            operation = item.operation

            for condition in (
                *operation.preconditions,
                *operation.postconditions,
            ):

                collected.append(
                    RetrievedCondition(
                        api_name=item.api_name,
                        operation_id=(
                            operation.operation_id
                        ),
                        condition=condition,
                    )
                )

        return collected

    # ==================================================================
    # API ranking
    # ==================================================================

    @staticmethod
    def _rank_apis(
        api_score: dict[str, float],
        operations: Sequence[
            RetrievedOperation
        ],
    ) -> list[str]:
        """Rank candidate APIs by accumulated retrieval evidence."""

        ranked = sorted(
            api_score,
            key=lambda api: api_score[api],
            reverse=True,
        )

        # Ensure APIs represented by reconstructed operations are present,
        # even if their documentation chunk had a lower semantic score.
        for item in operations:

            if (
                item.api_name
                and item.api_name
                not in ranked
            ):
                ranked.append(
                    item.api_name
                )

        return ranked


@lru_cache(maxsize=1)
def get_rag_retriever() -> RagRetriever:
    """Return the process-wide RAG retriever."""

    return RagRetriever.from_settings()


__all__ = [
    "RagRetriever",
    "RagContext",
    "RetrievedOperation",
    "RetrievedParameter",
    "RetrievedCondition",
    "get_rag_retriever",
]