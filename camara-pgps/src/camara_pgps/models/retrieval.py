"""RAG retrieval models.

Produced by the RAG Retrieval component (milestone 6) over the CAMARA
knowledge base (vector store + knowledge graph) and consumed by API selection
and parameter extraction.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievalQuery(BaseModel):
    """A request for context from the CAMARA knowledge base."""

    text: str
    top_k: int = 5
    filters: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A single scored passage returned by retrieval.

    ``api_name`` / ``operation_id`` link the chunk back to the API models when
    the source document is a parsed CAMARA spec.
    """

    id: str
    text: str
    score: float
    source: str | None = None
    api_name: str | None = None
    operation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """The ranked set of chunks retrieved for a query."""

    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)

    def top(self, n: int) -> list[RetrievedChunk]:
        """Return the ``n`` highest-scoring chunks."""

        return sorted(self.chunks, key=lambda c: c.score, reverse=True)[:n]

    def operation_ids(self) -> list[str]:
        """Return the distinct operation ids referenced by the chunks, in order."""

        seen: dict[str, None] = {}
        for chunk in self.chunks:
            if chunk.operation_id and chunk.operation_id not in seen:
                seen[chunk.operation_id] = None
        return list(seen)
