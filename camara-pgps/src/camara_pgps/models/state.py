"""Orchestration state.

:class:`AgentState` is the single object threaded through the LangGraph pipeline
(milestone 15). Each stage reads the fields it needs and writes its output back:
Intent Understanding sets ``intent``, RAG Retrieval sets ``rag`` / ``retrieval``,
API Selection sets ``selection`` / ``selected_operations``, Parameter Extraction
sets ``parameters``, the planner sets ``plan_request`` and ``plan``, and the
Execution Layer sets ``execution``. If information is missing the graph branches
to a clarification step and fills ``clarification``.

``rag`` / ``selection`` / ``parameters`` are typed ``Any`` on purpose: their
concrete types live in higher layers (``retrieval`` / ``agents``) that must not
be imported by ``models``.

``messages`` and ``errors`` are append-only: they use ``operator.add`` reducers
so LangGraph accumulates writes from concurrent branches instead of overwriting.
"""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, Field

from camara_pgps.models.api import CamaraOperation
from camara_pgps.models.execution import ExecutionResult
from camara_pgps.models.intent import Intent
from camara_pgps.models.plan import PlanRequest, PlanResult
from camara_pgps.models.retrieval import RetrievalResult


class PipelineStage(StrEnum):
    """The stage the orchestration graph last completed."""

    RECEIVED = "received"
    INTENT = "intent"
    RETRIEVAL = "retrieval"
    SELECTION = "selection"
    PARAMETERS = "parameters"
    PLANNING = "planning"
    CLARIFICATION = "clarification"
    VALIDATION = "validation"
    EXECUTION = "execution"
    DONE = "done"
    FAILED = "failed"


class AgentState(BaseModel):
    """Shared state for the end-to-end intent -> plan -> execution pipeline."""

    request_text: str
    stage: PipelineStage = PipelineStage.RECEIVED
    intent: Intent | None = None
    rag: Any | None = None
    retrieval: RetrievalResult | None = None
    selection: Any | None = None
    selected_operations: list[CamaraOperation] = Field(default_factory=list)
    parameters: Any | None = None
    supplied_arguments: dict[str, Any] = Field(default_factory=dict)
    plan_request: PlanRequest | None = None
    plan: PlanResult | None = None
    clarification: list[str] = Field(default_factory=list)
    execution: ExecutionResult | None = None
    facts: dict[str, Any] = Field(default_factory=dict)
    messages: Annotated[list[dict[str, Any]], operator.add] = Field(default_factory=list)
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
