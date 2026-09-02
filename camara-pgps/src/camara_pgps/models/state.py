"""Orchestration state.

:class:`AgentState` is the single object threaded through the LangGraph pipeline
(milestone 14). Each stage reads the fields it needs and writes its output back:
Intent Understanding sets ``intent``, RAG Retrieval sets ``retrieval``, API
Selection sets ``selected_operations``, the planner sets ``plan_request`` and
``plan``, and the Execution Layer sets ``execution``.

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
    VALIDATION = "validation"
    EXECUTION = "execution"
    DONE = "done"
    FAILED = "failed"


class AgentState(BaseModel):
    """Shared state for the end-to-end intent -> plan -> execution pipeline."""

    request_text: str
    stage: PipelineStage = PipelineStage.RECEIVED
    intent: Intent | None = None
    retrieval: RetrievalResult | None = None
    selected_operations: list[CamaraOperation] = Field(default_factory=list)
    plan_request: PlanRequest | None = None
    plan: PlanResult | None = None
    execution: ExecutionResult | None = None
    facts: dict[str, Any] = Field(default_factory=dict)
    messages: Annotated[list[dict[str, Any]], operator.add] = Field(default_factory=list)
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
