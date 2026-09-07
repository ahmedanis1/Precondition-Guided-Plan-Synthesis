"""Precondition / Alternative-Path Agent.

PGPS's planner only does one kind of reasoning: pure backward chaining over a
*fixed* goal ("operation B needs fact F; operation A produces F; so A runs
before B"). That covers data dependencies, but it can't answer a different
kind of question that shows up constantly in real requests: *which* operation
is actually the goal, when that depends on a fact nobody has checked yet.

Two concrete examples that motivated this agent:

* "Ensure best call quality for my interview" - `createSession` and
  `extendQosSessionDuration` are not a dependency chain, they're
  **alternatives**: whichever one is right depends on a runtime fact (does an
  active QoS session already exist for this device?) that a *different*
  operation (`retrieveSessionsByDevice`) can discover. A network operator
  will not let a device hold two overlapping QoS sessions - a model doesn't
  need that hand-encoded per API to know it, it's exactly the kind of
  domain knowledge a general-purpose telecom-literate LLM already has.
* "Increase my speed if I'm at Frankfurt" - `createSession` should not run
  at all unless `verifyLocation` first confirms the location. That's a
  **gate**, not a data dependency: the gated operation doesn't *consume* the
  verification result as an input parameter, its right to run at all is
  conditioned on it.

This agent does not call anything itself, and it does not decide anything by
itself either - it makes exactly one grounded judgement call, over the real
operations API Selection already found (never inventing an operation ID,
exactly like :class:`~camara_pgps.agents.api_selection_agent.LLMApiSelector`
never invents one), and hands back pure data describing what - if anything -
needs to be checked before the goal is genuinely settled. Everything that
happens next (actually running that check, deciding the branch from its real
result, planning and executing the rest) is the *existing* deterministic
planner/validator/executor pipeline; this agent changes what enters that
pipeline as the goal, never how the pipeline itself works.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from camara_pgps.agents._llm import build_chat_model, chat_content, extract_json
from camara_pgps.config import Settings, get_settings
from camara_pgps.models import CamaraOperation, Intent
from camara_pgps.pgps.conditions import ConditionModel

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """
You are a telecom network API expert (Deutsche Telekom / CAMARA APIs).

You are given a user's goal and a list of REAL candidate operations that
have already been retrieved for it. Decide whether anything needs to be
checked BEFORE committing to a final operation, in either of two ways:

(a) ALTERNATIVE - two candidates are different ways to reach the same goal,
    and which one is correct depends on a runtime fact that a *different*
    candidate operation can discover (e.g. whether a resource already
    exists). Use general domain knowledge here - for example, a telecom
    network will not let a device hold two overlapping sessions of the same
    kind, so a "create" operation and an "extend"/"modify" operation for the
    same kind of resource are normally alternatives, never both.

(b) GATE - a candidate operation should only run if another candidate
    confirms some fact first (a go/no-go check, e.g. verifying a location
    before acting on it because of it). The gated operation does not need
    the checked fact as an input parameter - it needs the fact to be true
    before it is allowed to run at all.

Rules:
- ONLY use operation_id values that literally appear in the candidate list
  below. Never invent one, and never reuse the same operation_id in two of
  the fields below.
- If nothing needs to be checked first, set "applies": false and leave every
  other field null.
- You are describing what should be checked and what it would mean - you are
  never performing the check and never guessing its outcome yourself.

Return JSON only, matching exactly this shape:
{
  "applies": true or false,
  "mode": "alternative" or "gate" or null,
  "resolver_operation": "<operation_id or null>",
  "if_resolved_operation": "<operation_id, alternative mode only, else null>",
  "if_unresolved_operation": "<operation_id, alternative mode only, else null>",
  "gated_operation": "<operation_id, gate mode only, else null>",
  "reason": "one short sentence"
}
"""

_MAX_DESCRIPTION_CHARS = 220


class PreconditionBrief(BaseModel):
    """The agent's grounded judgement. Pure data - describes, never executes."""

    applies: bool = False
    mode: Literal["alternative", "gate"] | None = None
    resolver_operation: str | None = None
    if_resolved_operation: str | None = None
    if_unresolved_operation: str | None = None
    gated_operation: str | None = None
    reason: str = ""


def _truncate(text: str | None, limit: int = _MAX_DESCRIPTION_CHARS) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


class PreconditionAgent:
    """Decides whether a check-then-branch or a gate applies to this request.

    Never blocks the pipeline: any failure (LLM unavailable, invalid output,
    an ungrounded operation_id) degrades to ``PreconditionBrief(applies=False)``
    - i.e. "nothing to check", which is exactly what the pipeline already does
    without this agent. This agent can only ever add reasoning, never remove
    the pipeline's ability to proceed.
    """

    def __init__(
        self,
        *,
        condition_model: ConditionModel,
        model: Any | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.condition_model = condition_model
        self.model = model or build_chat_model(self.settings, purpose="precondition agent")

    @classmethod
    def from_settings(
        cls,
        *,
        condition_model: ConditionModel,
        settings: Settings | None = None,
    ) -> "PreconditionAgent":
        settings = settings or get_settings()
        return cls(condition_model=condition_model, settings=settings)

    # -- public ---------------------------------------------------------

    def assess(
        self,
        *,
        intent: Intent,
        candidates: list[CamaraOperation],
    ) -> PreconditionBrief:
        by_id = {op.operation_id: op for op in candidates}
        if len(by_id) < 2:
            # nothing to check something *against* - an alternative or a
            # gate always needs at least two distinct real operations.
            return PreconditionBrief(applies=False)

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            human = self._format_candidates(intent, candidates)
            response = self.model.invoke(
                [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human)]
            )
            data = extract_json(chat_content(response))
            brief = PreconditionBrief.model_validate(data)
        except Exception as exc:  # noqa: BLE001 - must never block the pipeline
            logger.info("Precondition agent unavailable/unparsable (%s); skipping", exc)
            return PreconditionBrief(applies=False)

        return self._ground(brief, by_id)

    # -- internals --------------------------------------------------------

    def _format_candidates(self, intent: Intent, candidates: list[CamaraOperation]) -> str:
        lines = [
            f"USER GOAL: {getattr(intent, 'goal', '') or ''}",
            f"ACTION: {getattr(intent, 'action', '') or ''}",
            "",
            "CANDIDATE OPERATIONS",
            "---------------------",
        ]
        for op in candidates:
            conditions = self.condition_model.get(op.operation_id)
            requires = sorted(conditions.requires) if conditions else []
            produces = sorted(conditions.produces) if conditions else []
            lines.append(
                f"""
operation_id: {op.operation_id}
api_name: {getattr(op, "api_name", "") or ""}
method: {op.method}
path: {op.path}
description: {_truncate(op.description)}
requires: {requires}
produces: {produces}
"""
            )
        return "\n".join(lines)

    @staticmethod
    def _ground(brief: PreconditionBrief, by_id: dict[str, CamaraOperation]) -> PreconditionBrief:
        """Discard anything that isn't a real, distinct candidate operation_id.

        Same discipline as ``APISelectionAgent``'s validation of LLM output
        against its own candidate set - an agent allowed to reason freely
        over real data must never be allowed to *invent* an operation ID.
        """

        if not brief.applies or brief.mode is None:
            return PreconditionBrief(applies=False)

        if brief.resolver_operation not in by_id:
            return PreconditionBrief(applies=False)

        if brief.mode == "alternative":
            candidates_ok = (
                brief.if_resolved_operation in by_id
                and brief.if_unresolved_operation in by_id
                and len(
                    {
                        brief.resolver_operation,
                        brief.if_resolved_operation,
                        brief.if_unresolved_operation,
                    }
                )
                == 3
            )
            if not candidates_ok:
                return PreconditionBrief(applies=False)
            return PreconditionBrief(
                applies=True,
                mode="alternative",
                resolver_operation=brief.resolver_operation,
                if_resolved_operation=brief.if_resolved_operation,
                if_unresolved_operation=brief.if_unresolved_operation,
                reason=brief.reason,
            )

        if brief.mode == "gate":
            if brief.gated_operation not in by_id or brief.gated_operation == brief.resolver_operation:
                return PreconditionBrief(applies=False)
            return PreconditionBrief(
                applies=True,
                mode="gate",
                resolver_operation=brief.resolver_operation,
                gated_operation=brief.gated_operation,
                reason=brief.reason,
            )

        return PreconditionBrief(applies=False)


__all__ = ["PreconditionAgent", "PreconditionBrief"]
