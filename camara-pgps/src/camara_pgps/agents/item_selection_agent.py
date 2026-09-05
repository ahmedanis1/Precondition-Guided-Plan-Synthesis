"""Item Selection agent.

Some CAMARA operations resolve one fact by returning a *list* of candidate
resources rather than a single one - ``retrieveSessionsByDevice`` returns
every active QoS session for a device, ``getQosAssignmentByDevice`` every
active assignment. When PGPS chains such a "discovery" operation in front of
one that needs a specific id from it (e.g. ``extendQosSessionDuration``
needing ``qos.session.id``), something has to decide *which* item in that
list the id should come from - "extend my gaming session" needs the session
whose profile matches "gaming", not just whichever the API happened to list
first.

This agent makes that choice with an LLM, using the same grounding
discipline as :class:`~camara_pgps.agents.api_selection_agent.
APISelectionAgent`: it may only return the index of an item that was
actually in the candidate list, and a slow/unreachable/invalid LLM response
degrades to a deterministic fallback (the first item) rather than failing
execution - a plan should never break just because the disambiguation step
couldn't run.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from camara_pgps.agents._llm import build_chat_model, chat_content, extract_json
from camara_pgps.config import Settings

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are choosing which item from a list best matches the user's request.

Rules:
- You MUST choose exactly one item by its index (0-based) from the candidates given.
- NEVER invent an item or an index that is not in the candidate list.
- If nothing in the request distinguishes the items, choose index 0.
- Do NOT explain your reasoning outside the JSON.

Return JSON only: {"selected_index": <int>, "reason": "<short explanation>"}"""


class ItemSelection(BaseModel):
    """Structured output expected from the LLM."""

    selected_index: int = 0
    reason: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ItemSelectionAgent:
    """Pick the best-matching item out of a list of API response items."""

    def __init__(self, *, llm: Any | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.llm = (
            llm
            if llm is not None
            else build_chat_model(self.settings, purpose="item selection agent")
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> ItemSelectionAgent:
        settings = settings or Settings()
        return cls(settings=settings)

    def select(self, *, query: str, items: list[dict[str, Any]]) -> int:
        """Return the index into ``items`` that best matches ``query``.

        Always a valid index (falls back to ``0`` - the API's own ordering -
        whenever there is nothing to disambiguate, the LLM is unavailable, or
        its response can't be trusted), so callers can always safely do
        ``items[selected]`` without an extra bounds check.
        """

        if not items:
            raise ValueError("select() requires at least one item")
        if len(items) == 1 or self.llm is None:
            return 0

        candidate_text = "\n".join(f"{index}: {item}" for index, item in enumerate(items))
        prompt_body = f"""
USER REQUEST
------------
{query}

CANDIDATES
----------
{candidate_text}
"""

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            response = self.llm.invoke(
                [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=prompt_body),
                ]
            )
            data = extract_json(chat_content(response))
            selection = ItemSelection.model_validate(data)
        except Exception as exc:  # noqa: BLE001 - never fail execution over this
            logger.warning(
                "Item selection LLM call failed (%s: %s); defaulting to the first item",
                type(exc).__name__,
                exc,
            )
            return 0

        if 0 <= selection.selected_index < len(items):
            return selection.selected_index

        logger.warning(
            "Item selection model returned an out-of-range index (%s of %d candidates); "
            "defaulting to the first item",
            selection.selected_index,
            len(items),
        )
        return 0


__all__ = ["ItemSelectionAgent", "ItemSelection"]
