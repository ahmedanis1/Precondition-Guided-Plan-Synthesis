"""LLM agents. Each agent has a single, bounded responsibility.

* :mod:`camara_pgps.agents.intent_agent` - natural language -> structured Intent.
* :mod:`camara_pgps.agents.api_selection_agent` - Intent (+ RAG) -> registry-verified
  candidate APIs / operations.
"""

from camara_pgps.agents.api_selection_agent import (
    APISelection,
    APISelectionAgent,
    CandidateAPI,
    CandidateOperation,
    LLMApiSelector,
    RejectedSuggestion,
    SelectionSource,
    get_api_selection_agent,
)
from camara_pgps.agents.intent_agent import (
    IntentAgent,
    IntentDraft,
    IntentParser,
    LLMIntentParser,
    RuleBasedIntentParser,
    get_intent_agent,
)

__all__ = [
    "IntentAgent",
    "IntentDraft",
    "IntentParser",
    "LLMIntentParser",
    "RuleBasedIntentParser",
    "get_intent_agent",
    "APISelectionAgent",
    "APISelection",
    "CandidateAPI",
    "CandidateOperation",
    "RejectedSuggestion",
    "SelectionSource",
    "LLMApiSelector",
    "get_api_selection_agent",
]
