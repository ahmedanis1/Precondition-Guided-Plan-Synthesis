"""LLM agents. Each agent has a single, bounded responsibility.

* :mod:`camara_pgps.agents.intent_agent` - natural language -> structured Intent.
* :mod:`camara_pgps.agents.api_selection_agent` - Intent (+ RAG) -> registry-verified
  candidate APIs / operations.
* :mod:`camara_pgps.agents.parameter_agent` - request + candidate operations ->
  grounded parameter bindings (+ missing_information).
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
from camara_pgps.agents.parameter_agent import (
    LLMParameterExtractor,
    OperationParameters,
    ParameterExtraction,
    ParameterExtractionAgent,
    ParameterExtractor,
    RuleBasedParameterExtractor,
    get_parameter_agent,
)

__all__ = [
    "IntentAgent",
    "IntentDraft",
    "IntentParser",
    "LLMIntentParser",
    "RuleBasedIntentParser",
    "get_intent_agent",
    "ParameterExtractionAgent",
    "ParameterExtraction",
    "OperationParameters",
    "ParameterExtractor",
    "RuleBasedParameterExtractor",
    "LLMParameterExtractor",
    "get_parameter_agent",
    "APISelectionAgent",
    "APISelection",
    "CandidateAPI",
    "CandidateOperation",
    "RejectedSuggestion",
    "SelectionSource",
    "LLMApiSelector",
    "get_api_selection_agent",
]
