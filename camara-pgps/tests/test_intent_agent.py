"""Tests for the Intent Understanding agent (LLM-only)."""

from __future__ import annotations

import pytest

from camara_pgps.agents import IntentAgent, IntentDraft, LLMIntentParser
from camara_pgps.models import Intent
from camara_pgps.models.intent import IntentConstraint


class FakeChatModel:
    """Minimal stand-in for a LangChain chat model."""

    def __init__(self, response: str | Exception) -> None:
        self._response = response
        self.calls: list = []

    def invoke(self, messages: list) -> object:
        self.calls.append(messages)
        if isinstance(self._response, Exception):
            raise self._response

        class _Resp:
            content = self._response

        return _Resp()


_GAMING_JSON = """\
```json
{
  "goal": "get better network quality for a game",
  "action": "improve",
  "entities": {"activity": "gaming", "phoneNumber": "+49123456789"},
  "constraints": ["duration = 7200 seconds", "quality = best"]
}
```"""


def test_llm_output_is_converted_to_an_intent() -> None:
    model = FakeChatModel(_GAMING_JSON)
    intent = IntentAgent(llm=model).understand("I want the best network for 2h of gaming")

    assert model.calls
    assert isinstance(intent, Intent)
    assert intent.goal == "get better network quality for a game"
    assert intent.action == "improve"
    assert intent.get_entity("activity").value == "gaming"
    assert intent.entity_values()["phoneNumber"] == "+49123456789"
    # bare-string constraints are coerced to IntentConstraint
    assert all(isinstance(c, IntentConstraint) for c in intent.constraints)
    assert {c.value for c in intent.constraints} == {
        "duration = 7200 seconds",
        "quality = best",
    }
    assert intent.metadata["parser"] == "LLMIntentParser"


def test_to_understanding_shape() -> None:
    intent = IntentAgent(llm=FakeChatModel(_GAMING_JSON)).understand("x")
    understanding = intent.to_understanding()
    assert set(understanding) == {"goal", "action", "entities", "constraints", "confidence"}
    assert understanding["entities"]["activity"] == "gaming"


def test_understand_raw_returns_the_draft() -> None:
    draft = IntentAgent(llm=FakeChatModel(_GAMING_JSON)).understand_raw("x")
    assert isinstance(draft, IntentDraft)
    assert draft.goal == "get better network quality for a game"
    assert draft.entities["activity"] == "gaming"


def test_requires_an_llm_or_parser() -> None:
    with pytest.raises(ValueError, match="requires an LLM"):
        IntentAgent()


def test_from_settings_defaults_to_ollama() -> None:
    from camara_pgps.agents._llm import OllamaChatModel

    agent = IntentAgent.from_settings()
    assert isinstance(agent.parser, LLMIntentParser)
    assert isinstance(agent.parser._model, OllamaChatModel)
    assert agent.parser._model.model == "qwen3:8b"


def test_from_settings_with_provider_none_raises() -> None:
    from camara_pgps.config import Settings

    with pytest.raises(ValueError, match="requires an LLM"):
        IntentAgent.from_settings(Settings(llm={"provider": "none"}))  # type: ignore[arg-type]


def test_empty_text_is_rejected() -> None:
    agent = IntentAgent(llm=FakeChatModel(_GAMING_JSON))
    with pytest.raises(ValueError, match="must not be empty"):
        agent.understand("   ")


def test_custom_parser_is_used() -> None:
    class StubParser:
        def parse(self, text: str) -> IntentDraft:
            return IntentDraft(goal=text, action=None, entities={}, constraints=[])

    intent = IntentAgent(parser=StubParser()).understand("do the thing")
    assert intent.goal == "do the thing"
    assert intent.entities == []
    assert intent.constraints == []


def test_llm_parser_extracts_json_from_prose() -> None:
    parser = LLMIntentParser(
        FakeChatModel('Sure!\n{"goal": "g", "action": null, "entities": {}, "constraints": []}')
    )
    draft = parser.parse("hello")
    assert draft.goal == "g"
    assert draft.action is None
