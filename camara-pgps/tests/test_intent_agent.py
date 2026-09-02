"""Tests for the Intent Understanding agent (milestone 7)."""

from __future__ import annotations

import pytest

from camara_pgps.agents import (
    IntentAgent,
    IntentDraft,
    LLMIntentParser,
    RuleBasedIntentParser,
)
from camara_pgps.agents.intent_agent import _extract_json
from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import PROJECT_ROOT
from camara_pgps.models import Intent

REGISTRY_FILE = PROJECT_ROOT / "data" / "registry" / "camara_apis.json"

VIDEO_CALL = "My video call keeps freezing and I want better network quality."


@pytest.fixture(scope="module")
def registry() -> CamaraRegistry:
    return CamaraRegistry.from_file(REGISTRY_FILE)


@pytest.fixture
def agent(registry: CamaraRegistry) -> IntentAgent:
    return IntentAgent(registry=registry)  # no llm -> rule-based parser


class FakeChatModel:
    """Minimal stand-in for a LangChain chat model."""

    def __init__(self, response: str | Exception) -> None:
        self._response = response
        self.calls: list[list] = []

    def invoke(self, messages: list) -> object:
        self.calls.append(messages)
        if isinstance(self._response, Exception):
            raise self._response

        class _Resp:
            content = self._response

        return _Resp()


# --------------------------------------------------------------------------- #
# core behaviour
# --------------------------------------------------------------------------- #

def test_converts_text_into_structured_intent(agent: IntentAgent) -> None:
    intent = agent.understand(VIDEO_CALL)

    assert isinstance(intent, Intent)
    assert intent.text == VIDEO_CALL.strip()
    assert intent.goal  # non-empty paraphrase
    assert intent.required_capabilities == ["qos"]
    assert intent.target_domains == ["quality-on-demand"]
    assert intent.known_facts == []
    assert "device.identifier" in intent.missing_information
    assert "qos.profile" in intent.missing_information
    assert intent.metadata["parser"] == "RuleBasedIntentParser"


def test_to_understanding_matches_the_milestone_output_shape(agent: IntentAgent) -> None:
    understanding = agent.understand(VIDEO_CALL).to_understanding()
    assert set(understanding) == {
        "goal",
        "entities",
        "constraints",
        "required_capabilities",
        "known_facts",
        "missing_information",
    }
    assert understanding["required_capabilities"] == ["qos"]
    assert isinstance(understanding["entities"], dict)


def test_agent_does_not_decide_api_ordering(agent: IntentAgent) -> None:
    intent = agent.understand("quality on demand session for streaming, then check it")

    dumped = intent.model_dump()
    for planning_field in ("steps", "plan", "operations", "api_sequence", "order"):
        assert planning_field not in dumped

    # capabilities are coarse families from the closed vocabulary, not operations
    families = {c.split(".", 1)[0] for c in registry_capabilities()}
    assert set(intent.required_capabilities) <= families
    assert all("." not in cap for cap in intent.required_capabilities)


def registry_capabilities() -> list[str]:
    return CamaraRegistry.from_file(REGISTRY_FILE).capabilities()


# --------------------------------------------------------------------------- #
# entities / facts
# --------------------------------------------------------------------------- #

def test_extracts_entities_and_derives_known_facts(agent: IntentAgent) -> None:
    intent = agent.understand(
        "Give my phone +1 202 555 0123 a reliable low latency connection"
    )
    assert intent.get_entity("phoneNumber") is not None
    assert intent.get_entity("phoneNumber").value == "+12025550123"
    assert "device.identifier" in intent.known_facts
    # a satisfied fact must not also be reported as missing
    assert "device.identifier" not in intent.missing_information
    assert intent.constraints  # "reliable low latency connection" style clause


def test_location_request_maps_to_device_location(agent: IntentAgent) -> None:
    intent = agent.understand("Where is the device located right now?")
    assert "device-location" in intent.required_capabilities
    assert "device-location" in intent.target_domains


def test_empty_text_is_rejected(agent: IntentAgent) -> None:
    with pytest.raises(ValueError):
        agent.understand("   ")


# --------------------------------------------------------------------------- #
# rule-based parser unit
# --------------------------------------------------------------------------- #

def test_rule_based_parser_matches_qos_from_symptoms(registry: CamaraRegistry) -> None:
    draft = RuleBasedIntentParser(registry).parse("the stream keeps buffering and lagging")
    assert isinstance(draft, IntentDraft)
    assert draft.required_capabilities == ["qos"]


# --------------------------------------------------------------------------- #
# LLM parser path
# --------------------------------------------------------------------------- #

def test_llm_parser_json_is_validated_against_registry(registry: CamaraRegistry) -> None:
    payload = (
        '```json\n'
        '{"goal": "improve call quality", "entities": {},'
        ' "constraints": ["low latency"],'
        ' "required_capabilities": ["qos", "teleport"],'
        ' "known_facts": [], "missing_information": ["qos.profile"]}\n'
        '```'
    )
    model = FakeChatModel(payload)
    agent = IntentAgent(llm=model, registry=registry)
    assert isinstance(agent.parser, LLMIntentParser)

    intent = agent.understand("my call is bad")
    assert model.calls  # the model was actually invoked
    assert intent.goal == "improve call quality"
    assert intent.constraints == ["low latency"]
    assert intent.required_capabilities == ["qos"]  # "teleport" dropped
    assert intent.metadata["dropped_capabilities"] == ["teleport"]
    assert intent.missing_information == ["qos.profile"]


def test_llm_parser_falls_back_to_rules_on_failure(registry: CamaraRegistry) -> None:
    agent = IntentAgent(llm=FakeChatModel(RuntimeError("boom")), registry=registry)
    intent = agent.understand("the video keeps freezing")
    assert isinstance(intent, Intent)
    assert intent.required_capabilities == ["qos"]


def test_extract_json_strips_fences_and_prose() -> None:
    raw = 'Here you go:\n```json\n{"goal": "x"}\n```\nHope that helps!'
    assert _extract_json(raw) == {"goal": "x"}
    with pytest.raises(ValueError):
        _extract_json("no json here")
