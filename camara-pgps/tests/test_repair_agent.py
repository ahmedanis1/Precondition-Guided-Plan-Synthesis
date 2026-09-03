"""Tests for the Clarification / Repair agent (milestone 16)."""

from __future__ import annotations

import re

import pytest

from camara_pgps.agents import (
    ClarificationRequest,
    MissingItem,
    ParameterExtractionAgent,
    RepairAgent,
)
from camara_pgps.agents.repair_agent import LLMPhraser
from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import PROJECT_ROOT
from camara_pgps.models import Intent
from camara_pgps.models.state import AgentState

REGISTRY_FILE = PROJECT_ROOT / "data" / "registry" / "camara_apis.json"


@pytest.fixture(scope="module")
def registry() -> CamaraRegistry:
    return CamaraRegistry.from_file(REGISTRY_FILE)


@pytest.fixture
def agent(registry: CamaraRegistry) -> RepairAgent:
    return RepairAgent(registry=registry)  # rule-based phraser


class FakeChatModel:
    def __init__(self, response: str) -> None:
        self._response = response

    def invoke(self, messages: list) -> object:
        class _Resp:
            content = self._response

        return _Resp()


# --------------------------------------------------------------------------- #
# the milestone example
# --------------------------------------------------------------------------- #

def test_turns_missing_names_into_one_question(agent: RepairAgent) -> None:
    intent = Intent(
        text="Improve my network quality.", goal="improve your network quality"
    )
    request = agent.clarify(["device_id", "duration"], intent=intent)

    assert isinstance(request, ClarificationRequest)
    assert request.question.count(".") == 1  # a single sentence
    assert "device identifier" in request.question
    assert "duration" in request.question
    assert "network quality" in request.question
    assert [m.name for m in request.missing] == ["device_id", "duration"]


# --------------------------------------------------------------------------- #
# never fills in a value
# --------------------------------------------------------------------------- #

def test_missing_item_has_no_value_field() -> None:
    assert "value" not in MissingItem.model_fields
    assert "value" not in ClarificationRequest.model_fields


def test_does_not_invent_values(agent: RepairAgent) -> None:
    request = agent.clarify(["duration", "qosProfile"], operations=["createSession"])
    dump = request.model_dump()
    for item in dump["missing"]:
        assert "value" not in item
    # nothing numeric was put where a value would go
    assert not re.search(r"\b\d{2,}\b", request.question)
    assert request.provided == {}


def test_llm_phrasing_that_answers_is_discarded(registry: CamaraRegistry) -> None:
    model = FakeChatModel("Sure - I'll assume 1 hour and use device +10000000000.")
    agent = RepairAgent(registry=registry, llm=model)
    assert isinstance(agent.phraser, LLMPhraser)

    request = agent.clarify(
        ["duration", "device"],
        intent=Intent(text="improve quality", goal="improve quality"),
        operations=["createSession"],
    )
    # the model tried to answer -> we fell back to the deterministic phraser
    assert "1 hour" not in request.question
    assert "assume" not in request.question.lower()
    assert {m.name for m in request.missing} == {"duration", "device"}


def test_good_llm_phrasing_is_used(registry: CamaraRegistry) -> None:
    model = FakeChatModel("Could you share the session duration you'd like?")
    agent = RepairAgent(registry=registry, llm=model)
    request = agent.clarify(["duration"], operations=["createSession"])
    assert request.question == "Could you share the session duration you'd like?"


# --------------------------------------------------------------------------- #
# enrichment from the registry
# --------------------------------------------------------------------------- #

def test_enum_values_are_surfaced(agent: RepairAgent) -> None:
    request = agent.clarify(["qosProfile"], operations=["createSession"])
    item = request.missing[0]
    assert item.enum == ["QOS_E", "QOS_S", "QOS_M", "QOS_L"]
    assert item.description == "Name of the QoS profile to apply."
    assert "one of" in request.question


def test_labels_are_deduplicated(agent: RepairAgent) -> None:
    request = agent.clarify(["device", "device.identifier"], operations=["createSession"])
    assert [m.label for m in request.missing] == ["device identifier"]


# --------------------------------------------------------------------------- #
# input shapes
# --------------------------------------------------------------------------- #

def test_accepts_agent_state(agent: RepairAgent) -> None:
    state = AgentState(
        request_text="give my device a session",
        intent=Intent(text="give my device a session", goal="set up a QoS session"),
        clarification=["qosProfile", "duration"],
    )
    request = agent.clarify(state)
    assert {m.name for m in request.missing} == {"qosProfile", "duration"}
    assert request.intent_goal == "set up a QoS session"


def test_accepts_parameter_extraction(registry: CamaraRegistry, agent: RepairAgent) -> None:
    extraction = ParameterExtractionAgent(registry=registry).extract(
        "improve my quality", ["createSession"]
    )
    request = agent.clarify(extraction)
    assert {"device identifier", "application server address", "QoS profile",
            "session duration"} <= {m.label for m in request.missing}


def test_empty_missing_gives_a_generic_prompt(agent: RepairAgent) -> None:
    request = agent.clarify([])
    assert request.missing == []
    assert "rephrase" in request.question.lower()


def test_rule_based_phrasing_is_deterministic(agent: RepairAgent) -> None:
    first = agent.clarify(["duration", "device"], operations=["createSession"])
    second = agent.clarify(["duration", "device"], operations=["createSession"])
    assert first.question == second.question
