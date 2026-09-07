"""Tests for the Parameter Extraction agent (milestone 9)."""

from __future__ import annotations

import pytest

from camara_pgps.agents import (
    ParameterExtraction,
    ParameterExtractionAgent,
    RuleBasedParameterExtractor,
)
from camara_pgps.agents.api_selection_agent import APISelection, CandidateOperation
from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import PROJECT_ROOT
from camara_pgps.models import BindingSource, Intent, IntentEntity

REGISTRY_FILE = PROJECT_ROOT / "data" / "registry" / "camara_apis.json"


@pytest.fixture(scope="module")
def registry() -> CamaraRegistry:
    return CamaraRegistry.from_file(REGISTRY_FILE)


@pytest.fixture
def agent(registry: CamaraRegistry) -> ParameterExtractionAgent:
    return ParameterExtractionAgent(registry=registry)  # rule-based


class FakeChatModel:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list = []

    def invoke(self, messages: list) -> object:
        self.calls.append(messages)

        class _Resp:
            content = self._response

        return _Resp()


# --------------------------------------------------------------------------- #
# the milestone example: report missing, never invent
# --------------------------------------------------------------------------- #

def test_missing_parameters_are_reported_not_invented(agent: ParameterExtractionAgent) -> None:
    result = agent.extract("Improve my network quality.", ["createSession"])

    assert isinstance(result, ParameterExtraction)
    create = result.for_operation("createSession")
    assert create is not None
    assert create.arguments == {}  # nothing was fabricated
    assert create.missing_required == [
        "device",
        "applicationServer",
        "qosProfile",
        "duration",
    ]
    assert create.is_complete is False
    assert result.missing_information == [
        "device",
        "applicationServer",
        "qosProfile",
        "duration",
    ]
    # every required param has an explicit UNRESOLVED binding, no value
    unresolved = [b for b in create.bindings if b.source == BindingSource.UNRESOLVED]
    assert {b.parameter for b in unresolved} >= set(create.missing_required)
    assert all(b.value is None for b in unresolved)


def test_extracts_grounded_values_from_the_request(agent: ParameterExtractionAgent) -> None:
    result = agent.extract(
        "Give my phone +1 202 555 0123 a QOS_L session for 30 minutes",
        ["createSession"],
    )
    create = result.for_operation("createSession")
    assert create is not None
    assert create.arguments["device"] == "+12025550123"
    assert create.arguments["qosProfile"] == "QOS_L"
    assert create.arguments["duration"] == 1800
    assert create.missing_required == ["applicationServer"]  # genuinely absent
    assert result.known_facts.get("qos.profile") == "QOS_L"


def test_enum_value_not_in_schema_is_rejected(agent: ParameterExtractionAgent) -> None:
    intent = Intent(
        text="set up a QOS_ULTRA session for my phone +12025550123",
        goal="set up a session",
        entities=[IntentEntity(name="qosProfile", value="QOS_ULTRA")],
    )
    result = agent.extract(intent, ["createSession"])
    create = result.for_operation("createSession")
    assert create is not None
    assert "qosProfile" not in create.arguments
    assert "qosProfile" in create.missing_required
    assert any("QOS_ULTRA" in note for note in result.notes)


def test_known_facts_from_prior_steps_fill_parameters(
    agent: ParameterExtractionAgent,
) -> None:
    result = agent.extract(
        "check my quality on demand session",
        ["getSession"],
        known_facts={"qos.session.id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"},
    )
    get_session = result.for_operation("getSession")
    assert get_session is not None
    assert get_session.arguments["sessionId"] == "3fa85f64-5717-4562-b3fc-2c963f66afa6"
    assert get_session.is_complete is True
    binding = next(b for b in get_session.bindings if b.parameter == "sessionId")
    assert binding.source == BindingSource.PRIOR_STEP


def test_accepts_an_api_selection(
    registry: CamaraRegistry, agent: ParameterExtractionAgent
) -> None:
    selection = APISelection(
        query="x",
        operations=[
            CandidateOperation(
                operation_id="createSession",
                api_name="quality-on-demand",
                operation=registry.require_operation("createSession"),
            )
        ],
    )
    result = agent.extract(Intent(text="improve quality", goal="improve quality"), selection)
    assert result.for_operation("createSession") is not None


# --------------------------------------------------------------------------- #
# LLM path + grounding guard
# --------------------------------------------------------------------------- #

def test_llm_values_must_be_grounded_in_the_request(registry: CamaraRegistry) -> None:
    # duration 999999 is NOT in the request text -> must be dropped
    payload = (
        '{"createSession": {"device": "+12025550123", "qosProfile": "QOS_L",'
        ' "duration": 999999}}'
    )
    agent = ParameterExtractionAgent(registry=registry, llm=FakeChatModel(payload))
    result = agent.extract(
        "give my phone +1 202 555 0123 a QOS_L session", ["createSession"]
    )
    create = result.for_operation("createSession")
    assert create is not None
    assert create.arguments.get("device") == "+12025550123"
    assert create.arguments.get("qosProfile") == "QOS_L"
    assert "duration" not in create.arguments
    assert "duration" in create.missing_required
    assert any("ungrounded" in note and "duration" in note for note in result.notes)


def test_llm_cannot_fabricate_an_identifier(registry: CamaraRegistry) -> None:
    payload = '{"createSession": {"device": "SOME_MADE_UP_DEVICE_ID"}}'
    agent = ParameterExtractionAgent(registry=registry, llm=FakeChatModel(payload))
    result = agent.extract("please improve my connection", ["createSession"])
    create = result.for_operation("createSession")
    assert create is not None
    assert "device" not in create.arguments
    assert "device" in create.missing_required


def test_metadata_reports_extractor(agent: ParameterExtractionAgent) -> None:
    result = agent.extract("improve quality", ["createSession"])
    assert result.metadata["extractor"] == RuleBasedParameterExtractor.__name__
    assert result.metadata["operations"] == ["createSession"]


# --------------------------------------------------------------------------- #
# a place name is not a geofencing area (real Ollama-backed run regression)
# --------------------------------------------------------------------------- #

def test_a_place_name_does_not_ground_a_geofencing_area(
    agent: ParameterExtractionAgent,
) -> None:
    """Real bug from an Ollama-backed run of "Notify me whenever I leave the
    office.": the intent's own "location" entity ("office") was aliased
    straight into verifyLocation's ``area`` parameter, because
    ``_PARAM_ALIASES["area"]`` used to treat "location"/"place"/"region"/
    "zone" as synonyms for it. ``area`` is a structured geofencing circle
    (center coordinates + radius) - there is no way to derive one from a
    bare place name - so binding "office" into it produced a plan that
    looked complete but failed payload validation with a confusing
    "'office' is not of type 'object'" schema error instead of asking the
    user for real coordinates. Unlike "device" (a phone number *is*
    legitimately the value a device-object parameter needs, just not yet
    wrapped), a place name is never the right shape for an area, so the
    fix narrows the alias group rather than adding a blanket
    object/array-type guard - the latter was tried first and broke the
    device case above (`test_extracts_grounded_values_from_the_request`,
    `test_llm_values_must_be_grounded_in_the_request`).
    """
    intent = Intent(
        text="Notify me whenever I leave the office.",
        goal="Monitor departure from the office",
        action="monitor",
        entities=[IntentEntity(name="location", value="office")],
    )
    result = agent.extract(intent, ["verifyLocation"])
    verify = result.for_operation("verifyLocation")
    assert verify is not None
    assert "area" not in verify.arguments
    assert "area" in verify.missing_required
