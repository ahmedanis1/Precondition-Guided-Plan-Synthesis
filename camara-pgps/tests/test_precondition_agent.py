"""Tests for the Precondition / Alternative-Path agent.

These focus on the one property that matters most for a component allowed to
reason freely with an LLM: it must never be able to point PGPS at an
operation that wasn't actually offered to it, exactly like
``APISelectionAgent``'s own grounding of its LLM selector.
"""

from __future__ import annotations

import pytest

from camara_pgps.agents.precondition_agent import PreconditionAgent, PreconditionBrief
from camara_pgps.models import CamaraOperation, HTTPMethod, Intent


class _StubConditionModel:
    def get(self, operation_id: str):  # noqa: ANN001 - test stub
        return None


class FakeChatModel:
    def __init__(self, response: str | Exception) -> None:
        self._response = response

    def invoke(self, messages: list) -> object:
        if isinstance(self._response, Exception):
            raise self._response

        class _Resp:
            content = self._response

        return _Resp()


def _op(operation_id: str) -> CamaraOperation:
    return CamaraOperation(
        operation_id=operation_id,
        method=HTTPMethod.POST,
        path=f"/{operation_id}",
        description=f"{operation_id} description",
        api_name="Quality-On-Demand",
    )


CREATE = _op("createSession")
EXTEND = _op("extendQosSessionDuration")
RETRIEVE = _op("retrieveSessionsByDevice")
INTENT = Intent(text="ensure best quality", goal="ensure best quality", action="ensure")


def _agent(response: str) -> PreconditionAgent:
    return PreconditionAgent(condition_model=_StubConditionModel(), model=FakeChatModel(response))


def test_fewer_than_two_candidates_never_calls_the_model() -> None:
    agent = PreconditionAgent(
        condition_model=_StubConditionModel(),
        model=FakeChatModel(RuntimeError("should not be called")),
    )
    brief = agent.assess(intent=INTENT, candidates=[CREATE])
    assert brief.applies is False


def test_grounded_alternative_is_accepted() -> None:
    agent = _agent(
        '{"applies": true, "mode": "alternative", '
        '"resolver_operation": "retrieveSessionsByDevice", '
        '"if_resolved_operation": "extendQosSessionDuration", '
        '"if_unresolved_operation": "createSession", '
        '"reason": "no overlapping sessions"}'
    )
    brief = agent.assess(intent=INTENT, candidates=[CREATE, EXTEND, RETRIEVE])
    assert brief.applies is True
    assert brief.mode == "alternative"
    assert brief.resolver_operation == "retrieveSessionsByDevice"
    assert brief.if_resolved_operation == "extendQosSessionDuration"
    assert brief.if_unresolved_operation == "createSession"


def test_grounded_gate_is_accepted() -> None:
    agent = _agent(
        '{"applies": true, "mode": "gate", '
        '"resolver_operation": "retrieveSessionsByDevice", '
        '"gated_operation": "createSession", '
        '"reason": "check first"}'
    )
    brief = agent.assess(intent=INTENT, candidates=[CREATE, RETRIEVE])
    assert brief.applies is True
    assert brief.mode == "gate"
    assert brief.gated_operation == "createSession"


@pytest.mark.parametrize(
    "response",
    [
        # invented an operation_id that was never offered as a candidate
        '{"applies": true, "mode": "alternative", "resolver_operation": "deleteEverything", '
        '"if_resolved_operation": "extendQosSessionDuration", "if_unresolved_operation": "createSession", '
        '"reason": "x"}',
        # reuses the same operation_id in two roles
        '{"applies": true, "mode": "alternative", "resolver_operation": "retrieveSessionsByDevice", '
        '"if_resolved_operation": "retrieveSessionsByDevice", "if_unresolved_operation": "createSession", '
        '"reason": "x"}',
        # gate points its resolver and target at the same operation
        '{"applies": true, "mode": "gate", "resolver_operation": "createSession", '
        '"gated_operation": "createSession", "reason": "x"}',
        # malformed JSON
        "not json at all",
    ],
)
def test_ungrounded_or_unparsable_output_is_discarded(response: str) -> None:
    agent = _agent(response)
    brief = agent.assess(intent=INTENT, candidates=[CREATE, EXTEND, RETRIEVE])
    assert brief == PreconditionBrief(applies=False)


def test_model_exception_degrades_to_a_no_op() -> None:
    agent = PreconditionAgent(
        condition_model=_StubConditionModel(),
        model=FakeChatModel(ConnectionError("no ollama here")),
    )
    brief = agent.assess(intent=INTENT, candidates=[CREATE, EXTEND])
    assert brief.applies is False
