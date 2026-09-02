"""Smoke tests for the shared domain models (milestone 2)."""

from __future__ import annotations

from camara_pgps.models import (
    AgentState,
    APIParameter,
    BindingSource,
    CamaraAPI,
    CamaraOperation,
    Condition,
    ConditionKind,
    ExecutionResult,
    HTTPMethod,
    Intent,
    IntentEntity,
    ParameterBinding,
    ParameterLocation,
    PipelineStage,
    PlanRequest,
    PlanResult,
    PlanStatus,
    PlanStep,
    RetrievalResult,
    RetrievedChunk,
    StepExecution,
)


def _operation() -> CamaraOperation:
    return CamaraOperation(
        operation_id="retrieveLocation",
        method=HTTPMethod.POST,
        path="/location-retrieval/v0/retrieve",
        parameters=[
            APIParameter(name="device", location=ParameterLocation.BODY, required=True),
            APIParameter(name="maxAge", location=ParameterLocation.QUERY),
        ],
        preconditions=[
            Condition(
                kind=ConditionKind.PRECONDITION,
                description="an access token is available",
                requires=["auth.token"],
            )
        ],
        postconditions=[
            Condition(
                kind=ConditionKind.POSTCONDITION,
                description="the device location is known",
                produces=["device.location"],
            )
        ],
    )


def test_operation_helpers_and_schema_alias() -> None:
    op = _operation()
    assert [p.name for p in op.required_parameters()] == ["device"]
    assert op.get_parameter("maxAge").location is ParameterLocation.QUERY

    param = APIParameter.model_validate(
        {"name": "device", "location": "body", "schema": {"type": "object"}}
    )
    assert param.schema_ == {"type": "object"}
    assert param.model_dump(by_alias=True)["schema"] == {"type": "object"}


def test_camara_api_operation_lookup() -> None:
    api = CamaraAPI(name="device-location", version="0.3.0", operations=[_operation()])
    assert api.get_operation("retrieveLocation") is not None
    assert api.get_operation("missing") is None


def test_intent_entity_helpers() -> None:
    intent = Intent(
        text="where is my phone",
        goal="retrieve the location of the user's device",
        action="get_device_location",
        entities=[IntentEntity(name="device", value="+123456789", type="phoneNumber")],
    )
    assert intent.get_entity("device").value == "+123456789"
    assert intent.entity_values() == {"device": "+123456789"}


def test_retrieval_result_ordering() -> None:
    result = RetrievalResult(
        query="device location",
        chunks=[
            RetrievedChunk(id="a", text="...", score=0.2, operation_id="op1"),
            RetrievedChunk(id="b", text="...", score=0.9, operation_id="op2"),
            RetrievedChunk(id="c", text="...", score=0.5, operation_id="op1"),
        ],
    )
    assert len(result) == 3
    assert [c.id for c in result.top(2)] == ["b", "c"]
    assert result.operation_ids() == ["op1", "op2"]


def test_plan_request_and_result() -> None:
    intent = Intent(text="x", goal="y")
    request = PlanRequest(intent=intent, candidate_operations=[_operation()])
    assert request.max_steps == 20

    step = PlanStep(
        index=0,
        operation_id="retrieveLocation",
        method=HTTPMethod.POST,
        path="/location-retrieval/v0/retrieve",
        bindings=[
            ParameterBinding(parameter="device", source=BindingSource.INTENT, value="+1"),
            ParameterBinding(parameter="maxAge", source=BindingSource.UNRESOLVED),
        ],
        produces=["device.location"],
    )
    assert [b.parameter for b in step.unresolved_bindings()] == ["maxAge"]

    result = PlanResult(status=PlanStatus.VALID, steps=[step])
    assert result.is_executable is True
    assert result.get_step(0) is step


def test_execution_result_helpers() -> None:
    result = ExecutionResult(
        steps=[
            StepExecution(step_index=0, operation_id="a", succeeded=True),
            StepExecution(step_index=1, operation_id="b", succeeded=False, error="boom"),
        ]
    )
    assert result.failed_step.operation_id == "b"
    assert result.get_step(0).succeeded is True


def test_agent_state_defaults_and_log_reducers() -> None:
    state = AgentState(request_text="where is my phone")
    assert state.stage is PipelineStage.RECEIVED
    assert state.selected_operations == []

    # Annotated append-only fields carry an operator.add reducer for LangGraph.
    field = AgentState.model_fields["messages"]
    assert field.metadata  # reducer stored as field metadata
