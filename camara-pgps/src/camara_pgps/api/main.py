"""HTTP API for the CAMARA-PGPS agentic system.

    POST /api/v1/run        full pipeline (intent -> ... -> execution)
    POST /api/v1/plan       stop after PGPS planning
    POST /api/v1/validate   validate a supplied plan, or plan + validate a request
    POST /api/v1/execute    execute a supplied validated plan, or run a request end to end

    GET  /api/v1/apis                list the registered CAMARA APIs
    GET  /api/v1/apis/{api_name}     one API in full (operations, params, conditions)
    GET  /api/v1/health
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from camara_pgps.camara.registry import CamaraRegistry, get_registry
from camara_pgps.config import get_settings
from camara_pgps.models.plan import PlanResult
from camara_pgps.models.state import AgentState
from camara_pgps.workflow.graph import CamaraWorkflow, get_workflow

API_PREFIX = "/api/v1"


# --------------------------------------------------------------------------- #
# request bodies
# --------------------------------------------------------------------------- #

class RunRequest(BaseModel):
    request: str = Field(min_length=1, description="The natural-language request.")
    arguments: dict[str, dict[str, Any]] = Field(default_factory=dict)
    known_facts: dict[str, Any] = Field(default_factory=dict)


class PlanBasedRequest(BaseModel):
    request: str | None = None
    plan: dict[str, Any] | None = Field(
        default=None, description="A PlanResult to validate / execute directly."
    )
    arguments: dict[str, dict[str, Any]] = Field(default_factory=dict)
    known_facts: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _need_request_or_plan(self) -> PlanBasedRequest:
        if not self.request and self.plan is None:
            raise ValueError("provide either 'request' or 'plan'")
        return self


# --------------------------------------------------------------------------- #
# serialisation helpers
# --------------------------------------------------------------------------- #

def _dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def _explain(state: AgentState) -> str:
    parts: list[str] = []
    if state.intent is not None:
        caps = state.intent.required_capabilities
        parts.append(
            f"Understood the request as {state.intent.goal!r}"
            + (f" (capabilities: {caps})" if caps else "")
            + "."
        )
    if state.selection is not None:
        parts.append(f"Candidate API(s): {state.selection.api_names()}.")
    if state.plan is not None and state.plan.steps:
        sequence = " -> ".join(step.operation_id for step in state.plan.steps)
        parts.append(f"PGPS produced a {state.plan.status.value} plan: {sequence}.")
    elif state.plan is not None:
        parts.append(
            f"PGPS could not produce an executable plan "
            f"(status: {state.plan.status.value})."
        )
    if state.clarification:
        parts.append(f"Clarification needed: {state.clarification}.")
    if state.execution is not None:
        outcome = "succeeded" if state.execution.succeeded else "failed"
        parts.append(
            f"Execution {outcome} across {len(state.execution.steps)} step(s)."
        )
    parts.append(f"Final stage: {state.stage.value}.")
    return " ".join(parts)


def _state_response(state: AgentState) -> dict[str, Any]:
    return {
        "request": state.request_text,
        "stage": state.stage.value,
        "intent": _dump(state.intent),
        "retrieved": _dump(state.rag) or _dump(state.retrieval),
        "selected_apis": _dump(state.selection),
        "parameters": _dump(state.parameters),
        "plan": _dump(state.plan),
        "validation": state.metadata.get("validation_report"),
        "execution": _dump(state.execution),
        "clarification": state.clarification,
        "explanation": _explain(state),
        "messages": state.messages,
        "errors": state.errors,
    }


def _api_summary(api: Any) -> dict[str, Any]:
    return {
        "name": api.name,
        "title": api.title,
        "version": api.version,
        "base_path": api.base_path,
        "operations": [op.operation_id for op in api.operations],
    }


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #

def workflow_dependency() -> CamaraWorkflow:
    return get_workflow()


def registry_dependency() -> CamaraRegistry:
    return get_registry()


def create_app() -> FastAPI:
    app = FastAPI(
        title="CAMARA-PGPS API",
        version="0.1.0",
        summary="Natural language -> validated CAMARA plan -> execution.",
    )

    @app.get(f"{API_PREFIX}/health", tags=["meta"])
    def health(
        registry: CamaraRegistry = Depends(registry_dependency),
    ) -> dict[str, Any]:
        return {
            "status": "ok",
            "apis": registry.api_names(),
            "execution_mode": get_settings().execution.mode,
        }

    @app.get(f"{API_PREFIX}/apis", tags=["registry"])
    def list_apis(
        registry: CamaraRegistry = Depends(registry_dependency),
    ) -> dict[str, Any]:
        return {"apis": [_api_summary(api) for api in registry.list_apis()]}

    @app.get(f"{API_PREFIX}/apis/{{api_name}}", tags=["registry"])
    def get_api(
        api_name: str,
        registry: CamaraRegistry = Depends(registry_dependency),
    ) -> dict[str, Any]:
        api = registry.get_api(api_name)
        if api is None:
            raise HTTPException(
                status_code=404,
                detail=f"unknown CAMARA API {api_name!r}; known: {registry.api_names()}",
            )
        payload = api.model_dump(mode="json")
        payload["capabilities"] = {
            op.operation_id: registry.capability_of(op.operation_id)
            for op in api.operations
        }
        return payload

    @app.post(f"{API_PREFIX}/run", tags=["pipeline"])
    def run(
        body: RunRequest,
        workflow: CamaraWorkflow = Depends(workflow_dependency),
    ) -> dict[str, Any]:
        state = workflow.run(
            body.request, arguments=body.arguments, known_facts=body.known_facts
        )
        return _state_response(state)

    @app.post(f"{API_PREFIX}/plan", tags=["pipeline"])
    def plan(
        body: RunRequest,
        workflow: CamaraWorkflow = Depends(workflow_dependency),
    ) -> dict[str, Any]:
        state = workflow.plan(
            body.request, arguments=body.arguments, known_facts=body.known_facts
        )
        return _state_response(state)

    @app.post(f"{API_PREFIX}/validate", tags=["pipeline"])
    def validate(
        body: PlanBasedRequest,
        workflow: CamaraWorkflow = Depends(workflow_dependency),
    ) -> dict[str, Any]:
        plan_obj = PlanResult.model_validate(body.plan) if body.plan is not None else None
        state = workflow.validate(
            request_text=body.request,
            plan=plan_obj,
            arguments=body.arguments,
            known_facts=body.known_facts,
        )
        return _state_response(state)

    @app.post(f"{API_PREFIX}/execute", tags=["pipeline"])
    def execute(
        body: PlanBasedRequest,
        workflow: CamaraWorkflow = Depends(workflow_dependency),
    ) -> dict[str, Any]:
        plan_obj = PlanResult.model_validate(body.plan) if body.plan is not None else None
        state = workflow.execute(
            request_text=body.request,
            plan=plan_obj,
            arguments=body.arguments,
            known_facts=body.known_facts,
        )
        return _state_response(state)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "camara_pgps.api.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
    )


if __name__ == "__main__":
    main()
