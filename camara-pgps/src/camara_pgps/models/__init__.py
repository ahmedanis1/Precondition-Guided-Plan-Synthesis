"""Domain models shared by every camara-pgps component.

Later milestones import their data structures from here rather than defining
their own. The dependency direction is:

    api  intent  retrieval          (no cross-imports)
        \\   |   /
          plan                       (intent + api + retrieval)
       execution                     (api)
          state                      (all of the above)
"""

from camara_pgps.models.api import (
    APIParameter,
    CamaraAPI,
    CamaraOperation,
    Condition,
    ConditionKind,
    HTTPMethod,
    ParameterLocation,
)
from camara_pgps.models.execution import (
    ExecutionResult,
    HTTPRequestSpec,
    StepExecution,
)
from camara_pgps.models.intent import Intent, IntentEntity
from camara_pgps.models.plan import (
    BindingSource,
    ParameterBinding,
    PlanRequest,
    PlanResult,
    PlanStatus,
    PlanStep,
)
from camara_pgps.models.retrieval import (
    RetrievalQuery,
    RetrievalResult,
    RetrievedChunk,
)
from camara_pgps.models.state import AgentState, PipelineStage

__all__ = [
    # api
    "APIParameter",
    "CamaraAPI",
    "CamaraOperation",
    "Condition",
    "ConditionKind",
    "HTTPMethod",
    "ParameterLocation",
    # intent
    "Intent",
    "IntentEntity",
    # retrieval
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievedChunk",
    # plan
    "BindingSource",
    "ParameterBinding",
    "PlanRequest",
    "PlanResult",
    "PlanStatus",
    "PlanStep",
    # execution
    "ExecutionResult",
    "HTTPRequestSpec",
    "StepExecution",
    # state
    "AgentState",
    "PipelineStage",
]
