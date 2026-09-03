"""The LangGraph workflow that wires every component into one agentic system.

    START -> intent -> rag -> selection -> parameters -> build_plan_request
          -> pgps -> (clarification | validation) -> (clarification | execution) -> END

The state schema is :class:`~camara_pgps.models.state.AgentState` from milestone 2.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from camara_pgps.config import Settings, get_settings
from camara_pgps.models.state import AgentState
from camara_pgps.workflow.nodes import (
    WorkflowNodes,
    route_after_pgps,
    route_after_validation,
)


class CamaraWorkflow:
    """Compiles and runs the end-to-end LangGraph pipeline."""

    def __init__(
        self,
        *,
        nodes: WorkflowNodes | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.nodes = nodes or WorkflowNodes(settings=self.settings)
        self._compiled: Any | None = None

    # -- build ---------------------------------------------------------------

    def compile(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        node = self.nodes
        graph = StateGraph(AgentState)

        # node names deliberately differ from AgentState field names (LangGraph
        # forbids a node sharing a name with a state key)
        graph.add_node("understand_intent", node.intent_node)
        graph.add_node("rag_retrieval", node.rag_node)
        graph.add_node("api_selection", node.selection_node)
        graph.add_node("parameter_extraction", node.parameters_node)
        graph.add_node("assemble_plan_request", node.build_plan_request_node)
        graph.add_node("pgps_solve", node.pgps_node)
        graph.add_node("clarify", node.clarification_node)
        graph.add_node("validate", node.validation_node)
        graph.add_node("execute", node.execution_node)

        graph.add_edge(START, "understand_intent")
        graph.add_edge("understand_intent", "rag_retrieval")
        graph.add_edge("rag_retrieval", "api_selection")
        graph.add_edge("api_selection", "parameter_extraction")
        graph.add_edge("parameter_extraction", "assemble_plan_request")
        graph.add_edge("assemble_plan_request", "pgps_solve")
        graph.add_conditional_edges(
            "pgps_solve",
            route_after_pgps,
            {"clarify": "clarify", "validate": "validate"},
        )
        graph.add_conditional_edges(
            "validate",
            route_after_validation,
            {"clarify": "clarify", "execute": "execute"},
        )
        graph.add_edge("clarify", END)
        graph.add_edge("execute", END)
        return graph.compile()

    @property
    def app(self) -> Any:
        if self._compiled is None:
            self._compiled = self.compile()
        return self._compiled

    # -- run ---------------------------------------------------------------

    def run(
        self,
        request_text: str,
        *,
        arguments: dict[str, dict[str, Any]] | None = None,
        known_facts: dict[str, Any] | None = None,
    ) -> AgentState:
        # auth is handled by the executor (bearer header); treat it as a given so
        # it never blocks planning
        facts = {"auth.token": True, **(known_facts or {})}
        initial = AgentState(
            request_text=request_text,
            supplied_arguments=dict(arguments or {}),
            facts=facts,
        )
        raw = self.app.invoke(initial)
        if isinstance(raw, AgentState):
            return raw
        return AgentState.model_validate(raw)

    def mermaid(self) -> str:
        """Return the compiled graph as a Mermaid diagram."""

        return str(self.app.get_graph().draw_mermaid())


@lru_cache(maxsize=1)
def get_workflow() -> CamaraWorkflow:
    """Return the process-wide workflow built from configured settings."""

    return CamaraWorkflow()


def run_request(
    request_text: str,
    *,
    arguments: dict[str, dict[str, Any]] | None = None,
    known_facts: dict[str, Any] | None = None,
) -> AgentState:
    """One-shot convenience wrapper over :meth:`CamaraWorkflow.run`."""

    return get_workflow().run(
        request_text, arguments=arguments, known_facts=known_facts
    )


__all__ = ["CamaraWorkflow", "get_workflow", "run_request"]
