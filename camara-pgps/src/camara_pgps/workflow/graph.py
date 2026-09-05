"""The LangGraph workflow that wires every component into one agentic system.

    START -> intent -> rag -> selection -> parameters -> build_plan_request
          -> pgps -> (clarification | validation) -> (clarification | execution) -> END

The state schema is :class:`~camara_pgps.models.state.AgentState` from milestone 2.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

from camara_pgps.config import Settings, get_settings
from camara_pgps.models.plan import PlanResult
from camara_pgps.models.state import AgentState
from camara_pgps.workflow.nodes import (
    WorkflowNodes,
    facts_from_arguments,
    route_after_pgps,
    route_after_validation,
)

_PLAN_NODES = (
    "intent_node",
    "rag_node",
    "selection_node",
    "parameters_node",
    "build_plan_request_node",
    "pgps_node",
)


def _merge_state(state: AgentState, update: dict[str, Any]) -> AgentState:
    """Apply a node's partial update to the state (append-only fields accumulate)."""

    new = state.model_copy()
    for key, value in update.items():
        if key in ("messages", "errors"):
            setattr(new, key, [*getattr(new, key), *value])
        else:
            setattr(new, key, value)
    return new


def _make_step_printer() -> Callable[[str, AgentState], None]:
    """Build a stateful ``on_step`` callback that prints one line per node.

    Only the messages that are new since the last call are shown, so a node
    that adds nothing (e.g. ``build_plan_request_node``) still gets a line
    ("stage changed, nothing new to report") instead of repeating the
    previous node's message.
    """

    seen = {"count": 0}

    def _print_step(node_name: str, state: AgentState) -> None:
        new_messages = state.messages[seen["count"] :]
        seen["count"] = len(state.messages)
        label = f"[{node_name}] stage={state.stage.value}"
        if new_messages:
            label += " :: " + " | ".join(
                str(m.get("content", m)) for m in new_messages
            )
        print(label)

    return _print_step


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
        on_step: Callable[[str, AgentState], None] | None = None,
        verbose: bool = False,
    ) -> AgentState:
        """Run the full pipeline: intent -> rag -> selection -> parameters ->
        build_plan_request -> pgps -> (clarify | validate -> execute).

        By default this is unchanged: one call, the final state comes back.
        Pass ``verbose=True`` (or your own ``on_step`` callback) to watch it
        progress node-by-node instead of the whole graph running silently and
        only surfacing something at the very end (or in a crash). ``on_step``
        is called once per completed node with the node's name and the state
        as it stands right after that node ran.
        """

        # auth is handled by the executor (bearer header); treat it as a given so
        # it never blocks planning
        facts = {"auth.token": True, **(known_facts or {})}
        initial = AgentState(
            request_text=request_text,
            supplied_arguments=dict(arguments or {}),
            facts=facts,
        )

        if on_step is None and not verbose:
            raw = self.app.invoke(initial)
            if isinstance(raw, AgentState):
                return raw
            return AgentState.model_validate(raw)

        callback = on_step or _make_step_printer()

        state = initial
        for update in self.app.stream(initial, stream_mode="updates"):
            for node_name, partial in update.items():
                state = _merge_state(state, partial)
                callback(node_name, state)
        return state

    # -- partial runs (used by the HTTP API) --------------------------------

    def _initial(
        self,
        request_text: str,
        arguments: dict[str, dict[str, Any]] | None,
        known_facts: dict[str, Any] | None,
    ) -> AgentState:
        return AgentState(
            request_text=request_text,
            supplied_arguments=dict(arguments or {}),
            facts={
                "auth.token": True,
                **facts_from_arguments(arguments),
                **(known_facts or {}),
            },
        )

    def _pipeline(
        self,
        state: AgentState,
        node_names: tuple[str, ...],
        *,
        on_step: Callable[[str, AgentState], None] | None = None,
        verbose: bool = False,
    ) -> AgentState:
        callback = on_step or (_make_step_printer() if verbose else None)
        for name in node_names:
            state = _merge_state(state, getattr(self.nodes, name)(state))
            if callback is not None:
                callback(name, state)
        return state

    def plan(
        self,
        request_text: str,
        *,
        arguments: dict[str, dict[str, Any]] | None = None,
        known_facts: dict[str, Any] | None = None,
        on_step: Callable[[str, AgentState], None] | None = None,
        verbose: bool = False,
    ) -> AgentState:
        """Run intent -> RAG -> selection -> parameters -> PGPS (no validation/execution)."""

        state = self._initial(request_text, arguments, known_facts)
        return self._pipeline(state, _PLAN_NODES, on_step=on_step, verbose=verbose)

    def validate(
        self,
        *,
        request_text: str | None = None,
        plan: PlanResult | None = None,
        arguments: dict[str, dict[str, Any]] | None = None,
        known_facts: dict[str, Any] | None = None,
        on_step: Callable[[str, AgentState], None] | None = None,
        verbose: bool = False,
    ) -> AgentState:
        """Validate a supplied plan, or plan a request and then validate it."""

        if plan is not None:
            state = self._initial(request_text or "", arguments, known_facts)
            state.plan = plan
            return self._pipeline(
                state, ("validation_node",), on_step=on_step, verbose=verbose
            )
        state = self.plan(
            request_text or "",
            arguments=arguments,
            known_facts=known_facts,
            on_step=on_step,
            verbose=verbose,
        )
        step = "clarification_node" if route_after_pgps(state) == "clarify" else "validation_node"
        return self._pipeline(state, (step,), on_step=on_step, verbose=verbose)

    def execute(
        self,
        *,
        request_text: str | None = None,
        plan: PlanResult | None = None,
        arguments: dict[str, dict[str, Any]] | None = None,
        known_facts: dict[str, Any] | None = None,
        on_step: Callable[[str, AgentState], None] | None = None,
        verbose: bool = False,
    ) -> AgentState:
        """Execute a supplied validated plan, or run the full pipeline for a request."""

        if plan is None:
            return self.run(
                request_text or "",
                arguments=arguments,
                known_facts=known_facts,
                on_step=on_step,
                verbose=verbose,
            )
        state = self._initial(request_text or "", arguments, known_facts)
        state.plan = plan
        state = self._pipeline(
            state, ("validation_node",), on_step=on_step, verbose=verbose
        )
        follow = (
            "execution_node"
            if route_after_validation(state) == "execute"
            else "clarification_node"
        )
        return self._pipeline(state, (follow,), on_step=on_step, verbose=verbose)

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