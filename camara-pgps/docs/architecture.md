# Architecture

CAMARA-PGPS turns a natural-language request into a **validated, dependency-ordered
plan of CAMARA API calls** and executes it. The design keeps three concerns
strictly separate:

| Concern | Who owns it | Deterministic? |
|---|---|---|
| *What the user wants* | the LLM agents (`agents/`) | no (LLM, with rule-based fallbacks) |
| *What operations exist & how they depend on each other* | the registry + PGPS (`camara/`, `pgps/`) | **yes** |
| *Whether a plan is runnable* | the validator (`validator/`) | **yes** |

The LLM proposes; the registry and PGPS dispose.

## Pipeline

```
request text
   │
   ▼
IntentAgent ─────────► Intent            (agents/intent_agent.py)
   │                    goal, entities, required_capabilities, known/missing facts
   ▼
RagRetriever ────────► RagContext        (retrieval/rag.py)
   │                    doc chunks + candidate operations/params/conditions
   ▼
APISelectionAgent ──► APISelection       (agents/api_selection_agent.py)
   │                    candidate APIs/operations, EVERY name verified in the registry
   ▼
ParameterExtractionAgent ► ParameterExtraction   (agents/parameter_agent.py)
   │                    grounded parameter values; unresolved → missing_information
   ▼
build_plan_request  ─► PlanRequest       (workflow/nodes.py)
   ▼
PGPSPlanner ─────────► PlanResult        (pgps/planner.py → pgps/solver.py)
   │                    deterministic backward chaining over the fact graph
   ├── missing info / infeasible ──► RepairAgent ──► clarification question ──► END
   ▼
PlanValidator ──────► ValidationReport   (validator/) — structural + payload layers
   ▼
APIExecutor ────────► ExecutionResult    (execution/) — mock | rest | mcp
   ▼
END
```

`workflow/graph.py` wires this as a LangGraph `StateGraph`; the shared state is
`AgentState` (`models/state.py`). `workflow/report.py` renders a finished run for
the HTTP API (`api/`) and the CLI (`cli.py`).

## Layers (import direction is downward only)

```
api/  cli/  workflow/            entry points & orchestration
        │
        ▼
agents/   validator/   execution/    LLM agents & downstream processing
        │
        ▼
retrieval/     pgps/                  RAG knowledge base & the deterministic planner
        │        │
        ▼        ▼
      camara/                          registry, capability mapper, OpenAPI parser
        │
        ▼
      models/    config/               shared pydantic models & settings
```

`models/` never imports from any higher layer. `AgentState` therefore holds the
rich agent outputs (`rag`, `selection`, `parameters`) as `Any`.

## The PGPS core (`pgps/`)

* `conditions.py` — `Fact`, `FactSet`, `Precondition`/`Postcondition`,
  `OperationConditions`, and `ConditionModel` (a registry-wide `fact → producers /
  consumers` index).
* `dependency_graph.py` — a bipartite `operation ↔ fact` NetworkX graph;
  `operation_graph()` projects it to an operation DAG; `topological_order()` and
  `resolve_order()` do ordering and cycle/impossibility detection.
* `solver.py` — `PGPSSolver`: deterministic backward chaining. Invariant:
  `preconditions(op) ⊆ initial_facts ∪ postconditions(earlier ops)`.
* `planner.py` — `PGPSPlanner`: maps `PlanRequest` → `PlanResult` (real
  `method`/`path` from the registry, `depends_on`, `requires`/`produces`).
* `explain.py` — `explain_plan()`: a `PlanExplanation` derived **only** from the
  fact graph (`source = "pgps-dependency-graph"`), never from an LLM.

## Safety boundaries

* **API selection** — `_canonical_api()` / `_canonical_operation()` accept a name
  only if the registry recognises it; unknowns become `RejectedSuggestion`.
* **Parameter extraction** — a value is bound only if it is grounded in the
  request or in prior-step facts; LLM output is filtered by `_ground_filter`.
  Missing required values are reported, never invented.
* **Repair agent** — turns missing names into one question; `MissingItem` has no
  `value` field and an "I'll assume…" style LLM reply is discarded.
* **Execution** — refuses a plan whose status is `INVALID` / `INFEASIBLE`, and
  never mutates the plan.

## Offline / test posture

Every LLM-backed component has a deterministic fallback and accepts injected
collaborators:

| Component | Real backend | Fallback | Inject via |
|---|---|---|---|
| embeddings | `sentence-transformers` | `HashingEmbedder` (`embedding.provider = hash`) | `Retriever(embedder=…)` |
| intent / selection / params / repair | `ChatAnthropic` (extra `llm`) | rule-based | `IntentAgent(llm=…)` etc. |
| execution | `RESTExecutor` / `MCPExecutor` | `MockExecutor` (`execution.mode = mock`) | `build_executor(mode=…)` |
| LangGraph nodes | — | — | `WorkflowNodes(registry=…, rag_retriever=…, executor=…)` |
| HTTP API | — | — | `app.dependency_overrides[workflow_dependency]` |

`WorkflowNodes` never contains per-API or per-operation branching; capability
families come from `registry.capabilities()`. The one heuristic seed is
`agents/intent_agent._FAMILY_SYNONYMS` — keyword hints for the *offline* rule-based
parser only; the LLM path and the registry-derived keywords do not use it.
