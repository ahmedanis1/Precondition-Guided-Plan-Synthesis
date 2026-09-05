# Architecture

CAMARA-PGPS turns a natural-language request into a **validated, dependency-ordered
plan of CAMARA API calls** and executes it. The design keeps three concerns
strictly separate:

| Concern | Who owns it | Deterministic? |
|---|---|---|
| *What the user wants* | the LLM agents (`agents/`) | no (LLM, with rule-based fallbacks) |
| *What operations exist* | the registry (`camara/`) | **yes** |
| *How operations depend on each other (pre/postconditions)* | PGPS (`pgps/`), reading `data/conditions/` | **yes** |
| *Whether a plan is runnable* | the validator (`validator/`) | **yes** |

The LLM proposes; PGPS and the validator dispose. The registry only answers
"does this operation exist, and what does it look like" - it carries no
dependency information at all. PGPS's dependency knowledge always comes from
`data/conditions/` (or an explicitly-injected `ConditionModel`), completely
independent of the registry; the registry is consulted only for a real
`method`/`path` when PGPS chains in an operation that wasn't itself an
API-Selection candidate (see "The PGPS core" below).

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
  `OperationConditions`, and `ConditionModel` (a `fact → producers / consumers`
  index built from `data/conditions/`, independent of `CamaraRegistry`). A
  `Postcondition` can optionally declare `produces_from: {fact_key:
  response_field}`, so the executor can extract a real value out of *any*
  operation's response generically, from data alone — see
  `execution/executor.py::_extract_facts()` and
  `docs/design-decisions.md` §4.3.
* `parameter_facts.py` — the single, canonical parameter-name ↔ fact-key
  vocabulary (e.g. `sessionId` ↔ `qos.session.id`), shared by parameter
  extraction, the planner, and the executor so a fact recognised in one is
  always recognised in all of them — see `docs/design-decisions.md` §4.4.
* `dependency_graph.py` — a bipartite `operation ↔ fact` NetworkX graph;
  `operation_graph()` projects it to an operation DAG; `topological_order()` and
  `resolve_order()` do ordering and cycle/impossibility detection.
* `solver.py` — `PGPSSolver`: deterministic backward chaining. Invariant:
  `preconditions(op) ⊆ initial_facts ∪ postconditions(earlier ops)`.
* `planner.py` — `PGPSPlanner`: maps `PlanRequest` → `PlanResult`. Goal facts
  are the candidate operations' own `requires` (so a pure "sink" operation
  that produces nothing, e.g. `extendQosSessionDuration`, still drives
  backward chaining); the solver searches the *whole* `ConditionModel` for a
  producer, not just the API-Selection candidates, so PGPS can genuinely
  discover on its own that, say, extending a session first needs a
  `retrieveSessionsByDevice`-style call. Every candidate operation is always
  appended to the resulting plan once its preconditions are satisfiable - a
  sink operation is never "selected" by the solver's produces-a-fact
  mechanism, since it has nothing to produce. Each `PlanStep`'s real
  `method`/`path` comes from the matching candidate `CamaraOperation` when
  there is one, otherwise from a registry PGPS builds lazily purely for that
  structural lookup (`depends_on`, `requires`/`produces` always come from
  `ConditionModel`, never the registry). `_build_steps()` also records a
  `ParameterBinding(source=PRIOR_STEP)` on a step for every parameter whose
  fact is supplied by an earlier step in the same plan, via
  `parameter_facts.py` - this is what lets the validator and executor
  recognise a value like `sessionId` as already resolved instead of asking
  the user for something PGPS's own plan already arranged to fetch.

  Both `PGPSPlanner` and `PlanValidator` accept an explicit `condition_model=`
  and *must* be given one built from `data/conditions/` (e.g.
  `build_condition_model(settings=...)`) in production. Each also has a
  legacy `registry=`-only construction path, kept for backward compatibility
  with an earlier milestone, that silently builds an empty condition model
  from the registry's own (always-empty, for real operations)
  pre/postconditions instead - see `docs/design-decisions.md` §5 (#10, #13)
  for what happened when production code used that path by mistake.
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

## Model selection

Agents never name a provider or model. `settings.llm` decides:

| Setting | Default | Purpose |
|---|---|---|
| `llm.provider` | `ollama` | `ollama` (local) \| `anthropic` \| `openai` \| `none` |
| `llm.model` | `qwen3:8b` | change to `qwen3:14b` (or any tag) to compare models |
| `llm.base_url` | `http://localhost:11434` | the Ollama server |

`agents/_llm.build_chat_model(settings)` is the single construction point;
`OllamaChatModel` is a dependency-free HTTP client for `POST /api/chat`. The
thesis can benchmark models by changing `llm.model` only.

## Offline / test posture

| Component | Real backend | Test seam |
|---|---|---|
| embeddings | `sentence-transformers` / `ollama` (`embedding.provider`) | `HashingEmbedder` (`embedding.provider = hash`) / `Retriever(embedder=…)` |
| intent / repair / selection / params | Ollama / Anthropic / OpenAI via `build_chat_model` | `IntentAgent(parser=…)`, `RepairAgent(registry=…)` (rule-based), `APISelectionAgent(rag=…)` |
| execution | `RESTExecutor` / `MCPExecutor` | `MockExecutor` (`execution.mode = mock`) / `build_executor(mode=…)` |
| LangGraph nodes | — | `WorkflowNodes(registry=…, rag_retriever=…, executor=…, intent_agent=…, repair_agent=…)` — all lazy, so the graph compiles without an LLM |
| HTTP API | — | `app.dependency_overrides[workflow_dependency]` |

`WorkflowNodes` contains no per-API or per-operation branching. The intent agent
is LLM-only (a stub parser is injected for tests); `RepairAgent` and the
parameter/selection agents keep deterministic rule-based paths.
