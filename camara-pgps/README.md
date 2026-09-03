# camara-pgps

**Precondition-Guided Plan Synthesis: Agentic RAG for CAMARA API Orchestration**

An agentic layer over [CAMARA](https://camaraproject.org/) network APIs. It takes
a natural-language intent, retrieves the relevant CAMARA specifications with RAG,
selects operations and extracts parameters, and then synthesises a **plan** whose
step order is driven by the operations' *preconditions* and *postconditions*. The
plan is validated and executed against a CAMARA gateway (or the bundled mock).

> This repository accompanies a master's thesis and is built milestone by
> milestone. The section below tracks what is implemented.

## Status

| # | Milestone | State |
|---|-----------|-------|
| 1 | Project Foundation | ✅ done |
| 2 | Domain Models | ✅ done |
| 3 | CAMARA API Registry | ✅ done |
| 4 | CAMARA OpenAPI Parser | ✅ done |
| 5 | CAMARA Knowledge Ingestion Pipeline | ✅ done |
| 6 | RAG Retrieval | ✅ done |
| 7 | Intent Understanding | ✅ done |
| 8 | API Selection | ✅ done |
| 9 | Parameter Extraction | ✅ done |
| 10 | PGPS Dependency Model | ✅ done |
| 11 | PGPS Planner | ✅ done |
| 12 | Plan Validator | ✅ done |
| 13 | Execution Layer | ✅ done |
| 14 | Mock CAMARA Services | ✅ done |
| 15 | LangGraph Orchestration | ✅ done |
| 16 | Clarification / Repair Agent | ✅ done |
| 17 | FastAPI | ✅ done |
| 18 | CLI | ✅ done |
| 19 | Explainability & Provenance | ✅ done |
| 20 | End-to-End Integration | ⬜ planned |

## Requirements

- Python 3.11+
- [conda](https://docs.conda.io/) (Miniconda/Miniforge) for environment management

## Setup

```bash
conda env create -f environment.yml
conda activate camara-pgps

# or, using the helper script
./scripts/create_env.ps1     # Windows / PowerShell
./scripts/create_env.sh      # Linux / macOS
```

Conda provides the interpreter; every Python dependency is declared in
`pyproject.toml` and installed via `pip -e .[dev]`.

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

Verify the install:

```bash
camara-pgps --version
camara-pgps --show-config
pytest
```

Build the RAG knowledge base, then use the CLI or the services:

```bash
python -m camara_pgps ingest-camara --reset
python -m camara_pgps run --request "Improve the network quality for my video call"
python -m camara_pgps list-apis --api quality-on-demand
python -m camara_pgps show-config

mock-camara                     # mock CAMARA gateway  -> http://localhost:8080/docs
camara-pgps-api                 # the CAMARA-PGPS API  -> http://localhost:8000/docs
curl -s localhost:8000/api/v1/run -H 'content-type: application/json' \
  -d '{"request": "Improve the network quality for my video call."}'
```

## Configuration

Settings resolve in this order (highest precedence first):

1. Constructor arguments (tests)
2. `CAMARA_PGPS_*` environment variables (nested keys use `__`)
3. `.env`
4. `configs/config.yaml` (or the file named by `CAMARA_PGPS_CONFIG`)
5. Code defaults in `camara_pgps.config`

See `.env.example` for the full list of variables.

## Repository layout

```
camara-pgps/
├── src/camara_pgps/     application package
│   ├── __about__.py     version
│   ├── config.py        pydantic-settings configuration
│   ├── cli.py           `camara-pgps` CLI (run / list-apis / ingest-camara / show-config)
│   ├── models/          shared domain models (intent, api, plan, state, …)
│   ├── camara/          API registry, capability mapper & OpenAPI parser
│   ├── retrieval/       ingestion, embeddings, ChromaDB retriever, RAG retrieval
│   ├── agents/          LLM agents (intent, API selection, parameter extraction, repair)
│   ├── pgps/            condition model, dependency graph, deterministic solver, planner & explainer
│   ├── validator/       plan validation - structural (PGPS) + API/payload layers
│   ├── execution/       APIExecutor + mock / REST / MCP backends
│   ├── workflow/        LangGraph pipeline wiring every component into AgentState
│   └── api/             FastAPI app (/api/v1/run, /plan, /validate, /execute, /apis)
├── configs/             YAML configuration defaults
├── data/                local artefacts (vector store, knowledge graph) - gitignored
│   └── registry/        camara_apis.json - the local API registry (tracked)
├── scripts/             run_agent.py, ingest_camara.py, environment helpers
├── mock_camara/         runnable mock CAMARA gateway (main.py + services/)
├── docs/                architecture.md, adding-a-camara-api.md
├── tests/               test suite
├── .env.example         environment-variable template
├── environment.yml      conda environment definition
├── pyproject.toml       package metadata & dependencies
└── docker-compose.yml   optional dev stack (mock gateway + API)
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — layers, the PGPS core, safety boundaries, the offline/test posture
- [`docs/adding-a-camara-api.md`](docs/adding-a-camara-api.md) — how to register a new CAMARA API and its conditions

Every run carries a **graph-derived explanation** (`provenance` in the API/CLI
output): Goal → Selected APIs → Required facts → Produced facts → Dependencies →
Execution order, built from the PGPS dependency graph (`source =
"pgps-dependency-graph"`), never an LLM.

## Tech stack

Pydantic · LangChain · LangGraph · FastAPI · httpx · PyYAML · ChromaDB ·
sentence-transformers · NetworkX · jsonschema · Model Context Protocol

## License

MIT
