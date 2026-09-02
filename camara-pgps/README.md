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
| 7 | Intent Understanding | ⬜ planned |
| 8 | API Selection | ⬜ planned |
| 9 | Parameter Extraction | ⬜ planned |
| 10 | PGPS Dependency Model | ⬜ planned |
| 11 | PGPS Planner | ⬜ planned |
| 12 | Plan Validator | ⬜ planned |
| 13 | Execution Layer | ⬜ planned |
| 14 | LangGraph Orchestration | ⬜ planned |
| 15 | FastAPI + CLI | ⬜ planned |
| 16 | End-to-End Integration | ⬜ planned |

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
│   ├── cli.py           console-script entry point
│   ├── models/          shared domain models (intent, api, plan, state, …)
│   ├── camara/          API registry, capability mapper & OpenAPI parser
│   └── retrieval/       ingestion, embeddings, ChromaDB retriever, RAG retrieval
├── configs/             YAML configuration defaults
├── data/                local artefacts (vector store, knowledge graph) - gitignored
│   └── registry/        camara_apis.json - the local API registry (tracked)
├── scripts/             environment / maintenance helpers
├── mock_camara/         bundled mock CAMARA gateway
├── tests/               test suite
├── .env.example         environment-variable template
├── environment.yml      conda environment definition
├── pyproject.toml       package metadata & dependencies
└── docker-compose.yml   optional dev stack (mock gateway + API)
```

## Tech stack

Pydantic · LangChain · LangGraph · FastAPI · httpx · PyYAML · ChromaDB ·
sentence-transformers · NetworkX · Model Context Protocol

## License

MIT
