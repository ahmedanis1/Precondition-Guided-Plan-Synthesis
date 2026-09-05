"""Application configuration.

Resolution order (highest precedence first):

1. Constructor arguments (used in tests).
2. Environment variables prefixed ``CAMARA_PGPS_`` (nested via ``__``).
3. A local ``.env`` file.
4. ``configs/config.yaml`` (or the file named by ``CAMARA_PGPS_CONFIG``).
5. The defaults declared on the models below.

See ``.env.example`` for the full list of supported variables.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class LLMSettings(BaseModel):
    """Chat-model configuration for the LLM agents.

    ``provider``: ``ollama`` (default, local) | ``anthropic`` | ``openai`` |
    ``none`` (no LLM). The model is chosen here, never in agent code - change
    ``model`` (e.g. ``qwen3:8b`` -> ``qwen3:14b``) to compare models.
    """

    provider: str = "ollama"
    model: str = "qwen3:8b"
    base_url: str = "http://localhost:11434"  # used when provider = "ollama"
    api_key: str | None = None
    temperature: float = 0.0
    timeout: float = 300.0
    """Read timeout (seconds) for a single chat call. Was previously hardcoded
    to 120s inside OllamaChatModel and ignored for the ollama provider."""
    keep_alive: str | int | None = "30m"
    """Ollama-only: how long to keep the model resident after a call
    (e.g. "30m", -1 to never unload, 0 to unload immediately). Ignored by
    the anthropic/openai providers."""
    max_retries: int = 1
    """Automatic retries on a timeout/connection error before giving up."""


class EmbeddingSettings(BaseModel):
    """Embedding backend configuration.

    ``provider`` selects the implementation:

    * ``sentence-transformers`` (default) - a local model
    * ``ollama`` - an Ollama server; ``model_name`` is the Ollama model tag
      (e.g. ``nomic-embed-text``) and ``base_url`` points at the server
    * ``hash`` - a deterministic, dependency-free embedder for offline use / tests
    """

    provider: str = "sentence-transformers"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    base_url: str = "http://localhost:11434"  # used when provider = "ollama"


class VectorStoreSettings(BaseModel):
    """ChromaDB vector-store configuration."""

    persist_directory: Path = Path("data/chroma")
    collection_name: str = "camara_specs"
    distance: str = "cosine"


class RetrievalSettings(BaseModel):
    """RAG knowledge base configuration.

    ``knowledge_dir`` is a directory of pre-chunked ``*.jsonl`` corpus files that
    ``run_ingestion`` embeds into ChromaDB.
    """

    knowledge_dir: Path = Path("data/camara_docs")


class GraphSettings(BaseModel):
    """Knowledge-graph persistence configuration."""

    path: Path = Path("data/knowledge_graph.gml")


class RegistrySettings(BaseModel):
    """Local CAMARA API registry (authoritative source of executable APIs).

    ``CamaraRegistry.from_settings()`` builds the live registry straight from
    the real CAMARA OpenAPI specs in ``specs_dir`` - there is no separate
    hand-maintained JSON file to keep in sync as APIs are added. ``path`` is
    kept only because ``CamaraRegistry.from_file()`` still loads the old
    milestone-3 fixture explicitly in tests/test_registry.py; it is no
    longer read by ``from_settings()``.
    """

    path: Path = Path("data/registry/camara_apis.json")
    specs_dir: Path = Path("data/camara_docs")


class ConditionsSettings(BaseModel):
    """PGPS precondition/postcondition data.

    Deliberately decoupled from :class:`RegistrySettings`: RAG / API
    Selection never load this, and PGPS never loads the registry. Each file
    under ``directory`` is named ``<api-name>.conditions.json`` and keyed by
    ``operation_id`` (verbatim from the OpenAPI ``operationId``, matching the
    ids RAG hands back in ``PlanRequest.candidate_operations``).
    """

    directory: Path = Path("data/conditions")


class CamaraSettings(BaseModel):
    """Target CAMARA gateway (real deployment or the bundled mock)."""

    base_url: str = "http://localhost:8080"
    token: str | None = None
    timeout_seconds: float = 30.0


class ExecutionSettings(BaseModel):
    """How validated plans are executed against the CAMARA API."""

    mode: str = "mock"  # mock | rest | mcp
    stop_on_error: bool = True
    dry_run: bool = False


class MockCamaraSettings(BaseModel):
    """Bundled mock CAMARA server bind address."""

    host: str = "0.0.0.0"
    port: int = 8080


class ServerSettings(BaseModel):
    """Application FastAPI server bind address."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


class MCPSettings(BaseModel):
    """Model Context Protocol integration toggle."""

    enabled: bool = False


class Settings(BaseSettings):
    """Root settings object for the whole application."""

    model_config = SettingsConfigDict(
        env_prefix="CAMARA_PGPS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        yaml_file=os.getenv("CAMARA_PGPS_CONFIG", str(DEFAULT_CONFIG_PATH)),
        yaml_file_encoding="utf-8",
    )

    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    registry: RegistrySettings = Field(default_factory=RegistrySettings)
    conditions: ConditionsSettings = Field(default_factory=ConditionsSettings)
    camara: CamaraSettings = Field(default_factory=CamaraSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    mock_camara: MockCamaraSettings = Field(default_factory=MockCamaraSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""

    return Settings()