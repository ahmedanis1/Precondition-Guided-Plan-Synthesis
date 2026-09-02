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
    """Large-language-model provider configuration."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-5"
    api_key: str | None = None
    temperature: float = 0.0


class EmbeddingSettings(BaseModel):
    """sentence-transformers embedding configuration."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"


class VectorStoreSettings(BaseModel):
    """ChromaDB vector-store configuration."""

    persist_directory: Path = Path("data/chroma")
    collection_name: str = "camara_specs"


class GraphSettings(BaseModel):
    """Knowledge-graph persistence configuration."""

    path: Path = Path("data/knowledge_graph.gml")


class RegistrySettings(BaseModel):
    """Local CAMARA API registry (authoritative source of executable APIs)."""

    path: Path = Path("data/registry/camara_apis.json")


class CamaraSettings(BaseModel):
    """Target CAMARA gateway (real deployment or the bundled mock)."""

    base_url: str = "http://localhost:8080"
    token: str | None = None


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
    graph: GraphSettings = Field(default_factory=GraphSettings)
    registry: RegistrySettings = Field(default_factory=RegistrySettings)
    camara: CamaraSettings = Field(default_factory=CamaraSettings)
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
