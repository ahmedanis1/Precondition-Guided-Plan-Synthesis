"""Smoke tests for the project-foundation milestone."""

from __future__ import annotations

import pytest

import camara_pgps
from camara_pgps.__about__ import __version__
from camara_pgps.config import Settings, get_settings


def test_package_exposes_version() -> None:
    assert camara_pgps.__version__ == __version__
    assert __version__ == "0.1.0"


def test_settings_load_with_defaults() -> None:
    settings = Settings()
    assert settings.environment == "development"
    assert settings.llm.provider == "ollama"
    assert settings.llm.model == "qwen3:8b"
    assert settings.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.vector_store.collection_name == "camara_specs"
    assert settings.mock_camara.port == 8080
    assert settings.server.port == 8000
    assert settings.mcp.enabled is False


def test_env_overrides_nested_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAMARA_PGPS_LLM__MODEL", "claude-opus-5")
    monkeypatch.setenv("CAMARA_PGPS_SERVER__PORT", "9001")
    settings = Settings()
    assert settings.llm.model == "claude-opus-5"
    assert settings.server.port == 9001


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
