"""Tests for the shared LLM helpers - Ollama chat model + provider routing."""

from __future__ import annotations

import httpx

from camara_pgps.agents._llm import OllamaChatModel, build_chat_model
from camara_pgps.config import Settings


def test_ollama_chat_model_maps_messages_and_reads_content() -> None:
    import json

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.read()))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "hi"}})

    class _Sys:
        type = "system"
        content = "be terse"

    class _Human:
        type = "human"
        content = "hello"

    model = OllamaChatModel(
        "qwen3:8b",
        base_url="http://ollama:11434",
        temperature=0.2,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = model.invoke([_Sys(), _Human(), {"role": "assistant", "content": "prev"}])

    assert result.content == "hi"
    body = seen[0]
    assert body["model"] == "qwen3:8b"
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0.2
    assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant"]
    assert body["messages"][1]["content"] == "hello"


def test_build_chat_model_defaults_to_ollama_qwen3_8b() -> None:
    model = build_chat_model(Settings())
    assert isinstance(model, OllamaChatModel)
    assert model.model == "qwen3:8b"


def test_build_chat_model_honours_a_different_model() -> None:
    settings = Settings(llm={"provider": "ollama", "model": "qwen3:14b"})  # type: ignore[arg-type]
    model = build_chat_model(settings)
    assert isinstance(model, OllamaChatModel)
    assert model.model == "qwen3:14b"


def test_build_chat_model_none_provider_returns_none() -> None:
    assert build_chat_model(Settings(llm={"provider": "none"})) is None  # type: ignore[arg-type]
