"""Shared helpers for LLM-backed agents.

Chat-model construction and JSON extraction live here so agents never name a
provider or model. The model is chosen entirely by ``settings.llm`` - swap
``llm.model`` (e.g. ``qwen3:8b`` -> ``qwen3:14b``) or ``llm.provider`` without
touching agent code.

Default: a local Ollama server running ``qwen3:8b``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from camara_pgps.config import Settings

logger = logging.getLogger(__name__)

_RULE_ONLY_PROVIDERS = {"", "rule", "rules", "heuristic", "none", "off"}

_OLLAMA_ROLE = {"system": "system", "human": "user", "ai": "assistant", "user": "user"}


class _ChatResult:
    """A response object exposing ``.content`` (what :func:`chat_content` reads)."""

    __slots__ = ("content",)

    def __init__(self, content: str) -> None:
        self.content = content


class OllamaChatModel:
    """Minimal chat client for an Ollama server (``POST {base_url}/api/chat``).

    Implements just the ``invoke(messages)`` surface the agents use - each message
    is a LangChain message object (``.type`` / ``.content``) or a
    ``{"role", "content"}`` mapping. No streaming, no tools.
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        *,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout: float = 300.0,
        think: bool = False,
        keep_alive: str | int | None = "30m",
        max_retries: int = 1,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._timeout = timeout
        self._think = think
        self._keep_alive = keep_alive
        self._max_retries = max(0, max_retries)
        self._client = client

    def _http(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    @staticmethod
    def _to_role_content(message: Any) -> dict[str, str]:
        if isinstance(message, dict):
            return {
                "role": str(message.get("role", "user")),
                "content": str(message.get("content", "")),
            }
        role = _OLLAMA_ROLE.get(getattr(message, "type", "user"), "user")
        return {"role": role, "content": str(getattr(message, "content", message))}

    def invoke(self, messages: list[Any]) -> _ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._to_role_content(m) for m in messages],
            "stream": False,
            "think": self._think,
            "options": {"temperature": self._temperature},
        }
        if self._keep_alive is not None:
            # Keeps the model resident in Ollama between calls so a later
            # agent in the same pipeline run doesn't pay a cold-load penalty
            # (Ollama's own default is to unload after 5m of inactivity).
            payload["keep_alive"] = self._keep_alive

        import httpx

        attempt = 0
        last_exc: Exception | None = None
        while attempt <= self._max_retries:
            try:
                response = self._http().post(f"{self._base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                content = (data.get("message") or {}).get("content", "")
                return _ChatResult(str(content))
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                logger.warning(
                    "Ollama call to %s timed out/unreachable (attempt %d/%d): %s",
                    self.model,
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
                attempt += 1

        assert last_exc is not None
        raise last_exc


def extract_json(raw: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response (tolerates prose / fences)."""

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in model response")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("model response JSON is not an object")
    return data


def chat_content(response: Any) -> str:
    """Normalise a chat-model response to plain text."""

    content = getattr(response, "content", response)
    if isinstance(content, list):  # content-block style responses
        content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def build_chat_model(settings: Settings, *, purpose: str = "agent") -> Any | None:
    """Build the chat model from ``settings.llm``, or ``None`` for no LLM.

    ``provider = "ollama"`` (default) talks to a local Ollama server;
    ``anthropic`` / ``openai`` need an API key; anything in
    :data:`_RULE_ONLY_PROVIDERS` returns ``None``.
    """

    provider = (settings.llm.provider or "").lower()
    if provider in _RULE_ONLY_PROVIDERS:
        return None

    if provider == "ollama":
        # getattr(...) with defaults: these fields may not exist yet on every
        # deployed Settings.llm, and previously weren't wired through at all
        # (Ollama silently ignored any timeout you configured).
        return OllamaChatModel(
            model=settings.llm.model,
            base_url=settings.llm.base_url,
            temperature=settings.llm.temperature,
            timeout=float(getattr(settings.llm, "timeout", 300.0) or 300.0),
            keep_alive=getattr(settings.llm, "keep_alive", "30m"),
            max_retries=int(getattr(settings.llm, "max_retries", 1) or 0),
        )

    api_key = (
        settings.llm.api_key
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        logger.info("no LLM API key configured; %s will use rule-based logic", purpose)
        return None
    try:
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=settings.llm.model,
                temperature=settings.llm.temperature,
                timeout=60,
                stop=None,
                api_key=api_key,
            )
        if provider in {"openai", "azure-openai"}:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.llm.model,
                temperature=settings.llm.temperature,
                api_key=api_key,
            )
    except ImportError as exc:
        logger.warning(
            "LLM provider %r is not installed (%s); %s will use rule-based logic",
            provider,
            exc,
            purpose,
        )
        return None
    logger.warning("unsupported LLM provider %r; %s will use rule-based logic", provider, purpose)
    return None