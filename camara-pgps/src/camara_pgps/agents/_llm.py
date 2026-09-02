"""Shared helpers for LLM-backed agents.

Keeps chat-model construction and JSON extraction in one place so every agent
degrades to a deterministic rule-based path the same way when no LLM (or no API
key, or no provider package) is available.
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
    """Build a LangChain chat model from settings, or ``None`` to use rule-based logic."""

    provider = (settings.llm.provider or "").lower()
    if provider in _RULE_ONLY_PROVIDERS:
        return None
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
