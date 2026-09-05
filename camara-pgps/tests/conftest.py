"""Test configuration: repo-root import path + offline stubs."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from camara_pgps.agents import IntentAgent, IntentDraft  # noqa: E402


class StubIntentParser:
    """Deterministic offline intent parser (the LLM-only agent needs a parser)."""

    def parse(self, text: str) -> IntentDraft:
        return IntentDraft(goal=text, action=None, entities={}, constraints=[])


def stub_intent_agent() -> IntentAgent:
    return IntentAgent(parser=StubIntentParser())
